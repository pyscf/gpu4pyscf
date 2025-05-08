#include <cuda.h>
#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks.cu"


__device__ static
void _rys_j_0_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_0_0 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    gout_0_0 += fac * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+0, gout_0_0*dm[kl_pair0+0]);
            atomicAdd(vj+kl_pair0+0, gout_0_0*dm[ij_pair0+0]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
static void rys_j_0_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_0_0(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_1_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_1_0 = 0.;
        double gout_2_0 = 0.;
        double gout_3_0 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout_1_0 += fac * 1 * trr_10z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout_2_0 += fac * trr_10y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_3_0 += trr_10x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+1, gout_1_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+2, gout_2_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+3, gout_3_0*dm[kl_pair0+0]);
            atomicAdd(vj+kl_pair0+0, gout_1_0*dm[ij_pair0+1] + gout_2_0*dm[ij_pair0+2] + gout_3_0*dm[ij_pair0+3]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
static void rys_j_1_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_1_0(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_1_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_1_1 = 0.;
        double gout_1_2 = 0.;
        double gout_1_3 = 0.;
        double gout_2_1 = 0.;
        double gout_2_2 = 0.;
        double gout_2_3 = 0.;
        double gout_3_1 = 0.;
        double gout_3_2 = 0.;
        double gout_3_3 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpz = zqc + zpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout_1_1 += fac * 1 * trr_11z;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    gout_1_2 += fac * trr_01y * trr_10z;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    gout_1_3 += trr_01x * 1 * trr_10z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_01z = cpz * wt;
                    gout_2_1 += fac * trr_10y * trr_01z;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout_2_2 += fac * trr_11y * wt;
                    gout_2_3 += trr_01x * trr_10y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_3_1 += trr_10x * 1 * trr_01z;
                    gout_3_2 += trr_10x * trr_01y * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    gout_3_3 += trr_11x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+1, gout_1_1*dm[kl_pair0+1] + gout_1_2*dm[kl_pair0+2] + gout_1_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+2, gout_2_1*dm[kl_pair0+1] + gout_2_2*dm[kl_pair0+2] + gout_2_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+3, gout_3_1*dm[kl_pair0+1] + gout_3_2*dm[kl_pair0+2] + gout_3_3*dm[kl_pair0+3]);
            atomicAdd(vj+kl_pair0+1, gout_1_1*dm[ij_pair0+1] + gout_2_1*dm[ij_pair0+2] + gout_3_1*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+2, gout_1_2*dm[ij_pair0+1] + gout_2_2*dm[ij_pair0+2] + gout_3_2*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+3, gout_1_3*dm[ij_pair0+1] + gout_2_3*dm[ij_pair0+2] + gout_3_3*dm[ij_pair0+3]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
static void rys_j_1_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_1_1(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_1_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_1_1 = 0.;
        double gout_1_2 = 0.;
        double gout_1_3 = 0.;
        double gout_1_4 = 0.;
        double gout_1_5 = 0.;
        double gout_1_6 = 0.;
        double gout_1_7 = 0.;
        double gout_1_8 = 0.;
        double gout_1_9 = 0.;
        double gout_2_1 = 0.;
        double gout_2_2 = 0.;
        double gout_2_3 = 0.;
        double gout_2_4 = 0.;
        double gout_2_5 = 0.;
        double gout_2_6 = 0.;
        double gout_2_7 = 0.;
        double gout_2_8 = 0.;
        double gout_2_9 = 0.;
        double gout_3_1 = 0.;
        double gout_3_2 = 0.;
        double gout_3_3 = 0.;
        double gout_3_4 = 0.;
        double gout_3_5 = 0.;
        double gout_3_6 = 0.;
        double gout_3_7 = 0.;
        double gout_3_8 = 0.;
        double gout_3_9 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout_1_1 += fac * 1 * trr_11z;
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    gout_1_2 += fac * 1 * trr_12z;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    gout_1_3 += fac * trr_01y * trr_10z;
                    gout_1_4 += fac * trr_01y * trr_11z;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    gout_1_5 += fac * trr_02y * trr_10z;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    gout_1_6 += trr_01x * 1 * trr_10z;
                    gout_1_7 += trr_01x * 1 * trr_11z;
                    gout_1_8 += trr_01x * trr_01y * trr_10z;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    gout_1_9 += trr_02x * 1 * trr_10z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout_2_1 += fac * trr_10y * trr_01z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout_2_2 += fac * trr_10y * trr_02z;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout_2_3 += fac * trr_11y * wt;
                    gout_2_4 += fac * trr_11y * trr_01z;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    gout_2_5 += fac * trr_12y * wt;
                    gout_2_6 += trr_01x * trr_10y * wt;
                    gout_2_7 += trr_01x * trr_10y * trr_01z;
                    gout_2_8 += trr_01x * trr_11y * wt;
                    gout_2_9 += trr_02x * trr_10y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_3_1 += trr_10x * 1 * trr_01z;
                    gout_3_2 += trr_10x * 1 * trr_02z;
                    gout_3_3 += trr_10x * trr_01y * wt;
                    gout_3_4 += trr_10x * trr_01y * trr_01z;
                    gout_3_5 += trr_10x * trr_02y * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    gout_3_6 += trr_11x * 1 * wt;
                    gout_3_7 += trr_11x * 1 * trr_01z;
                    gout_3_8 += trr_11x * trr_01y * wt;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    gout_3_9 += trr_12x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+1, gout_1_1*dm[kl_pair0+1] + gout_1_2*dm[kl_pair0+2] + gout_1_3*dm[kl_pair0+3] + gout_1_4*dm[kl_pair0+4] + gout_1_5*dm[kl_pair0+5] + gout_1_6*dm[kl_pair0+6] + gout_1_7*dm[kl_pair0+7] + gout_1_8*dm[kl_pair0+8] + gout_1_9*dm[kl_pair0+9]);
            atomicAdd(vj+ij_pair0+2, gout_2_1*dm[kl_pair0+1] + gout_2_2*dm[kl_pair0+2] + gout_2_3*dm[kl_pair0+3] + gout_2_4*dm[kl_pair0+4] + gout_2_5*dm[kl_pair0+5] + gout_2_6*dm[kl_pair0+6] + gout_2_7*dm[kl_pair0+7] + gout_2_8*dm[kl_pair0+8] + gout_2_9*dm[kl_pair0+9]);
            atomicAdd(vj+ij_pair0+3, gout_3_1*dm[kl_pair0+1] + gout_3_2*dm[kl_pair0+2] + gout_3_3*dm[kl_pair0+3] + gout_3_4*dm[kl_pair0+4] + gout_3_5*dm[kl_pair0+5] + gout_3_6*dm[kl_pair0+6] + gout_3_7*dm[kl_pair0+7] + gout_3_8*dm[kl_pair0+8] + gout_3_9*dm[kl_pair0+9]);
            atomicAdd(vj+kl_pair0+1, gout_1_1*dm[ij_pair0+1] + gout_2_1*dm[ij_pair0+2] + gout_3_1*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+2, gout_1_2*dm[ij_pair0+1] + gout_2_2*dm[ij_pair0+2] + gout_3_2*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+3, gout_1_3*dm[ij_pair0+1] + gout_2_3*dm[ij_pair0+2] + gout_3_3*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+4, gout_1_4*dm[ij_pair0+1] + gout_2_4*dm[ij_pair0+2] + gout_3_4*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+5, gout_1_5*dm[ij_pair0+1] + gout_2_5*dm[ij_pair0+2] + gout_3_5*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+6, gout_1_6*dm[ij_pair0+1] + gout_2_6*dm[ij_pair0+2] + gout_3_6*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+7, gout_1_7*dm[ij_pair0+1] + gout_2_7*dm[ij_pair0+2] + gout_3_7*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+8, gout_1_8*dm[ij_pair0+1] + gout_2_8*dm[ij_pair0+2] + gout_3_8*dm[ij_pair0+3]);
            atomicAdd(vj+kl_pair0+9, gout_1_9*dm[ij_pair0+1] + gout_2_9*dm[ij_pair0+2] + gout_3_9*dm[ij_pair0+3]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
__global__
static void rys_j_1_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_1_2(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_2_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_1_0 = 0.;
        double gout_2_0 = 0.;
        double gout_3_0 = 0.;
        double gout_4_0 = 0.;
        double gout_5_0 = 0.;
        double gout_6_0 = 0.;
        double gout_7_0 = 0.;
        double gout_8_0 = 0.;
        double gout_9_0 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout_1_0 += fac * 1 * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout_2_0 += fac * 1 * trr_20z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout_3_0 += fac * trr_10y * wt;
                    gout_4_0 += fac * trr_10y * trr_10z;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout_5_0 += fac * trr_20y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_6_0 += trr_10x * 1 * wt;
                    gout_7_0 += trr_10x * 1 * trr_10z;
                    gout_8_0 += trr_10x * trr_10y * wt;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    gout_9_0 += trr_20x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+1, gout_1_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+2, gout_2_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+3, gout_3_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+4, gout_4_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+5, gout_5_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+6, gout_6_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+7, gout_7_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+8, gout_8_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+9, gout_9_0*dm[kl_pair0+0]);
            atomicAdd(vj+kl_pair0+0, gout_1_0*dm[ij_pair0+1] + gout_2_0*dm[ij_pair0+2] + gout_3_0*dm[ij_pair0+3] + gout_4_0*dm[ij_pair0+4] + gout_5_0*dm[ij_pair0+5] + gout_6_0*dm[ij_pair0+6] + gout_7_0*dm[ij_pair0+7] + gout_8_0*dm[ij_pair0+8] + gout_9_0*dm[ij_pair0+9]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
#if CUDA_VERSION >= 12040
__global__ __maxnreg__(128)
#else
__global__
#endif
static void rys_j_2_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_2_0(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_2_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_1_1 = 0.;
        double gout_1_2 = 0.;
        double gout_1_3 = 0.;
        double gout_2_1 = 0.;
        double gout_2_2 = 0.;
        double gout_2_3 = 0.;
        double gout_3_1 = 0.;
        double gout_3_2 = 0.;
        double gout_3_3 = 0.;
        double gout_4_1 = 0.;
        double gout_4_2 = 0.;
        double gout_4_3 = 0.;
        double gout_5_1 = 0.;
        double gout_5_2 = 0.;
        double gout_5_3 = 0.;
        double gout_6_1 = 0.;
        double gout_6_2 = 0.;
        double gout_6_3 = 0.;
        double gout_7_1 = 0.;
        double gout_7_2 = 0.;
        double gout_7_3 = 0.;
        double gout_8_1 = 0.;
        double gout_8_2 = 0.;
        double gout_8_3 = 0.;
        double gout_9_1 = 0.;
        double gout_9_2 = 0.;
        double gout_9_3 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpz = zqc + zpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout_1_1 += fac * 1 * trr_11z;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    gout_1_2 += fac * trr_01y * trr_10z;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    gout_1_3 += trr_01x * 1 * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout_2_1 += fac * 1 * trr_21z;
                    gout_2_2 += fac * trr_01y * trr_20z;
                    gout_2_3 += trr_01x * 1 * trr_20z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_01z = cpz * wt;
                    gout_3_1 += fac * trr_10y * trr_01z;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout_3_2 += fac * trr_11y * wt;
                    gout_3_3 += trr_01x * trr_10y * wt;
                    gout_4_1 += fac * trr_10y * trr_11z;
                    gout_4_2 += fac * trr_11y * trr_10z;
                    gout_4_3 += trr_01x * trr_10y * trr_10z;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout_5_1 += fac * trr_20y * trr_01z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout_5_2 += fac * trr_21y * wt;
                    gout_5_3 += trr_01x * trr_20y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_6_1 += trr_10x * 1 * trr_01z;
                    gout_6_2 += trr_10x * trr_01y * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    gout_6_3 += trr_11x * 1 * wt;
                    gout_7_1 += trr_10x * 1 * trr_11z;
                    gout_7_2 += trr_10x * trr_01y * trr_10z;
                    gout_7_3 += trr_11x * 1 * trr_10z;
                    gout_8_1 += trr_10x * trr_10y * trr_01z;
                    gout_8_2 += trr_10x * trr_11y * wt;
                    gout_8_3 += trr_11x * trr_10y * wt;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    gout_9_1 += trr_20x * 1 * trr_01z;
                    gout_9_2 += trr_20x * trr_01y * wt;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    gout_9_3 += trr_21x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+1, gout_1_1*dm[kl_pair0+1] + gout_1_2*dm[kl_pair0+2] + gout_1_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+2, gout_2_1*dm[kl_pair0+1] + gout_2_2*dm[kl_pair0+2] + gout_2_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+3, gout_3_1*dm[kl_pair0+1] + gout_3_2*dm[kl_pair0+2] + gout_3_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+4, gout_4_1*dm[kl_pair0+1] + gout_4_2*dm[kl_pair0+2] + gout_4_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+5, gout_5_1*dm[kl_pair0+1] + gout_5_2*dm[kl_pair0+2] + gout_5_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+6, gout_6_1*dm[kl_pair0+1] + gout_6_2*dm[kl_pair0+2] + gout_6_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+7, gout_7_1*dm[kl_pair0+1] + gout_7_2*dm[kl_pair0+2] + gout_7_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+8, gout_8_1*dm[kl_pair0+1] + gout_8_2*dm[kl_pair0+2] + gout_8_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+9, gout_9_1*dm[kl_pair0+1] + gout_9_2*dm[kl_pair0+2] + gout_9_3*dm[kl_pair0+3]);
            atomicAdd(vj+kl_pair0+1, gout_1_1*dm[ij_pair0+1] + gout_2_1*dm[ij_pair0+2] + gout_3_1*dm[ij_pair0+3] + gout_4_1*dm[ij_pair0+4] + gout_5_1*dm[ij_pair0+5] + gout_6_1*dm[ij_pair0+6] + gout_7_1*dm[ij_pair0+7] + gout_8_1*dm[ij_pair0+8] + gout_9_1*dm[ij_pair0+9]);
            atomicAdd(vj+kl_pair0+2, gout_1_2*dm[ij_pair0+1] + gout_2_2*dm[ij_pair0+2] + gout_3_2*dm[ij_pair0+3] + gout_4_2*dm[ij_pair0+4] + gout_5_2*dm[ij_pair0+5] + gout_6_2*dm[ij_pair0+6] + gout_7_2*dm[ij_pair0+7] + gout_8_2*dm[ij_pair0+8] + gout_9_2*dm[ij_pair0+9]);
            atomicAdd(vj+kl_pair0+3, gout_1_3*dm[ij_pair0+1] + gout_2_3*dm[ij_pair0+2] + gout_3_3*dm[ij_pair0+3] + gout_4_3*dm[ij_pair0+4] + gout_5_3*dm[ij_pair0+5] + gout_6_3*dm[ij_pair0+6] + gout_7_3*dm[ij_pair0+7] + gout_8_3*dm[ij_pair0+8] + gout_9_3*dm[ij_pair0+9]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
__global__
static void rys_j_2_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_2_1(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_2_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 10*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 10*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_001 = dm[kl_pair0+1];
        double dm_kl_002 = dm[kl_pair0+2];
        double dm_kl_010 = dm[kl_pair0+3];
        double dm_kl_011 = dm[kl_pair0+4];
        double dm_kl_020 = dm[kl_pair0+5];
        double dm_kl_100 = dm[kl_pair0+6];
        double dm_kl_101 = dm[kl_pair0+7];
        double dm_kl_110 = dm[kl_pair0+8];
        double dm_kl_200 = dm[kl_pair0+9];
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        double vj_kl_001 = 0;
        double vj_kl_002 = 0;
        double vj_kl_010 = 0;
        double vj_kl_011 = 0;
        double vj_kl_020 = 0;
        double vj_kl_100 = 0;
        double vj_kl_101 = 0;
        double vj_kl_110 = 0;
        double vj_kl_200 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double dot_lij_z_000 = trr_10z * dm_ij_cache[sh_ij+1*TILE2] + trr_20z * dm_ij_cache[sh_ij+2*TILE2];
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double dot_lij_z_001 = trr_11z * dm_ij_cache[sh_ij+1*TILE2] + trr_21z * dm_ij_cache[sh_ij+2*TILE2];
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double dot_lij_z_002 = trr_12z * dm_ij_cache[sh_ij+1*TILE2] + trr_22z * dm_ij_cache[sh_ij+2*TILE2];
                    double dot_lij_z_010 = wt * dm_ij_cache[sh_ij+3*TILE2] + trr_10z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_011 = trr_01z * dm_ij_cache[sh_ij+3*TILE2] + trr_11z * dm_ij_cache[sh_ij+4*TILE2];
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double dot_lij_z_012 = trr_02z * dm_ij_cache[sh_ij+3*TILE2] + trr_12z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_022 = trr_02z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_100 = wt * dm_ij_cache[sh_ij+6*TILE2] + trr_10z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_101 = trr_01z * dm_ij_cache[sh_ij+6*TILE2] + trr_11z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_102 = trr_02z * dm_ij_cache[sh_ij+6*TILE2] + trr_12z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_112 = trr_02z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_202 = trr_02z * dm_ij_cache[sh_ij+9*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021;
                    double dot_lij_y_002 = 1 * dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020;
                    double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111;
                    double dot_lij_y_102 = 1 * dot_lij_z_102 + trr_10y * dot_lij_z_112;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110;
                    double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111;
                    double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110;
                    double dot_lij_y_200 = 1 * dot_lij_z_200;
                    double dot_lij_y_201 = 1 * dot_lij_z_201;
                    double dot_lij_y_202 = 1 * dot_lij_z_202;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200;
                    double dot_lij_y_211 = trr_01y * dot_lij_z_201;
                    double dot_lij_y_220 = trr_02y * dot_lij_z_200;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    vj_kl_001 += fac * dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201;
                    vj_kl_002 += fac * dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202;
                    vj_kl_010 += fac * dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210;
                    vj_kl_011 += fac * dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211;
                    vj_kl_020 += fac * dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200;
                    vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201;
                    vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200;
                    double dot_lkl_z_000 = trr_01z * dm_kl_001 + trr_02z * dm_kl_002;
                    double dot_lkl_z_001 = trr_11z * dm_kl_001 + trr_12z * dm_kl_002;
                    double dot_lkl_z_002 = trr_21z * dm_kl_001 + trr_22z * dm_kl_002;
                    double dot_lkl_z_010 = wt * dm_kl_010 + trr_01z * dm_kl_011;
                    double dot_lkl_z_011 = trr_10z * dm_kl_010 + trr_11z * dm_kl_011;
                    double dot_lkl_z_012 = trr_20z * dm_kl_010 + trr_21z * dm_kl_011;
                    double dot_lkl_z_020 = wt * dm_kl_020;
                    double dot_lkl_z_021 = trr_10z * dm_kl_020;
                    double dot_lkl_z_022 = trr_20z * dm_kl_020;
                    double dot_lkl_z_100 = wt * dm_kl_100 + trr_01z * dm_kl_101;
                    double dot_lkl_z_101 = trr_10z * dm_kl_100 + trr_11z * dm_kl_101;
                    double dot_lkl_z_102 = trr_20z * dm_kl_100 + trr_21z * dm_kl_101;
                    double dot_lkl_z_110 = wt * dm_kl_110;
                    double dot_lkl_z_111 = trr_10z * dm_kl_110;
                    double dot_lkl_z_112 = trr_20z * dm_kl_110;
                    double dot_lkl_z_200 = wt * dm_kl_200;
                    double dot_lkl_z_201 = trr_10z * dm_kl_200;
                    double dot_lkl_z_202 = trr_20z * dm_kl_200;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102 + trr_01y * dot_lkl_z_112;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110;
                    double dot_lkl_y_200 = 1 * dot_lkl_z_200;
                    double dot_lkl_y_201 = 1 * dot_lkl_z_201;
                    double dot_lkl_y_202 = 1 * dot_lkl_z_202;
                    double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                    double dot_lkl_y_211 = trr_10y * dot_lkl_z_201;
                    double dot_lkl_y_220 = trr_20y * dot_lkl_z_200;
                    vj_ij_001 += fac * dot_lkl_y_001 + trr_01x * dot_lkl_y_101 + trr_02x * dot_lkl_y_201;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202;
                    vj_ij_010 += fac * dot_lkl_y_010 + trr_01x * dot_lkl_y_110 + trr_02x * dot_lkl_y_210;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220;
                    vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100 + trr_12x * dot_lkl_y_200;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+1, vj_ij_001);
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_010);
        atomicAdd(vj+ij_pair0+4, vj_ij_011);
        atomicAdd(vj+ij_pair0+5, vj_ij_020);
        atomicAdd(vj+ij_pair0+6, vj_ij_100);
        atomicAdd(vj+ij_pair0+7, vj_ij_101);
        atomicAdd(vj+ij_pair0+8, vj_ij_110);
        atomicAdd(vj+ij_pair0+9, vj_ij_200);
        atomicAdd(vj+kl_pair0+1, vj_kl_001);
        atomicAdd(vj+kl_pair0+2, vj_kl_002);
        atomicAdd(vj+kl_pair0+3, vj_kl_010);
        atomicAdd(vj+kl_pair0+4, vj_kl_011);
        atomicAdd(vj+kl_pair0+5, vj_kl_020);
        atomicAdd(vj+kl_pair0+6, vj_kl_100);
        atomicAdd(vj+kl_pair0+7, vj_kl_101);
        atomicAdd(vj+kl_pair0+8, vj_kl_110);
        atomicAdd(vj+kl_pair0+9, vj_kl_200);
    }
}
__global__
static void rys_j_2_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_2_2(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_2_3(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 10*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 10*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_002 = dm[kl_pair0+2];
        double dm_kl_003 = dm[kl_pair0+3];
        double dm_kl_011 = dm[kl_pair0+5];
        double dm_kl_012 = dm[kl_pair0+6];
        double dm_kl_020 = dm[kl_pair0+7];
        double dm_kl_021 = dm[kl_pair0+8];
        double dm_kl_030 = dm[kl_pair0+9];
        double dm_kl_101 = dm[kl_pair0+11];
        double dm_kl_102 = dm[kl_pair0+12];
        double dm_kl_110 = dm[kl_pair0+13];
        double dm_kl_111 = dm[kl_pair0+14];
        double dm_kl_120 = dm[kl_pair0+15];
        double dm_kl_200 = dm[kl_pair0+16];
        double dm_kl_201 = dm[kl_pair0+17];
        double dm_kl_210 = dm[kl_pair0+18];
        double dm_kl_300 = dm[kl_pair0+19];
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        double vj_kl_002 = 0;
        double vj_kl_003 = 0;
        double vj_kl_011 = 0;
        double vj_kl_012 = 0;
        double vj_kl_020 = 0;
        double vj_kl_021 = 0;
        double vj_kl_030 = 0;
        double vj_kl_101 = 0;
        double vj_kl_102 = 0;
        double vj_kl_110 = 0;
        double vj_kl_111 = 0;
        double vj_kl_120 = 0;
        double vj_kl_200 = 0;
        double vj_kl_201 = 0;
        double vj_kl_210 = 0;
        double vj_kl_300 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double dot_lij_z_000 = trr_10z * dm_ij_cache[sh_ij+1*TILE2] + trr_20z * dm_ij_cache[sh_ij+2*TILE2];
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double dot_lij_z_001 = trr_11z * dm_ij_cache[sh_ij+1*TILE2] + trr_21z * dm_ij_cache[sh_ij+2*TILE2];
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double dot_lij_z_002 = trr_12z * dm_ij_cache[sh_ij+1*TILE2] + trr_22z * dm_ij_cache[sh_ij+2*TILE2];
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                    double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                    double dot_lij_z_003 = trr_13z * dm_ij_cache[sh_ij+1*TILE2] + trr_23z * dm_ij_cache[sh_ij+2*TILE2];
                    double dot_lij_z_010 = wt * dm_ij_cache[sh_ij+3*TILE2] + trr_10z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_011 = trr_01z * dm_ij_cache[sh_ij+3*TILE2] + trr_11z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_012 = trr_02z * dm_ij_cache[sh_ij+3*TILE2] + trr_12z * dm_ij_cache[sh_ij+4*TILE2];
                    double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                    double dot_lij_z_013 = trr_03z * dm_ij_cache[sh_ij+3*TILE2] + trr_13z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_022 = trr_02z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_023 = trr_03z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_100 = wt * dm_ij_cache[sh_ij+6*TILE2] + trr_10z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_101 = trr_01z * dm_ij_cache[sh_ij+6*TILE2] + trr_11z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_102 = trr_02z * dm_ij_cache[sh_ij+6*TILE2] + trr_12z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_103 = trr_03z * dm_ij_cache[sh_ij+6*TILE2] + trr_13z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_112 = trr_02z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_113 = trr_03z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_202 = trr_02z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_203 = trr_03z * dm_ij_cache[sh_ij+9*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021;
                    double dot_lij_y_002 = 1 * dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022;
                    double dot_lij_y_003 = 1 * dot_lij_z_003 + trr_10y * dot_lij_z_013 + trr_20y * dot_lij_z_023;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020;
                    double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021;
                    double dot_lij_y_012 = trr_01y * dot_lij_z_002 + trr_11y * dot_lij_z_012 + trr_21y * dot_lij_z_022;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020;
                    double dot_lij_y_021 = trr_02y * dot_lij_z_001 + trr_12y * dot_lij_z_011 + trr_22y * dot_lij_z_021;
                    double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                    double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                    double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                    double dot_lij_y_030 = trr_03y * dot_lij_z_000 + trr_13y * dot_lij_z_010 + trr_23y * dot_lij_z_020;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111;
                    double dot_lij_y_102 = 1 * dot_lij_z_102 + trr_10y * dot_lij_z_112;
                    double dot_lij_y_103 = 1 * dot_lij_z_103 + trr_10y * dot_lij_z_113;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110;
                    double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111;
                    double dot_lij_y_112 = trr_01y * dot_lij_z_102 + trr_11y * dot_lij_z_112;
                    double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110;
                    double dot_lij_y_121 = trr_02y * dot_lij_z_101 + trr_12y * dot_lij_z_111;
                    double dot_lij_y_130 = trr_03y * dot_lij_z_100 + trr_13y * dot_lij_z_110;
                    double dot_lij_y_200 = 1 * dot_lij_z_200;
                    double dot_lij_y_201 = 1 * dot_lij_z_201;
                    double dot_lij_y_202 = 1 * dot_lij_z_202;
                    double dot_lij_y_203 = 1 * dot_lij_z_203;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200;
                    double dot_lij_y_211 = trr_01y * dot_lij_z_201;
                    double dot_lij_y_212 = trr_01y * dot_lij_z_202;
                    double dot_lij_y_220 = trr_02y * dot_lij_z_200;
                    double dot_lij_y_221 = trr_02y * dot_lij_z_201;
                    double dot_lij_y_230 = trr_03y * dot_lij_z_200;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    vj_kl_002 += fac * dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202;
                    vj_kl_003 += fac * dot_lij_y_003 + trr_10x * dot_lij_y_103 + trr_20x * dot_lij_y_203;
                    vj_kl_011 += fac * dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211;
                    vj_kl_012 += fac * dot_lij_y_012 + trr_10x * dot_lij_y_112 + trr_20x * dot_lij_y_212;
                    vj_kl_020 += fac * dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220;
                    vj_kl_021 += fac * dot_lij_y_021 + trr_10x * dot_lij_y_121 + trr_20x * dot_lij_y_221;
                    vj_kl_030 += fac * dot_lij_y_030 + trr_10x * dot_lij_y_130 + trr_20x * dot_lij_y_230;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201;
                    vj_kl_102 += trr_01x * dot_lij_y_002 + trr_11x * dot_lij_y_102 + trr_21x * dot_lij_y_202;
                    vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210;
                    vj_kl_111 += trr_01x * dot_lij_y_011 + trr_11x * dot_lij_y_111 + trr_21x * dot_lij_y_211;
                    vj_kl_120 += trr_01x * dot_lij_y_020 + trr_11x * dot_lij_y_120 + trr_21x * dot_lij_y_220;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200;
                    vj_kl_201 += trr_02x * dot_lij_y_001 + trr_12x * dot_lij_y_101 + trr_22x * dot_lij_y_201;
                    vj_kl_210 += trr_02x * dot_lij_y_010 + trr_12x * dot_lij_y_110 + trr_22x * dot_lij_y_210;
                    double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                    double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                    double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                    vj_kl_300 += trr_03x * dot_lij_y_000 + trr_13x * dot_lij_y_100 + trr_23x * dot_lij_y_200;
                    double dot_lkl_z_000 = trr_02z * dm_kl_002 + trr_03z * dm_kl_003;
                    double dot_lkl_z_001 = trr_12z * dm_kl_002 + trr_13z * dm_kl_003;
                    double dot_lkl_z_002 = trr_22z * dm_kl_002 + trr_23z * dm_kl_003;
                    double dot_lkl_z_010 = trr_01z * dm_kl_011 + trr_02z * dm_kl_012;
                    double dot_lkl_z_011 = trr_11z * dm_kl_011 + trr_12z * dm_kl_012;
                    double dot_lkl_z_012 = trr_21z * dm_kl_011 + trr_22z * dm_kl_012;
                    double dot_lkl_z_020 = wt * dm_kl_020 + trr_01z * dm_kl_021;
                    double dot_lkl_z_021 = trr_10z * dm_kl_020 + trr_11z * dm_kl_021;
                    double dot_lkl_z_022 = trr_20z * dm_kl_020 + trr_21z * dm_kl_021;
                    double dot_lkl_z_030 = wt * dm_kl_030;
                    double dot_lkl_z_031 = trr_10z * dm_kl_030;
                    double dot_lkl_z_032 = trr_20z * dm_kl_030;
                    double dot_lkl_z_100 = trr_01z * dm_kl_101 + trr_02z * dm_kl_102;
                    double dot_lkl_z_101 = trr_11z * dm_kl_101 + trr_12z * dm_kl_102;
                    double dot_lkl_z_102 = trr_21z * dm_kl_101 + trr_22z * dm_kl_102;
                    double dot_lkl_z_110 = wt * dm_kl_110 + trr_01z * dm_kl_111;
                    double dot_lkl_z_111 = trr_10z * dm_kl_110 + trr_11z * dm_kl_111;
                    double dot_lkl_z_112 = trr_20z * dm_kl_110 + trr_21z * dm_kl_111;
                    double dot_lkl_z_120 = wt * dm_kl_120;
                    double dot_lkl_z_121 = trr_10z * dm_kl_120;
                    double dot_lkl_z_122 = trr_20z * dm_kl_120;
                    double dot_lkl_z_200 = wt * dm_kl_200 + trr_01z * dm_kl_201;
                    double dot_lkl_z_201 = trr_10z * dm_kl_200 + trr_11z * dm_kl_201;
                    double dot_lkl_z_202 = trr_20z * dm_kl_200 + trr_21z * dm_kl_201;
                    double dot_lkl_z_210 = wt * dm_kl_210;
                    double dot_lkl_z_211 = trr_10z * dm_kl_210;
                    double dot_lkl_z_212 = trr_20z * dm_kl_210;
                    double dot_lkl_z_300 = wt * dm_kl_300;
                    double dot_lkl_z_301 = trr_10z * dm_kl_300;
                    double dot_lkl_z_302 = trr_20z * dm_kl_300;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020 + trr_03y * dot_lkl_z_030;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021 + trr_03y * dot_lkl_z_031;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022 + trr_03y * dot_lkl_z_032;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020 + trr_13y * dot_lkl_z_030;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021 + trr_13y * dot_lkl_z_031;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020 + trr_23y * dot_lkl_z_030;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100 + trr_01y * dot_lkl_z_110 + trr_02y * dot_lkl_z_120;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101 + trr_01y * dot_lkl_z_111 + trr_02y * dot_lkl_z_121;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102 + trr_01y * dot_lkl_z_112 + trr_02y * dot_lkl_z_122;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110 + trr_12y * dot_lkl_z_120;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111 + trr_12y * dot_lkl_z_121;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110 + trr_22y * dot_lkl_z_120;
                    double dot_lkl_y_200 = 1 * dot_lkl_z_200 + trr_01y * dot_lkl_z_210;
                    double dot_lkl_y_201 = 1 * dot_lkl_z_201 + trr_01y * dot_lkl_z_211;
                    double dot_lkl_y_202 = 1 * dot_lkl_z_202 + trr_01y * dot_lkl_z_212;
                    double dot_lkl_y_210 = trr_10y * dot_lkl_z_200 + trr_11y * dot_lkl_z_210;
                    double dot_lkl_y_211 = trr_10y * dot_lkl_z_201 + trr_11y * dot_lkl_z_211;
                    double dot_lkl_y_220 = trr_20y * dot_lkl_z_200 + trr_21y * dot_lkl_z_210;
                    double dot_lkl_y_300 = 1 * dot_lkl_z_300;
                    double dot_lkl_y_301 = 1 * dot_lkl_z_301;
                    double dot_lkl_y_302 = 1 * dot_lkl_z_302;
                    double dot_lkl_y_310 = trr_10y * dot_lkl_z_300;
                    double dot_lkl_y_311 = trr_10y * dot_lkl_z_301;
                    double dot_lkl_y_320 = trr_20y * dot_lkl_z_300;
                    vj_ij_001 += fac * dot_lkl_y_001 + trr_01x * dot_lkl_y_101 + trr_02x * dot_lkl_y_201 + trr_03x * dot_lkl_y_301;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202 + trr_03x * dot_lkl_y_302;
                    vj_ij_010 += fac * dot_lkl_y_010 + trr_01x * dot_lkl_y_110 + trr_02x * dot_lkl_y_210 + trr_03x * dot_lkl_y_310;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211 + trr_03x * dot_lkl_y_311;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220 + trr_03x * dot_lkl_y_320;
                    vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100 + trr_12x * dot_lkl_y_200 + trr_13x * dot_lkl_y_300;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201 + trr_13x * dot_lkl_y_301;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210 + trr_13x * dot_lkl_y_310;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200 + trr_23x * dot_lkl_y_300;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+1, vj_ij_001);
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_010);
        atomicAdd(vj+ij_pair0+4, vj_ij_011);
        atomicAdd(vj+ij_pair0+5, vj_ij_020);
        atomicAdd(vj+ij_pair0+6, vj_ij_100);
        atomicAdd(vj+ij_pair0+7, vj_ij_101);
        atomicAdd(vj+ij_pair0+8, vj_ij_110);
        atomicAdd(vj+ij_pair0+9, vj_ij_200);
        atomicAdd(vj+kl_pair0+2, vj_kl_002);
        atomicAdd(vj+kl_pair0+3, vj_kl_003);
        atomicAdd(vj+kl_pair0+5, vj_kl_011);
        atomicAdd(vj+kl_pair0+6, vj_kl_012);
        atomicAdd(vj+kl_pair0+7, vj_kl_020);
        atomicAdd(vj+kl_pair0+8, vj_kl_021);
        atomicAdd(vj+kl_pair0+9, vj_kl_030);
        atomicAdd(vj+kl_pair0+11, vj_kl_101);
        atomicAdd(vj+kl_pair0+12, vj_kl_102);
        atomicAdd(vj+kl_pair0+13, vj_kl_110);
        atomicAdd(vj+kl_pair0+14, vj_kl_111);
        atomicAdd(vj+kl_pair0+15, vj_kl_120);
        atomicAdd(vj+kl_pair0+16, vj_kl_200);
        atomicAdd(vj+kl_pair0+17, vj_kl_201);
        atomicAdd(vj+kl_pair0+18, vj_kl_210);
        atomicAdd(vj+kl_pair0+19, vj_kl_300);
    }
}
__global__
static void rys_j_2_3(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_2_3(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_2_4(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 10*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 10*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_002 = dm[kl_pair0+2];
        double dm_kl_003 = dm[kl_pair0+3];
        double dm_kl_004 = dm[kl_pair0+4];
        double dm_kl_011 = dm[kl_pair0+6];
        double dm_kl_012 = dm[kl_pair0+7];
        double dm_kl_013 = dm[kl_pair0+8];
        double dm_kl_020 = dm[kl_pair0+9];
        double dm_kl_021 = dm[kl_pair0+10];
        double dm_kl_022 = dm[kl_pair0+11];
        double dm_kl_030 = dm[kl_pair0+12];
        double dm_kl_031 = dm[kl_pair0+13];
        double dm_kl_040 = dm[kl_pair0+14];
        double dm_kl_101 = dm[kl_pair0+16];
        double dm_kl_102 = dm[kl_pair0+17];
        double dm_kl_103 = dm[kl_pair0+18];
        double dm_kl_110 = dm[kl_pair0+19];
        double dm_kl_111 = dm[kl_pair0+20];
        double dm_kl_112 = dm[kl_pair0+21];
        double dm_kl_120 = dm[kl_pair0+22];
        double dm_kl_121 = dm[kl_pair0+23];
        double dm_kl_130 = dm[kl_pair0+24];
        double dm_kl_200 = dm[kl_pair0+25];
        double dm_kl_201 = dm[kl_pair0+26];
        double dm_kl_202 = dm[kl_pair0+27];
        double dm_kl_210 = dm[kl_pair0+28];
        double dm_kl_211 = dm[kl_pair0+29];
        double dm_kl_220 = dm[kl_pair0+30];
        double dm_kl_300 = dm[kl_pair0+31];
        double dm_kl_301 = dm[kl_pair0+32];
        double dm_kl_310 = dm[kl_pair0+33];
        double dm_kl_400 = dm[kl_pair0+34];
        double vj_ij_001 = 0;
        double vj_ij_002 = 0;
        double vj_ij_010 = 0;
        double vj_ij_011 = 0;
        double vj_ij_020 = 0;
        double vj_ij_100 = 0;
        double vj_ij_101 = 0;
        double vj_ij_110 = 0;
        double vj_ij_200 = 0;
        double vj_kl_002 = 0;
        double vj_kl_003 = 0;
        double vj_kl_004 = 0;
        double vj_kl_011 = 0;
        double vj_kl_012 = 0;
        double vj_kl_013 = 0;
        double vj_kl_020 = 0;
        double vj_kl_021 = 0;
        double vj_kl_022 = 0;
        double vj_kl_030 = 0;
        double vj_kl_031 = 0;
        double vj_kl_040 = 0;
        double vj_kl_101 = 0;
        double vj_kl_102 = 0;
        double vj_kl_103 = 0;
        double vj_kl_110 = 0;
        double vj_kl_111 = 0;
        double vj_kl_112 = 0;
        double vj_kl_120 = 0;
        double vj_kl_121 = 0;
        double vj_kl_130 = 0;
        double vj_kl_200 = 0;
        double vj_kl_201 = 0;
        double vj_kl_202 = 0;
        double vj_kl_210 = 0;
        double vj_kl_211 = 0;
        double vj_kl_220 = 0;
        double vj_kl_300 = 0;
        double vj_kl_301 = 0;
        double vj_kl_310 = 0;
        double vj_kl_400 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double dot_lij_z_000 = trr_10z * dm_ij_cache[sh_ij+1*TILE2] + trr_20z * dm_ij_cache[sh_ij+2*TILE2];
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double dot_lij_z_001 = trr_11z * dm_ij_cache[sh_ij+1*TILE2] + trr_21z * dm_ij_cache[sh_ij+2*TILE2];
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double dot_lij_z_002 = trr_12z * dm_ij_cache[sh_ij+1*TILE2] + trr_22z * dm_ij_cache[sh_ij+2*TILE2];
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                    double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                    double dot_lij_z_003 = trr_13z * dm_ij_cache[sh_ij+1*TILE2] + trr_23z * dm_ij_cache[sh_ij+2*TILE2];
                    double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                    double trr_14z = cpz * trr_13z + 3*b01 * trr_12z + 1*b00 * trr_03z;
                    double trr_24z = cpz * trr_23z + 3*b01 * trr_22z + 2*b00 * trr_13z;
                    double dot_lij_z_004 = trr_14z * dm_ij_cache[sh_ij+1*TILE2] + trr_24z * dm_ij_cache[sh_ij+2*TILE2];
                    double dot_lij_z_010 = wt * dm_ij_cache[sh_ij+3*TILE2] + trr_10z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_011 = trr_01z * dm_ij_cache[sh_ij+3*TILE2] + trr_11z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_012 = trr_02z * dm_ij_cache[sh_ij+3*TILE2] + trr_12z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_013 = trr_03z * dm_ij_cache[sh_ij+3*TILE2] + trr_13z * dm_ij_cache[sh_ij+4*TILE2];
                    double trr_04z = cpz * trr_03z + 3*b01 * trr_02z;
                    double dot_lij_z_014 = trr_04z * dm_ij_cache[sh_ij+3*TILE2] + trr_14z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_022 = trr_02z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_023 = trr_03z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_024 = trr_04z * dm_ij_cache[sh_ij+5*TILE2];
                    double dot_lij_z_100 = wt * dm_ij_cache[sh_ij+6*TILE2] + trr_10z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_101 = trr_01z * dm_ij_cache[sh_ij+6*TILE2] + trr_11z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_102 = trr_02z * dm_ij_cache[sh_ij+6*TILE2] + trr_12z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_103 = trr_03z * dm_ij_cache[sh_ij+6*TILE2] + trr_13z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_104 = trr_04z * dm_ij_cache[sh_ij+6*TILE2] + trr_14z * dm_ij_cache[sh_ij+7*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_112 = trr_02z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_113 = trr_03z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_114 = trr_04z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_202 = trr_02z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_203 = trr_03z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_204 = trr_04z * dm_ij_cache[sh_ij+9*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021;
                    double dot_lij_y_002 = 1 * dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022;
                    double dot_lij_y_003 = 1 * dot_lij_z_003 + trr_10y * dot_lij_z_013 + trr_20y * dot_lij_z_023;
                    double dot_lij_y_004 = 1 * dot_lij_z_004 + trr_10y * dot_lij_z_014 + trr_20y * dot_lij_z_024;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020;
                    double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021;
                    double dot_lij_y_012 = trr_01y * dot_lij_z_002 + trr_11y * dot_lij_z_012 + trr_21y * dot_lij_z_022;
                    double dot_lij_y_013 = trr_01y * dot_lij_z_003 + trr_11y * dot_lij_z_013 + trr_21y * dot_lij_z_023;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020;
                    double dot_lij_y_021 = trr_02y * dot_lij_z_001 + trr_12y * dot_lij_z_011 + trr_22y * dot_lij_z_021;
                    double dot_lij_y_022 = trr_02y * dot_lij_z_002 + trr_12y * dot_lij_z_012 + trr_22y * dot_lij_z_022;
                    double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                    double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                    double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                    double dot_lij_y_030 = trr_03y * dot_lij_z_000 + trr_13y * dot_lij_z_010 + trr_23y * dot_lij_z_020;
                    double dot_lij_y_031 = trr_03y * dot_lij_z_001 + trr_13y * dot_lij_z_011 + trr_23y * dot_lij_z_021;
                    double trr_04y = cpy * trr_03y + 3*b01 * trr_02y;
                    double trr_14y = cpy * trr_13y + 3*b01 * trr_12y + 1*b00 * trr_03y;
                    double trr_24y = cpy * trr_23y + 3*b01 * trr_22y + 2*b00 * trr_13y;
                    double dot_lij_y_040 = trr_04y * dot_lij_z_000 + trr_14y * dot_lij_z_010 + trr_24y * dot_lij_z_020;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111;
                    double dot_lij_y_102 = 1 * dot_lij_z_102 + trr_10y * dot_lij_z_112;
                    double dot_lij_y_103 = 1 * dot_lij_z_103 + trr_10y * dot_lij_z_113;
                    double dot_lij_y_104 = 1 * dot_lij_z_104 + trr_10y * dot_lij_z_114;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110;
                    double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111;
                    double dot_lij_y_112 = trr_01y * dot_lij_z_102 + trr_11y * dot_lij_z_112;
                    double dot_lij_y_113 = trr_01y * dot_lij_z_103 + trr_11y * dot_lij_z_113;
                    double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110;
                    double dot_lij_y_121 = trr_02y * dot_lij_z_101 + trr_12y * dot_lij_z_111;
                    double dot_lij_y_122 = trr_02y * dot_lij_z_102 + trr_12y * dot_lij_z_112;
                    double dot_lij_y_130 = trr_03y * dot_lij_z_100 + trr_13y * dot_lij_z_110;
                    double dot_lij_y_131 = trr_03y * dot_lij_z_101 + trr_13y * dot_lij_z_111;
                    double dot_lij_y_140 = trr_04y * dot_lij_z_100 + trr_14y * dot_lij_z_110;
                    double dot_lij_y_200 = 1 * dot_lij_z_200;
                    double dot_lij_y_201 = 1 * dot_lij_z_201;
                    double dot_lij_y_202 = 1 * dot_lij_z_202;
                    double dot_lij_y_203 = 1 * dot_lij_z_203;
                    double dot_lij_y_204 = 1 * dot_lij_z_204;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200;
                    double dot_lij_y_211 = trr_01y * dot_lij_z_201;
                    double dot_lij_y_212 = trr_01y * dot_lij_z_202;
                    double dot_lij_y_213 = trr_01y * dot_lij_z_203;
                    double dot_lij_y_220 = trr_02y * dot_lij_z_200;
                    double dot_lij_y_221 = trr_02y * dot_lij_z_201;
                    double dot_lij_y_222 = trr_02y * dot_lij_z_202;
                    double dot_lij_y_230 = trr_03y * dot_lij_z_200;
                    double dot_lij_y_231 = trr_03y * dot_lij_z_201;
                    double dot_lij_y_240 = trr_04y * dot_lij_z_200;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    vj_kl_002 += fac * dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202;
                    vj_kl_003 += fac * dot_lij_y_003 + trr_10x * dot_lij_y_103 + trr_20x * dot_lij_y_203;
                    vj_kl_004 += fac * dot_lij_y_004 + trr_10x * dot_lij_y_104 + trr_20x * dot_lij_y_204;
                    vj_kl_011 += fac * dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211;
                    vj_kl_012 += fac * dot_lij_y_012 + trr_10x * dot_lij_y_112 + trr_20x * dot_lij_y_212;
                    vj_kl_013 += fac * dot_lij_y_013 + trr_10x * dot_lij_y_113 + trr_20x * dot_lij_y_213;
                    vj_kl_020 += fac * dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220;
                    vj_kl_021 += fac * dot_lij_y_021 + trr_10x * dot_lij_y_121 + trr_20x * dot_lij_y_221;
                    vj_kl_022 += fac * dot_lij_y_022 + trr_10x * dot_lij_y_122 + trr_20x * dot_lij_y_222;
                    vj_kl_030 += fac * dot_lij_y_030 + trr_10x * dot_lij_y_130 + trr_20x * dot_lij_y_230;
                    vj_kl_031 += fac * dot_lij_y_031 + trr_10x * dot_lij_y_131 + trr_20x * dot_lij_y_231;
                    vj_kl_040 += fac * dot_lij_y_040 + trr_10x * dot_lij_y_140 + trr_20x * dot_lij_y_240;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201;
                    vj_kl_102 += trr_01x * dot_lij_y_002 + trr_11x * dot_lij_y_102 + trr_21x * dot_lij_y_202;
                    vj_kl_103 += trr_01x * dot_lij_y_003 + trr_11x * dot_lij_y_103 + trr_21x * dot_lij_y_203;
                    vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210;
                    vj_kl_111 += trr_01x * dot_lij_y_011 + trr_11x * dot_lij_y_111 + trr_21x * dot_lij_y_211;
                    vj_kl_112 += trr_01x * dot_lij_y_012 + trr_11x * dot_lij_y_112 + trr_21x * dot_lij_y_212;
                    vj_kl_120 += trr_01x * dot_lij_y_020 + trr_11x * dot_lij_y_120 + trr_21x * dot_lij_y_220;
                    vj_kl_121 += trr_01x * dot_lij_y_021 + trr_11x * dot_lij_y_121 + trr_21x * dot_lij_y_221;
                    vj_kl_130 += trr_01x * dot_lij_y_030 + trr_11x * dot_lij_y_130 + trr_21x * dot_lij_y_230;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200;
                    vj_kl_201 += trr_02x * dot_lij_y_001 + trr_12x * dot_lij_y_101 + trr_22x * dot_lij_y_201;
                    vj_kl_202 += trr_02x * dot_lij_y_002 + trr_12x * dot_lij_y_102 + trr_22x * dot_lij_y_202;
                    vj_kl_210 += trr_02x * dot_lij_y_010 + trr_12x * dot_lij_y_110 + trr_22x * dot_lij_y_210;
                    vj_kl_211 += trr_02x * dot_lij_y_011 + trr_12x * dot_lij_y_111 + trr_22x * dot_lij_y_211;
                    vj_kl_220 += trr_02x * dot_lij_y_020 + trr_12x * dot_lij_y_120 + trr_22x * dot_lij_y_220;
                    double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                    double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                    double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                    vj_kl_300 += trr_03x * dot_lij_y_000 + trr_13x * dot_lij_y_100 + trr_23x * dot_lij_y_200;
                    vj_kl_301 += trr_03x * dot_lij_y_001 + trr_13x * dot_lij_y_101 + trr_23x * dot_lij_y_201;
                    vj_kl_310 += trr_03x * dot_lij_y_010 + trr_13x * dot_lij_y_110 + trr_23x * dot_lij_y_210;
                    double trr_04x = cpx * trr_03x + 3*b01 * trr_02x;
                    double trr_14x = cpx * trr_13x + 3*b01 * trr_12x + 1*b00 * trr_03x;
                    double trr_24x = cpx * trr_23x + 3*b01 * trr_22x + 2*b00 * trr_13x;
                    vj_kl_400 += trr_04x * dot_lij_y_000 + trr_14x * dot_lij_y_100 + trr_24x * dot_lij_y_200;
                    double dot_lkl_z_000 = trr_02z * dm_kl_002 + trr_03z * dm_kl_003 + trr_04z * dm_kl_004;
                    double dot_lkl_z_001 = trr_12z * dm_kl_002 + trr_13z * dm_kl_003 + trr_14z * dm_kl_004;
                    double dot_lkl_z_002 = trr_22z * dm_kl_002 + trr_23z * dm_kl_003 + trr_24z * dm_kl_004;
                    double dot_lkl_z_010 = trr_01z * dm_kl_011 + trr_02z * dm_kl_012 + trr_03z * dm_kl_013;
                    double dot_lkl_z_011 = trr_11z * dm_kl_011 + trr_12z * dm_kl_012 + trr_13z * dm_kl_013;
                    double dot_lkl_z_012 = trr_21z * dm_kl_011 + trr_22z * dm_kl_012 + trr_23z * dm_kl_013;
                    double dot_lkl_z_020 = wt * dm_kl_020 + trr_01z * dm_kl_021 + trr_02z * dm_kl_022;
                    double dot_lkl_z_021 = trr_10z * dm_kl_020 + trr_11z * dm_kl_021 + trr_12z * dm_kl_022;
                    double dot_lkl_z_022 = trr_20z * dm_kl_020 + trr_21z * dm_kl_021 + trr_22z * dm_kl_022;
                    double dot_lkl_z_030 = wt * dm_kl_030 + trr_01z * dm_kl_031;
                    double dot_lkl_z_031 = trr_10z * dm_kl_030 + trr_11z * dm_kl_031;
                    double dot_lkl_z_032 = trr_20z * dm_kl_030 + trr_21z * dm_kl_031;
                    double dot_lkl_z_040 = wt * dm_kl_040;
                    double dot_lkl_z_041 = trr_10z * dm_kl_040;
                    double dot_lkl_z_042 = trr_20z * dm_kl_040;
                    double dot_lkl_z_100 = trr_01z * dm_kl_101 + trr_02z * dm_kl_102 + trr_03z * dm_kl_103;
                    double dot_lkl_z_101 = trr_11z * dm_kl_101 + trr_12z * dm_kl_102 + trr_13z * dm_kl_103;
                    double dot_lkl_z_102 = trr_21z * dm_kl_101 + trr_22z * dm_kl_102 + trr_23z * dm_kl_103;
                    double dot_lkl_z_110 = wt * dm_kl_110 + trr_01z * dm_kl_111 + trr_02z * dm_kl_112;
                    double dot_lkl_z_111 = trr_10z * dm_kl_110 + trr_11z * dm_kl_111 + trr_12z * dm_kl_112;
                    double dot_lkl_z_112 = trr_20z * dm_kl_110 + trr_21z * dm_kl_111 + trr_22z * dm_kl_112;
                    double dot_lkl_z_120 = wt * dm_kl_120 + trr_01z * dm_kl_121;
                    double dot_lkl_z_121 = trr_10z * dm_kl_120 + trr_11z * dm_kl_121;
                    double dot_lkl_z_122 = trr_20z * dm_kl_120 + trr_21z * dm_kl_121;
                    double dot_lkl_z_130 = wt * dm_kl_130;
                    double dot_lkl_z_131 = trr_10z * dm_kl_130;
                    double dot_lkl_z_132 = trr_20z * dm_kl_130;
                    double dot_lkl_z_200 = wt * dm_kl_200 + trr_01z * dm_kl_201 + trr_02z * dm_kl_202;
                    double dot_lkl_z_201 = trr_10z * dm_kl_200 + trr_11z * dm_kl_201 + trr_12z * dm_kl_202;
                    double dot_lkl_z_202 = trr_20z * dm_kl_200 + trr_21z * dm_kl_201 + trr_22z * dm_kl_202;
                    double dot_lkl_z_210 = wt * dm_kl_210 + trr_01z * dm_kl_211;
                    double dot_lkl_z_211 = trr_10z * dm_kl_210 + trr_11z * dm_kl_211;
                    double dot_lkl_z_212 = trr_20z * dm_kl_210 + trr_21z * dm_kl_211;
                    double dot_lkl_z_220 = wt * dm_kl_220;
                    double dot_lkl_z_221 = trr_10z * dm_kl_220;
                    double dot_lkl_z_222 = trr_20z * dm_kl_220;
                    double dot_lkl_z_300 = wt * dm_kl_300 + trr_01z * dm_kl_301;
                    double dot_lkl_z_301 = trr_10z * dm_kl_300 + trr_11z * dm_kl_301;
                    double dot_lkl_z_302 = trr_20z * dm_kl_300 + trr_21z * dm_kl_301;
                    double dot_lkl_z_310 = wt * dm_kl_310;
                    double dot_lkl_z_311 = trr_10z * dm_kl_310;
                    double dot_lkl_z_312 = trr_20z * dm_kl_310;
                    double dot_lkl_z_400 = wt * dm_kl_400;
                    double dot_lkl_z_401 = trr_10z * dm_kl_400;
                    double dot_lkl_z_402 = trr_20z * dm_kl_400;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020 + trr_03y * dot_lkl_z_030 + trr_04y * dot_lkl_z_040;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021 + trr_03y * dot_lkl_z_031 + trr_04y * dot_lkl_z_041;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022 + trr_03y * dot_lkl_z_032 + trr_04y * dot_lkl_z_042;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020 + trr_13y * dot_lkl_z_030 + trr_14y * dot_lkl_z_040;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021 + trr_13y * dot_lkl_z_031 + trr_14y * dot_lkl_z_041;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020 + trr_23y * dot_lkl_z_030 + trr_24y * dot_lkl_z_040;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100 + trr_01y * dot_lkl_z_110 + trr_02y * dot_lkl_z_120 + trr_03y * dot_lkl_z_130;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101 + trr_01y * dot_lkl_z_111 + trr_02y * dot_lkl_z_121 + trr_03y * dot_lkl_z_131;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102 + trr_01y * dot_lkl_z_112 + trr_02y * dot_lkl_z_122 + trr_03y * dot_lkl_z_132;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110 + trr_12y * dot_lkl_z_120 + trr_13y * dot_lkl_z_130;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111 + trr_12y * dot_lkl_z_121 + trr_13y * dot_lkl_z_131;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110 + trr_22y * dot_lkl_z_120 + trr_23y * dot_lkl_z_130;
                    double dot_lkl_y_200 = 1 * dot_lkl_z_200 + trr_01y * dot_lkl_z_210 + trr_02y * dot_lkl_z_220;
                    double dot_lkl_y_201 = 1 * dot_lkl_z_201 + trr_01y * dot_lkl_z_211 + trr_02y * dot_lkl_z_221;
                    double dot_lkl_y_202 = 1 * dot_lkl_z_202 + trr_01y * dot_lkl_z_212 + trr_02y * dot_lkl_z_222;
                    double dot_lkl_y_210 = trr_10y * dot_lkl_z_200 + trr_11y * dot_lkl_z_210 + trr_12y * dot_lkl_z_220;
                    double dot_lkl_y_211 = trr_10y * dot_lkl_z_201 + trr_11y * dot_lkl_z_211 + trr_12y * dot_lkl_z_221;
                    double dot_lkl_y_220 = trr_20y * dot_lkl_z_200 + trr_21y * dot_lkl_z_210 + trr_22y * dot_lkl_z_220;
                    double dot_lkl_y_300 = 1 * dot_lkl_z_300 + trr_01y * dot_lkl_z_310;
                    double dot_lkl_y_301 = 1 * dot_lkl_z_301 + trr_01y * dot_lkl_z_311;
                    double dot_lkl_y_302 = 1 * dot_lkl_z_302 + trr_01y * dot_lkl_z_312;
                    double dot_lkl_y_310 = trr_10y * dot_lkl_z_300 + trr_11y * dot_lkl_z_310;
                    double dot_lkl_y_311 = trr_10y * dot_lkl_z_301 + trr_11y * dot_lkl_z_311;
                    double dot_lkl_y_320 = trr_20y * dot_lkl_z_300 + trr_21y * dot_lkl_z_310;
                    double dot_lkl_y_400 = 1 * dot_lkl_z_400;
                    double dot_lkl_y_401 = 1 * dot_lkl_z_401;
                    double dot_lkl_y_402 = 1 * dot_lkl_z_402;
                    double dot_lkl_y_410 = trr_10y * dot_lkl_z_400;
                    double dot_lkl_y_411 = trr_10y * dot_lkl_z_401;
                    double dot_lkl_y_420 = trr_20y * dot_lkl_z_400;
                    vj_ij_001 += fac * dot_lkl_y_001 + trr_01x * dot_lkl_y_101 + trr_02x * dot_lkl_y_201 + trr_03x * dot_lkl_y_301 + trr_04x * dot_lkl_y_401;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202 + trr_03x * dot_lkl_y_302 + trr_04x * dot_lkl_y_402;
                    vj_ij_010 += fac * dot_lkl_y_010 + trr_01x * dot_lkl_y_110 + trr_02x * dot_lkl_y_210 + trr_03x * dot_lkl_y_310 + trr_04x * dot_lkl_y_410;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211 + trr_03x * dot_lkl_y_311 + trr_04x * dot_lkl_y_411;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220 + trr_03x * dot_lkl_y_320 + trr_04x * dot_lkl_y_420;
                    vj_ij_100 += trr_10x * dot_lkl_y_000 + trr_11x * dot_lkl_y_100 + trr_12x * dot_lkl_y_200 + trr_13x * dot_lkl_y_300 + trr_14x * dot_lkl_y_400;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201 + trr_13x * dot_lkl_y_301 + trr_14x * dot_lkl_y_401;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210 + trr_13x * dot_lkl_y_310 + trr_14x * dot_lkl_y_410;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200 + trr_23x * dot_lkl_y_300 + trr_24x * dot_lkl_y_400;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+1, vj_ij_001);
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_010);
        atomicAdd(vj+ij_pair0+4, vj_ij_011);
        atomicAdd(vj+ij_pair0+5, vj_ij_020);
        atomicAdd(vj+ij_pair0+6, vj_ij_100);
        atomicAdd(vj+ij_pair0+7, vj_ij_101);
        atomicAdd(vj+ij_pair0+8, vj_ij_110);
        atomicAdd(vj+ij_pair0+9, vj_ij_200);
        atomicAdd(vj+kl_pair0+2, vj_kl_002);
        atomicAdd(vj+kl_pair0+3, vj_kl_003);
        atomicAdd(vj+kl_pair0+4, vj_kl_004);
        atomicAdd(vj+kl_pair0+6, vj_kl_011);
        atomicAdd(vj+kl_pair0+7, vj_kl_012);
        atomicAdd(vj+kl_pair0+8, vj_kl_013);
        atomicAdd(vj+kl_pair0+9, vj_kl_020);
        atomicAdd(vj+kl_pair0+10, vj_kl_021);
        atomicAdd(vj+kl_pair0+11, vj_kl_022);
        atomicAdd(vj+kl_pair0+12, vj_kl_030);
        atomicAdd(vj+kl_pair0+13, vj_kl_031);
        atomicAdd(vj+kl_pair0+14, vj_kl_040);
        atomicAdd(vj+kl_pair0+16, vj_kl_101);
        atomicAdd(vj+kl_pair0+17, vj_kl_102);
        atomicAdd(vj+kl_pair0+18, vj_kl_103);
        atomicAdd(vj+kl_pair0+19, vj_kl_110);
        atomicAdd(vj+kl_pair0+20, vj_kl_111);
        atomicAdd(vj+kl_pair0+21, vj_kl_112);
        atomicAdd(vj+kl_pair0+22, vj_kl_120);
        atomicAdd(vj+kl_pair0+23, vj_kl_121);
        atomicAdd(vj+kl_pair0+24, vj_kl_130);
        atomicAdd(vj+kl_pair0+25, vj_kl_200);
        atomicAdd(vj+kl_pair0+26, vj_kl_201);
        atomicAdd(vj+kl_pair0+27, vj_kl_202);
        atomicAdd(vj+kl_pair0+28, vj_kl_210);
        atomicAdd(vj+kl_pair0+29, vj_kl_211);
        atomicAdd(vj+kl_pair0+30, vj_kl_220);
        atomicAdd(vj+kl_pair0+31, vj_kl_300);
        atomicAdd(vj+kl_pair0+32, vj_kl_301);
        atomicAdd(vj+kl_pair0+33, vj_kl_310);
        atomicAdd(vj+kl_pair0+34, vj_kl_400);
    }
}
__global__
static void rys_j_2_4(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_2_4(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_3_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_2_0 = 0.;
        double gout_3_0 = 0.;
        double gout_5_0 = 0.;
        double gout_6_0 = 0.;
        double gout_7_0 = 0.;
        double gout_8_0 = 0.;
        double gout_9_0 = 0.;
        double gout_11_0 = 0.;
        double gout_12_0 = 0.;
        double gout_13_0 = 0.;
        double gout_14_0 = 0.;
        double gout_15_0 = 0.;
        double gout_16_0 = 0.;
        double gout_17_0 = 0.;
        double gout_18_0 = 0.;
        double gout_19_0 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout_2_0 += fac * 1 * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout_3_0 += fac * 1 * trr_30z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout_5_0 += fac * trr_10y * trr_10z;
                    gout_6_0 += fac * trr_10y * trr_20z;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout_7_0 += fac * trr_20y * wt;
                    gout_8_0 += fac * trr_20y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout_9_0 += fac * trr_30y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_11_0 += trr_10x * 1 * trr_10z;
                    gout_12_0 += trr_10x * 1 * trr_20z;
                    gout_13_0 += trr_10x * trr_10y * wt;
                    gout_14_0 += trr_10x * trr_10y * trr_10z;
                    gout_15_0 += trr_10x * trr_20y * wt;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    gout_16_0 += trr_20x * 1 * wt;
                    gout_17_0 += trr_20x * 1 * trr_10z;
                    gout_18_0 += trr_20x * trr_10y * wt;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    gout_19_0 += trr_30x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+2, gout_2_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+3, gout_3_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+5, gout_5_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+6, gout_6_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+7, gout_7_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+8, gout_8_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+9, gout_9_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+11, gout_11_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+12, gout_12_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+13, gout_13_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+14, gout_14_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+15, gout_15_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+16, gout_16_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+17, gout_17_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+18, gout_18_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+19, gout_19_0*dm[kl_pair0+0]);
            atomicAdd(vj+kl_pair0+0, gout_2_0*dm[ij_pair0+2] + gout_3_0*dm[ij_pair0+3] + gout_5_0*dm[ij_pair0+5] + gout_6_0*dm[ij_pair0+6] + gout_7_0*dm[ij_pair0+7] + gout_8_0*dm[ij_pair0+8] + gout_9_0*dm[ij_pair0+9] + gout_11_0*dm[ij_pair0+11] + gout_12_0*dm[ij_pair0+12] + gout_13_0*dm[ij_pair0+13] + gout_14_0*dm[ij_pair0+14] + gout_15_0*dm[ij_pair0+15] + gout_16_0*dm[ij_pair0+16] + gout_17_0*dm[ij_pair0+17] + gout_18_0*dm[ij_pair0+18] + gout_19_0*dm[ij_pair0+19]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
__global__
static void rys_j_3_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_3_0(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_3_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_2_1 = 0.;
        double gout_2_2 = 0.;
        double gout_2_3 = 0.;
        double gout_3_1 = 0.;
        double gout_3_2 = 0.;
        double gout_3_3 = 0.;
        double gout_5_1 = 0.;
        double gout_5_2 = 0.;
        double gout_5_3 = 0.;
        double gout_6_1 = 0.;
        double gout_6_2 = 0.;
        double gout_6_3 = 0.;
        double gout_7_1 = 0.;
        double gout_7_2 = 0.;
        double gout_7_3 = 0.;
        double gout_8_1 = 0.;
        double gout_8_2 = 0.;
        double gout_8_3 = 0.;
        double gout_9_1 = 0.;
        double gout_9_2 = 0.;
        double gout_9_3 = 0.;
        double gout_11_1 = 0.;
        double gout_11_2 = 0.;
        double gout_11_3 = 0.;
        double gout_12_1 = 0.;
        double gout_12_2 = 0.;
        double gout_12_3 = 0.;
        double gout_13_1 = 0.;
        double gout_13_2 = 0.;
        double gout_13_3 = 0.;
        double gout_14_1 = 0.;
        double gout_14_2 = 0.;
        double gout_14_3 = 0.;
        double gout_15_1 = 0.;
        double gout_15_2 = 0.;
        double gout_15_3 = 0.;
        double gout_16_1 = 0.;
        double gout_16_2 = 0.;
        double gout_16_3 = 0.;
        double gout_17_1 = 0.;
        double gout_17_2 = 0.;
        double gout_17_3 = 0.;
        double gout_18_1 = 0.;
        double gout_18_2 = 0.;
        double gout_18_3 = 0.;
        double gout_19_1 = 0.;
        double gout_19_2 = 0.;
        double gout_19_3 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpz = zqc + zpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout_2_1 += fac * 1 * trr_21z;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    gout_2_2 += fac * trr_01y * trr_20z;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    gout_2_3 += trr_01x * 1 * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    gout_3_1 += fac * 1 * trr_31z;
                    gout_3_2 += fac * trr_01y * trr_30z;
                    gout_3_3 += trr_01x * 1 * trr_30z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout_5_1 += fac * trr_10y * trr_11z;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    gout_5_2 += fac * trr_11y * trr_10z;
                    gout_5_3 += trr_01x * trr_10y * trr_10z;
                    gout_6_1 += fac * trr_10y * trr_21z;
                    gout_6_2 += fac * trr_11y * trr_20z;
                    gout_6_3 += trr_01x * trr_10y * trr_20z;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double trr_01z = cpz * wt;
                    gout_7_1 += fac * trr_20y * trr_01z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout_7_2 += fac * trr_21y * wt;
                    gout_7_3 += trr_01x * trr_20y * wt;
                    gout_8_1 += fac * trr_20y * trr_11z;
                    gout_8_2 += fac * trr_21y * trr_10z;
                    gout_8_3 += trr_01x * trr_20y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout_9_1 += fac * trr_30y * trr_01z;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    gout_9_2 += fac * trr_31y * wt;
                    gout_9_3 += trr_01x * trr_30y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_11_1 += trr_10x * 1 * trr_11z;
                    gout_11_2 += trr_10x * trr_01y * trr_10z;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    gout_11_3 += trr_11x * 1 * trr_10z;
                    gout_12_1 += trr_10x * 1 * trr_21z;
                    gout_12_2 += trr_10x * trr_01y * trr_20z;
                    gout_12_3 += trr_11x * 1 * trr_20z;
                    gout_13_1 += trr_10x * trr_10y * trr_01z;
                    gout_13_2 += trr_10x * trr_11y * wt;
                    gout_13_3 += trr_11x * trr_10y * wt;
                    gout_14_1 += trr_10x * trr_10y * trr_11z;
                    gout_14_2 += trr_10x * trr_11y * trr_10z;
                    gout_14_3 += trr_11x * trr_10y * trr_10z;
                    gout_15_1 += trr_10x * trr_20y * trr_01z;
                    gout_15_2 += trr_10x * trr_21y * wt;
                    gout_15_3 += trr_11x * trr_20y * wt;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    gout_16_1 += trr_20x * 1 * trr_01z;
                    gout_16_2 += trr_20x * trr_01y * wt;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    gout_16_3 += trr_21x * 1 * wt;
                    gout_17_1 += trr_20x * 1 * trr_11z;
                    gout_17_2 += trr_20x * trr_01y * trr_10z;
                    gout_17_3 += trr_21x * 1 * trr_10z;
                    gout_18_1 += trr_20x * trr_10y * trr_01z;
                    gout_18_2 += trr_20x * trr_11y * wt;
                    gout_18_3 += trr_21x * trr_10y * wt;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    gout_19_1 += trr_30x * 1 * trr_01z;
                    gout_19_2 += trr_30x * trr_01y * wt;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    gout_19_3 += trr_31x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+2, gout_2_1*dm[kl_pair0+1] + gout_2_2*dm[kl_pair0+2] + gout_2_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+3, gout_3_1*dm[kl_pair0+1] + gout_3_2*dm[kl_pair0+2] + gout_3_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+5, gout_5_1*dm[kl_pair0+1] + gout_5_2*dm[kl_pair0+2] + gout_5_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+6, gout_6_1*dm[kl_pair0+1] + gout_6_2*dm[kl_pair0+2] + gout_6_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+7, gout_7_1*dm[kl_pair0+1] + gout_7_2*dm[kl_pair0+2] + gout_7_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+8, gout_8_1*dm[kl_pair0+1] + gout_8_2*dm[kl_pair0+2] + gout_8_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+9, gout_9_1*dm[kl_pair0+1] + gout_9_2*dm[kl_pair0+2] + gout_9_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+11, gout_11_1*dm[kl_pair0+1] + gout_11_2*dm[kl_pair0+2] + gout_11_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+12, gout_12_1*dm[kl_pair0+1] + gout_12_2*dm[kl_pair0+2] + gout_12_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+13, gout_13_1*dm[kl_pair0+1] + gout_13_2*dm[kl_pair0+2] + gout_13_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+14, gout_14_1*dm[kl_pair0+1] + gout_14_2*dm[kl_pair0+2] + gout_14_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+15, gout_15_1*dm[kl_pair0+1] + gout_15_2*dm[kl_pair0+2] + gout_15_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+16, gout_16_1*dm[kl_pair0+1] + gout_16_2*dm[kl_pair0+2] + gout_16_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+17, gout_17_1*dm[kl_pair0+1] + gout_17_2*dm[kl_pair0+2] + gout_17_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+18, gout_18_1*dm[kl_pair0+1] + gout_18_2*dm[kl_pair0+2] + gout_18_3*dm[kl_pair0+3]);
            atomicAdd(vj+ij_pair0+19, gout_19_1*dm[kl_pair0+1] + gout_19_2*dm[kl_pair0+2] + gout_19_3*dm[kl_pair0+3]);
            atomicAdd(vj+kl_pair0+1, gout_2_1*dm[ij_pair0+2] + gout_3_1*dm[ij_pair0+3] + gout_5_1*dm[ij_pair0+5] + gout_6_1*dm[ij_pair0+6] + gout_7_1*dm[ij_pair0+7] + gout_8_1*dm[ij_pair0+8] + gout_9_1*dm[ij_pair0+9] + gout_11_1*dm[ij_pair0+11] + gout_12_1*dm[ij_pair0+12] + gout_13_1*dm[ij_pair0+13] + gout_14_1*dm[ij_pair0+14] + gout_15_1*dm[ij_pair0+15] + gout_16_1*dm[ij_pair0+16] + gout_17_1*dm[ij_pair0+17] + gout_18_1*dm[ij_pair0+18] + gout_19_1*dm[ij_pair0+19]);
            atomicAdd(vj+kl_pair0+2, gout_2_2*dm[ij_pair0+2] + gout_3_2*dm[ij_pair0+3] + gout_5_2*dm[ij_pair0+5] + gout_6_2*dm[ij_pair0+6] + gout_7_2*dm[ij_pair0+7] + gout_8_2*dm[ij_pair0+8] + gout_9_2*dm[ij_pair0+9] + gout_11_2*dm[ij_pair0+11] + gout_12_2*dm[ij_pair0+12] + gout_13_2*dm[ij_pair0+13] + gout_14_2*dm[ij_pair0+14] + gout_15_2*dm[ij_pair0+15] + gout_16_2*dm[ij_pair0+16] + gout_17_2*dm[ij_pair0+17] + gout_18_2*dm[ij_pair0+18] + gout_19_2*dm[ij_pair0+19]);
            atomicAdd(vj+kl_pair0+3, gout_2_3*dm[ij_pair0+2] + gout_3_3*dm[ij_pair0+3] + gout_5_3*dm[ij_pair0+5] + gout_6_3*dm[ij_pair0+6] + gout_7_3*dm[ij_pair0+7] + gout_8_3*dm[ij_pair0+8] + gout_9_3*dm[ij_pair0+9] + gout_11_3*dm[ij_pair0+11] + gout_12_3*dm[ij_pair0+12] + gout_13_3*dm[ij_pair0+13] + gout_14_3*dm[ij_pair0+14] + gout_15_3*dm[ij_pair0+15] + gout_16_3*dm[ij_pair0+16] + gout_17_3*dm[ij_pair0+17] + gout_18_3*dm[ij_pair0+18] + gout_19_3*dm[ij_pair0+19]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
__global__
static void rys_j_3_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_3_1(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_3_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 20*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 20*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_001 = dm[kl_pair0+1];
        double dm_kl_002 = dm[kl_pair0+2];
        double dm_kl_010 = dm[kl_pair0+3];
        double dm_kl_011 = dm[kl_pair0+4];
        double dm_kl_020 = dm[kl_pair0+5];
        double dm_kl_100 = dm[kl_pair0+6];
        double dm_kl_101 = dm[kl_pair0+7];
        double dm_kl_110 = dm[kl_pair0+8];
        double dm_kl_200 = dm[kl_pair0+9];
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_030 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_120 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_210 = 0;
        double vj_ij_300 = 0;
        double vj_kl_001 = 0;
        double vj_kl_002 = 0;
        double vj_kl_010 = 0;
        double vj_kl_011 = 0;
        double vj_kl_020 = 0;
        double vj_kl_100 = 0;
        double vj_kl_101 = 0;
        double vj_kl_110 = 0;
        double vj_kl_200 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double dot_lij_z_000 = trr_20z * dm_ij_cache[sh_ij+2*TILE2] + trr_30z * dm_ij_cache[sh_ij+3*TILE2];
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double dot_lij_z_001 = trr_21z * dm_ij_cache[sh_ij+2*TILE2] + trr_31z * dm_ij_cache[sh_ij+3*TILE2];
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                    double dot_lij_z_002 = trr_22z * dm_ij_cache[sh_ij+2*TILE2] + trr_32z * dm_ij_cache[sh_ij+3*TILE2];
                    double dot_lij_z_010 = trr_10z * dm_ij_cache[sh_ij+5*TILE2] + trr_20z * dm_ij_cache[sh_ij+6*TILE2];
                    double dot_lij_z_011 = trr_11z * dm_ij_cache[sh_ij+5*TILE2] + trr_21z * dm_ij_cache[sh_ij+6*TILE2];
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double dot_lij_z_012 = trr_12z * dm_ij_cache[sh_ij+5*TILE2] + trr_22z * dm_ij_cache[sh_ij+6*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+7*TILE2] + trr_10z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+7*TILE2] + trr_11z * dm_ij_cache[sh_ij+8*TILE2];
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double dot_lij_z_022 = trr_02z * dm_ij_cache[sh_ij+7*TILE2] + trr_12z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_030 = wt * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_031 = trr_01z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_032 = trr_02z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_100 = trr_10z * dm_ij_cache[sh_ij+11*TILE2] + trr_20z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_101 = trr_11z * dm_ij_cache[sh_ij+11*TILE2] + trr_21z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_102 = trr_12z * dm_ij_cache[sh_ij+11*TILE2] + trr_22z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+13*TILE2] + trr_10z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+13*TILE2] + trr_11z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_112 = trr_02z * dm_ij_cache[sh_ij+13*TILE2] + trr_12z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_120 = wt * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_121 = trr_01z * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_122 = trr_02z * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+16*TILE2] + trr_10z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+16*TILE2] + trr_11z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_202 = trr_02z * dm_ij_cache[sh_ij+16*TILE2] + trr_12z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_210 = wt * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_211 = trr_01z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_212 = trr_02z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_300 = wt * dm_ij_cache[sh_ij+19*TILE2];
                    double dot_lij_z_301 = trr_01z * dm_ij_cache[sh_ij+19*TILE2];
                    double dot_lij_z_302 = trr_02z * dm_ij_cache[sh_ij+19*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031;
                    double dot_lij_y_002 = 1 * dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022 + trr_30y * dot_lij_z_032;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030;
                    double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021 + trr_31y * dot_lij_z_031;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                    double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020 + trr_32y * dot_lij_z_030;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121;
                    double dot_lij_y_102 = 1 * dot_lij_z_102 + trr_10y * dot_lij_z_112 + trr_20y * dot_lij_z_122;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120;
                    double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111 + trr_21y * dot_lij_z_121;
                    double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110 + trr_22y * dot_lij_z_120;
                    double dot_lij_y_200 = 1 * dot_lij_z_200 + trr_10y * dot_lij_z_210;
                    double dot_lij_y_201 = 1 * dot_lij_z_201 + trr_10y * dot_lij_z_211;
                    double dot_lij_y_202 = 1 * dot_lij_z_202 + trr_10y * dot_lij_z_212;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210;
                    double dot_lij_y_211 = trr_01y * dot_lij_z_201 + trr_11y * dot_lij_z_211;
                    double dot_lij_y_220 = trr_02y * dot_lij_z_200 + trr_12y * dot_lij_z_210;
                    double dot_lij_y_300 = 1 * dot_lij_z_300;
                    double dot_lij_y_301 = 1 * dot_lij_z_301;
                    double dot_lij_y_302 = 1 * dot_lij_z_302;
                    double dot_lij_y_310 = trr_01y * dot_lij_z_300;
                    double dot_lij_y_311 = trr_01y * dot_lij_z_301;
                    double dot_lij_y_320 = trr_02y * dot_lij_z_300;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    vj_kl_001 += fac * dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301;
                    vj_kl_002 += fac * dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202 + trr_30x * dot_lij_y_302;
                    vj_kl_010 += fac * dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310;
                    vj_kl_011 += fac * dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211 + trr_30x * dot_lij_y_311;
                    vj_kl_020 += fac * dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220 + trr_30x * dot_lij_y_320;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300;
                    vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201 + trr_31x * dot_lij_y_301;
                    vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210 + trr_31x * dot_lij_y_310;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                    vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200 + trr_32x * dot_lij_y_300;
                    double dot_lkl_z_000 = trr_01z * dm_kl_001 + trr_02z * dm_kl_002;
                    double dot_lkl_z_001 = trr_11z * dm_kl_001 + trr_12z * dm_kl_002;
                    double dot_lkl_z_002 = trr_21z * dm_kl_001 + trr_22z * dm_kl_002;
                    double dot_lkl_z_003 = trr_31z * dm_kl_001 + trr_32z * dm_kl_002;
                    double dot_lkl_z_010 = wt * dm_kl_010 + trr_01z * dm_kl_011;
                    double dot_lkl_z_011 = trr_10z * dm_kl_010 + trr_11z * dm_kl_011;
                    double dot_lkl_z_012 = trr_20z * dm_kl_010 + trr_21z * dm_kl_011;
                    double dot_lkl_z_013 = trr_30z * dm_kl_010 + trr_31z * dm_kl_011;
                    double dot_lkl_z_020 = wt * dm_kl_020;
                    double dot_lkl_z_021 = trr_10z * dm_kl_020;
                    double dot_lkl_z_022 = trr_20z * dm_kl_020;
                    double dot_lkl_z_023 = trr_30z * dm_kl_020;
                    double dot_lkl_z_100 = wt * dm_kl_100 + trr_01z * dm_kl_101;
                    double dot_lkl_z_101 = trr_10z * dm_kl_100 + trr_11z * dm_kl_101;
                    double dot_lkl_z_102 = trr_20z * dm_kl_100 + trr_21z * dm_kl_101;
                    double dot_lkl_z_103 = trr_30z * dm_kl_100 + trr_31z * dm_kl_101;
                    double dot_lkl_z_110 = wt * dm_kl_110;
                    double dot_lkl_z_111 = trr_10z * dm_kl_110;
                    double dot_lkl_z_112 = trr_20z * dm_kl_110;
                    double dot_lkl_z_113 = trr_30z * dm_kl_110;
                    double dot_lkl_z_200 = wt * dm_kl_200;
                    double dot_lkl_z_201 = trr_10z * dm_kl_200;
                    double dot_lkl_z_202 = trr_20z * dm_kl_200;
                    double dot_lkl_z_203 = trr_30z * dm_kl_200;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022;
                    double dot_lkl_y_003 = 1 * dot_lkl_z_003 + trr_01y * dot_lkl_z_013 + trr_02y * dot_lkl_z_023;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021;
                    double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012 + trr_12y * dot_lkl_z_022;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020;
                    double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011 + trr_22y * dot_lkl_z_021;
                    double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010 + trr_32y * dot_lkl_z_020;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102 + trr_01y * dot_lkl_z_112;
                    double dot_lkl_y_103 = 1 * dot_lkl_z_103 + trr_01y * dot_lkl_z_113;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111;
                    double dot_lkl_y_112 = trr_10y * dot_lkl_z_102 + trr_11y * dot_lkl_z_112;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110;
                    double dot_lkl_y_121 = trr_20y * dot_lkl_z_101 + trr_21y * dot_lkl_z_111;
                    double dot_lkl_y_130 = trr_30y * dot_lkl_z_100 + trr_31y * dot_lkl_z_110;
                    double dot_lkl_y_200 = 1 * dot_lkl_z_200;
                    double dot_lkl_y_201 = 1 * dot_lkl_z_201;
                    double dot_lkl_y_202 = 1 * dot_lkl_z_202;
                    double dot_lkl_y_203 = 1 * dot_lkl_z_203;
                    double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                    double dot_lkl_y_211 = trr_10y * dot_lkl_z_201;
                    double dot_lkl_y_212 = trr_10y * dot_lkl_z_202;
                    double dot_lkl_y_220 = trr_20y * dot_lkl_z_200;
                    double dot_lkl_y_221 = trr_20y * dot_lkl_z_201;
                    double dot_lkl_y_230 = trr_30y * dot_lkl_z_200;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202;
                    vj_ij_003 += fac * dot_lkl_y_003 + trr_01x * dot_lkl_y_103 + trr_02x * dot_lkl_y_203;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211;
                    vj_ij_012 += fac * dot_lkl_y_012 + trr_01x * dot_lkl_y_112 + trr_02x * dot_lkl_y_212;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220;
                    vj_ij_021 += fac * dot_lkl_y_021 + trr_01x * dot_lkl_y_121 + trr_02x * dot_lkl_y_221;
                    vj_ij_030 += fac * dot_lkl_y_030 + trr_01x * dot_lkl_y_130 + trr_02x * dot_lkl_y_230;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201;
                    vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102 + trr_12x * dot_lkl_y_202;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210;
                    vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111 + trr_12x * dot_lkl_y_211;
                    vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120 + trr_12x * dot_lkl_y_220;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200;
                    vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101 + trr_22x * dot_lkl_y_201;
                    vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110 + trr_22x * dot_lkl_y_210;
                    vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100 + trr_32x * dot_lkl_y_200;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_003);
        atomicAdd(vj+ij_pair0+5, vj_ij_011);
        atomicAdd(vj+ij_pair0+6, vj_ij_012);
        atomicAdd(vj+ij_pair0+7, vj_ij_020);
        atomicAdd(vj+ij_pair0+8, vj_ij_021);
        atomicAdd(vj+ij_pair0+9, vj_ij_030);
        atomicAdd(vj+ij_pair0+11, vj_ij_101);
        atomicAdd(vj+ij_pair0+12, vj_ij_102);
        atomicAdd(vj+ij_pair0+13, vj_ij_110);
        atomicAdd(vj+ij_pair0+14, vj_ij_111);
        atomicAdd(vj+ij_pair0+15, vj_ij_120);
        atomicAdd(vj+ij_pair0+16, vj_ij_200);
        atomicAdd(vj+ij_pair0+17, vj_ij_201);
        atomicAdd(vj+ij_pair0+18, vj_ij_210);
        atomicAdd(vj+ij_pair0+19, vj_ij_300);
        atomicAdd(vj+kl_pair0+1, vj_kl_001);
        atomicAdd(vj+kl_pair0+2, vj_kl_002);
        atomicAdd(vj+kl_pair0+3, vj_kl_010);
        atomicAdd(vj+kl_pair0+4, vj_kl_011);
        atomicAdd(vj+kl_pair0+5, vj_kl_020);
        atomicAdd(vj+kl_pair0+6, vj_kl_100);
        atomicAdd(vj+kl_pair0+7, vj_kl_101);
        atomicAdd(vj+kl_pair0+8, vj_kl_110);
        atomicAdd(vj+kl_pair0+9, vj_kl_200);
    }
}
__global__
static void rys_j_3_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_3_2(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_3_3(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 20*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 20*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_002 = dm[kl_pair0+2];
        double dm_kl_003 = dm[kl_pair0+3];
        double dm_kl_011 = dm[kl_pair0+5];
        double dm_kl_012 = dm[kl_pair0+6];
        double dm_kl_020 = dm[kl_pair0+7];
        double dm_kl_021 = dm[kl_pair0+8];
        double dm_kl_030 = dm[kl_pair0+9];
        double dm_kl_101 = dm[kl_pair0+11];
        double dm_kl_102 = dm[kl_pair0+12];
        double dm_kl_110 = dm[kl_pair0+13];
        double dm_kl_111 = dm[kl_pair0+14];
        double dm_kl_120 = dm[kl_pair0+15];
        double dm_kl_200 = dm[kl_pair0+16];
        double dm_kl_201 = dm[kl_pair0+17];
        double dm_kl_210 = dm[kl_pair0+18];
        double dm_kl_300 = dm[kl_pair0+19];
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_030 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_120 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_210 = 0;
        double vj_ij_300 = 0;
        double vj_kl_002 = 0;
        double vj_kl_003 = 0;
        double vj_kl_011 = 0;
        double vj_kl_012 = 0;
        double vj_kl_020 = 0;
        double vj_kl_021 = 0;
        double vj_kl_030 = 0;
        double vj_kl_101 = 0;
        double vj_kl_102 = 0;
        double vj_kl_110 = 0;
        double vj_kl_111 = 0;
        double vj_kl_120 = 0;
        double vj_kl_200 = 0;
        double vj_kl_201 = 0;
        double vj_kl_210 = 0;
        double vj_kl_300 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double dot_lij_z_000 = trr_20z * dm_ij_cache[sh_ij+2*TILE2] + trr_30z * dm_ij_cache[sh_ij+3*TILE2];
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double dot_lij_z_001 = trr_21z * dm_ij_cache[sh_ij+2*TILE2] + trr_31z * dm_ij_cache[sh_ij+3*TILE2];
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                    double dot_lij_z_002 = trr_22z * dm_ij_cache[sh_ij+2*TILE2] + trr_32z * dm_ij_cache[sh_ij+3*TILE2];
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                    double trr_33z = cpz * trr_32z + 2*b01 * trr_31z + 3*b00 * trr_22z;
                    double dot_lij_z_003 = trr_23z * dm_ij_cache[sh_ij+2*TILE2] + trr_33z * dm_ij_cache[sh_ij+3*TILE2];
                    double dot_lij_z_010 = trr_10z * dm_ij_cache[sh_ij+5*TILE2] + trr_20z * dm_ij_cache[sh_ij+6*TILE2];
                    double dot_lij_z_011 = trr_11z * dm_ij_cache[sh_ij+5*TILE2] + trr_21z * dm_ij_cache[sh_ij+6*TILE2];
                    double dot_lij_z_012 = trr_12z * dm_ij_cache[sh_ij+5*TILE2] + trr_22z * dm_ij_cache[sh_ij+6*TILE2];
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                    double dot_lij_z_013 = trr_13z * dm_ij_cache[sh_ij+5*TILE2] + trr_23z * dm_ij_cache[sh_ij+6*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+7*TILE2] + trr_10z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+7*TILE2] + trr_11z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_022 = trr_02z * dm_ij_cache[sh_ij+7*TILE2] + trr_12z * dm_ij_cache[sh_ij+8*TILE2];
                    double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                    double dot_lij_z_023 = trr_03z * dm_ij_cache[sh_ij+7*TILE2] + trr_13z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_030 = wt * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_031 = trr_01z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_032 = trr_02z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_033 = trr_03z * dm_ij_cache[sh_ij+9*TILE2];
                    double dot_lij_z_100 = trr_10z * dm_ij_cache[sh_ij+11*TILE2] + trr_20z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_101 = trr_11z * dm_ij_cache[sh_ij+11*TILE2] + trr_21z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_102 = trr_12z * dm_ij_cache[sh_ij+11*TILE2] + trr_22z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_103 = trr_13z * dm_ij_cache[sh_ij+11*TILE2] + trr_23z * dm_ij_cache[sh_ij+12*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+13*TILE2] + trr_10z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+13*TILE2] + trr_11z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_112 = trr_02z * dm_ij_cache[sh_ij+13*TILE2] + trr_12z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_113 = trr_03z * dm_ij_cache[sh_ij+13*TILE2] + trr_13z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_120 = wt * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_121 = trr_01z * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_122 = trr_02z * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_123 = trr_03z * dm_ij_cache[sh_ij+15*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+16*TILE2] + trr_10z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+16*TILE2] + trr_11z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_202 = trr_02z * dm_ij_cache[sh_ij+16*TILE2] + trr_12z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_203 = trr_03z * dm_ij_cache[sh_ij+16*TILE2] + trr_13z * dm_ij_cache[sh_ij+17*TILE2];
                    double dot_lij_z_210 = wt * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_211 = trr_01z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_212 = trr_02z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_213 = trr_03z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_300 = wt * dm_ij_cache[sh_ij+19*TILE2];
                    double dot_lij_z_301 = trr_01z * dm_ij_cache[sh_ij+19*TILE2];
                    double dot_lij_z_302 = trr_02z * dm_ij_cache[sh_ij+19*TILE2];
                    double dot_lij_z_303 = trr_03z * dm_ij_cache[sh_ij+19*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031;
                    double dot_lij_y_002 = 1 * dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022 + trr_30y * dot_lij_z_032;
                    double dot_lij_y_003 = 1 * dot_lij_z_003 + trr_10y * dot_lij_z_013 + trr_20y * dot_lij_z_023 + trr_30y * dot_lij_z_033;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030;
                    double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021 + trr_31y * dot_lij_z_031;
                    double dot_lij_y_012 = trr_01y * dot_lij_z_002 + trr_11y * dot_lij_z_012 + trr_21y * dot_lij_z_022 + trr_31y * dot_lij_z_032;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                    double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020 + trr_32y * dot_lij_z_030;
                    double dot_lij_y_021 = trr_02y * dot_lij_z_001 + trr_12y * dot_lij_z_011 + trr_22y * dot_lij_z_021 + trr_32y * dot_lij_z_031;
                    double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                    double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                    double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                    double trr_33y = cpy * trr_32y + 2*b01 * trr_31y + 3*b00 * trr_22y;
                    double dot_lij_y_030 = trr_03y * dot_lij_z_000 + trr_13y * dot_lij_z_010 + trr_23y * dot_lij_z_020 + trr_33y * dot_lij_z_030;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121;
                    double dot_lij_y_102 = 1 * dot_lij_z_102 + trr_10y * dot_lij_z_112 + trr_20y * dot_lij_z_122;
                    double dot_lij_y_103 = 1 * dot_lij_z_103 + trr_10y * dot_lij_z_113 + trr_20y * dot_lij_z_123;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120;
                    double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111 + trr_21y * dot_lij_z_121;
                    double dot_lij_y_112 = trr_01y * dot_lij_z_102 + trr_11y * dot_lij_z_112 + trr_21y * dot_lij_z_122;
                    double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110 + trr_22y * dot_lij_z_120;
                    double dot_lij_y_121 = trr_02y * dot_lij_z_101 + trr_12y * dot_lij_z_111 + trr_22y * dot_lij_z_121;
                    double dot_lij_y_130 = trr_03y * dot_lij_z_100 + trr_13y * dot_lij_z_110 + trr_23y * dot_lij_z_120;
                    double dot_lij_y_200 = 1 * dot_lij_z_200 + trr_10y * dot_lij_z_210;
                    double dot_lij_y_201 = 1 * dot_lij_z_201 + trr_10y * dot_lij_z_211;
                    double dot_lij_y_202 = 1 * dot_lij_z_202 + trr_10y * dot_lij_z_212;
                    double dot_lij_y_203 = 1 * dot_lij_z_203 + trr_10y * dot_lij_z_213;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210;
                    double dot_lij_y_211 = trr_01y * dot_lij_z_201 + trr_11y * dot_lij_z_211;
                    double dot_lij_y_212 = trr_01y * dot_lij_z_202 + trr_11y * dot_lij_z_212;
                    double dot_lij_y_220 = trr_02y * dot_lij_z_200 + trr_12y * dot_lij_z_210;
                    double dot_lij_y_221 = trr_02y * dot_lij_z_201 + trr_12y * dot_lij_z_211;
                    double dot_lij_y_230 = trr_03y * dot_lij_z_200 + trr_13y * dot_lij_z_210;
                    double dot_lij_y_300 = 1 * dot_lij_z_300;
                    double dot_lij_y_301 = 1 * dot_lij_z_301;
                    double dot_lij_y_302 = 1 * dot_lij_z_302;
                    double dot_lij_y_303 = 1 * dot_lij_z_303;
                    double dot_lij_y_310 = trr_01y * dot_lij_z_300;
                    double dot_lij_y_311 = trr_01y * dot_lij_z_301;
                    double dot_lij_y_312 = trr_01y * dot_lij_z_302;
                    double dot_lij_y_320 = trr_02y * dot_lij_z_300;
                    double dot_lij_y_321 = trr_02y * dot_lij_z_301;
                    double dot_lij_y_330 = trr_03y * dot_lij_z_300;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    vj_kl_002 += fac * dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202 + trr_30x * dot_lij_y_302;
                    vj_kl_003 += fac * dot_lij_y_003 + trr_10x * dot_lij_y_103 + trr_20x * dot_lij_y_203 + trr_30x * dot_lij_y_303;
                    vj_kl_011 += fac * dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211 + trr_30x * dot_lij_y_311;
                    vj_kl_012 += fac * dot_lij_y_012 + trr_10x * dot_lij_y_112 + trr_20x * dot_lij_y_212 + trr_30x * dot_lij_y_312;
                    vj_kl_020 += fac * dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220 + trr_30x * dot_lij_y_320;
                    vj_kl_021 += fac * dot_lij_y_021 + trr_10x * dot_lij_y_121 + trr_20x * dot_lij_y_221 + trr_30x * dot_lij_y_321;
                    vj_kl_030 += fac * dot_lij_y_030 + trr_10x * dot_lij_y_130 + trr_20x * dot_lij_y_230 + trr_30x * dot_lij_y_330;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201 + trr_31x * dot_lij_y_301;
                    vj_kl_102 += trr_01x * dot_lij_y_002 + trr_11x * dot_lij_y_102 + trr_21x * dot_lij_y_202 + trr_31x * dot_lij_y_302;
                    vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210 + trr_31x * dot_lij_y_310;
                    vj_kl_111 += trr_01x * dot_lij_y_011 + trr_11x * dot_lij_y_111 + trr_21x * dot_lij_y_211 + trr_31x * dot_lij_y_311;
                    vj_kl_120 += trr_01x * dot_lij_y_020 + trr_11x * dot_lij_y_120 + trr_21x * dot_lij_y_220 + trr_31x * dot_lij_y_320;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                    vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200 + trr_32x * dot_lij_y_300;
                    vj_kl_201 += trr_02x * dot_lij_y_001 + trr_12x * dot_lij_y_101 + trr_22x * dot_lij_y_201 + trr_32x * dot_lij_y_301;
                    vj_kl_210 += trr_02x * dot_lij_y_010 + trr_12x * dot_lij_y_110 + trr_22x * dot_lij_y_210 + trr_32x * dot_lij_y_310;
                    double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                    double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                    double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                    double trr_33x = cpx * trr_32x + 2*b01 * trr_31x + 3*b00 * trr_22x;
                    vj_kl_300 += trr_03x * dot_lij_y_000 + trr_13x * dot_lij_y_100 + trr_23x * dot_lij_y_200 + trr_33x * dot_lij_y_300;
                    double dot_lkl_z_000 = trr_02z * dm_kl_002 + trr_03z * dm_kl_003;
                    double dot_lkl_z_001 = trr_12z * dm_kl_002 + trr_13z * dm_kl_003;
                    double dot_lkl_z_002 = trr_22z * dm_kl_002 + trr_23z * dm_kl_003;
                    double dot_lkl_z_003 = trr_32z * dm_kl_002 + trr_33z * dm_kl_003;
                    double dot_lkl_z_010 = trr_01z * dm_kl_011 + trr_02z * dm_kl_012;
                    double dot_lkl_z_011 = trr_11z * dm_kl_011 + trr_12z * dm_kl_012;
                    double dot_lkl_z_012 = trr_21z * dm_kl_011 + trr_22z * dm_kl_012;
                    double dot_lkl_z_013 = trr_31z * dm_kl_011 + trr_32z * dm_kl_012;
                    double dot_lkl_z_020 = wt * dm_kl_020 + trr_01z * dm_kl_021;
                    double dot_lkl_z_021 = trr_10z * dm_kl_020 + trr_11z * dm_kl_021;
                    double dot_lkl_z_022 = trr_20z * dm_kl_020 + trr_21z * dm_kl_021;
                    double dot_lkl_z_023 = trr_30z * dm_kl_020 + trr_31z * dm_kl_021;
                    double dot_lkl_z_030 = wt * dm_kl_030;
                    double dot_lkl_z_031 = trr_10z * dm_kl_030;
                    double dot_lkl_z_032 = trr_20z * dm_kl_030;
                    double dot_lkl_z_033 = trr_30z * dm_kl_030;
                    double dot_lkl_z_100 = trr_01z * dm_kl_101 + trr_02z * dm_kl_102;
                    double dot_lkl_z_101 = trr_11z * dm_kl_101 + trr_12z * dm_kl_102;
                    double dot_lkl_z_102 = trr_21z * dm_kl_101 + trr_22z * dm_kl_102;
                    double dot_lkl_z_103 = trr_31z * dm_kl_101 + trr_32z * dm_kl_102;
                    double dot_lkl_z_110 = wt * dm_kl_110 + trr_01z * dm_kl_111;
                    double dot_lkl_z_111 = trr_10z * dm_kl_110 + trr_11z * dm_kl_111;
                    double dot_lkl_z_112 = trr_20z * dm_kl_110 + trr_21z * dm_kl_111;
                    double dot_lkl_z_113 = trr_30z * dm_kl_110 + trr_31z * dm_kl_111;
                    double dot_lkl_z_120 = wt * dm_kl_120;
                    double dot_lkl_z_121 = trr_10z * dm_kl_120;
                    double dot_lkl_z_122 = trr_20z * dm_kl_120;
                    double dot_lkl_z_123 = trr_30z * dm_kl_120;
                    double dot_lkl_z_200 = wt * dm_kl_200 + trr_01z * dm_kl_201;
                    double dot_lkl_z_201 = trr_10z * dm_kl_200 + trr_11z * dm_kl_201;
                    double dot_lkl_z_202 = trr_20z * dm_kl_200 + trr_21z * dm_kl_201;
                    double dot_lkl_z_203 = trr_30z * dm_kl_200 + trr_31z * dm_kl_201;
                    double dot_lkl_z_210 = wt * dm_kl_210;
                    double dot_lkl_z_211 = trr_10z * dm_kl_210;
                    double dot_lkl_z_212 = trr_20z * dm_kl_210;
                    double dot_lkl_z_213 = trr_30z * dm_kl_210;
                    double dot_lkl_z_300 = wt * dm_kl_300;
                    double dot_lkl_z_301 = trr_10z * dm_kl_300;
                    double dot_lkl_z_302 = trr_20z * dm_kl_300;
                    double dot_lkl_z_303 = trr_30z * dm_kl_300;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020 + trr_03y * dot_lkl_z_030;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021 + trr_03y * dot_lkl_z_031;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022 + trr_03y * dot_lkl_z_032;
                    double dot_lkl_y_003 = 1 * dot_lkl_z_003 + trr_01y * dot_lkl_z_013 + trr_02y * dot_lkl_z_023 + trr_03y * dot_lkl_z_033;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020 + trr_13y * dot_lkl_z_030;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021 + trr_13y * dot_lkl_z_031;
                    double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012 + trr_12y * dot_lkl_z_022 + trr_13y * dot_lkl_z_032;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020 + trr_23y * dot_lkl_z_030;
                    double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011 + trr_22y * dot_lkl_z_021 + trr_23y * dot_lkl_z_031;
                    double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010 + trr_32y * dot_lkl_z_020 + trr_33y * dot_lkl_z_030;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100 + trr_01y * dot_lkl_z_110 + trr_02y * dot_lkl_z_120;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101 + trr_01y * dot_lkl_z_111 + trr_02y * dot_lkl_z_121;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102 + trr_01y * dot_lkl_z_112 + trr_02y * dot_lkl_z_122;
                    double dot_lkl_y_103 = 1 * dot_lkl_z_103 + trr_01y * dot_lkl_z_113 + trr_02y * dot_lkl_z_123;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110 + trr_12y * dot_lkl_z_120;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111 + trr_12y * dot_lkl_z_121;
                    double dot_lkl_y_112 = trr_10y * dot_lkl_z_102 + trr_11y * dot_lkl_z_112 + trr_12y * dot_lkl_z_122;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110 + trr_22y * dot_lkl_z_120;
                    double dot_lkl_y_121 = trr_20y * dot_lkl_z_101 + trr_21y * dot_lkl_z_111 + trr_22y * dot_lkl_z_121;
                    double dot_lkl_y_130 = trr_30y * dot_lkl_z_100 + trr_31y * dot_lkl_z_110 + trr_32y * dot_lkl_z_120;
                    double dot_lkl_y_200 = 1 * dot_lkl_z_200 + trr_01y * dot_lkl_z_210;
                    double dot_lkl_y_201 = 1 * dot_lkl_z_201 + trr_01y * dot_lkl_z_211;
                    double dot_lkl_y_202 = 1 * dot_lkl_z_202 + trr_01y * dot_lkl_z_212;
                    double dot_lkl_y_203 = 1 * dot_lkl_z_203 + trr_01y * dot_lkl_z_213;
                    double dot_lkl_y_210 = trr_10y * dot_lkl_z_200 + trr_11y * dot_lkl_z_210;
                    double dot_lkl_y_211 = trr_10y * dot_lkl_z_201 + trr_11y * dot_lkl_z_211;
                    double dot_lkl_y_212 = trr_10y * dot_lkl_z_202 + trr_11y * dot_lkl_z_212;
                    double dot_lkl_y_220 = trr_20y * dot_lkl_z_200 + trr_21y * dot_lkl_z_210;
                    double dot_lkl_y_221 = trr_20y * dot_lkl_z_201 + trr_21y * dot_lkl_z_211;
                    double dot_lkl_y_230 = trr_30y * dot_lkl_z_200 + trr_31y * dot_lkl_z_210;
                    double dot_lkl_y_300 = 1 * dot_lkl_z_300;
                    double dot_lkl_y_301 = 1 * dot_lkl_z_301;
                    double dot_lkl_y_302 = 1 * dot_lkl_z_302;
                    double dot_lkl_y_303 = 1 * dot_lkl_z_303;
                    double dot_lkl_y_310 = trr_10y * dot_lkl_z_300;
                    double dot_lkl_y_311 = trr_10y * dot_lkl_z_301;
                    double dot_lkl_y_312 = trr_10y * dot_lkl_z_302;
                    double dot_lkl_y_320 = trr_20y * dot_lkl_z_300;
                    double dot_lkl_y_321 = trr_20y * dot_lkl_z_301;
                    double dot_lkl_y_330 = trr_30y * dot_lkl_z_300;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202 + trr_03x * dot_lkl_y_302;
                    vj_ij_003 += fac * dot_lkl_y_003 + trr_01x * dot_lkl_y_103 + trr_02x * dot_lkl_y_203 + trr_03x * dot_lkl_y_303;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211 + trr_03x * dot_lkl_y_311;
                    vj_ij_012 += fac * dot_lkl_y_012 + trr_01x * dot_lkl_y_112 + trr_02x * dot_lkl_y_212 + trr_03x * dot_lkl_y_312;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220 + trr_03x * dot_lkl_y_320;
                    vj_ij_021 += fac * dot_lkl_y_021 + trr_01x * dot_lkl_y_121 + trr_02x * dot_lkl_y_221 + trr_03x * dot_lkl_y_321;
                    vj_ij_030 += fac * dot_lkl_y_030 + trr_01x * dot_lkl_y_130 + trr_02x * dot_lkl_y_230 + trr_03x * dot_lkl_y_330;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201 + trr_13x * dot_lkl_y_301;
                    vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102 + trr_12x * dot_lkl_y_202 + trr_13x * dot_lkl_y_302;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210 + trr_13x * dot_lkl_y_310;
                    vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111 + trr_12x * dot_lkl_y_211 + trr_13x * dot_lkl_y_311;
                    vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120 + trr_12x * dot_lkl_y_220 + trr_13x * dot_lkl_y_320;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200 + trr_23x * dot_lkl_y_300;
                    vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101 + trr_22x * dot_lkl_y_201 + trr_23x * dot_lkl_y_301;
                    vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110 + trr_22x * dot_lkl_y_210 + trr_23x * dot_lkl_y_310;
                    vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100 + trr_32x * dot_lkl_y_200 + trr_33x * dot_lkl_y_300;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_003);
        atomicAdd(vj+ij_pair0+5, vj_ij_011);
        atomicAdd(vj+ij_pair0+6, vj_ij_012);
        atomicAdd(vj+ij_pair0+7, vj_ij_020);
        atomicAdd(vj+ij_pair0+8, vj_ij_021);
        atomicAdd(vj+ij_pair0+9, vj_ij_030);
        atomicAdd(vj+ij_pair0+11, vj_ij_101);
        atomicAdd(vj+ij_pair0+12, vj_ij_102);
        atomicAdd(vj+ij_pair0+13, vj_ij_110);
        atomicAdd(vj+ij_pair0+14, vj_ij_111);
        atomicAdd(vj+ij_pair0+15, vj_ij_120);
        atomicAdd(vj+ij_pair0+16, vj_ij_200);
        atomicAdd(vj+ij_pair0+17, vj_ij_201);
        atomicAdd(vj+ij_pair0+18, vj_ij_210);
        atomicAdd(vj+ij_pair0+19, vj_ij_300);
        atomicAdd(vj+kl_pair0+2, vj_kl_002);
        atomicAdd(vj+kl_pair0+3, vj_kl_003);
        atomicAdd(vj+kl_pair0+5, vj_kl_011);
        atomicAdd(vj+kl_pair0+6, vj_kl_012);
        atomicAdd(vj+kl_pair0+7, vj_kl_020);
        atomicAdd(vj+kl_pair0+8, vj_kl_021);
        atomicAdd(vj+kl_pair0+9, vj_kl_030);
        atomicAdd(vj+kl_pair0+11, vj_kl_101);
        atomicAdd(vj+kl_pair0+12, vj_kl_102);
        atomicAdd(vj+kl_pair0+13, vj_kl_110);
        atomicAdd(vj+kl_pair0+14, vj_kl_111);
        atomicAdd(vj+kl_pair0+15, vj_kl_120);
        atomicAdd(vj+kl_pair0+16, vj_kl_200);
        atomicAdd(vj+kl_pair0+17, vj_kl_201);
        atomicAdd(vj+kl_pair0+18, vj_kl_210);
        atomicAdd(vj+kl_pair0+19, vj_kl_300);
    }
}
__global__
static void rys_j_3_3(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_3_3(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_4_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double gout_2_0 = 0.;
        double gout_3_0 = 0.;
        double gout_4_0 = 0.;
        double gout_6_0 = 0.;
        double gout_7_0 = 0.;
        double gout_8_0 = 0.;
        double gout_9_0 = 0.;
        double gout_10_0 = 0.;
        double gout_11_0 = 0.;
        double gout_12_0 = 0.;
        double gout_13_0 = 0.;
        double gout_14_0 = 0.;
        double gout_16_0 = 0.;
        double gout_17_0 = 0.;
        double gout_18_0 = 0.;
        double gout_19_0 = 0.;
        double gout_20_0 = 0.;
        double gout_21_0 = 0.;
        double gout_22_0 = 0.;
        double gout_23_0 = 0.;
        double gout_24_0 = 0.;
        double gout_25_0 = 0.;
        double gout_26_0 = 0.;
        double gout_27_0 = 0.;
        double gout_28_0 = 0.;
        double gout_29_0 = 0.;
        double gout_30_0 = 0.;
        double gout_31_0 = 0.;
        double gout_32_0 = 0.;
        double gout_33_0 = 0.;
        double gout_34_0 = 0.;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout_2_0 += fac * 1 * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout_3_0 += fac * 1 * trr_30z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    gout_4_0 += fac * 1 * trr_40z;
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    gout_6_0 += fac * trr_10y * trr_10z;
                    gout_7_0 += fac * trr_10y * trr_20z;
                    gout_8_0 += fac * trr_10y * trr_30z;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    gout_9_0 += fac * trr_20y * wt;
                    gout_10_0 += fac * trr_20y * trr_10z;
                    gout_11_0 += fac * trr_20y * trr_20z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout_12_0 += fac * trr_30y * wt;
                    gout_13_0 += fac * trr_30y * trr_10z;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    gout_14_0 += fac * trr_40y * wt;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    gout_16_0 += trr_10x * 1 * trr_10z;
                    gout_17_0 += trr_10x * 1 * trr_20z;
                    gout_18_0 += trr_10x * 1 * trr_30z;
                    gout_19_0 += trr_10x * trr_10y * wt;
                    gout_20_0 += trr_10x * trr_10y * trr_10z;
                    gout_21_0 += trr_10x * trr_10y * trr_20z;
                    gout_22_0 += trr_10x * trr_20y * wt;
                    gout_23_0 += trr_10x * trr_20y * trr_10z;
                    gout_24_0 += trr_10x * trr_30y * wt;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    gout_25_0 += trr_20x * 1 * wt;
                    gout_26_0 += trr_20x * 1 * trr_10z;
                    gout_27_0 += trr_20x * 1 * trr_20z;
                    gout_28_0 += trr_20x * trr_10y * wt;
                    gout_29_0 += trr_20x * trr_10y * trr_10z;
                    gout_30_0 += trr_20x * trr_20y * wt;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    gout_31_0 += trr_30x * 1 * wt;
                    gout_32_0 += trr_30x * 1 * trr_10z;
                    gout_33_0 += trr_30x * trr_10y * wt;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    gout_34_0 += trr_40x * 1 * wt;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int nao_pairs = pair_loc[nbas*nbas];
        double *vj = jk.vj;
        double *dm = jk.dm;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
            atomicAdd(vj+ij_pair0+2, gout_2_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+3, gout_3_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+4, gout_4_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+6, gout_6_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+7, gout_7_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+8, gout_8_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+9, gout_9_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+10, gout_10_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+11, gout_11_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+12, gout_12_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+13, gout_13_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+14, gout_14_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+16, gout_16_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+17, gout_17_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+18, gout_18_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+19, gout_19_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+20, gout_20_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+21, gout_21_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+22, gout_22_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+23, gout_23_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+24, gout_24_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+25, gout_25_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+26, gout_26_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+27, gout_27_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+28, gout_28_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+29, gout_29_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+30, gout_30_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+31, gout_31_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+32, gout_32_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+33, gout_33_0*dm[kl_pair0+0]);
            atomicAdd(vj+ij_pair0+34, gout_34_0*dm[kl_pair0+0]);
            atomicAdd(vj+kl_pair0+0, gout_2_0*dm[ij_pair0+2] + gout_3_0*dm[ij_pair0+3] + gout_4_0*dm[ij_pair0+4] + gout_6_0*dm[ij_pair0+6] + gout_7_0*dm[ij_pair0+7] + gout_8_0*dm[ij_pair0+8] + gout_9_0*dm[ij_pair0+9] + gout_10_0*dm[ij_pair0+10] + gout_11_0*dm[ij_pair0+11] + gout_12_0*dm[ij_pair0+12] + gout_13_0*dm[ij_pair0+13] + gout_14_0*dm[ij_pair0+14] + gout_16_0*dm[ij_pair0+16] + gout_17_0*dm[ij_pair0+17] + gout_18_0*dm[ij_pair0+18] + gout_19_0*dm[ij_pair0+19] + gout_20_0*dm[ij_pair0+20] + gout_21_0*dm[ij_pair0+21] + gout_22_0*dm[ij_pair0+22] + gout_23_0*dm[ij_pair0+23] + gout_24_0*dm[ij_pair0+24] + gout_25_0*dm[ij_pair0+25] + gout_26_0*dm[ij_pair0+26] + gout_27_0*dm[ij_pair0+27] + gout_28_0*dm[ij_pair0+28] + gout_29_0*dm[ij_pair0+29] + gout_30_0*dm[ij_pair0+30] + gout_31_0*dm[ij_pair0+31] + gout_32_0*dm[ij_pair0+32] + gout_33_0*dm[ij_pair0+33] + gout_34_0*dm[ij_pair0+34]);
            vj += nao_pairs;
            dm += nao_pairs;
        }
    }
}
__global__
static void rys_j_4_0(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_4_0(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_4_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 35*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 35*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_001 = dm[kl_pair0+1];
        double dm_kl_010 = dm[kl_pair0+2];
        double dm_kl_100 = dm[kl_pair0+3];
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_004 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_013 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_022 = 0;
        double vj_ij_030 = 0;
        double vj_ij_031 = 0;
        double vj_ij_040 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_103 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_112 = 0;
        double vj_ij_120 = 0;
        double vj_ij_121 = 0;
        double vj_ij_130 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_202 = 0;
        double vj_ij_210 = 0;
        double vj_ij_211 = 0;
        double vj_ij_220 = 0;
        double vj_ij_300 = 0;
        double vj_ij_301 = 0;
        double vj_ij_310 = 0;
        double vj_ij_400 = 0;
        double vj_kl_001 = 0;
        double vj_kl_010 = 0;
        double vj_kl_100 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double dot_lij_z_000 = trr_20z * dm_ij_cache[sh_ij+2*TILE2] + trr_30z * dm_ij_cache[sh_ij+3*TILE2] + trr_40z * dm_ij_cache[sh_ij+4*TILE2];
                    double rt_akl = rt_aa * aij;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double trr_41z = cpz * trr_40z + 4*b00 * trr_30z;
                    double dot_lij_z_001 = trr_21z * dm_ij_cache[sh_ij+2*TILE2] + trr_31z * dm_ij_cache[sh_ij+3*TILE2] + trr_41z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_010 = trr_10z * dm_ij_cache[sh_ij+6*TILE2] + trr_20z * dm_ij_cache[sh_ij+7*TILE2] + trr_30z * dm_ij_cache[sh_ij+8*TILE2];
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double dot_lij_z_011 = trr_11z * dm_ij_cache[sh_ij+6*TILE2] + trr_21z * dm_ij_cache[sh_ij+7*TILE2] + trr_31z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+9*TILE2] + trr_10z * dm_ij_cache[sh_ij+10*TILE2] + trr_20z * dm_ij_cache[sh_ij+11*TILE2];
                    double trr_01z = cpz * wt;
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+9*TILE2] + trr_11z * dm_ij_cache[sh_ij+10*TILE2] + trr_21z * dm_ij_cache[sh_ij+11*TILE2];
                    double dot_lij_z_030 = wt * dm_ij_cache[sh_ij+12*TILE2] + trr_10z * dm_ij_cache[sh_ij+13*TILE2];
                    double dot_lij_z_031 = trr_01z * dm_ij_cache[sh_ij+12*TILE2] + trr_11z * dm_ij_cache[sh_ij+13*TILE2];
                    double dot_lij_z_040 = wt * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_041 = trr_01z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_100 = trr_10z * dm_ij_cache[sh_ij+16*TILE2] + trr_20z * dm_ij_cache[sh_ij+17*TILE2] + trr_30z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_101 = trr_11z * dm_ij_cache[sh_ij+16*TILE2] + trr_21z * dm_ij_cache[sh_ij+17*TILE2] + trr_31z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+19*TILE2] + trr_10z * dm_ij_cache[sh_ij+20*TILE2] + trr_20z * dm_ij_cache[sh_ij+21*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+19*TILE2] + trr_11z * dm_ij_cache[sh_ij+20*TILE2] + trr_21z * dm_ij_cache[sh_ij+21*TILE2];
                    double dot_lij_z_120 = wt * dm_ij_cache[sh_ij+22*TILE2] + trr_10z * dm_ij_cache[sh_ij+23*TILE2];
                    double dot_lij_z_121 = trr_01z * dm_ij_cache[sh_ij+22*TILE2] + trr_11z * dm_ij_cache[sh_ij+23*TILE2];
                    double dot_lij_z_130 = wt * dm_ij_cache[sh_ij+24*TILE2];
                    double dot_lij_z_131 = trr_01z * dm_ij_cache[sh_ij+24*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+25*TILE2] + trr_10z * dm_ij_cache[sh_ij+26*TILE2] + trr_20z * dm_ij_cache[sh_ij+27*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+25*TILE2] + trr_11z * dm_ij_cache[sh_ij+26*TILE2] + trr_21z * dm_ij_cache[sh_ij+27*TILE2];
                    double dot_lij_z_210 = wt * dm_ij_cache[sh_ij+28*TILE2] + trr_10z * dm_ij_cache[sh_ij+29*TILE2];
                    double dot_lij_z_211 = trr_01z * dm_ij_cache[sh_ij+28*TILE2] + trr_11z * dm_ij_cache[sh_ij+29*TILE2];
                    double dot_lij_z_220 = wt * dm_ij_cache[sh_ij+30*TILE2];
                    double dot_lij_z_221 = trr_01z * dm_ij_cache[sh_ij+30*TILE2];
                    double dot_lij_z_300 = wt * dm_ij_cache[sh_ij+31*TILE2] + trr_10z * dm_ij_cache[sh_ij+32*TILE2];
                    double dot_lij_z_301 = trr_01z * dm_ij_cache[sh_ij+31*TILE2] + trr_11z * dm_ij_cache[sh_ij+32*TILE2];
                    double dot_lij_z_310 = wt * dm_ij_cache[sh_ij+33*TILE2];
                    double dot_lij_z_311 = trr_01z * dm_ij_cache[sh_ij+33*TILE2];
                    double dot_lij_z_400 = wt * dm_ij_cache[sh_ij+34*TILE2];
                    double dot_lij_z_401 = trr_01z * dm_ij_cache[sh_ij+34*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030 + trr_40y * dot_lij_z_040;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031 + trr_40y * dot_lij_z_041;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double trr_41y = cpy * trr_40y + 4*b00 * trr_30y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030 + trr_41y * dot_lij_z_040;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120 + trr_30y * dot_lij_z_130;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121 + trr_30y * dot_lij_z_131;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120 + trr_31y * dot_lij_z_130;
                    double dot_lij_y_200 = 1 * dot_lij_z_200 + trr_10y * dot_lij_z_210 + trr_20y * dot_lij_z_220;
                    double dot_lij_y_201 = 1 * dot_lij_z_201 + trr_10y * dot_lij_z_211 + trr_20y * dot_lij_z_221;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210 + trr_21y * dot_lij_z_220;
                    double dot_lij_y_300 = 1 * dot_lij_z_300 + trr_10y * dot_lij_z_310;
                    double dot_lij_y_301 = 1 * dot_lij_z_301 + trr_10y * dot_lij_z_311;
                    double dot_lij_y_310 = trr_01y * dot_lij_z_300 + trr_11y * dot_lij_z_310;
                    double dot_lij_y_400 = 1 * dot_lij_z_400;
                    double dot_lij_y_401 = 1 * dot_lij_z_401;
                    double dot_lij_y_410 = trr_01y * dot_lij_z_400;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    vj_kl_001 += fac * dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301 + trr_40x * dot_lij_y_401;
                    vj_kl_010 += fac * dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310 + trr_40x * dot_lij_y_410;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double trr_41x = cpx * trr_40x + 4*b00 * trr_30x;
                    vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300 + trr_41x * dot_lij_y_400;
                    double dot_lkl_z_000 = trr_01z * dm_kl_001;
                    double dot_lkl_z_001 = trr_11z * dm_kl_001;
                    double dot_lkl_z_002 = trr_21z * dm_kl_001;
                    double dot_lkl_z_003 = trr_31z * dm_kl_001;
                    double dot_lkl_z_004 = trr_41z * dm_kl_001;
                    double dot_lkl_z_010 = wt * dm_kl_010;
                    double dot_lkl_z_011 = trr_10z * dm_kl_010;
                    double dot_lkl_z_012 = trr_20z * dm_kl_010;
                    double dot_lkl_z_013 = trr_30z * dm_kl_010;
                    double dot_lkl_z_014 = trr_40z * dm_kl_010;
                    double dot_lkl_z_100 = wt * dm_kl_100;
                    double dot_lkl_z_101 = trr_10z * dm_kl_100;
                    double dot_lkl_z_102 = trr_20z * dm_kl_100;
                    double dot_lkl_z_103 = trr_30z * dm_kl_100;
                    double dot_lkl_z_104 = trr_40z * dm_kl_100;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012;
                    double dot_lkl_y_003 = 1 * dot_lkl_z_003 + trr_01y * dot_lkl_z_013;
                    double dot_lkl_y_004 = 1 * dot_lkl_z_004 + trr_01y * dot_lkl_z_014;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011;
                    double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012;
                    double dot_lkl_y_013 = trr_10y * dot_lkl_z_003 + trr_11y * dot_lkl_z_013;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010;
                    double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011;
                    double dot_lkl_y_022 = trr_20y * dot_lkl_z_002 + trr_21y * dot_lkl_z_012;
                    double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010;
                    double dot_lkl_y_031 = trr_30y * dot_lkl_z_001 + trr_31y * dot_lkl_z_011;
                    double dot_lkl_y_040 = trr_40y * dot_lkl_z_000 + trr_41y * dot_lkl_z_010;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102;
                    double dot_lkl_y_103 = 1 * dot_lkl_z_103;
                    double dot_lkl_y_104 = 1 * dot_lkl_z_104;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101;
                    double dot_lkl_y_112 = trr_10y * dot_lkl_z_102;
                    double dot_lkl_y_113 = trr_10y * dot_lkl_z_103;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100;
                    double dot_lkl_y_121 = trr_20y * dot_lkl_z_101;
                    double dot_lkl_y_122 = trr_20y * dot_lkl_z_102;
                    double dot_lkl_y_130 = trr_30y * dot_lkl_z_100;
                    double dot_lkl_y_131 = trr_30y * dot_lkl_z_101;
                    double dot_lkl_y_140 = trr_40y * dot_lkl_z_100;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102;
                    vj_ij_003 += fac * dot_lkl_y_003 + trr_01x * dot_lkl_y_103;
                    vj_ij_004 += fac * dot_lkl_y_004 + trr_01x * dot_lkl_y_104;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111;
                    vj_ij_012 += fac * dot_lkl_y_012 + trr_01x * dot_lkl_y_112;
                    vj_ij_013 += fac * dot_lkl_y_013 + trr_01x * dot_lkl_y_113;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120;
                    vj_ij_021 += fac * dot_lkl_y_021 + trr_01x * dot_lkl_y_121;
                    vj_ij_022 += fac * dot_lkl_y_022 + trr_01x * dot_lkl_y_122;
                    vj_ij_030 += fac * dot_lkl_y_030 + trr_01x * dot_lkl_y_130;
                    vj_ij_031 += fac * dot_lkl_y_031 + trr_01x * dot_lkl_y_131;
                    vj_ij_040 += fac * dot_lkl_y_040 + trr_01x * dot_lkl_y_140;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101;
                    vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102;
                    vj_ij_103 += trr_10x * dot_lkl_y_003 + trr_11x * dot_lkl_y_103;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110;
                    vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111;
                    vj_ij_112 += trr_10x * dot_lkl_y_012 + trr_11x * dot_lkl_y_112;
                    vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120;
                    vj_ij_121 += trr_10x * dot_lkl_y_021 + trr_11x * dot_lkl_y_121;
                    vj_ij_130 += trr_10x * dot_lkl_y_030 + trr_11x * dot_lkl_y_130;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100;
                    vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101;
                    vj_ij_202 += trr_20x * dot_lkl_y_002 + trr_21x * dot_lkl_y_102;
                    vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110;
                    vj_ij_211 += trr_20x * dot_lkl_y_011 + trr_21x * dot_lkl_y_111;
                    vj_ij_220 += trr_20x * dot_lkl_y_020 + trr_21x * dot_lkl_y_120;
                    vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100;
                    vj_ij_301 += trr_30x * dot_lkl_y_001 + trr_31x * dot_lkl_y_101;
                    vj_ij_310 += trr_30x * dot_lkl_y_010 + trr_31x * dot_lkl_y_110;
                    vj_ij_400 += trr_40x * dot_lkl_y_000 + trr_41x * dot_lkl_y_100;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_003);
        atomicAdd(vj+ij_pair0+4, vj_ij_004);
        atomicAdd(vj+ij_pair0+6, vj_ij_011);
        atomicAdd(vj+ij_pair0+7, vj_ij_012);
        atomicAdd(vj+ij_pair0+8, vj_ij_013);
        atomicAdd(vj+ij_pair0+9, vj_ij_020);
        atomicAdd(vj+ij_pair0+10, vj_ij_021);
        atomicAdd(vj+ij_pair0+11, vj_ij_022);
        atomicAdd(vj+ij_pair0+12, vj_ij_030);
        atomicAdd(vj+ij_pair0+13, vj_ij_031);
        atomicAdd(vj+ij_pair0+14, vj_ij_040);
        atomicAdd(vj+ij_pair0+16, vj_ij_101);
        atomicAdd(vj+ij_pair0+17, vj_ij_102);
        atomicAdd(vj+ij_pair0+18, vj_ij_103);
        atomicAdd(vj+ij_pair0+19, vj_ij_110);
        atomicAdd(vj+ij_pair0+20, vj_ij_111);
        atomicAdd(vj+ij_pair0+21, vj_ij_112);
        atomicAdd(vj+ij_pair0+22, vj_ij_120);
        atomicAdd(vj+ij_pair0+23, vj_ij_121);
        atomicAdd(vj+ij_pair0+24, vj_ij_130);
        atomicAdd(vj+ij_pair0+25, vj_ij_200);
        atomicAdd(vj+ij_pair0+26, vj_ij_201);
        atomicAdd(vj+ij_pair0+27, vj_ij_202);
        atomicAdd(vj+ij_pair0+28, vj_ij_210);
        atomicAdd(vj+ij_pair0+29, vj_ij_211);
        atomicAdd(vj+ij_pair0+30, vj_ij_220);
        atomicAdd(vj+ij_pair0+31, vj_ij_300);
        atomicAdd(vj+ij_pair0+32, vj_ij_301);
        atomicAdd(vj+ij_pair0+33, vj_ij_310);
        atomicAdd(vj+ij_pair0+34, vj_ij_400);
        atomicAdd(vj+kl_pair0+1, vj_kl_001);
        atomicAdd(vj+kl_pair0+2, vj_kl_010);
        atomicAdd(vj+kl_pair0+3, vj_kl_100);
    }
}
__global__
static void rys_j_4_1(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_4_1(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_j_4_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int nroots = bounds.nroots;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    int *pair_loc = envs.ao_loc;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *dm_ij_cache = Rpa_cicj + iprim*jprim*TILE2*4;
    double *rw = dm_ij_cache + 35*TILE2 + sq_id;
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
    double *dm = jk.dm;
    for (int n = sq_id; n < 35*TILE2; n += nsq_per_block) {
        int m = n / TILE2;
        int ij_sh = n % TILE2;
        int ish = ish0 + ij_sh / TILE;
        int jsh = jsh0 + ij_sh % TILE;
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        dm_ij_cache[ij_sh+m*TILE2] = dm[ij_pair0+m];
    }

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
        int ij_pair0 = pair_loc[ish*nbas+jsh];
        int kl_pair0 = pair_loc[ksh*nbas+lsh];
        double dm_kl_001 = dm[kl_pair0+1];
        double dm_kl_002 = dm[kl_pair0+2];
        double dm_kl_010 = dm[kl_pair0+3];
        double dm_kl_011 = dm[kl_pair0+4];
        double dm_kl_020 = dm[kl_pair0+5];
        double dm_kl_100 = dm[kl_pair0+6];
        double dm_kl_101 = dm[kl_pair0+7];
        double dm_kl_110 = dm[kl_pair0+8];
        double dm_kl_200 = dm[kl_pair0+9];
        double vj_ij_002 = 0;
        double vj_ij_003 = 0;
        double vj_ij_004 = 0;
        double vj_ij_011 = 0;
        double vj_ij_012 = 0;
        double vj_ij_013 = 0;
        double vj_ij_020 = 0;
        double vj_ij_021 = 0;
        double vj_ij_022 = 0;
        double vj_ij_030 = 0;
        double vj_ij_031 = 0;
        double vj_ij_040 = 0;
        double vj_ij_101 = 0;
        double vj_ij_102 = 0;
        double vj_ij_103 = 0;
        double vj_ij_110 = 0;
        double vj_ij_111 = 0;
        double vj_ij_112 = 0;
        double vj_ij_120 = 0;
        double vj_ij_121 = 0;
        double vj_ij_130 = 0;
        double vj_ij_200 = 0;
        double vj_ij_201 = 0;
        double vj_ij_202 = 0;
        double vj_ij_210 = 0;
        double vj_ij_211 = 0;
        double vj_ij_220 = 0;
        double vj_ij_300 = 0;
        double vj_ij_301 = 0;
        double vj_ij_310 = 0;
        double vj_ij_400 = 0;
        double vj_kl_001 = 0;
        double vj_kl_002 = 0;
        double vj_kl_010 = 0;
        double vj_kl_011 = 0;
        double vj_kl_020 = 0;
        double vj_kl_100 = 0;
        double vj_kl_101 = 0;
        double vj_kl_110 = 0;
        double vj_kl_200 = 0;
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
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
                rys_roots_rs(nroots, theta, rr, omega, rw, nsq_per_block, 0, 1);
                for (int irys = 0; irys < nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double b00 = .5 * rt_aa;
                    double rt_aij = rt_aa * akl;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0z = Rpa[sh_ij+2*TILE2] - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double dot_lij_z_000 = trr_20z * dm_ij_cache[sh_ij+2*TILE2] + trr_30z * dm_ij_cache[sh_ij+3*TILE2] + trr_40z * dm_ij_cache[sh_ij+4*TILE2];
                    double rt_akl = rt_aa * aij;
                    double b01 = .5/akl * (1 - rt_akl);
                    double cpz = zqc + zpq*rt_akl;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double trr_41z = cpz * trr_40z + 4*b00 * trr_30z;
                    double dot_lij_z_001 = trr_21z * dm_ij_cache[sh_ij+2*TILE2] + trr_31z * dm_ij_cache[sh_ij+3*TILE2] + trr_41z * dm_ij_cache[sh_ij+4*TILE2];
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                    double trr_42z = cpz * trr_41z + 1*b01 * trr_40z + 4*b00 * trr_31z;
                    double dot_lij_z_002 = trr_22z * dm_ij_cache[sh_ij+2*TILE2] + trr_32z * dm_ij_cache[sh_ij+3*TILE2] + trr_42z * dm_ij_cache[sh_ij+4*TILE2];
                    double dot_lij_z_010 = trr_10z * dm_ij_cache[sh_ij+6*TILE2] + trr_20z * dm_ij_cache[sh_ij+7*TILE2] + trr_30z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_011 = trr_11z * dm_ij_cache[sh_ij+6*TILE2] + trr_21z * dm_ij_cache[sh_ij+7*TILE2] + trr_31z * dm_ij_cache[sh_ij+8*TILE2];
                    double trr_01z = cpz * wt;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double dot_lij_z_012 = trr_12z * dm_ij_cache[sh_ij+6*TILE2] + trr_22z * dm_ij_cache[sh_ij+7*TILE2] + trr_32z * dm_ij_cache[sh_ij+8*TILE2];
                    double dot_lij_z_020 = wt * dm_ij_cache[sh_ij+9*TILE2] + trr_10z * dm_ij_cache[sh_ij+10*TILE2] + trr_20z * dm_ij_cache[sh_ij+11*TILE2];
                    double dot_lij_z_021 = trr_01z * dm_ij_cache[sh_ij+9*TILE2] + trr_11z * dm_ij_cache[sh_ij+10*TILE2] + trr_21z * dm_ij_cache[sh_ij+11*TILE2];
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double dot_lij_z_022 = trr_02z * dm_ij_cache[sh_ij+9*TILE2] + trr_12z * dm_ij_cache[sh_ij+10*TILE2] + trr_22z * dm_ij_cache[sh_ij+11*TILE2];
                    double dot_lij_z_030 = wt * dm_ij_cache[sh_ij+12*TILE2] + trr_10z * dm_ij_cache[sh_ij+13*TILE2];
                    double dot_lij_z_031 = trr_01z * dm_ij_cache[sh_ij+12*TILE2] + trr_11z * dm_ij_cache[sh_ij+13*TILE2];
                    double dot_lij_z_032 = trr_02z * dm_ij_cache[sh_ij+12*TILE2] + trr_12z * dm_ij_cache[sh_ij+13*TILE2];
                    double dot_lij_z_040 = wt * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_041 = trr_01z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_042 = trr_02z * dm_ij_cache[sh_ij+14*TILE2];
                    double dot_lij_z_100 = trr_10z * dm_ij_cache[sh_ij+16*TILE2] + trr_20z * dm_ij_cache[sh_ij+17*TILE2] + trr_30z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_101 = trr_11z * dm_ij_cache[sh_ij+16*TILE2] + trr_21z * dm_ij_cache[sh_ij+17*TILE2] + trr_31z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_102 = trr_12z * dm_ij_cache[sh_ij+16*TILE2] + trr_22z * dm_ij_cache[sh_ij+17*TILE2] + trr_32z * dm_ij_cache[sh_ij+18*TILE2];
                    double dot_lij_z_110 = wt * dm_ij_cache[sh_ij+19*TILE2] + trr_10z * dm_ij_cache[sh_ij+20*TILE2] + trr_20z * dm_ij_cache[sh_ij+21*TILE2];
                    double dot_lij_z_111 = trr_01z * dm_ij_cache[sh_ij+19*TILE2] + trr_11z * dm_ij_cache[sh_ij+20*TILE2] + trr_21z * dm_ij_cache[sh_ij+21*TILE2];
                    double dot_lij_z_112 = trr_02z * dm_ij_cache[sh_ij+19*TILE2] + trr_12z * dm_ij_cache[sh_ij+20*TILE2] + trr_22z * dm_ij_cache[sh_ij+21*TILE2];
                    double dot_lij_z_120 = wt * dm_ij_cache[sh_ij+22*TILE2] + trr_10z * dm_ij_cache[sh_ij+23*TILE2];
                    double dot_lij_z_121 = trr_01z * dm_ij_cache[sh_ij+22*TILE2] + trr_11z * dm_ij_cache[sh_ij+23*TILE2];
                    double dot_lij_z_122 = trr_02z * dm_ij_cache[sh_ij+22*TILE2] + trr_12z * dm_ij_cache[sh_ij+23*TILE2];
                    double dot_lij_z_130 = wt * dm_ij_cache[sh_ij+24*TILE2];
                    double dot_lij_z_131 = trr_01z * dm_ij_cache[sh_ij+24*TILE2];
                    double dot_lij_z_132 = trr_02z * dm_ij_cache[sh_ij+24*TILE2];
                    double dot_lij_z_200 = wt * dm_ij_cache[sh_ij+25*TILE2] + trr_10z * dm_ij_cache[sh_ij+26*TILE2] + trr_20z * dm_ij_cache[sh_ij+27*TILE2];
                    double dot_lij_z_201 = trr_01z * dm_ij_cache[sh_ij+25*TILE2] + trr_11z * dm_ij_cache[sh_ij+26*TILE2] + trr_21z * dm_ij_cache[sh_ij+27*TILE2];
                    double dot_lij_z_202 = trr_02z * dm_ij_cache[sh_ij+25*TILE2] + trr_12z * dm_ij_cache[sh_ij+26*TILE2] + trr_22z * dm_ij_cache[sh_ij+27*TILE2];
                    double dot_lij_z_210 = wt * dm_ij_cache[sh_ij+28*TILE2] + trr_10z * dm_ij_cache[sh_ij+29*TILE2];
                    double dot_lij_z_211 = trr_01z * dm_ij_cache[sh_ij+28*TILE2] + trr_11z * dm_ij_cache[sh_ij+29*TILE2];
                    double dot_lij_z_212 = trr_02z * dm_ij_cache[sh_ij+28*TILE2] + trr_12z * dm_ij_cache[sh_ij+29*TILE2];
                    double dot_lij_z_220 = wt * dm_ij_cache[sh_ij+30*TILE2];
                    double dot_lij_z_221 = trr_01z * dm_ij_cache[sh_ij+30*TILE2];
                    double dot_lij_z_222 = trr_02z * dm_ij_cache[sh_ij+30*TILE2];
                    double dot_lij_z_300 = wt * dm_ij_cache[sh_ij+31*TILE2] + trr_10z * dm_ij_cache[sh_ij+32*TILE2];
                    double dot_lij_z_301 = trr_01z * dm_ij_cache[sh_ij+31*TILE2] + trr_11z * dm_ij_cache[sh_ij+32*TILE2];
                    double dot_lij_z_302 = trr_02z * dm_ij_cache[sh_ij+31*TILE2] + trr_12z * dm_ij_cache[sh_ij+32*TILE2];
                    double dot_lij_z_310 = wt * dm_ij_cache[sh_ij+33*TILE2];
                    double dot_lij_z_311 = trr_01z * dm_ij_cache[sh_ij+33*TILE2];
                    double dot_lij_z_312 = trr_02z * dm_ij_cache[sh_ij+33*TILE2];
                    double dot_lij_z_400 = wt * dm_ij_cache[sh_ij+34*TILE2];
                    double dot_lij_z_401 = trr_01z * dm_ij_cache[sh_ij+34*TILE2];
                    double dot_lij_z_402 = trr_02z * dm_ij_cache[sh_ij+34*TILE2];
                    double c0y = Rpa[sh_ij+1*TILE2] - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double dot_lij_y_000 = 1 * dot_lij_z_000 + trr_10y * dot_lij_z_010 + trr_20y * dot_lij_z_020 + trr_30y * dot_lij_z_030 + trr_40y * dot_lij_z_040;
                    double dot_lij_y_001 = 1 * dot_lij_z_001 + trr_10y * dot_lij_z_011 + trr_20y * dot_lij_z_021 + trr_30y * dot_lij_z_031 + trr_40y * dot_lij_z_041;
                    double dot_lij_y_002 = 1 * dot_lij_z_002 + trr_10y * dot_lij_z_012 + trr_20y * dot_lij_z_022 + trr_30y * dot_lij_z_032 + trr_40y * dot_lij_z_042;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double trr_41y = cpy * trr_40y + 4*b00 * trr_30y;
                    double dot_lij_y_010 = trr_01y * dot_lij_z_000 + trr_11y * dot_lij_z_010 + trr_21y * dot_lij_z_020 + trr_31y * dot_lij_z_030 + trr_41y * dot_lij_z_040;
                    double dot_lij_y_011 = trr_01y * dot_lij_z_001 + trr_11y * dot_lij_z_011 + trr_21y * dot_lij_z_021 + trr_31y * dot_lij_z_031 + trr_41y * dot_lij_z_041;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                    double trr_42y = cpy * trr_41y + 1*b01 * trr_40y + 4*b00 * trr_31y;
                    double dot_lij_y_020 = trr_02y * dot_lij_z_000 + trr_12y * dot_lij_z_010 + trr_22y * dot_lij_z_020 + trr_32y * dot_lij_z_030 + trr_42y * dot_lij_z_040;
                    double dot_lij_y_100 = 1 * dot_lij_z_100 + trr_10y * dot_lij_z_110 + trr_20y * dot_lij_z_120 + trr_30y * dot_lij_z_130;
                    double dot_lij_y_101 = 1 * dot_lij_z_101 + trr_10y * dot_lij_z_111 + trr_20y * dot_lij_z_121 + trr_30y * dot_lij_z_131;
                    double dot_lij_y_102 = 1 * dot_lij_z_102 + trr_10y * dot_lij_z_112 + trr_20y * dot_lij_z_122 + trr_30y * dot_lij_z_132;
                    double dot_lij_y_110 = trr_01y * dot_lij_z_100 + trr_11y * dot_lij_z_110 + trr_21y * dot_lij_z_120 + trr_31y * dot_lij_z_130;
                    double dot_lij_y_111 = trr_01y * dot_lij_z_101 + trr_11y * dot_lij_z_111 + trr_21y * dot_lij_z_121 + trr_31y * dot_lij_z_131;
                    double dot_lij_y_120 = trr_02y * dot_lij_z_100 + trr_12y * dot_lij_z_110 + trr_22y * dot_lij_z_120 + trr_32y * dot_lij_z_130;
                    double dot_lij_y_200 = 1 * dot_lij_z_200 + trr_10y * dot_lij_z_210 + trr_20y * dot_lij_z_220;
                    double dot_lij_y_201 = 1 * dot_lij_z_201 + trr_10y * dot_lij_z_211 + trr_20y * dot_lij_z_221;
                    double dot_lij_y_202 = 1 * dot_lij_z_202 + trr_10y * dot_lij_z_212 + trr_20y * dot_lij_z_222;
                    double dot_lij_y_210 = trr_01y * dot_lij_z_200 + trr_11y * dot_lij_z_210 + trr_21y * dot_lij_z_220;
                    double dot_lij_y_211 = trr_01y * dot_lij_z_201 + trr_11y * dot_lij_z_211 + trr_21y * dot_lij_z_221;
                    double dot_lij_y_220 = trr_02y * dot_lij_z_200 + trr_12y * dot_lij_z_210 + trr_22y * dot_lij_z_220;
                    double dot_lij_y_300 = 1 * dot_lij_z_300 + trr_10y * dot_lij_z_310;
                    double dot_lij_y_301 = 1 * dot_lij_z_301 + trr_10y * dot_lij_z_311;
                    double dot_lij_y_302 = 1 * dot_lij_z_302 + trr_10y * dot_lij_z_312;
                    double dot_lij_y_310 = trr_01y * dot_lij_z_300 + trr_11y * dot_lij_z_310;
                    double dot_lij_y_311 = trr_01y * dot_lij_z_301 + trr_11y * dot_lij_z_311;
                    double dot_lij_y_320 = trr_02y * dot_lij_z_300 + trr_12y * dot_lij_z_310;
                    double dot_lij_y_400 = 1 * dot_lij_z_400;
                    double dot_lij_y_401 = 1 * dot_lij_z_401;
                    double dot_lij_y_402 = 1 * dot_lij_z_402;
                    double dot_lij_y_410 = trr_01y * dot_lij_z_400;
                    double dot_lij_y_411 = trr_01y * dot_lij_z_401;
                    double dot_lij_y_420 = trr_02y * dot_lij_z_400;
                    double c0x = Rpa[sh_ij+0*TILE2] - xpq*rt_aij;
                    double trr_10x = c0x * fac;
                    double trr_20x = c0x * trr_10x + 1*b10 * fac;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    vj_kl_001 += fac * dot_lij_y_001 + trr_10x * dot_lij_y_101 + trr_20x * dot_lij_y_201 + trr_30x * dot_lij_y_301 + trr_40x * dot_lij_y_401;
                    vj_kl_002 += fac * dot_lij_y_002 + trr_10x * dot_lij_y_102 + trr_20x * dot_lij_y_202 + trr_30x * dot_lij_y_302 + trr_40x * dot_lij_y_402;
                    vj_kl_010 += fac * dot_lij_y_010 + trr_10x * dot_lij_y_110 + trr_20x * dot_lij_y_210 + trr_30x * dot_lij_y_310 + trr_40x * dot_lij_y_410;
                    vj_kl_011 += fac * dot_lij_y_011 + trr_10x * dot_lij_y_111 + trr_20x * dot_lij_y_211 + trr_30x * dot_lij_y_311 + trr_40x * dot_lij_y_411;
                    vj_kl_020 += fac * dot_lij_y_020 + trr_10x * dot_lij_y_120 + trr_20x * dot_lij_y_220 + trr_30x * dot_lij_y_320 + trr_40x * dot_lij_y_420;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * fac;
                    double trr_11x = cpx * trr_10x + 1*b00 * fac;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double trr_41x = cpx * trr_40x + 4*b00 * trr_30x;
                    vj_kl_100 += trr_01x * dot_lij_y_000 + trr_11x * dot_lij_y_100 + trr_21x * dot_lij_y_200 + trr_31x * dot_lij_y_300 + trr_41x * dot_lij_y_400;
                    vj_kl_101 += trr_01x * dot_lij_y_001 + trr_11x * dot_lij_y_101 + trr_21x * dot_lij_y_201 + trr_31x * dot_lij_y_301 + trr_41x * dot_lij_y_401;
                    vj_kl_110 += trr_01x * dot_lij_y_010 + trr_11x * dot_lij_y_110 + trr_21x * dot_lij_y_210 + trr_31x * dot_lij_y_310 + trr_41x * dot_lij_y_410;
                    double trr_02x = cpx * trr_01x + 1*b01 * fac;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                    double trr_42x = cpx * trr_41x + 1*b01 * trr_40x + 4*b00 * trr_31x;
                    vj_kl_200 += trr_02x * dot_lij_y_000 + trr_12x * dot_lij_y_100 + trr_22x * dot_lij_y_200 + trr_32x * dot_lij_y_300 + trr_42x * dot_lij_y_400;
                    double dot_lkl_z_000 = trr_01z * dm_kl_001 + trr_02z * dm_kl_002;
                    double dot_lkl_z_001 = trr_11z * dm_kl_001 + trr_12z * dm_kl_002;
                    double dot_lkl_z_002 = trr_21z * dm_kl_001 + trr_22z * dm_kl_002;
                    double dot_lkl_z_003 = trr_31z * dm_kl_001 + trr_32z * dm_kl_002;
                    double dot_lkl_z_004 = trr_41z * dm_kl_001 + trr_42z * dm_kl_002;
                    double dot_lkl_z_010 = wt * dm_kl_010 + trr_01z * dm_kl_011;
                    double dot_lkl_z_011 = trr_10z * dm_kl_010 + trr_11z * dm_kl_011;
                    double dot_lkl_z_012 = trr_20z * dm_kl_010 + trr_21z * dm_kl_011;
                    double dot_lkl_z_013 = trr_30z * dm_kl_010 + trr_31z * dm_kl_011;
                    double dot_lkl_z_014 = trr_40z * dm_kl_010 + trr_41z * dm_kl_011;
                    double dot_lkl_z_020 = wt * dm_kl_020;
                    double dot_lkl_z_021 = trr_10z * dm_kl_020;
                    double dot_lkl_z_022 = trr_20z * dm_kl_020;
                    double dot_lkl_z_023 = trr_30z * dm_kl_020;
                    double dot_lkl_z_024 = trr_40z * dm_kl_020;
                    double dot_lkl_z_100 = wt * dm_kl_100 + trr_01z * dm_kl_101;
                    double dot_lkl_z_101 = trr_10z * dm_kl_100 + trr_11z * dm_kl_101;
                    double dot_lkl_z_102 = trr_20z * dm_kl_100 + trr_21z * dm_kl_101;
                    double dot_lkl_z_103 = trr_30z * dm_kl_100 + trr_31z * dm_kl_101;
                    double dot_lkl_z_104 = trr_40z * dm_kl_100 + trr_41z * dm_kl_101;
                    double dot_lkl_z_110 = wt * dm_kl_110;
                    double dot_lkl_z_111 = trr_10z * dm_kl_110;
                    double dot_lkl_z_112 = trr_20z * dm_kl_110;
                    double dot_lkl_z_113 = trr_30z * dm_kl_110;
                    double dot_lkl_z_114 = trr_40z * dm_kl_110;
                    double dot_lkl_z_200 = wt * dm_kl_200;
                    double dot_lkl_z_201 = trr_10z * dm_kl_200;
                    double dot_lkl_z_202 = trr_20z * dm_kl_200;
                    double dot_lkl_z_203 = trr_30z * dm_kl_200;
                    double dot_lkl_z_204 = trr_40z * dm_kl_200;
                    double dot_lkl_y_000 = 1 * dot_lkl_z_000 + trr_01y * dot_lkl_z_010 + trr_02y * dot_lkl_z_020;
                    double dot_lkl_y_001 = 1 * dot_lkl_z_001 + trr_01y * dot_lkl_z_011 + trr_02y * dot_lkl_z_021;
                    double dot_lkl_y_002 = 1 * dot_lkl_z_002 + trr_01y * dot_lkl_z_012 + trr_02y * dot_lkl_z_022;
                    double dot_lkl_y_003 = 1 * dot_lkl_z_003 + trr_01y * dot_lkl_z_013 + trr_02y * dot_lkl_z_023;
                    double dot_lkl_y_004 = 1 * dot_lkl_z_004 + trr_01y * dot_lkl_z_014 + trr_02y * dot_lkl_z_024;
                    double dot_lkl_y_010 = trr_10y * dot_lkl_z_000 + trr_11y * dot_lkl_z_010 + trr_12y * dot_lkl_z_020;
                    double dot_lkl_y_011 = trr_10y * dot_lkl_z_001 + trr_11y * dot_lkl_z_011 + trr_12y * dot_lkl_z_021;
                    double dot_lkl_y_012 = trr_10y * dot_lkl_z_002 + trr_11y * dot_lkl_z_012 + trr_12y * dot_lkl_z_022;
                    double dot_lkl_y_013 = trr_10y * dot_lkl_z_003 + trr_11y * dot_lkl_z_013 + trr_12y * dot_lkl_z_023;
                    double dot_lkl_y_020 = trr_20y * dot_lkl_z_000 + trr_21y * dot_lkl_z_010 + trr_22y * dot_lkl_z_020;
                    double dot_lkl_y_021 = trr_20y * dot_lkl_z_001 + trr_21y * dot_lkl_z_011 + trr_22y * dot_lkl_z_021;
                    double dot_lkl_y_022 = trr_20y * dot_lkl_z_002 + trr_21y * dot_lkl_z_012 + trr_22y * dot_lkl_z_022;
                    double dot_lkl_y_030 = trr_30y * dot_lkl_z_000 + trr_31y * dot_lkl_z_010 + trr_32y * dot_lkl_z_020;
                    double dot_lkl_y_031 = trr_30y * dot_lkl_z_001 + trr_31y * dot_lkl_z_011 + trr_32y * dot_lkl_z_021;
                    double dot_lkl_y_040 = trr_40y * dot_lkl_z_000 + trr_41y * dot_lkl_z_010 + trr_42y * dot_lkl_z_020;
                    double dot_lkl_y_100 = 1 * dot_lkl_z_100 + trr_01y * dot_lkl_z_110;
                    double dot_lkl_y_101 = 1 * dot_lkl_z_101 + trr_01y * dot_lkl_z_111;
                    double dot_lkl_y_102 = 1 * dot_lkl_z_102 + trr_01y * dot_lkl_z_112;
                    double dot_lkl_y_103 = 1 * dot_lkl_z_103 + trr_01y * dot_lkl_z_113;
                    double dot_lkl_y_104 = 1 * dot_lkl_z_104 + trr_01y * dot_lkl_z_114;
                    double dot_lkl_y_110 = trr_10y * dot_lkl_z_100 + trr_11y * dot_lkl_z_110;
                    double dot_lkl_y_111 = trr_10y * dot_lkl_z_101 + trr_11y * dot_lkl_z_111;
                    double dot_lkl_y_112 = trr_10y * dot_lkl_z_102 + trr_11y * dot_lkl_z_112;
                    double dot_lkl_y_113 = trr_10y * dot_lkl_z_103 + trr_11y * dot_lkl_z_113;
                    double dot_lkl_y_120 = trr_20y * dot_lkl_z_100 + trr_21y * dot_lkl_z_110;
                    double dot_lkl_y_121 = trr_20y * dot_lkl_z_101 + trr_21y * dot_lkl_z_111;
                    double dot_lkl_y_122 = trr_20y * dot_lkl_z_102 + trr_21y * dot_lkl_z_112;
                    double dot_lkl_y_130 = trr_30y * dot_lkl_z_100 + trr_31y * dot_lkl_z_110;
                    double dot_lkl_y_131 = trr_30y * dot_lkl_z_101 + trr_31y * dot_lkl_z_111;
                    double dot_lkl_y_140 = trr_40y * dot_lkl_z_100 + trr_41y * dot_lkl_z_110;
                    double dot_lkl_y_200 = 1 * dot_lkl_z_200;
                    double dot_lkl_y_201 = 1 * dot_lkl_z_201;
                    double dot_lkl_y_202 = 1 * dot_lkl_z_202;
                    double dot_lkl_y_203 = 1 * dot_lkl_z_203;
                    double dot_lkl_y_204 = 1 * dot_lkl_z_204;
                    double dot_lkl_y_210 = trr_10y * dot_lkl_z_200;
                    double dot_lkl_y_211 = trr_10y * dot_lkl_z_201;
                    double dot_lkl_y_212 = trr_10y * dot_lkl_z_202;
                    double dot_lkl_y_213 = trr_10y * dot_lkl_z_203;
                    double dot_lkl_y_220 = trr_20y * dot_lkl_z_200;
                    double dot_lkl_y_221 = trr_20y * dot_lkl_z_201;
                    double dot_lkl_y_222 = trr_20y * dot_lkl_z_202;
                    double dot_lkl_y_230 = trr_30y * dot_lkl_z_200;
                    double dot_lkl_y_231 = trr_30y * dot_lkl_z_201;
                    double dot_lkl_y_240 = trr_40y * dot_lkl_z_200;
                    vj_ij_002 += fac * dot_lkl_y_002 + trr_01x * dot_lkl_y_102 + trr_02x * dot_lkl_y_202;
                    vj_ij_003 += fac * dot_lkl_y_003 + trr_01x * dot_lkl_y_103 + trr_02x * dot_lkl_y_203;
                    vj_ij_004 += fac * dot_lkl_y_004 + trr_01x * dot_lkl_y_104 + trr_02x * dot_lkl_y_204;
                    vj_ij_011 += fac * dot_lkl_y_011 + trr_01x * dot_lkl_y_111 + trr_02x * dot_lkl_y_211;
                    vj_ij_012 += fac * dot_lkl_y_012 + trr_01x * dot_lkl_y_112 + trr_02x * dot_lkl_y_212;
                    vj_ij_013 += fac * dot_lkl_y_013 + trr_01x * dot_lkl_y_113 + trr_02x * dot_lkl_y_213;
                    vj_ij_020 += fac * dot_lkl_y_020 + trr_01x * dot_lkl_y_120 + trr_02x * dot_lkl_y_220;
                    vj_ij_021 += fac * dot_lkl_y_021 + trr_01x * dot_lkl_y_121 + trr_02x * dot_lkl_y_221;
                    vj_ij_022 += fac * dot_lkl_y_022 + trr_01x * dot_lkl_y_122 + trr_02x * dot_lkl_y_222;
                    vj_ij_030 += fac * dot_lkl_y_030 + trr_01x * dot_lkl_y_130 + trr_02x * dot_lkl_y_230;
                    vj_ij_031 += fac * dot_lkl_y_031 + trr_01x * dot_lkl_y_131 + trr_02x * dot_lkl_y_231;
                    vj_ij_040 += fac * dot_lkl_y_040 + trr_01x * dot_lkl_y_140 + trr_02x * dot_lkl_y_240;
                    vj_ij_101 += trr_10x * dot_lkl_y_001 + trr_11x * dot_lkl_y_101 + trr_12x * dot_lkl_y_201;
                    vj_ij_102 += trr_10x * dot_lkl_y_002 + trr_11x * dot_lkl_y_102 + trr_12x * dot_lkl_y_202;
                    vj_ij_103 += trr_10x * dot_lkl_y_003 + trr_11x * dot_lkl_y_103 + trr_12x * dot_lkl_y_203;
                    vj_ij_110 += trr_10x * dot_lkl_y_010 + trr_11x * dot_lkl_y_110 + trr_12x * dot_lkl_y_210;
                    vj_ij_111 += trr_10x * dot_lkl_y_011 + trr_11x * dot_lkl_y_111 + trr_12x * dot_lkl_y_211;
                    vj_ij_112 += trr_10x * dot_lkl_y_012 + trr_11x * dot_lkl_y_112 + trr_12x * dot_lkl_y_212;
                    vj_ij_120 += trr_10x * dot_lkl_y_020 + trr_11x * dot_lkl_y_120 + trr_12x * dot_lkl_y_220;
                    vj_ij_121 += trr_10x * dot_lkl_y_021 + trr_11x * dot_lkl_y_121 + trr_12x * dot_lkl_y_221;
                    vj_ij_130 += trr_10x * dot_lkl_y_030 + trr_11x * dot_lkl_y_130 + trr_12x * dot_lkl_y_230;
                    vj_ij_200 += trr_20x * dot_lkl_y_000 + trr_21x * dot_lkl_y_100 + trr_22x * dot_lkl_y_200;
                    vj_ij_201 += trr_20x * dot_lkl_y_001 + trr_21x * dot_lkl_y_101 + trr_22x * dot_lkl_y_201;
                    vj_ij_202 += trr_20x * dot_lkl_y_002 + trr_21x * dot_lkl_y_102 + trr_22x * dot_lkl_y_202;
                    vj_ij_210 += trr_20x * dot_lkl_y_010 + trr_21x * dot_lkl_y_110 + trr_22x * dot_lkl_y_210;
                    vj_ij_211 += trr_20x * dot_lkl_y_011 + trr_21x * dot_lkl_y_111 + trr_22x * dot_lkl_y_211;
                    vj_ij_220 += trr_20x * dot_lkl_y_020 + trr_21x * dot_lkl_y_120 + trr_22x * dot_lkl_y_220;
                    vj_ij_300 += trr_30x * dot_lkl_y_000 + trr_31x * dot_lkl_y_100 + trr_32x * dot_lkl_y_200;
                    vj_ij_301 += trr_30x * dot_lkl_y_001 + trr_31x * dot_lkl_y_101 + trr_32x * dot_lkl_y_201;
                    vj_ij_310 += trr_30x * dot_lkl_y_010 + trr_31x * dot_lkl_y_110 + trr_32x * dot_lkl_y_210;
                    vj_ij_400 += trr_40x * dot_lkl_y_000 + trr_41x * dot_lkl_y_100 + trr_42x * dot_lkl_y_200;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *vj = jk.vj;
        atomicAdd(vj+ij_pair0+2, vj_ij_002);
        atomicAdd(vj+ij_pair0+3, vj_ij_003);
        atomicAdd(vj+ij_pair0+4, vj_ij_004);
        atomicAdd(vj+ij_pair0+6, vj_ij_011);
        atomicAdd(vj+ij_pair0+7, vj_ij_012);
        atomicAdd(vj+ij_pair0+8, vj_ij_013);
        atomicAdd(vj+ij_pair0+9, vj_ij_020);
        atomicAdd(vj+ij_pair0+10, vj_ij_021);
        atomicAdd(vj+ij_pair0+11, vj_ij_022);
        atomicAdd(vj+ij_pair0+12, vj_ij_030);
        atomicAdd(vj+ij_pair0+13, vj_ij_031);
        atomicAdd(vj+ij_pair0+14, vj_ij_040);
        atomicAdd(vj+ij_pair0+16, vj_ij_101);
        atomicAdd(vj+ij_pair0+17, vj_ij_102);
        atomicAdd(vj+ij_pair0+18, vj_ij_103);
        atomicAdd(vj+ij_pair0+19, vj_ij_110);
        atomicAdd(vj+ij_pair0+20, vj_ij_111);
        atomicAdd(vj+ij_pair0+21, vj_ij_112);
        atomicAdd(vj+ij_pair0+22, vj_ij_120);
        atomicAdd(vj+ij_pair0+23, vj_ij_121);
        atomicAdd(vj+ij_pair0+24, vj_ij_130);
        atomicAdd(vj+ij_pair0+25, vj_ij_200);
        atomicAdd(vj+ij_pair0+26, vj_ij_201);
        atomicAdd(vj+ij_pair0+27, vj_ij_202);
        atomicAdd(vj+ij_pair0+28, vj_ij_210);
        atomicAdd(vj+ij_pair0+29, vj_ij_211);
        atomicAdd(vj+ij_pair0+30, vj_ij_220);
        atomicAdd(vj+ij_pair0+31, vj_ij_300);
        atomicAdd(vj+ij_pair0+32, vj_ij_301);
        atomicAdd(vj+ij_pair0+33, vj_ij_310);
        atomicAdd(vj+ij_pair0+34, vj_ij_400);
        atomicAdd(vj+kl_pair0+1, vj_kl_001);
        atomicAdd(vj+kl_pair0+2, vj_kl_002);
        atomicAdd(vj+kl_pair0+3, vj_kl_010);
        atomicAdd(vj+kl_pair0+4, vj_kl_011);
        atomicAdd(vj+kl_pair0+5, vj_kl_020);
        atomicAdd(vj+kl_pair0+6, vj_kl_100);
        atomicAdd(vj+kl_pair0+7, vj_kl_101);
        atomicAdd(vj+kl_pair0+8, vj_kl_110);
        atomicAdd(vj+kl_pair0+9, vj_kl_200);
    }
}
__global__
static void rys_j_4_2(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            _rys_j_4_2(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_j_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                   ShellQuartet *pool, uint32_t *batch_head,
                   int *scheme, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int lij = li + lj;
    int lkl = lk + ll;
    int threads = 256;
    int nroots = bounds->nroots;
    int nf3_ij = (lij+1)*(lij+2)*(lij+3)/6;
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ijkl = lij*9 + lkl;

#if CUDA_VERSION >= 12040
    switch (ijkl) {
    case 0: threads *= 2; break;
    case 9: threads *= 2; break;
    case 10: threads *= 2; break;
    case 18: threads *= 2; break;
    }
#endif

    int buflen = (nroots*2) * threads + iprim*jprim*TILE2*4 + nf3_ij*TILE2;
    switch (ijkl) {
    case 0: rys_j_0_0<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 9: rys_j_1_0<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 10: rys_j_1_1<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 11: rys_j_1_2<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 18: rys_j_2_0<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 19: rys_j_2_1<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 20: rys_j_2_2<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 21: rys_j_2_3<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 22: rys_j_2_4<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 27: rys_j_3_0<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 28: rys_j_3_1<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 29: rys_j_3_2<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 30: rys_j_3_3<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 36: rys_j_4_0<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 37: rys_j_4_1<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 38: rys_j_4_2<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
