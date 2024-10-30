#include <cuda.h>
#include "vhf.cuh"
#include "gamma_inc_unrolled.cu"
#include "create_tasks.cu"
int os_jk_unrolled_lmax = 1;
int os_jk_unrolled_max_order = 0;


__device__ static
void _os_jk_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x + blockDim.x * threadIdx.y;
    int nsq_per_block = blockDim.x * blockDim.y;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double Rpa_cicj[];
    double *gamma_inc = Rpa_cicj + iprim*jprim*TILE2*4;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    for (int n = t_id; n < iprim*jprim*TILE2; n += nsq_per_block) {
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
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
                double yij = ri[1] + ypa;
                double zij = ri[2] + zpa;
                double xkl = rk[0] + xqc;
                double ykl = rk[1] + yqc;
                double zkl = rk[2] + zqc;
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta = aij * akl / (aij + akl);
                double theta_rr = theta * rr;
                if (omega == 0) {
                    eval_gamma_inc_fn(gamma_inc, theta_rr, 0);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    eval_gamma_inc_fn(gamma_inc, theta_fac*theta_rr, 0);
                    double scale = sqrt(theta_fac);
                    for (int n = 0 ; n <= 0; ++n) {
                        gamma_inc[sq_id+n*nsq_per_block] *= scale;
                        scale *= theta_fac;
                    }
                } else { // omega < 0
                    eval_gamma_inc_fn(gamma_inc, theta_rr, 0);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    double *gamma_inc1 = gamma_inc + nsq_per_block * 1;
                    eval_gamma_inc_fn(gamma_inc1, theta_fac*theta_rr, 0);
                    __syncthreads();
                    double scale = sqrt(theta_fac);
                    for (int n = 0 ; n <= 0; ++n) {
                        gamma_inc[sq_id+n*nsq_per_block] -= scale * gamma_inc1[sq_id+n*nsq_per_block];
                        scale *= theta_fac;
                    }
                }
                __syncthreads();
                if (task_id < ntasks) {
                    double vrr_0_000 = fac * gamma_inc[sq_id+0*nsq_per_block];
                    gout0 += vrr_0_000;
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
void os_jk_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
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
        int nbas = envs.nbas;
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
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _os_jk_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int os_jk_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                   ShellQuartet *pool, uint32_t *batch_head,
                   int *scheme, int workers, double omega)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int threads = scheme[0] * scheme[1];
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int order = li + lj + lk + ll;
    int buflen = (order + 1) * threads + ij_prims*TILE2*4;
    if (omega < 0) {
        buflen += (order + 1) * threads;
    }
    int ijkl = li*8 + lj*4 + lk*2 + ll;
    switch (ijkl) {
    case 0: os_jk_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 1;
    }
    return 0;
}
