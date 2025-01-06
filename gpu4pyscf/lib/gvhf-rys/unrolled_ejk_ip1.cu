#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks_ip1.cu"


__device__ static
void _rys_ejk_ip1_0000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 1 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 1*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 1;
        int j = ij / 1;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(1, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 1; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 1; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 2*nsq_per_block;
                    rys_roots(1, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(1, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 1; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double trr_01x = cpx * 1;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_0000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_1000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 3 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 3*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 3;
        int j = ij / 3;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double b00 = .5 * rt_aa;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_01x = cpx * 1;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_1000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_1000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_1010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 3 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 3*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 3;
        int j = ij / 3;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        double dm_lk_0_1 = dm[(l0+0)*nao+(k0+1)];
        double dm_lk_0_2 = dm[(l0+0)*nao+(k0+2)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
            dm_lk_0_1 += dm[nao2+(l0+0)*nao+(k0+1)];
            dm_lk_0_2 += dm[nao2+(l0+0)*nao+(k0+2)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jk_0_1 = dm[(j0+0)*nao+(k0+1)];
        double dm_jk_0_2 = dm[(j0+0)*nao+(k0+2)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_0_1 = dm[(i0+0)*nao+(k0+1)];
        double dm_ik_0_2 = dm[(i0+0)*nao+(k0+2)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_1_1 = dm[(i0+1)*nao+(k0+1)];
        double dm_ik_1_2 = dm[(i0+1)*nao+(k0+2)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_ik_2_1 = dm[(i0+2)*nao+(k0+1)];
        double dm_ik_2_2 = dm[(i0+2)*nao+(k0+2)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b00 = .5 * rt_aa;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_11x * 1 * dd;
                    prod_xz = trr_11x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    fxi = ai2 * trr_21x;
                    double trr_01x = cpx * 1;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double b01 = .5/akl * (1 - rt_akl);
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * trr_10y * dd;
                    prod_xz = trr_01x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * 1 * dd;
                    prod_xz = trr_01x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = trr_10x * trr_01y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = trr_01y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_11y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = trr_11y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_01y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = trr_01y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * trr_01z * dd;
                    prod_yz = 1 * trr_01z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * trr_01z * dd;
                    prod_yz = trr_10y * trr_01z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * trr_11z * dd;
                    prod_yz = 1 * trr_11z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_1010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_1010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_1011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 3 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 3*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 3;
        int j = ij / 3;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        double dm_lk_0_1 = dm[(l0+0)*nao+(k0+1)];
        double dm_lk_0_2 = dm[(l0+0)*nao+(k0+2)];
        double dm_lk_1_0 = dm[(l0+1)*nao+(k0+0)];
        double dm_lk_1_1 = dm[(l0+1)*nao+(k0+1)];
        double dm_lk_1_2 = dm[(l0+1)*nao+(k0+2)];
        double dm_lk_2_0 = dm[(l0+2)*nao+(k0+0)];
        double dm_lk_2_1 = dm[(l0+2)*nao+(k0+1)];
        double dm_lk_2_2 = dm[(l0+2)*nao+(k0+2)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
            dm_lk_0_1 += dm[nao2+(l0+0)*nao+(k0+1)];
            dm_lk_0_2 += dm[nao2+(l0+0)*nao+(k0+2)];
            dm_lk_1_0 += dm[nao2+(l0+1)*nao+(k0+0)];
            dm_lk_1_1 += dm[nao2+(l0+1)*nao+(k0+1)];
            dm_lk_1_2 += dm[nao2+(l0+1)*nao+(k0+2)];
            dm_lk_2_0 += dm[nao2+(l0+2)*nao+(k0+0)];
            dm_lk_2_1 += dm[nao2+(l0+2)*nao+(k0+1)];
            dm_lk_2_2 += dm[nao2+(l0+2)*nao+(k0+2)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jk_0_1 = dm[(j0+0)*nao+(k0+1)];
        double dm_jk_0_2 = dm[(j0+0)*nao+(k0+2)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_jl_0_1 = dm[(j0+0)*nao+(l0+1)];
        double dm_jl_0_2 = dm[(j0+0)*nao+(l0+2)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_0_1 = dm[(i0+0)*nao+(k0+1)];
        double dm_ik_0_2 = dm[(i0+0)*nao+(k0+2)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_1_1 = dm[(i0+1)*nao+(k0+1)];
        double dm_ik_1_2 = dm[(i0+1)*nao+(k0+2)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_ik_2_1 = dm[(i0+2)*nao+(k0+1)];
        double dm_ik_2_2 = dm[(i0+2)*nao+(k0+2)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_0_1 = dm[(i0+0)*nao+(l0+1)];
        double dm_il_0_2 = dm[(i0+0)*nao+(l0+2)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_1_1 = dm[(i0+1)*nao+(l0+1)];
        double dm_il_1_2 = dm[(i0+1)*nao+(l0+2)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dm_il_2_1 = dm[(i0+2)*nao+(l0+1)];
        double dm_il_2_2 = dm[(i0+2)*nao+(l0+2)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b00 = .5 * rt_aa;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double b01 = .5/akl * (1 - rt_akl);
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_1011x * 1 * dd;
                    prod_xz = hrr_1011x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double hrr_2011x = trr_22x - xlxk * trr_21x;
                    fxi = ai2 * hrr_2011x;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    fxi -= 1 * hrr_0011x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                    fxj = aj2 * hrr_1111x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                    double hrr_1021x = trr_13x - xlxk * trr_12x;
                    fxk = ak2 * hrr_1021x;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    fxk -= 1 * hrr_1001x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1012x = hrr_1021x - xlxk * hrr_1011x;
                    fxl = al2 * hrr_1012x;
                    fxl -= 1 * trr_11x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0011x * trr_10y * dd;
                    prod_xz = hrr_0011x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * hrr_1011x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                    fxj = aj2 * hrr_0111x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                    double hrr_0021x = trr_03x - xlxk * trr_02x;
                    fxk = ak2 * hrr_0021x;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    fxk -= 1 * hrr_0001x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0012x = hrr_0021x - xlxk * hrr_0011x;
                    fxl = al2 * hrr_0012x;
                    fxl -= 1 * trr_01x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0011x * 1 * dd;
                    prod_xz = hrr_0011x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * hrr_1011x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0111x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0021x;
                    fxk -= 1 * hrr_0001x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0012x;
                    fxl -= 1 * trr_01x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = hrr_1001x * trr_01y * dd;
                    prod_xz = hrr_1001x * wt * dd;
                    prod_yz = trr_01y * wt * dd;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    fxi = ai2 * hrr_2001x;
                    fxi -= 1 * hrr_0001x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                    fxj = aj2 * hrr_1101x;
                    v_jx += fxj * prod_yz;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_1011x;
                    v_kx += fxk * prod_yz;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                    fxl = al2 * hrr_1002x;
                    fxl -= 1 * trr_10x;
                    v_lx += fxl * prod_yz;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = hrr_0001x * trr_11y * dd;
                    prod_xz = hrr_0001x * wt * dd;
                    prod_yz = trr_11y * wt * dd;
                    fxi = ai2 * hrr_1001x;
                    v_ix += fxi * prod_yz;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                    fxj = aj2 * hrr_0101x;
                    v_jx += fxj * prod_yz;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0011x;
                    v_kx += fxk * prod_yz;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                    fxl = al2 * hrr_0002x;
                    fxl -= 1 * 1;
                    v_lx += fxl * prod_yz;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = hrr_0001x * trr_01y * dd;
                    prod_xz = hrr_0001x * trr_10z * dd;
                    prod_yz = trr_01y * trr_10z * dd;
                    fxi = ai2 * hrr_1001x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0101x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0011x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0002x;
                    fxl -= 1 * 1;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = hrr_1001x * 1 * dd;
                    prod_xz = hrr_1001x * trr_01z * dd;
                    prod_yz = 1 * trr_01z * dd;
                    fxi = ai2 * hrr_2001x;
                    fxi -= 1 * hrr_0001x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1101x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_1011x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1002x;
                    fxl -= 1 * trr_10x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = hrr_0001x * trr_10y * dd;
                    prod_xz = hrr_0001x * trr_01z * dd;
                    prod_yz = trr_10y * trr_01z * dd;
                    fxi = ai2 * hrr_1001x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0101x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0011x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0002x;
                    fxl -= 1 * 1;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = hrr_0001x * 1 * dd;
                    prod_xz = hrr_0001x * trr_11z * dd;
                    prod_yz = 1 * trr_11z * dd;
                    fxi = ai2 * hrr_1001x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0101x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0011x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0002x;
                    fxl -= 1 * 1;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_1;
                        dd += dm_jl_0_1 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_1_0;
                    }
                    prod_xy = trr_11x * hrr_0001y * dd;
                    prod_xz = trr_11x * wt * dd;
                    prod_yz = hrr_0001y * wt * dd;
                    fxi = ai2 * trr_21x;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1001y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                    fyj = aj2 * hrr_0101y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0011y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                    fyl = al2 * hrr_0002y;
                    fyl -= 1 * 1;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_1;
                        dd += dm_jl_0_1 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_1_0;
                    }
                    prod_xy = trr_01x * hrr_1001y * dd;
                    prod_xz = trr_01x * wt * dd;
                    prod_yz = hrr_1001y * wt * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    fyi = ai2 * hrr_2001y;
                    fyi -= 1 * hrr_0001y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                    fyj = aj2 * hrr_1101y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_1011y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                    fyl = al2 * hrr_1002y;
                    fyl -= 1 * trr_10y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_1;
                        dd += dm_jl_0_1 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_1_0;
                    }
                    prod_xy = trr_01x * hrr_0001y * dd;
                    prod_xz = trr_01x * trr_10z * dd;
                    prod_yz = hrr_0001y * trr_10z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1001y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0101y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0011y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0002y;
                    fyl -= 1 * 1;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_0_1;
                        dd += dm_jl_0_1 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_1_1;
                    }
                    prod_xy = trr_10x * hrr_0011y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = hrr_0011y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1011y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                    fyj = aj2 * hrr_0111y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                    double hrr_0021y = trr_03y - ylyk * trr_02y;
                    fyk = ak2 * hrr_0021y;
                    fyk -= 1 * hrr_0001y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0012y = hrr_0021y - ylyk * hrr_0011y;
                    fyl = al2 * hrr_0012y;
                    fyl -= 1 * trr_01y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_1_1;
                        dd += dm_jl_0_1 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_1_1;
                    }
                    prod_xy = 1 * hrr_1011y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = hrr_1011y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double hrr_2011y = trr_22y - ylyk * trr_21y;
                    fyi = ai2 * hrr_2011y;
                    fyi -= 1 * hrr_0011y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                    fyj = aj2 * hrr_1111y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                    double hrr_1021y = trr_13y - ylyk * trr_12y;
                    fyk = ak2 * hrr_1021y;
                    fyk -= 1 * hrr_1001y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_1012y = hrr_1021y - ylyk * hrr_1011y;
                    fyl = al2 * hrr_1012y;
                    fyl -= 1 * trr_11y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_2_1;
                        dd += dm_jl_0_1 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_1_1;
                    }
                    prod_xy = 1 * hrr_0011y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = hrr_0011y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1011y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0111y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0021y;
                    fyk -= 1 * hrr_0001y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0012y;
                    fyl -= 1 * trr_01y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_0_1;
                        dd += dm_jl_0_1 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_1_2;
                    }
                    prod_xy = trr_10x * hrr_0001y * dd;
                    prod_xz = trr_10x * trr_01z * dd;
                    prod_yz = hrr_0001y * trr_01z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1001y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0101y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0011y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0002y;
                    fyl -= 1 * 1;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_1_1;
                        dd += dm_jl_0_1 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_1_2;
                    }
                    prod_xy = 1 * hrr_1001y * dd;
                    prod_xz = 1 * trr_01z * dd;
                    prod_yz = hrr_1001y * trr_01z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_2001y;
                    fyi -= 1 * hrr_0001y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1101y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_1011y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1002y;
                    fyl -= 1 * trr_10y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_2_1;
                        dd += dm_jl_0_1 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+1];
                            dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_1_2;
                    }
                    prod_xy = 1 * hrr_0001y * dd;
                    prod_xz = 1 * trr_11z * dd;
                    prod_yz = hrr_0001y * trr_11z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1001y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0101y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0011y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0002y;
                    fyl -= 1 * 1;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_2;
                        dd += dm_jl_0_2 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_2_0;
                    }
                    prod_xy = trr_11x * 1 * dd;
                    prod_xz = trr_11x * hrr_0001z * dd;
                    prod_yz = 1 * hrr_0001z * dd;
                    fxi = ai2 * trr_21x;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1001z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                    fzj = aj2 * hrr_0101z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0011z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                    fzl = al2 * hrr_0002z;
                    fzl -= 1 * wt;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_2;
                        dd += dm_jl_0_2 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_2_0;
                    }
                    prod_xy = trr_01x * trr_10y * dd;
                    prod_xz = trr_01x * hrr_0001z * dd;
                    prod_yz = trr_10y * hrr_0001z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1001z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0101z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0011z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0002z;
                    fzl -= 1 * wt;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_2;
                        dd += dm_jl_0_2 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_2_0;
                    }
                    prod_xy = trr_01x * 1 * dd;
                    prod_xz = trr_01x * hrr_1001z * dd;
                    prod_yz = 1 * hrr_1001z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    fzi = ai2 * hrr_2001z;
                    fzi -= 1 * hrr_0001z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                    fzj = aj2 * hrr_1101z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_1011z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                    fzl = al2 * hrr_1002z;
                    fzl -= 1 * trr_10z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_0_2;
                        dd += dm_jl_0_2 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_2_1;
                    }
                    prod_xy = trr_10x * trr_01y * dd;
                    prod_xz = trr_10x * hrr_0001z * dd;
                    prod_yz = trr_01y * hrr_0001z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1001z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0101z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0011z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0002z;
                    fzl -= 1 * wt;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_1_2;
                        dd += dm_jl_0_2 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_2_1;
                    }
                    prod_xy = 1 * trr_11y * dd;
                    prod_xz = 1 * hrr_0001z * dd;
                    prod_yz = trr_11y * hrr_0001z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1001z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0101z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0011z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0002z;
                    fzl -= 1 * wt;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_2_2;
                        dd += dm_jl_0_2 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_2_1;
                    }
                    prod_xy = 1 * trr_01y * dd;
                    prod_xz = 1 * hrr_1001z * dd;
                    prod_yz = trr_01y * hrr_1001z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_2001z;
                    fzi -= 1 * hrr_0001z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1101z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_1011z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1002z;
                    fzl -= 1 * trr_10z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_0_2;
                        dd += dm_jl_0_2 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_2_2;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * hrr_0011z * dd;
                    prod_yz = 1 * hrr_0011z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1011z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                    fzj = aj2 * hrr_0111z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                    double hrr_0021z = trr_03z - zlzk * trr_02z;
                    fzk = ak2 * hrr_0021z;
                    fzk -= 1 * hrr_0001z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0012z = hrr_0021z - zlzk * hrr_0011z;
                    fzl = al2 * hrr_0012z;
                    fzl -= 1 * trr_01z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_1_2;
                        dd += dm_jl_0_2 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_2_2;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * hrr_0011z * dd;
                    prod_yz = trr_10y * hrr_0011z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1011z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0111z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0021z;
                    fzk -= 1 * hrr_0001z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0012z;
                    fzl -= 1 * trr_01z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_2_2;
                        dd += dm_jl_0_2 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+2];
                            dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_2_2;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * hrr_1011z * dd;
                    prod_yz = 1 * hrr_1011z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double hrr_2011z = trr_22z - zlzk * trr_21z;
                    fzi = ai2 * hrr_2011z;
                    fzi -= 1 * hrr_0011z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                    fzj = aj2 * hrr_1111z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                    double hrr_1021z = trr_13z - zlzk * trr_12z;
                    fzk = ak2 * hrr_1021z;
                    fzk -= 1 * hrr_1001z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1012z = hrr_1021z - zlzk * hrr_1011z;
                    fzl = al2 * hrr_1012z;
                    fzl -= 1 * trr_11z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_1011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_1011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_1100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 9 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 9*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 3;
        int j = ij / 3;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jk_1_0 = dm[(j0+1)*nao+(k0+0)];
        double dm_jk_2_0 = dm[(j0+2)*nao+(k0+0)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_jl_1_0 = dm[(j0+1)*nao+(l0+0)];
        double dm_jl_2_0 = dm[(j0+2)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_1100x * 1 * dd;
                    prod_xz = hrr_1100x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    fxi = ai2 * hrr_2100x;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxi -= 1 * hrr_0100x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                    fxj = aj2 * hrr_1200x;
                    fxj -= 1 * trr_10x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double b00 = .5 * rt_aa;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    fxk = ak2 * hrr_1110x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                    fxl = al2 * hrr_1101x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0100x * trr_10y * dd;
                    prod_xz = hrr_0100x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_01x = cpx * 1;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0100x * 1 * dd;
                    prod_xz = hrr_0100x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_0_0;
                        dd += dm_jl_1_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * hrr_0100y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = hrr_0100y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_1_0;
                        dd += dm_jl_1_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * hrr_1100y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = hrr_1100y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    fyi = ai2 * hrr_2100y;
                    fyi -= 1 * hrr_0100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                    fyj = aj2 * hrr_1200y;
                    fyj -= 1 * trr_10y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    fyk = ak2 * hrr_1110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                    fyl = al2 * hrr_1101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_2_0;
                        dd += dm_jl_1_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * hrr_0100y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = hrr_0100y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_0_0;
                        dd += dm_jl_2_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * hrr_0100z * dd;
                    prod_yz = 1 * hrr_0100z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_1_0;
                        dd += dm_jl_2_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * hrr_0100z * dd;
                    prod_yz = trr_10y * hrr_0100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_2_0;
                        dd += dm_jl_2_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * hrr_1100z * dd;
                    prod_yz = 1 * hrr_1100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    fzi = ai2 * hrr_2100z;
                    fzi -= 1 * hrr_0100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                    fzj = aj2 * hrr_1200z;
                    fzj -= 1 * trr_10z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    fzk = ak2 * hrr_1110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                    fzl = al2 * hrr_1101z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_1100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_1100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_1110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 9 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 9*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 3;
        int j = ij / 3;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        double dm_lk_0_1 = dm[(l0+0)*nao+(k0+1)];
        double dm_lk_0_2 = dm[(l0+0)*nao+(k0+2)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
            dm_lk_0_1 += dm[nao2+(l0+0)*nao+(k0+1)];
            dm_lk_0_2 += dm[nao2+(l0+0)*nao+(k0+2)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jk_0_1 = dm[(j0+0)*nao+(k0+1)];
        double dm_jk_0_2 = dm[(j0+0)*nao+(k0+2)];
        double dm_jk_1_0 = dm[(j0+1)*nao+(k0+0)];
        double dm_jk_1_1 = dm[(j0+1)*nao+(k0+1)];
        double dm_jk_1_2 = dm[(j0+1)*nao+(k0+2)];
        double dm_jk_2_0 = dm[(j0+2)*nao+(k0+0)];
        double dm_jk_2_1 = dm[(j0+2)*nao+(k0+1)];
        double dm_jk_2_2 = dm[(j0+2)*nao+(k0+2)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_jl_1_0 = dm[(j0+1)*nao+(l0+0)];
        double dm_jl_2_0 = dm[(j0+2)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_0_1 = dm[(i0+0)*nao+(k0+1)];
        double dm_ik_0_2 = dm[(i0+0)*nao+(k0+2)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_1_1 = dm[(i0+1)*nao+(k0+1)];
        double dm_ik_1_2 = dm[(i0+1)*nao+(k0+2)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_ik_2_1 = dm[(i0+2)*nao+(k0+1)];
        double dm_ik_2_2 = dm[(i0+2)*nao+(k0+2)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double b00 = .5 * rt_aa;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_1110x * 1 * dd;
                    prod_xz = hrr_1110x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double hrr_2110x = trr_31x - xjxi * trr_21x;
                    fxi = ai2 * hrr_2110x;
                    double trr_01x = cpx * 1;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    fxi -= 1 * hrr_0110x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                    fxj = aj2 * hrr_1210x;
                    fxj -= 1 * trr_11x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double b01 = .5/akl * (1 - rt_akl);
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double hrr_1120x = trr_22x - xjxi * trr_12x;
                    fxk = ak2 * hrr_1120x;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxk -= 1 * hrr_1100x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_2011x = trr_22x - xlxk * trr_21x;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                    fxl = al2 * hrr_1111x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0110x * trr_10y * dd;
                    prod_xz = hrr_0110x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * hrr_1110x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                    fxj = aj2 * hrr_0210x;
                    fxj -= 1 * trr_01x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double hrr_0120x = trr_12x - xjxi * trr_02x;
                    fxk = ak2 * hrr_0120x;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxk -= 1 * hrr_0100x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                    fxl = al2 * hrr_0111x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0110x * 1 * dd;
                    prod_xz = hrr_0110x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * hrr_1110x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0210x;
                    fxj -= 1 * trr_01x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0120x;
                    fxk -= 1 * hrr_0100x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0111x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_0_0;
                        dd += dm_jl_1_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_11x * hrr_0100y * dd;
                    prod_xz = trr_11x * wt * dd;
                    prod_yz = hrr_0100y * wt * dd;
                    fxi = ai2 * trr_21x;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_1_0;
                        dd += dm_jl_1_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * hrr_1100y * dd;
                    prod_xz = trr_01x * wt * dd;
                    prod_yz = hrr_1100y * wt * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    fyi = ai2 * hrr_2100y;
                    fyi -= 1 * hrr_0100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                    fyj = aj2 * hrr_1200y;
                    fyj -= 1 * trr_10y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    fyk = ak2 * hrr_1110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                    fyl = al2 * hrr_1101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_2_0;
                        dd += dm_jl_1_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * hrr_0100y * dd;
                    prod_xz = trr_01x * trr_10z * dd;
                    prod_yz = hrr_0100y * trr_10z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_0_0;
                        dd += dm_jl_2_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_11x * 1 * dd;
                    prod_xz = trr_11x * hrr_0100z * dd;
                    prod_yz = 1 * hrr_0100z * dd;
                    fxi = ai2 * trr_21x;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_1_0;
                        dd += dm_jl_2_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * trr_10y * dd;
                    prod_xz = trr_01x * hrr_0100z * dd;
                    prod_yz = trr_10y * hrr_0100z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_2_0;
                        dd += dm_jl_2_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * 1 * dd;
                    prod_xz = trr_01x * hrr_1100z * dd;
                    prod_yz = 1 * hrr_1100z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    fzi = ai2 * hrr_2100z;
                    fzi -= 1 * hrr_0100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                    fzj = aj2 * hrr_1200z;
                    fzj -= 1 * trr_10z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    fzk = ak2 * hrr_1110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                    fzl = al2 * hrr_1101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = hrr_1100x * trr_01y * dd;
                    prod_xz = hrr_1100x * wt * dd;
                    prod_yz = trr_01y * wt * dd;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    fxi = ai2 * hrr_2100x;
                    fxi -= 1 * hrr_0100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                    fxj = aj2 * hrr_1200x;
                    fxj -= 1 * trr_10x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_1110x;
                    v_kx += fxk * prod_yz;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                    fxl = al2 * hrr_1101x;
                    v_lx += fxl * prod_yz;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = hrr_0100x * trr_11y * dd;
                    prod_xz = hrr_0100x * wt * dd;
                    prod_yz = trr_11y * wt * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = hrr_0100x * trr_01y * dd;
                    prod_xz = hrr_0100x * trr_10z * dd;
                    prod_yz = trr_01y * trr_10z * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_1 * dm_il_0_0;
                        dd += dm_jl_1_0 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = trr_10x * hrr_0110y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = hrr_0110y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1110y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                    fyj = aj2 * hrr_0210y;
                    fyj -= 1 * trr_01y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double hrr_0120y = trr_12y - yjyi * trr_02y;
                    fyk = ak2 * hrr_0120y;
                    fyk -= 1 * hrr_0100y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                    fyl = al2 * hrr_0111y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_1 * dm_il_1_0;
                        dd += dm_jl_1_0 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * hrr_1110y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = hrr_1110y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double hrr_2110y = trr_31y - yjyi * trr_21y;
                    fyi = ai2 * hrr_2110y;
                    fyi -= 1 * hrr_0110y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                    fyj = aj2 * hrr_1210y;
                    fyj -= 1 * trr_11y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double hrr_1120y = trr_22y - yjyi * trr_12y;
                    fyk = ak2 * hrr_1120y;
                    fyk -= 1 * hrr_1100y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_2011y = trr_22y - ylyk * trr_21y;
                    double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                    fyl = al2 * hrr_1111y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_1 * dm_il_2_0;
                        dd += dm_jl_1_0 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * hrr_0110y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = hrr_0110y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1110y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0210y;
                    fyj -= 1 * trr_01y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0120y;
                    fyk -= 1 * hrr_0100y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0111y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_1 * dm_il_0_0;
                        dd += dm_jl_2_0 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = trr_10x * trr_01y * dd;
                    prod_xz = trr_10x * hrr_0100z * dd;
                    prod_yz = trr_01y * hrr_0100z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_1 * dm_il_1_0;
                        dd += dm_jl_2_0 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_11y * dd;
                    prod_xz = 1 * hrr_0100z * dd;
                    prod_yz = trr_11y * hrr_0100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_1 * dm_il_2_0;
                        dd += dm_jl_2_0 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_01y * dd;
                    prod_xz = 1 * hrr_1100z * dd;
                    prod_yz = trr_01y * hrr_1100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_2100z;
                    fzi -= 1 * hrr_0100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1200z;
                    fzj -= 1 * trr_10z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_1110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = hrr_1100x * 1 * dd;
                    prod_xz = hrr_1100x * trr_01z * dd;
                    prod_yz = 1 * trr_01z * dd;
                    fxi = ai2 * hrr_2100x;
                    fxi -= 1 * hrr_0100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1200x;
                    fxj -= 1 * trr_10x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_1110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = hrr_0100x * trr_10y * dd;
                    prod_xz = hrr_0100x * trr_01z * dd;
                    prod_yz = trr_10y * trr_01z * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = hrr_0100x * 1 * dd;
                    prod_xz = hrr_0100x * trr_11z * dd;
                    prod_yz = 1 * trr_11z * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_2 * dm_il_0_0;
                        dd += dm_jl_1_0 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = trr_10x * hrr_0100y * dd;
                    prod_xz = trr_10x * trr_01z * dd;
                    prod_yz = hrr_0100y * trr_01z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_2 * dm_il_1_0;
                        dd += dm_jl_1_0 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * hrr_1100y * dd;
                    prod_xz = 1 * trr_01z * dd;
                    prod_yz = hrr_1100y * trr_01z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_2100y;
                    fyi -= 1 * hrr_0100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1200y;
                    fyj -= 1 * trr_10y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_1110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_2 * dm_il_2_0;
                        dd += dm_jl_1_0 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * hrr_0100y * dd;
                    prod_xz = 1 * trr_11z * dd;
                    prod_yz = hrr_0100y * trr_11z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_2 * dm_il_0_0;
                        dd += dm_jl_2_0 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * hrr_0110z * dd;
                    prod_yz = 1 * hrr_0110z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1110z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                    fzj = aj2 * hrr_0210z;
                    fzj -= 1 * trr_01z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double hrr_0120z = trr_12z - zjzi * trr_02z;
                    fzk = ak2 * hrr_0120z;
                    fzk -= 1 * hrr_0100z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                    fzl = al2 * hrr_0111z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_2 * dm_il_1_0;
                        dd += dm_jl_2_0 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * hrr_0110z * dd;
                    prod_yz = trr_10y * hrr_0110z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1110z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0210z;
                    fzj -= 1 * trr_01z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0120z;
                    fzk -= 1 * hrr_0100z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0111z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_2 * dm_il_2_0;
                        dd += dm_jl_2_0 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * hrr_1110z * dd;
                    prod_yz = 1 * hrr_1110z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double hrr_2110z = trr_31z - zjzi * trr_21z;
                    fzi = ai2 * hrr_2110z;
                    fzi -= 1 * hrr_0110z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                    fzj = aj2 * hrr_1210z;
                    fzj -= 1 * trr_11z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double hrr_1120z = trr_22z - zjzi * trr_12z;
                    fzk = ak2 * hrr_1120z;
                    fzk -= 1 * hrr_1100z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2011z = trr_22z - zlzk * trr_21z;
                    double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                    fzl = al2 * hrr_1111z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_1110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_1110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_1111(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 9 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 64 * bounds.nroots;
    double *gy = gx + 1152;
    double *gz = gy + 1152;
    int nao2 = nao * nao;
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    int _ik, _il, _jk, _jl, _lk;
    double s0, s1, s2;
    double Rpq[3];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 9*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 3;
        int j = ij / 3;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double Ix, Iy, Iz;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*64];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 8) {
                        if (n == 2) {
                            gz[0] = rw[irys*64+32];
                        }
                        double *_gx = gx + n * 1152;
                        double xjxi = rj[n] - ri[n];
                        double Rpa = xjxi * aj_aij;
                        double c0x = Rpa - rt_aij * Rpq[n];
                        s0 = _gx[0];
                        s1 = c0x * s0;
                        _gx[32] = s1;
                        s2 = c0x * s1 + 1 * b10 * s0;
                        _gx[64] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[96] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
                        s0 = _gx[0];
                        s1 = cpx * s0;
                        _gx[192] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        _gx[384] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        _gx[576] = s2;
                        s0 = _gx[32];
                        s1 = cpx * s0;
                        s1 += 1 * b00 * _gx[0];
                        _gx[224] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 1 * b00 * _gx[192];
                        _gx[416] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 1 * b00 * _gx[384];
                        _gx[608] = s2;
                        s0 = _gx[64];
                        s1 = cpx * s0;
                        s1 += 2 * b00 * _gx[32];
                        _gx[256] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 2 * b00 * _gx[224];
                        _gx[448] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 2 * b00 * _gx[416];
                        _gx[640] = s2;
                        s0 = _gx[96];
                        s1 = cpx * s0;
                        s1 += 3 * b00 * _gx[64];
                        _gx[288] = s1;
                        s2 = cpx*s1 + 1 * b01 *s0;
                        s2 += 3 * b00 * _gx[256];
                        _gx[480] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = cpx*s1 + 2 * b01 *s0;
                        s2 += 3 * b00 * _gx[448];
                        _gx[672] = s2;
                        s1 = _gx[96];
                        s0 = _gx[64];
                        _gx[160] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[32];
                        _gx[128] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[96] = s1 - xjxi * s0;
                        s1 = _gx[288];
                        s0 = _gx[256];
                        _gx[352] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[224];
                        _gx[320] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[192];
                        _gx[288] = s1 - xjxi * s0;
                        s1 = _gx[480];
                        s0 = _gx[448];
                        _gx[544] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[416];
                        _gx[512] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[384];
                        _gx[480] = s1 - xjxi * s0;
                        s1 = _gx[672];
                        s0 = _gx[640];
                        _gx[736] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[608];
                        _gx[704] = s1 - xjxi * s0;
                        s1 = s0;
                        s0 = _gx[576];
                        _gx[672] = s1 - xjxi * s0;
                        s1 = _gx[576];
                        s0 = _gx[384];
                        _gx[960] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[192];
                        _gx[768] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[576] = s1 - xlxk * s0;
                        s1 = _gx[608];
                        s0 = _gx[416];
                        _gx[992] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[224];
                        _gx[800] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[32];
                        _gx[608] = s1 - xlxk * s0;
                        s1 = _gx[640];
                        s0 = _gx[448];
                        _gx[1024] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[256];
                        _gx[832] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[640] = s1 - xlxk * s0;
                        s1 = _gx[672];
                        s0 = _gx[480];
                        _gx[1056] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[288];
                        _gx[864] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[96];
                        _gx[672] = s1 - xlxk * s0;
                        s1 = _gx[704];
                        s0 = _gx[512];
                        _gx[1088] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[320];
                        _gx[896] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[704] = s1 - xlxk * s0;
                        s1 = _gx[736];
                        s0 = _gx[544];
                        _gx[1120] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[352];
                        _gx[928] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[160];
                        _gx[736] = s1 - xlxk * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[896];
                        Iy = gy[0];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[928] - 1 * gx[864]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[928] - xjxi * Ix) - 1 * gx[800]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1088] - 1 * gx[704]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[1088] - xlxk * Ix) - 1 * gx[320]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[0];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[800] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[160] - 1 * gz[96]) * prod_xy;
                        v_jx += aj2 * (gx[800] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[160] - zjzi * Iz) - 1 * gz[32]) * prod_xy;
                        v_kx += (ak2 * gx[960] - 1 * gx[576]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += (al2 * (gx[960] - xlxk * Ix) - 1 * gx[192]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[224];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += (ai2 * gy[256] - 1 * gy[192]) * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[256] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[416] - 1 * gy[32]) * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[416] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[608];
                        Iy = gy[0];
                        Iz = gz[288];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[192]) * prod_xy;
                        v_kx += ak2 * gx[800] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[480] - 1 * gz[96]) * prod_xy;
                        v_lx += (al2 * (gx[800] - xlxk * Ix) - 1 * gx[32]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[480] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[672];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += ai2 * gy[704] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[704] - yjyi * Iy) - 1 * gy[576]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[864] * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[864] - ylyk * Iy) - 1 * gy[96]) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[896];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[928] - 1 * gy[864]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[928] - yjyi * Iy) - 1 * gy[800]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[1088] - 1 * gy[704]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1088] - ylyk * Iy) - 1 * gy[320]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[672];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[704] * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[704] - yjyi * Iy) - 1 * gy[576]) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += ak2 * gy[864] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[864] - ylyk * Iy) - 1 * gy[96]) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[288];
                        Iy = gy[0];
                        Iz = gz[608];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[576]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[192]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[480] - 1 * gx[96]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[800] * prod_xy;
                        v_lx += al2 * (gx[480] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[800] - zlzk * Iz) - 1 * gz[32]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[224];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += (ai2 * gy[256] - 1 * gy[192]) * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[256] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += (ak2 * gy[416] - 1 * gy[32]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[416] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[0];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[160] - 1 * gx[96]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[800] * prod_xy;
                        v_jx += (aj2 * (gx[160] - xjxi * Ix) - 1 * gx[32]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[800] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[960] - 1 * gz[576]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[960] - zlzk * Iz) - 1 * gz[192]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[896];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[928] - 1 * gz[864]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[928] - zjzi * Iz) - 1 * gz[800]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[1088] - 1 * gz[704]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1088] - zlzk * Iz) - 1 * gz[320]) * prod_xy;
                        break;
                    case 1:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[864];
                        Iy = gy[32];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[896] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[896] - xjxi * Ix) - 1 * gx[768]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1056] - 1 * gx[672]) * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[1056] - xlxk * Ix) - 1 * gx[288]) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[704];
                        Iy = gy[192];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[736] - 1 * gx[672]) * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[736] - xjxi * Ix) - 1 * gx[608]) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[896] - xlxk * Ix) - 1 * gx[128]) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[192];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += (ai2 * gz[160] - 1 * gz[96]) * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[160] - zjzi * Iz) - 1 * gz[32]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[32];
                        Iz = gz[288];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[192]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += (ak2 * gz[480] - 1 * gz[96]) * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[480] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[224];
                        Iy = gy[576];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[256] - 1 * gx[192]) * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[256] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[416] - 1 * gx[32]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += al2 * (gx[416] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[864];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[896] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[896] - yjyi * Iy) - 1 * gy[768]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[1056] - 1 * gy[672]) * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1056] - ylyk * Iy) - 1 * gy[288]) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[704];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[736] - 1 * gy[672]) * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[736] - yjyi * Iy) - 1 * gy[608]) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[896] - ylyk * Iy) - 1 * gy[128]) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[224];
                        Iy = gy[96];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[256] - 1 * gx[192]) * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += aj2 * (gx[256] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[416] - 1 * gx[32]) * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[416] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[192];
                        Iz = gz[608];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[576]) * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[800] * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[800] - zlzk * Iz) - 1 * gz[32]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[32];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[800] * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[800] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += (ak2 * gz[960] - 1 * gz[576]) * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[960] - zlzk * Iz) - 1 * gz[192]) * prod_xy;
                        break;
                    case 2:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[864];
                        Iy = gy[0];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[896] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[896] - xjxi * Ix) - 1 * gx[768]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1056] - 1 * gx[672]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += (al2 * (gx[1056] - xlxk * Ix) - 1 * gx[288]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[672];
                        Iy = gy[224];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[704] * prod_yz;
                        v_iy += (ai2 * gy[256] - 1 * gy[192]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                        v_jy += aj2 * (gy[256] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[864] * prod_yz;
                        v_ky += (ak2 * gy[416] - 1 * gy[32]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[864] - xlxk * Ix) - 1 * gx[96]) * prod_yz;
                        v_ly += al2 * (gy[416] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[704];
                        Iy = gy[0];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[736] - 1 * gx[672]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += (aj2 * (gx[736] - xjxi * Ix) - 1 * gx[608]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[896] - xlxk * Ix) - 1 * gx[128]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[0];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[352] - 1 * gz[288]) * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[352] - zjzi * Iz) - 1 * gz[224]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[128]) * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[608];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[800] * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[800] - ylyk * Iy) - 1 * gy[32]) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[768];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[800] * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[800] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += (ak2 * gy[960] - 1 * gy[576]) * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[960] - ylyk * Iy) - 1 * gy[192]) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[672];
                        Iz = gz[224];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[704] * prod_xz;
                        v_iz += (ai2 * gz[256] - 1 * gz[192]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[704] - yjyi * Iy) - 1 * gy[576]) * prod_xz;
                        v_jz += aj2 * (gz[256] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[864] * prod_xz;
                        v_kz += (ak2 * gz[416] - 1 * gz[32]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[864] - ylyk * Iy) - 1 * gy[96]) * prod_xz;
                        v_lz += al2 * (gz[416] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[128];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += (ai2 * gy[160] - 1 * gy[96]) * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[160] - yjyi * Iy) - 1 * gy[32]) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[288];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[192]) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += (ak2 * gy[480] - 1 * gy[96]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[480] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[0];
                        Iz = gz[800];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[832] - 1 * gz[768]) * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[992] - 1 * gz[608]) * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[992] - zlzk * Iz) - 1 * gz[224]) * prod_xy;
                        break;
                    case 3:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[800];
                        Iy = gy[96];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[832] - 1 * gx[768]) * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[992] - 1 * gx[608]) * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[992] - xlxk * Ix) - 1 * gx[224]) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[672];
                        Iy = gy[192];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[704] * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[864] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += (al2 * (gx[864] - xlxk * Ix) - 1 * gx[96]) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[672];
                        Iy = gy[32];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[704] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[864] * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[864] - xlxk * Ix) - 1 * gx[96]) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[576];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[352] - 1 * gx[288]) * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[352] - xjxi * Ix) - 1 * gx[224]) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[576];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += (ai2 * gz[160] - 1 * gz[96]) * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[160] - zjzi * Iz) - 1 * gz[32]) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[800];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[832] - 1 * gy[768]) * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[992] - 1 * gy[608]) * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[992] - ylyk * Iy) - 1 * gy[224]) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[576];
                        Iz = gz[288];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[192]) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[480] - 1 * gz[96]) * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[480] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[96];
                        Iz = gz[608];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[576]) * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += ak2 * gz[800] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[800] - zlzk * Iz) - 1 * gz[32]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[320];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[352] - 1 * gy[288]) * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[352] - yjyi * Iy) - 1 * gy[224]) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[96];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += ai2 * gz[800] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[800] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += (ak2 * gz[960] - 1 * gz[576]) * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[960] - zlzk * Iz) - 1 * gz[192]) * prod_xy;
                        break;
                    case 4:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[128];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[800] * prod_yz;
                        v_iy += (ai2 * gy[160] - 1 * gy[96]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[800] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[160] - yjyi * Iy) - 1 * gy[32]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[960] - 1 * gx[576]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[960] - xlxk * Ix) - 1 * gx[192]) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[608];
                        Iy = gy[288];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[192]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[800] * prod_yz;
                        v_ky += (ak2 * gy[480] - 1 * gy[96]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[800] - xlxk * Ix) - 1 * gx[32]) * prod_yz;
                        v_ly += al2 * (gy[480] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[672];
                        Iy = gy[0];
                        Iz = gz[224];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[704] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[256] - 1 * gz[192]) * prod_xy;
                        v_jx += (aj2 * (gx[704] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[256] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[864] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[416] - 1 * gz[32]) * prod_xy;
                        v_lx += (al2 * (gx[864] - xlxk * Ix) - 1 * gx[96]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[416] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[288];
                        Iy = gy[608];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[192]) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[480] - 1 * gx[96]) * prod_yz;
                        v_ky += ak2 * gy[800] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[480] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[800] - ylyk * Iy) - 1 * gy[32]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[768];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[160] - 1 * gx[96]) * prod_yz;
                        v_iy += ai2 * gy[800] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[160] - xjxi * Ix) - 1 * gx[32]) * prod_yz;
                        v_jy += aj2 * (gy[800] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[960] - 1 * gy[576]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[960] - ylyk * Iy) - 1 * gy[192]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[768];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[800] * prod_xz;
                        v_iz += (ai2 * gz[160] - 1 * gz[96]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[800] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[160] - zjzi * Iz) - 1 * gz[32]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[960] - 1 * gy[576]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[960] - ylyk * Iy) - 1 * gy[192]) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[608];
                        Iz = gz[288];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[192]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[800] * prod_xz;
                        v_kz += (ak2 * gz[480] - 1 * gz[96]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[800] - ylyk * Iy) - 1 * gy[32]) * prod_xz;
                        v_lz += al2 * (gz[480] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[224];
                        Iy = gy[0];
                        Iz = gz[672];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[256] - 1 * gx[192]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[704] * prod_xy;
                        v_jx += aj2 * (gx[256] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[704] - zjzi * Iz) - 1 * gz[576]) * prod_xy;
                        v_kx += (ak2 * gx[416] - 1 * gx[32]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[864] * prod_xy;
                        v_lx += al2 * (gx[416] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[864] - zlzk * Iz) - 1 * gz[96]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[288];
                        Iz = gz[608];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[576]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[192]) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[480] - 1 * gy[96]) * prod_xz;
                        v_kz += ak2 * gz[800] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[480] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[800] - zlzk * Iz) - 1 * gz[32]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[128];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[160] - 1 * gy[96]) * prod_xz;
                        v_iz += ai2 * gz[800] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[160] - yjyi * Iy) - 1 * gy[32]) * prod_xz;
                        v_jz += aj2 * (gz[800] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[960] - 1 * gz[576]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[960] - zlzk * Iz) - 1 * gz[192]) * prod_xy;
                        break;
                    case 5:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[96];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[800] * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[800] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[960] - 1 * gx[576]) * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += (al2 * (gx[960] - xlxk * Ix) - 1 * gx[192]) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[320];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += (ai2 * gy[352] - 1 * gy[288]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[352] - yjyi * Iy) - 1 * gy[224]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[608];
                        Iy = gy[96];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[800] * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[800] - xlxk * Ix) - 1 * gx[32]) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[288];
                        Iy = gy[576];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[192]) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[480] - 1 * gx[96]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += al2 * (gx[480] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[800];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += (ai2 * gy[832] - 1 * gy[768]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += (ak2 * gy[992] - 1 * gy[608]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[992] - ylyk * Iy) - 1 * gy[224]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[576];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[160] - 1 * gx[96]) * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += (aj2 * (gx[160] - xjxi * Ix) - 1 * gx[32]) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+1);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[576];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += (ai2 * gz[352] - 1 * gz[288]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[352] - zjzi * Iz) - 1 * gz[224]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[128]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[32];
                        Iz = gz[672];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[704] * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[704] - zjzi * Iz) - 1 * gz[576]) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += ak2 * gz[864] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[864] - zlzk * Iz) - 1 * gz[96]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[192];
                        Iz = gz[672];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += ai2 * gz[704] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[704] - zjzi * Iz) - 1 * gz[576]) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[864] * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[864] - zlzk * Iz) - 1 * gz[96]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+2);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[96];
                        Iz = gz[800];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += (ai2 * gz[832] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += (ak2 * gz[992] - 1 * gz[608]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[992] - zlzk * Iz) - 1 * gz[224]) * prod_xy;
                        break;
                    case 6:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[800];
                        Iy = gy[0];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[832] - 1 * gx[768]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[992] - 1 * gx[608]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += (al2 * (gx[992] - xlxk * Ix) - 1 * gx[224]) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[288];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[192]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[480] - 1 * gy[96]) * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[480] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[128];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += (ai2 * gy[160] - 1 * gy[96]) * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[160] - yjyi * Iy) - 1 * gy[32]) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[224];
                        Iy = gy[672];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[256] - 1 * gx[192]) * prod_yz;
                        v_iy += ai2 * gy[704] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[256] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[704] - yjyi * Iy) - 1 * gy[576]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[416] - 1 * gx[32]) * prod_yz;
                        v_ky += ak2 * gy[864] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[416] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[864] - ylyk * Iy) - 1 * gy[96]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[768];
                        Iz = gz[32];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += ai2 * gy[800] * prod_xz;
                        v_iz += (ai2 * gz[64] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[800] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += (ak2 * gy[960] - 1 * gy[576]) * prod_xz;
                        v_kz += ak2 * gz[224] * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[960] - ylyk * Iy) - 1 * gy[192]) * prod_xz;
                        v_lz += al2 * (gz[224] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[608];
                        Iz = gz[192];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[224] * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[224] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += ak2 * gy[800] * prod_xz;
                        v_kz += (ak2 * gz[384] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[800] - ylyk * Iy) - 1 * gy[32]) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[0];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[352] - 1 * gx[288]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += (aj2 * (gx[352] - xjxi * Ix) - 1 * gx[224]) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[0];
                        Iz = gz[704];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += (ai2 * gz[736] - 1 * gz[672]) * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[736] - zjzi * Iz) - 1 * gz[608]) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[896] - zlzk * Iz) - 1 * gz[128]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[224];
                        Iz = gz[672];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[256] - 1 * gy[192]) * prod_xz;
                        v_iz += ai2 * gz[704] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[256] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[704] - zjzi * Iz) - 1 * gz[576]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[416] - 1 * gy[32]) * prod_xz;
                        v_kz += ak2 * gz[864] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[416] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[864] - zlzk * Iz) - 1 * gz[96]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[0];
                        Iz = gz[864];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[32] * prod_xz;
                        v_iz += ai2 * gz[896] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[32] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[896] - zjzi * Iz) - 1 * gz[768]) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += ak2 * gy[192] * prod_xz;
                        v_kz += (ak2 * gz[1056] - 1 * gz[672]) * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[192] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1056] - zlzk * Iz) - 1 * gz[288]) * prod_xy;
                        break;
                    case 7:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[32];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[800] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[800] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[960] - 1 * gx[576]) * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += (al2 * (gx[960] - xlxk * Ix) - 1 * gx[192]) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[608];
                        Iy = gy[192];
                        Iz = gz[96];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += ai2 * gz[128] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[128] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[800] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[288] * prod_xy;
                        v_lx += (al2 * (gx[800] - xlxk * Ix) - 1 * gx[32]) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[288] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[96];
                        Iz = gz[224];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[608] * prod_yz;
                        v_iy += ai2 * gy[128] * prod_xz;
                        v_iz += (ai2 * gz[256] - 1 * gz[192]) * prod_xy;
                        v_jx += aj2 * (gx[608] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[128] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[256] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[288] * prod_xz;
                        v_kz += (ak2 * gz[416] - 1 * gz[32]) * prod_xy;
                        v_lx += (al2 * (gx[768] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[288] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[416] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[192];
                        Iy = gy[704];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[224] * prod_yz;
                        v_iy += (ai2 * gy[736] - 1 * gy[672]) * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[224] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[736] - yjyi * Iy) - 1 * gy[608]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[384] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[896] - ylyk * Iy) - 1 * gy[128]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+1);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[32];
                        Iy = gy[864];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[64] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[896] * prod_xz;
                        v_iz += ai2 * gz[32] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[896] - yjyi * Iy) - 1 * gy[768]) * prod_xz;
                        v_jz += aj2 * (gz[32] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[224] * prod_yz;
                        v_ky += (ak2 * gy[1056] - 1 * gy[672]) * prod_xz;
                        v_kz += ak2 * gz[192] * prod_xy;
                        v_lx += al2 * (gx[224] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1056] - ylyk * Iy) - 1 * gy[288]) * prod_xz;
                        v_lz += al2 * (gz[192] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[96];
                        Iy = gy[576];
                        Iz = gz[224];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[128] * prod_yz;
                        v_iy += ai2 * gy[608] * prod_xz;
                        v_iz += (ai2 * gz[256] - 1 * gz[192]) * prod_xy;
                        v_jx += (aj2 * (gx[128] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[608] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[256] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[288] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[416] - 1 * gz[32]) * prod_xy;
                        v_lx += al2 * (gx[288] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[768] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[416] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[288];
                        Iy = gy[32];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[192]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[480] - 1 * gx[96]) * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[480] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[192];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[160] - 1 * gx[96]) * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += ai2 * gz[608] * prod_xy;
                        v_jx += (aj2 * (gx[160] - xjxi * Ix) - 1 * gx[32]) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[608] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[768] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[192];
                        Iz = gz[704];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += ai2 * gy[224] * prod_xz;
                        v_iz += (ai2 * gz[736] - 1 * gz[672]) * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[224] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[736] - zjzi * Iz) - 1 * gz[608]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += (ak2 * gy[384] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[896] - zlzk * Iz) - 1 * gz[128]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+2);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[32];
                        Iz = gz[864];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[32] * prod_yz;
                        v_iy += (ai2 * gy[64] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[896] * prod_xy;
                        v_jx += aj2 * (gx[32] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[896] - zjzi * Iz) - 1 * gz[768]) * prod_xy;
                        v_kx += ak2 * gx[192] * prod_yz;
                        v_ky += ak2 * gy[224] * prod_xz;
                        v_kz += (ak2 * gz[1056] - 1 * gz[672]) * prod_xy;
                        v_lx += al2 * (gx[192] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[224] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1056] - zlzk * Iz) - 1 * gz[288]) * prod_xy;
                        break;
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_1111(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_1111(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 6 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 6*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_ik_3_0 = dm[(i0+3)*nao+(k0+0)];
        double dm_ik_4_0 = dm[(i0+4)*nao+(k0+0)];
        double dm_ik_5_0 = dm[(i0+5)*nao+(k0+0)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dm_il_3_0 = dm[(i0+3)*nao+(l0+0)];
        double dm_il_4_0 = dm[(i0+4)*nao+(l0+0)];
        double dm_il_5_0 = dm[(i0+5)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 4*nsq_per_block;
                    rys_roots(2, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 2; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_20x * 1 * dd;
                    prod_xz = trr_20x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    fxi = ai2 * trr_30x;
                    fxi -= 2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    fxj = aj2 * hrr_2100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double b00 = .5 * rt_aa;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    fxk = ak2 * trr_21x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    fxl = al2 * hrr_2001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * trr_10y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_3_0;
                        dd += dm_jl_0_0 * dm_ik_3_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * trr_20y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = trr_20y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    fyi = ai2 * trr_30y;
                    fyi -= 2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    fyj = aj2 * hrr_2100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_01x = cpx * 1;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    fyk = ak2 * trr_21y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    fyl = al2 * hrr_2001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_4_0;
                        dd += dm_jl_0_0 * dm_ik_4_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = trr_10y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_5_0;
                        dd += dm_jl_0_0 * dm_ik_5_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * trr_20z * dd;
                    prod_yz = 1 * trr_20z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    fzi = ai2 * trr_30z;
                    fzi -= 2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    fzj = aj2 * hrr_2100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    fzk = ak2 * trr_21z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    fzl = al2 * hrr_2001z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 6 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 6*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        double dm_lk_0_1 = dm[(l0+0)*nao+(k0+1)];
        double dm_lk_0_2 = dm[(l0+0)*nao+(k0+2)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
            dm_lk_0_1 += dm[nao2+(l0+0)*nao+(k0+1)];
            dm_lk_0_2 += dm[nao2+(l0+0)*nao+(k0+2)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jk_0_1 = dm[(j0+0)*nao+(k0+1)];
        double dm_jk_0_2 = dm[(j0+0)*nao+(k0+2)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_0_1 = dm[(i0+0)*nao+(k0+1)];
        double dm_ik_0_2 = dm[(i0+0)*nao+(k0+2)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_1_1 = dm[(i0+1)*nao+(k0+1)];
        double dm_ik_1_2 = dm[(i0+1)*nao+(k0+2)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_ik_2_1 = dm[(i0+2)*nao+(k0+1)];
        double dm_ik_2_2 = dm[(i0+2)*nao+(k0+2)];
        double dm_ik_3_0 = dm[(i0+3)*nao+(k0+0)];
        double dm_ik_3_1 = dm[(i0+3)*nao+(k0+1)];
        double dm_ik_3_2 = dm[(i0+3)*nao+(k0+2)];
        double dm_ik_4_0 = dm[(i0+4)*nao+(k0+0)];
        double dm_ik_4_1 = dm[(i0+4)*nao+(k0+1)];
        double dm_ik_4_2 = dm[(i0+4)*nao+(k0+2)];
        double dm_ik_5_0 = dm[(i0+5)*nao+(k0+0)];
        double dm_ik_5_1 = dm[(i0+5)*nao+(k0+1)];
        double dm_ik_5_2 = dm[(i0+5)*nao+(k0+2)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dm_il_3_0 = dm[(i0+3)*nao+(l0+0)];
        double dm_il_4_0 = dm[(i0+4)*nao+(l0+0)];
        double dm_il_5_0 = dm[(i0+5)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double b00 = .5 * rt_aa;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_21x * 1 * dd;
                    prod_xz = trr_21x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    fxi = ai2 * trr_31x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    fxi -= 2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_2110x = trr_31x - xjxi * trr_21x;
                    fxj = aj2 * hrr_2110x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double b01 = .5/akl * (1 - rt_akl);
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    fxk = ak2 * trr_22x;
                    fxk -= 1 * trr_20x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_2011x = trr_22x - xlxk * trr_21x;
                    fxl = al2 * hrr_2011x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_11x * trr_10y * dd;
                    prod_xz = trr_11x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * trr_21x;
                    double trr_01x = cpx * 1;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_11x * 1 * dd;
                    prod_xz = trr_11x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * trr_21x;
                    fxi -= 1 * trr_01x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_12x;
                    fxk -= 1 * trr_10x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_3_0;
                        dd += dm_jl_0_0 * dm_ik_3_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * trr_20y * dd;
                    prod_xz = trr_01x * wt * dd;
                    prod_yz = trr_20y * wt * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    fyi = ai2 * trr_30y;
                    fyi -= 2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    fyj = aj2 * hrr_2100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
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
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    fyl = al2 * hrr_2001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_4_0;
                        dd += dm_jl_0_0 * dm_ik_4_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * trr_10y * dd;
                    prod_xz = trr_01x * trr_10z * dd;
                    prod_yz = trr_10y * trr_10z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_5_0;
                        dd += dm_jl_0_0 * dm_ik_5_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_01x * 1 * dd;
                    prod_xz = trr_01x * trr_20z * dd;
                    prod_yz = 1 * trr_20z * dd;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    fzi = ai2 * trr_30z;
                    fzi -= 2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0110x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    fzj = aj2 * hrr_2100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_02x;
                    fxk -= 1 * 1;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    fzk = ak2 * trr_21z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0011x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    fzl = al2 * hrr_2001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = trr_20x * trr_01y * dd;
                    prod_xz = trr_20x * wt * dd;
                    prod_yz = trr_01y * wt * dd;
                    fxi = ai2 * trr_30x;
                    fxi -= 2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    fxj = aj2 * hrr_2100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_21x;
                    v_kx += fxk * prod_yz;
                    double trr_02y = cpy * trr_01y + 1*b01 * 1;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    fxl = al2 * hrr_2001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = trr_10x * trr_11y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = trr_11y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = trr_10x * trr_01y * dd;
                    prod_xz = trr_10x * trr_10z * dd;
                    prod_yz = trr_01y * trr_10z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_3_0;
                        dd += dm_jl_0_0 * dm_ik_3_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_21y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = trr_21y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    fyi = ai2 * trr_31y;
                    fyi -= 2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_2110y = trr_31y - yjyi * trr_21y;
                    fyj = aj2 * hrr_2110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    fyk = ak2 * trr_22y;
                    fyk -= 1 * trr_20y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_2011y = trr_22y - ylyk * trr_21y;
                    fyl = al2 * hrr_2011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_4_0;
                        dd += dm_jl_0_0 * dm_ik_4_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_11y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = trr_11y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_21y;
                    fyi -= 1 * trr_01y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_12y;
                    fyk -= 1 * trr_10y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_1 * dm_il_5_0;
                        dd += dm_jl_0_0 * dm_ik_5_1;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+1];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_1;
                    }
                    prod_xy = 1 * trr_01y * dd;
                    prod_xz = 1 * trr_20z * dd;
                    prod_yz = trr_01y * trr_20z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_30z;
                    fzi -= 2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_2100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_02y;
                    fyk -= 1 * 1;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_21z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0011y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_2001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = trr_20x * 1 * dd;
                    prod_xz = trr_20x * trr_01z * dd;
                    prod_yz = 1 * trr_01z * dd;
                    fxi = ai2 * trr_30x;
                    fxi -= 2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_2100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_21x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_2001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = trr_10x * trr_10y * dd;
                    prod_xz = trr_10x * trr_01z * dd;
                    prod_yz = trr_10y * trr_01z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * trr_11z * dd;
                    prod_yz = 1 * trr_11z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_3_0;
                        dd += dm_jl_0_0 * dm_ik_3_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * trr_20y * dd;
                    prod_xz = 1 * trr_01z * dd;
                    prod_yz = trr_20y * trr_01z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_30y;
                    fyi -= 2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_2100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_21y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_02z;
                    fzk -= 1 * wt;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_2001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_4_0;
                        dd += dm_jl_0_0 * dm_ik_4_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * trr_11z * dd;
                    prod_yz = trr_10y * trr_11z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_21z;
                    fzi -= 1 * trr_01z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_12z;
                    fzk -= 1 * trr_10z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1011z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_2 * dm_il_5_0;
                        dd += dm_jl_0_0 * dm_ik_5_2;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+2];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_2;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * trr_21z * dd;
                    prod_yz = 1 * trr_21z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    fzi = ai2 * trr_31z;
                    fzi -= 2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_2110z = trr_31z - zjzi * trr_21z;
                    fzj = aj2 * hrr_2110z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    fzk = ak2 * trr_22z;
                    fzk -= 1 * trr_20z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2011z = trr_22z - zlzk * trr_21z;
                    fzl = al2 * hrr_2011z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 6 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 128 * bounds.nroots;
    double *gy = gx + 1536;
    double *gz = gy + 1536;
    int nao2 = nao * nao;
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    int _ik, _il, _jk, _jl, _lk;
    double s0, s1, s2;
    double Rpq[3];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 6*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double Ix, Iy, Iz;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gz[0] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1536;
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
                        s1 = _gx[768];
                        s0 = _gx[512];
                        _gx[1280] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[256];
                        _gx[1024] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[0];
                        _gx[768] = s1 - xlxk * s0;
                        s1 = _gx[832];
                        s0 = _gx[576];
                        _gx[1344] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[320];
                        _gx[1088] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[64];
                        _gx[832] = s1 - xlxk * s0;
                        s1 = _gx[896];
                        s0 = _gx[640];
                        _gx[1408] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[384];
                        _gx[1152] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[128];
                        _gx[896] = s1 - xlxk * s0;
                        s1 = _gx[960];
                        s0 = _gx[704];
                        _gx[1472] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[448];
                        _gx[1216] = s1 - xlxk * s0;
                        s1 = s0;
                        s0 = _gx[192];
                        _gx[960] = s1 - xlxk * s0;
                    }
                    __syncthreads();
                    switch (gout_id) {
                    case 0:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[1152];
                        Iy = gy[0];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[1216] - 2 * gx[1088]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[1216] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1408] - 1 * gx[896]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += (al2 * (gx[1408] - xlxk * Ix) - 1 * gx[384]) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[1024];
                        Iy = gy[64];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[1088] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[1088] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1280] - 1 * gx[768]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += (al2 * (gx[1280] - xlxk * Ix) - 1 * gx[256]) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[832];
                        Iy = gy[256];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[896] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += (al2 * (gx[1088] - xlxk * Ix) - 1 * gx[64]) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[896];
                        Iy = gy[0];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[960] - 2 * gx[832]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[960] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1152] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[1152] - xlxk * Ix) - 1 * gx[128]) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[64];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += (al2 * (gx[1024] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[768];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1024] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[1024];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[1088] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[1088] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += (ak2 * gy[1280] - 1 * gy[768]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1280] - ylyk * Iy) - 1 * gy[256]) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+1);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[1088];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[1152] - 1 * gy[1024]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[1152] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[1344] - 1 * gy[832]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1344] - ylyk * Iy) - 1 * gy[320]) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[768];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1024] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[0];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[640] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1024] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+2);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[64];
                        Iz = gz[832];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[896] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[896] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1088] - zlzk * Iz) - 1 * gz[64]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[256];
                        Iz = gz[832];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[896] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[896] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1088] - zlzk * Iz) - 1 * gz[64]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[0];
                        Iz = gz[1024];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[1088] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[1088] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[1280] - 1 * gz[768]) * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1280] - zlzk * Iz) - 1 * gz[256]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+2);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[64];
                        Iz = gz[1088];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[1152] - 1 * gz[1024]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[1152] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[1344] - 1 * gz[832]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1344] - zlzk * Iz) - 1 * gz[320]) * prod_xy;
                        break;
                    case 1:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[1088];
                        Iy = gy[64];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[1152] - 1 * gx[1024]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[1152] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1344] - 1 * gx[832]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += (al2 * (gx[1344] - xlxk * Ix) - 1 * gx[320]) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[1024];
                        Iy = gy[0];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[1088] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[1088] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1280] - 1 * gx[768]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += (al2 * (gx[1280] - xlxk * Ix) - 1 * gx[256]) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[384];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[448] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += (ak2 * gy[640] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += (al2 * (gx[1024] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[832];
                        Iy = gy[64];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[896] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[1088] - xlxk * Ix) - 1 * gx[64]) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[0];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[448] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[640] - 1 * gz[128]) * prod_xy;
                        v_lx += (al2 * (gx[1024] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+1);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[896];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[960] - 2 * gy[832]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[960] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[1152] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1152] - ylyk * Iy) - 1 * gy[128]) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[1088];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[1152] - 1 * gy[1024]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[1152] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[1344] - 1 * gy[832]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1344] - ylyk * Iy) - 1 * gy[320]) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+1);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[1024];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[1088] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[1088] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[1280] - 1 * gy[768]) * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1280] - ylyk * Iy) - 1 * gy[256]) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+1);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[896];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[960] - 2 * gy[832]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[960] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[1152] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1152] - ylyk * Iy) - 1 * gy[128]) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[64];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1024] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+2);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[0];
                        Iz = gz[896];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[960] - 2 * gz[832]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[960] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[1152] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1152] - zlzk * Iz) - 1 * gz[128]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+2);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[384];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[448] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[640] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1024] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[64];
                        Iz = gz[1024];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[1088] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[1088] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[1280] - 1 * gz[768]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1280] - zlzk * Iz) - 1 * gz[256]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+2);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[1152];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[1216] - 2 * gz[1088]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[1216] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[1408] - 1 * gz[896]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1408] - zlzk * Iz) - 1 * gz[384]) * prod_xy;
                        break;
                    case 2:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[1088];
                        Iy = gy[0];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[1152] - 1 * gx[1024]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[1152] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1344] - 1 * gx[832]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += (al2 * (gx[1344] - xlxk * Ix) - 1 * gx[320]) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[896];
                        Iy = gy[256];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[960] - 2 * gx[832]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[960] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1152] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += (al2 * (gx[1152] - xlxk * Ix) - 1 * gx[128]) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[320];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += (al2 * (gx[1024] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[832];
                        Iy = gy[0];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[896] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += (al2 * (gx[1088] - xlxk * Ix) - 1 * gx[64]) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[768];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[640] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1024] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+1);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[832];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[896] - 1 * gy[768]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[896] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1088] - ylyk * Iy) - 1 * gy[64]) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+1);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[1024];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[1088] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[1088] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[1280] - 1 * gy[768]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1280] - ylyk * Iy) - 1 * gy[256]) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+1);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[768];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1024] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+1);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[832];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[896] - 1 * gy[768]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[896] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1088] - ylyk * Iy) - 1 * gy[64]) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[0];
                        Iz = gz[832];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[896] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[896] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1088] - zlzk * Iz) - 1 * gz[64]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+2);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[256];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1024] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+2);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[320];
                        Iz = gz[832];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[896] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[896] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1088] - zlzk * Iz) - 1 * gz[64]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+2);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[0];
                        Iz = gz[1088];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[1152] - 1 * gz[1024]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[1152] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[1344] - 1 * gz[832]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1344] - zlzk * Iz) - 1 * gz[320]) * prod_xy;
                        break;
                    case 3:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[1024];
                        Iy = gy[128];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[1088] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[1088] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1280] - 1 * gx[768]) * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += (al2 * (gx[1280] - xlxk * Ix) - 1 * gx[256]) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[832];
                        Iy = gy[320];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[896] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += (al2 * (gx[1088] - xlxk * Ix) - 1 * gx[64]) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[256];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += (al2 * (gx[1024] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[128];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[832] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += (al2 * (gx[1024] - xlxk * Ix) - 1 * gx[0]) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[832];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[896] - 1 * gy[768]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[896] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1088] - ylyk * Iy) - 1 * gy[64]) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+1);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[768];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1024] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+1);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[1152];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[1216] - 2 * gy[1088]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[1216] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[1408] - 1 * gy[896]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1408] - ylyk * Iy) - 1 * gy[384]) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+1);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[832];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[896] - 1 * gy[768]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[896] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1088] - ylyk * Iy) - 1 * gy[64]) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+1);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+1);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+1)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[768];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[832] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[448] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += (ak2 * gz[640] - 1 * gz[128]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += (al2 * (gy[1024] - ylyk * Iy) - 1 * gy[0]) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+2);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[128];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1024] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+2);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[320];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[832] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1024] - zlzk * Iz) - 1 * gz[0]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+2);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[256];
                        Iz = gz[896];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[960] - 2 * gz[832]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[960] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[1152] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1152] - zlzk * Iz) - 1 * gz[128]) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+2);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+2);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+2)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[128];
                        Iz = gz[1024];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[1088] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[1088] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += (ak2 * gz[1280] - 1 * gz[768]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += (al2 * (gz[1280] - zlzk * Iz) - 1 * gz[256]) * prod_xy;
                        break;
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2020(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 6 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 128 * bounds.nroots;
    double *gy = gx + 1024;
    double *gz = gy + 1024;
    int nao2 = nao * nao;
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    int _ik, _il, _jk, _jl, _lk;
    double s0, s1, s2;
    double Rpq[3];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 6*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double Ix, Iy, Iz;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
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
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 2 * b10 * s0;
                        _gx[192] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
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
                    switch (gout_id) {
                    case 0:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[640];
                        Iy = gy[0];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[704] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[896] - 2 * gx[384]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[896] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[64];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[768] - 2 * gx[256]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[256];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[0];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[640] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[64];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+3);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+3);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+3);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[512];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[768] - 2 * gy[256]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+4);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+4);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+4);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[256];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+4);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+4);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+4);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[320];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+5);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+5);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+5);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[0];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[832] - 2 * gz[320]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        break;
                    case 1:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[64];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[832] - 2 * gx[320]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[0];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[768] - 2 * gx[256]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[384];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[448] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += (ak2 * gy[640] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[64];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[0];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[448] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[640] - 1 * gz[128]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+3);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+3);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+3);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[640];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[704] - 2 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[704] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[896] - 2 * gy[384]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[896] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+4);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+4);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+4);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[320];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+4);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+4);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+4);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[256];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[448] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += (ak2 * gz[640] - 1 * gz[128]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+5);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+5);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+5);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[128];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += (ak2 * gz[768] - 2 * gz[256]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        break;
                    case 2:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[0];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[832] - 2 * gx[320]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[256];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[448] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[640] - 1 * gx[128]) * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[320];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[0];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+3);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+3);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+3);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[512];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += (ak2 * gy[768] - 2 * gy[256]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+3);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+3);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+3);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[576];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[832] - 2 * gy[320]) * prod_xz;
                        v_kz += ak2 * gz[320] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[320] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+4);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+4);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+4);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[256];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[384] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += (ak2 * gz[576] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+5);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+5);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+5);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[0];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[384] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[768] - 2 * gz[256]) * prod_xy;
                        v_lx += al2 * (gx[384] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+5);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+5);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+5);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[64];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[832] - 2 * gz[320]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        break;
                    case 3:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[128];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[768] - 2 * gx[256]) * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[320];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[384] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[384] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[576] - 1 * gx[64]) * prod_yz;
                        v_ky += (ak2 * gy[576] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[256];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[320] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += (ak2 * gy[512] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[128];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[320] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[512] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[384] * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[384] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+3);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+3);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+3);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[576];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += (ak2 * gy[832] - 2 * gy[320]) * prod_xz;
                        v_kz += ak2 * gz[256] * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[256] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+3);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+3);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+3);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[512];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[768] - 2 * gy[256]) * prod_xz;
                        v_kz += ak2 * gz[384] * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[384] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+4);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+4);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+4);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[384];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[448] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[320] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += (ak2 * gy[640] - 1 * gy[128]) * prod_xz;
                        v_kz += (ak2 * gz[512] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+5);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+5);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+5);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[64];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[320] * prod_yz;
                        v_ky += ak2 * gy[320] * prod_xz;
                        v_kz += (ak2 * gz[768] - 2 * gz[256]) * prod_xy;
                        v_lx += al2 * (gx[320] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[320] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+5);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+5);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+5);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[640];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[704] - 2 * gz[576]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[704] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[256] * prod_yz;
                        v_ky += ak2 * gy[256] * prod_xz;
                        v_kz += (ak2 * gz[896] - 2 * gz[384]) * prod_xy;
                        v_lx += al2 * (gx[256] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[256] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[896] - zlzk * Iz) * prod_xy;
                        break;
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2020(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2020(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 18 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 18*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jk_1_0 = dm[(j0+1)*nao+(k0+0)];
        double dm_jk_2_0 = dm[(j0+2)*nao+(k0+0)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_jl_1_0 = dm[(j0+1)*nao+(l0+0)];
        double dm_jl_2_0 = dm[(j0+2)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_ik_1_0 = dm[(i0+1)*nao+(k0+0)];
        double dm_ik_2_0 = dm[(i0+2)*nao+(k0+0)];
        double dm_ik_3_0 = dm[(i0+3)*nao+(k0+0)];
        double dm_ik_4_0 = dm[(i0+4)*nao+(k0+0)];
        double dm_ik_5_0 = dm[(i0+5)*nao+(k0+0)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        double dm_il_1_0 = dm[(i0+1)*nao+(l0+0)];
        double dm_il_2_0 = dm[(i0+2)*nao+(l0+0)];
        double dm_il_3_0 = dm[(i0+3)*nao+(l0+0)];
        double dm_il_4_0 = dm[(i0+4)*nao+(l0+0)];
        double dm_il_5_0 = dm[(i0+5)*nao+(l0+0)];
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double fxi, fyi, fzi;
        double fxj, fyj, fzj;
        double fxk, fyk, fzk;
        double fxl, fyl, fzl;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                if (task_id >= ntasks) {
                    continue;
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    double wt = rw[(2*irys+1)*nsq_per_block];
                    double rt = rw[ 2*irys   *nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double c0x = xpa - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double b10 = .5/aij * (1 - rt_aij);
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_0_0;
                        dd += dm_jl_0_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_2100x * 1 * dd;
                    prod_xz = hrr_2100x * wt * dd;
                    prod_yz = 1 * wt * dd;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    double hrr_3100x = trr_40x - xjxi * trr_30x;
                    fxi = ai2 * hrr_3100x;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    fxi -= 2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    double c0y = ypa - ypq*rt_aij;
                    double trr_10y = c0y * 1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zpa - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                    fxj = aj2 * hrr_2200x;
                    fxj -= 1 * trr_20x;
                    v_jx += fxj * prod_yz;
                    double hrr_0100y = trr_10y - yjyi * 1;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double rt_akl = rt_aa * aij;
                    double cpx = xqc + xpq*rt_akl;
                    double b00 = .5 * rt_aa;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double hrr_2110x = trr_31x - xjxi * trr_21x;
                    fxk = ak2 * hrr_2110x;
                    v_kx += fxk * prod_yz;
                    double cpy = yqc + ypq*rt_akl;
                    double trr_01y = cpy * 1;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double cpz = zqc + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_3001x = trr_31x - xlxk * trr_30x;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    double hrr_2101x = hrr_3001x - xjxi * hrr_2001x;
                    fxl = al2 * hrr_2101x;
                    v_lx += fxl * prod_yz;
                    double hrr_0001y = trr_01y - ylyk * 1;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_1_0;
                        dd += dm_jl_0_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_1100x * trr_10y * dd;
                    prod_xz = hrr_1100x * wt * dd;
                    prod_yz = trr_10y * wt * dd;
                    fxi = ai2 * hrr_2100x;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    fxi -= 1 * hrr_0100x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * 1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                    fxj = aj2 * hrr_1200x;
                    fxj -= 1 * trr_10x;
                    v_jx += fxj * prod_yz;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    fxk = ak2 * hrr_1110x;
                    v_kx += fxk * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * 1;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                    fxl = al2 * hrr_1101x;
                    v_lx += fxl * prod_yz;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_2_0;
                        dd += dm_jl_0_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_1100x * 1 * dd;
                    prod_xz = hrr_1100x * trr_10z * dd;
                    prod_yz = 1 * trr_10z * dd;
                    fxi = ai2 * hrr_2100x;
                    fxi -= 1 * hrr_0100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1200x;
                    fxj -= 1 * trr_10x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_1110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_3_0;
                        dd += dm_jl_0_0 * dm_ik_3_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0100x * trr_20y * dd;
                    prod_xz = hrr_0100x * wt * dd;
                    prod_yz = trr_20y * wt * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    fyi = ai2 * trr_30y;
                    fyi -= 2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    fyj = aj2 * hrr_2100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    double trr_01x = cpx * 1;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    fyk = ak2 * trr_21y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    fyl = al2 * hrr_2001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_4_0;
                        dd += dm_jl_0_0 * dm_ik_4_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0100x * trr_10y * dd;
                    prod_xz = hrr_0100x * trr_10z * dd;
                    prod_yz = trr_10y * trr_10z * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_0_0 * dm_il_5_0;
                        dd += dm_jl_0_0 * dm_ik_5_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = hrr_0100x * 1 * dd;
                    prod_xz = hrr_0100x * trr_20z * dd;
                    prod_yz = 1 * trr_20z * dd;
                    fxi = ai2 * hrr_1100x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    fzi = ai2 * trr_30z;
                    fzi -= 2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0200x;
                    fxj -= 1 * 1;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    fzj = aj2 * hrr_2100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * hrr_0110x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    fzk = ak2 * trr_21z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0101x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    fzl = al2 * hrr_2001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_0_0;
                        dd += dm_jl_1_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_20x * hrr_0100y * dd;
                    prod_xz = trr_20x * wt * dd;
                    prod_yz = hrr_0100y * wt * dd;
                    fxi = ai2 * trr_30x;
                    fxi -= 2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_2100x;
                    v_jx += fxj * prod_yz;
                    double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_21x;
                    v_kx += fxk * prod_yz;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_2001x;
                    v_lx += fxl * prod_yz;
                    double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_1_0;
                        dd += dm_jl_1_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * hrr_1100y * dd;
                    prod_xz = trr_10x * wt * dd;
                    prod_yz = hrr_1100y * wt * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_2100y;
                    fyi -= 1 * hrr_0100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                    fyj = aj2 * hrr_1200y;
                    fyj -= 1 * trr_10y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    fyk = ak2 * hrr_1110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                    fyl = al2 * hrr_1101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_2_0;
                        dd += dm_jl_1_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * hrr_0100y * dd;
                    prod_xz = trr_10x * trr_10z * dd;
                    prod_yz = hrr_0100y * trr_10z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_3_0;
                        dd += dm_jl_1_0 * dm_ik_3_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * hrr_2100y * dd;
                    prod_xz = 1 * wt * dd;
                    prod_yz = hrr_2100y * wt * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double hrr_3100y = trr_40y - yjyi * trr_30y;
                    fyi = ai2 * hrr_3100y;
                    fyi -= 2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                    fyj = aj2 * hrr_2200y;
                    fyj -= 1 * trr_20y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double hrr_2110y = trr_31y - yjyi * trr_21y;
                    fyk = ak2 * hrr_2110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_01z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    double hrr_3001y = trr_31y - ylyk * trr_30y;
                    double hrr_2101y = hrr_3001y - yjyi * hrr_2001y;
                    fyl = al2 * hrr_2101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_4_0;
                        dd += dm_jl_1_0 * dm_ik_4_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * hrr_1100y * dd;
                    prod_xz = 1 * trr_10z * dd;
                    prod_yz = hrr_1100y * trr_10z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_2100y;
                    fyi -= 1 * hrr_0100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1200y;
                    fyj -= 1 * trr_10y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_1110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_11z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_1_0 * dm_il_5_0;
                        dd += dm_jl_1_0 * dm_ik_5_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * hrr_0100y * dd;
                    prod_xz = 1 * trr_20z * dd;
                    prod_yz = hrr_0100y * trr_20z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * hrr_1100y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_30z;
                    fzi -= 2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0200y;
                    fyj -= 1 * 1;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_2100z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * hrr_0110y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * trr_21z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0101y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_2001z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_0_0;
                        dd += dm_jl_2_0 * dm_ik_0_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_20x * 1 * dd;
                    prod_xz = trr_20x * hrr_0100z * dd;
                    prod_yz = 1 * hrr_0100z * dd;
                    fxi = ai2 * trr_30x;
                    fxi -= 2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_2100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_21x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_2001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_1_0;
                        dd += dm_jl_2_0 * dm_ik_1_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * trr_10y * dd;
                    prod_xz = trr_10x * hrr_0100z * dd;
                    prod_yz = trr_10y * hrr_0100z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_2_0;
                        dd += dm_jl_2_0 * dm_ik_2_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = trr_10x * 1 * dd;
                    prod_xz = trr_10x * hrr_1100z * dd;
                    prod_yz = 1 * hrr_1100z * dd;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_2100z;
                    fzi -= 1 * hrr_0100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_1100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                    fzj = aj2 * hrr_1200z;
                    fzj -= 1 * trr_10z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_11x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    fzk = ak2 * hrr_1110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_1001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                    fzl = al2 * hrr_1101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_3_0;
                        dd += dm_jl_2_0 * dm_ik_3_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+3)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+3)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * trr_20y * dd;
                    prod_xz = 1 * hrr_0100z * dd;
                    prod_yz = trr_20y * hrr_0100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_30y;
                    fyi -= 2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_2100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_0200z;
                    fzj -= 1 * wt;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_21y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_0110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_2001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_0101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_4_0;
                        dd += dm_jl_2_0 * dm_ik_4_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+4)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+4)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * trr_10y * dd;
                    prod_xz = 1 * hrr_1100z * dd;
                    prod_yz = trr_10y * hrr_1100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * 1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * hrr_2100z;
                    fzi -= 1 * hrr_0100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_1100y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_1200z;
                    fzj -= 1 * trr_10z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_11y;
                    v_ky += fyk * prod_xz;
                    fzk = ak2 * hrr_1110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_1001y;
                    v_ly += fyl * prod_xz;
                    fzl = al2 * hrr_1101z;
                    v_lz += fzl * prod_xy;
                    dd = 0.;
                    if (do_k) {
                        dd  = dm_jk_2_0 * dm_il_5_0;
                        dd += dm_jl_2_0 * dm_ik_5_0;
                        if (jk.n_dm > 1) {
                            dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+5)*nao+l0+0];
                            dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+5)*nao+k0+0];
                        }
                        dd *= jk.k_factor;
                    }
                    if (do_j) {
                        dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * dm_lk_0_0;
                    }
                    prod_xy = 1 * 1 * dd;
                    prod_xz = 1 * hrr_2100z * dd;
                    prod_yz = 1 * hrr_2100z * dd;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double hrr_3100z = trr_40z - zjzi * trr_30z;
                    fzi = ai2 * hrr_3100z;
                    fzi -= 2 * hrr_1100z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_0100x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_0100y;
                    v_jy += fyj * prod_xz;
                    double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                    fzj = aj2 * hrr_2200z;
                    fzj -= 1 * trr_20z;
                    v_jz += fzj * prod_xy;
                    fxk = ak2 * trr_01x;
                    v_kx += fxk * prod_yz;
                    fyk = ak2 * trr_01y;
                    v_ky += fyk * prod_xz;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double hrr_2110z = trr_31z - zjzi * trr_21z;
                    fzk = ak2 * hrr_2110z;
                    v_kz += fzk * prod_xy;
                    fxl = al2 * hrr_0001x;
                    v_lx += fxl * prod_yz;
                    fyl = al2 * hrr_0001y;
                    v_ly += fyl * prod_xz;
                    double hrr_3001z = trr_31z - zlzk * trr_30z;
                    double hrr_2101z = hrr_3001z - zjzi * hrr_2001z;
                    fzl = al2 * hrr_2101z;
                    v_lz += fzl * prod_xy;
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 18 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 128 * bounds.nroots;
    double *gy = gx + 1536;
    double *gz = gy + 1536;
    int nao2 = nao * nao;
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    int _ik, _il, _jk, _jl, _lk;
    double s0, s1, s2;
    double Rpq[3];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 18*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double Ix, Iy, Iz;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gz[0] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1536;
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
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 3 * b10 * s0;
                        _gx[256] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
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
                    switch (gout_id) {
                    case 0:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[896];
                        Iy = gy[0];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[960] - 2 * gx[832]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[960] - xjxi * Ix) - 1 * gx[640]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1408] - 1 * gx[384]) * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[1408] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[64];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[832] - xjxi * Ix) - 1 * gx[512]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1280] - 1 * gx[256]) * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[1280] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[256];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[640];
                        Iy = gy[0];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[704] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[1152] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1152] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[64];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[512];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += (ak2 * gy[1024] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[768];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[832] - yjyi * Iy) - 1 * gy[512]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[640] * prod_yz;
                        v_ky += (ak2 * gy[1280] - 1 * gy[256]) * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1280] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[832];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[896] - 1 * gy[768]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[896] - yjyi * Iy) - 1 * gy[576]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += (ak2 * gy[1344] - 1 * gy[320]) * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1344] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[512];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += (ak2 * gy[1024] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[0];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += (ak2 * gz[1024] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[896] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[64];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += (ak2 * gz[1088] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[256];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[1088] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[0];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[832] - zjzi * Iz) - 1 * gz[512]) * prod_xy;
                        v_kx += ak2 * gx[640] * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += (ak2 * gz[1280] - 1 * gz[256]) * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1280] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[64];
                        Iz = gz[832];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[896] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[896] - zjzi * Iz) - 1 * gz[576]) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += (ak2 * gz[1344] - 1 * gz[320]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1344] - zlzk * Iz) * prod_xy;
                        break;
                    case 1:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[832];
                        Iy = gy[64];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[896] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1344] - 1 * gx[320]) * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[1344] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[0];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += (aj2 * (gx[832] - xjxi * Ix) - 1 * gx[512]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1280] - 1 * gx[256]) * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += ak2 * gz[640] * prod_xy;
                        v_lx += al2 * (gx[1280] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[384];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[448] - yjyi * Iy) - 1 * gy[128]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[896] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[64];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[0];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[448] - zjzi * Iz) - 1 * gz[128]) * prod_xy;
                        v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[896] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[640];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[704] - 2 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[704] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[1152] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1152] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[832];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[896] - 1 * gy[768]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[896] - yjyi * Iy) - 1 * gy[576]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += (ak2 * gy[1344] - 1 * gy[320]) * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1344] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[768];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[832] - yjyi * Iy) - 1 * gy[512]) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += (ak2 * gy[1280] - 1 * gy[256]) * prod_xz;
                        v_kz += ak2 * gz[640] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1280] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[640];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[704] - 2 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[704] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += (ak2 * gy[1152] - 1 * gy[128]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1152] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[64];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += (ak2 * gz[1024] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[0];
                        Iz = gz[640];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[704] - 2 * gz[576]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[704] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += (ak2 * gz[1152] - 1 * gz[128]) * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1152] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[384];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[448] - yjyi * Iy) - 1 * gy[128]) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += (ak2 * gz[1024] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[896] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[64];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[832] - zjzi * Iz) - 1 * gz[512]) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += ak2 * gy[576] * prod_xz;
                        v_kz += (ak2 * gz[1280] - 1 * gz[256]) * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[576] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1280] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[896];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[960] - 2 * gz[832]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[960] - zjzi * Iz) - 1 * gz[640]) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += (ak2 * gz[1408] - 1 * gz[384]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1408] - zlzk * Iz) * prod_xy;
                        break;
                    case 2:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[832];
                        Iy = gy[0];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[896] - 1 * gx[768]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[896] - xjxi * Ix) - 1 * gx[576]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1344] - 1 * gx[320]) * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[1344] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[640];
                        Iy = gy[256];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[704] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1152] - 1 * gx[128]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[1152] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[320];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[0];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[512];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += (ak2 * gy[1024] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[896] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[576];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[1088] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[768];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[832] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[832] - yjyi * Iy) - 1 * gy[512]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += (ak2 * gy[1280] - 1 * gy[256]) * prod_xz;
                        v_kz += ak2 * gz[576] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1280] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[576] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[512];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[640] * prod_yz;
                        v_ky += (ak2 * gy[1024] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[576];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += (ak2 * gy[1088] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[0];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += (ak2 * gz[1088] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[256];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[640] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[1024] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[640] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[320];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += aj2 * (gz[640] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += (ak2 * gz[1088] - 1 * gz[64]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[0];
                        Iz = gz[832];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[896] - 1 * gz[768]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[896] - zjzi * Iz) - 1 * gz[576]) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += ak2 * gy[512] * prod_xz;
                        v_kz += (ak2 * gz[1344] - 1 * gz[320]) * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[512] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1344] - zlzk * Iz) * prod_xy;
                        break;
                    case 3:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[768];
                        Iy = gy[128];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[832] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[832] - xjxi * Ix) - 1 * gx[512]) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1280] - 1 * gx[256]) * prod_yz;
                        v_ky += ak2 * gy[640] * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[1280] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[320];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[640] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1088] - 1 * gx[64]) * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[256];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[640] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[128];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[576] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += (ak2 * gx[1024] - 1 * gx[0]) * prod_yz;
                        v_ky += ak2 * gy[640] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[576];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += (ak2 * gy[1088] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[512];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += (ak2 * gy[1024] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[640] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[640] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+1);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[896];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[960] - 2 * gy[832]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[960] - yjyi * Iy) - 1 * gy[640]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += (ak2 * gy[1408] - 1 * gy[384]) * prod_xz;
                        v_kz += ak2 * gz[512] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1408] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[512] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[576];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[640] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += (ak2 * gy[1088] - 1 * gy[64]) * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+1);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+1);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+1);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[512];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[576] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[448] - zjzi * Iz) - 1 * gz[128]) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += (ak2 * gy[1024] - 1 * gy[0]) * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[896] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[128];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[640] * prod_xz;
                        v_kz += (ak2 * gz[1024] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[320];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += aj2 * (gz[576] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[576] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += (ak2 * gz[1024] - 1 * gz[0]) * prod_xy;
                        v_lx += al2 * (gx[576] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+2);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[256];
                        Iz = gz[640];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[704] - 2 * gz[576]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[704] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += (ak2 * gz[1152] - 1 * gz[128]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1152] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+2);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+2);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+2);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[128];
                        Iz = gz[768];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[832] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[832] - zjzi * Iz) - 1 * gz[512]) * prod_xy;
                        v_kx += ak2 * gx[512] * prod_yz;
                        v_ky += ak2 * gy[640] * prod_xz;
                        v_kz += (ak2 * gz[1280] - 1 * gz[256]) * prod_xy;
                        v_lx += al2 * (gx[512] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[640] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1280] - zlzk * Iz) * prod_xy;
                        break;
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip1_2200(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
                ShellQuartet *shl_quartet_idx, int ntasks, int ish0, int jsh0)
{
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int nsq_per_block = blockDim.x;
    int gout_stride = blockDim.y;
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
    int do_j = jk.j_factor != NULL;
    int do_k = jk.k_factor != NULL;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *cicj_cache = dm_cache + 36 * TILE2;
    double *rw = cicj_cache + iprim*jprim*TILE2 + sq_id;
    double *gx = rw + 128 * bounds.nroots;
    double *gy = gx + 1536;
    double *gz = gy + 1536;
    int nao2 = nao * nao;
    if (gout_id == 0) {
        gx[0] = 1.;
        gy[0] = 1.;
    }
    int _ik, _il, _jk, _jl, _lk;
    double s0, s1, s2;
    double Rpq[3];
    int thread_id = nsq_per_block * gout_id + sq_id;
    int threads = nsq_per_block * gout_stride;
    for (int n = thread_id; n < iprim*jprim*TILE2; n += threads) {
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
        cicj_cache[sh_ij+ijp*TILE2] = ci[ip] * cj[jp] * Kab;
    }
    for (int n = thread_id; n < 36*TILE2; n += threads) {
        int ij = n / TILE2;
        int sh_ij = n % TILE2;
        int i = ij % 6;
        int j = ij / 6;
        int ish = ish0 + sh_ij / TILE;
        int jsh = jsh0 + sh_ij % TILE;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        if (jk.n_dm == 1) {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i];
        } else {
            dm_cache[sh_ij+ij*TILE2] = dm[(j0+j)*nao+i0+i] + dm[(nao+j0+j)*nao+i0+i];
        }
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
        double v_ix = 0;
        double v_iy = 0;
        double v_iz = 0;
        double v_jx = 0;
        double v_jy = 0;
        double v_jz = 0;
        double v_kx = 0;
        double v_ky = 0;
        double v_kz = 0;
        double v_lx = 0;
        double v_ly = 0;
        double v_lz = 0;
        double dd;
        double prod_xy;
        double prod_xz;
        double prod_yz;
        double Ix, Iy, Iz;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double ak2 = ak * 2;
            double al2 = al * 2;
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
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double cicj = cicj_cache[sh_ij+ijp*TILE2];
                double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                double xpa = xjxi * aj_aij;
                double ypa = yjyi * aj_aij;
                double zpa = zjzi * aj_aij;
                double xij = ri[0] + xpa; // (ai*xi+aj*xj)/aij
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
                __syncthreads();
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[(irys*2+1)*nsq_per_block] *= fac;
                    }
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    double *rw1 = rw + 6*nsq_per_block;
                    rys_roots(3, theta_rr, rw1, nsq_per_block, gout_id, gout_stride);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw, nsq_per_block, gout_id, gout_stride);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac) * fac;
                    for (int irys = gout_id; irys < 3; irys += gout_stride) {
                        rw[ irys*2   *nsq_per_block] *= theta_fac;
                        rw[(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                        rw1[(irys*2+1)*nsq_per_block] *= fac;
                    }
                }
                for (int irys = 0; irys < bounds.nroots; ++irys) {
                    __syncthreads();
                    double rt = rw[irys*128];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    for (int n = gout_id; n < 3; n += 4) {
                        if (n == 2) {
                            gz[0] = rw[irys*128+64];
                        }
                        double *_gx = gx + n * 1536;
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
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 3 * b10 * s0;
                        _gx[256] = s2;
                        s0 = s1;
                        s1 = s2;
                        s2 = c0x * s1 + 4 * b10 * s0;
                        _gx[320] = s2;
                        double xlxk = rl[n] - rk[n];
                        double Rqc = xlxk * al_akl;
                        double cpx = Rqc + rt_akl * Rpq[n];
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
                    switch (gout_id) {
                    case 0:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[0*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[640];
                        Iy = gy[0];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[704] - 2 * gx[576]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[704] - xjxi * Ix) - 2 * gx[384]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1408] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1408] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[4*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[64];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[576] - xjxi * Ix) - 2 * gx[256]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1280] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[1280] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[8*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[256];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[12*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[0];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[1152] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[1152] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[16*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[64];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+3)*nao+(l0+0);
                            _jk = (j0+3)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[20*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[20*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[512];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[576] - yjyi * Iy) - 2 * gy[256]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[1280] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1280] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+4)*nao+(l0+0);
                            _jk = (j0+4)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[24*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[24*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[256];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[896] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+4)*nao+(l0+0);
                            _jk = (j0+4)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[28*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[28*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[320];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+5)*nao+(l0+0);
                            _jk = (j0+5)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[32*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[32*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[0];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[640] - zjzi * Iz) - 2 * gz[320]) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[1344] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1344] - zlzk * Iz) * prod_xy;
                        break;
                    case 1:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[1*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[64];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[640] - xjxi * Ix) - 2 * gx[320]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1344] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1344] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[5*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[0];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += (aj2 * (gx[576] - xjxi * Ix) - 2 * gx[256]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1280] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[1280] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[896] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[9*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[384];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += (aj2 * (gy[448] - yjyi * Iy) - 1 * gy[128]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[1152] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1152] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[13*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[64];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[17*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[0];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[448] - zjzi * Iz) - 1 * gz[128]) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[1152] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1152] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+3)*nao+(l0+0);
                            _jk = (j0+3)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[21*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[21*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[640];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[704] - 2 * gy[576]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[704] - yjyi * Iy) - 2 * gy[384]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[1408] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1408] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+4)*nao+(l0+0);
                            _jk = (j0+4)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[25*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[25*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[320];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+4)*nao+(l0+0);
                            _jk = (j0+4)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[29*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[29*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[256];
                        Iz = gz[384];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[448] - 2 * gz[320]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += (aj2 * (gz[448] - zjzi * Iz) - 1 * gz[128]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[1152] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1152] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+5)*nao+(l0+0);
                            _jk = (j0+5)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[33*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[33*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[128];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[576] - zjzi * Iz) - 2 * gz[256]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += ak2 * gz[1280] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[896] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1280] - zlzk * Iz) * prod_xy;
                        break;
                    case 2:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[2*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[576];
                        Iy = gy[0];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[640] - 1 * gx[512]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[640] - xjxi * Ix) - 2 * gx[320]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1344] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[1344] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[6*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[384];
                        Iy = gy[256];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[448] - 2 * gx[320]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[448] - xjxi * Ix) - 1 * gx[128]) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1152] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1152] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[10*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[320];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[14*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[0];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+3)*nao+(l0+0);
                            _jk = (j0+3)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[18*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[18*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[512];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[576] - yjyi * Iy) - 2 * gy[256]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += ak2 * gy[1280] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[896] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1280] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+3)*nao+(l0+0);
                            _jk = (j0+3)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[22*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[22*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[576];
                        Iz = gz[64];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += (ai2 * gz[128] - 1 * gz[0]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[640] - yjyi * Iy) - 2 * gy[320]) * prod_xz;
                        v_jz += aj2 * (gz[128] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[1344] * prod_xz;
                        v_kz += ak2 * gz[832] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1344] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[832] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+4)*nao+(l0+0);
                            _jk = (j0+4)*nao+(k0+0);
                            _il = (i0+2)*nao+(l0+0);
                            _ik = (i0+2)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[26*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[26*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[256];
                        Iz = gz[320];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[384] - 1 * gz[256]) * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += (aj2 * (gz[384] - zjzi * Iz) - 1 * gz[64]) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[1088] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1088] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+5)*nao+(l0+0);
                            _jk = (j0+5)*nao+(k0+0);
                            _il = (i0+0)*nao+(l0+0);
                            _ik = (i0+0)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[30*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[30*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[128];
                        Iy = gy[0];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[192] - 2 * gx[64]) * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[192] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[576] - zjzi * Iz) - 2 * gz[256]) * prod_xy;
                        v_kx += ak2 * gx[896] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[1280] * prod_xy;
                        v_lx += al2 * (gx[896] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1280] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+5)*nao+(l0+0);
                            _jk = (j0+5)*nao+(k0+0);
                            _il = (i0+4)*nao+(l0+0);
                            _ik = (i0+4)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[34*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[34*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[64];
                        Iz = gz[576];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += (ai2 * gz[640] - 1 * gz[512]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[640] - zjzi * Iz) - 2 * gz[320]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[1344] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1344] - zlzk * Iz) * prod_xy;
                        break;
                    case 3:
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+0)*nao+(l0+0);
                            _jk = (j0+0)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[3*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[512];
                        Iy = gy[128];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[576] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[576] - xjxi * Ix) - 2 * gx[256]) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1280] * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1280] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[896] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[7*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[320];
                        Iy = gy[320];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[384] - 1 * gx[256]) * prod_yz;
                        v_iy += (ai2 * gy[384] - 1 * gy[256]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += (aj2 * (gx[384] - xjxi * Ix) - 1 * gx[64]) * prod_yz;
                        v_jy += (aj2 * (gy[384] - yjyi * Iy) - 1 * gy[64]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1088] * prod_yz;
                        v_ky += ak2 * gy[1088] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[1088] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1088] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+1)*nao+(l0+0);
                            _jk = (j0+1)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[11*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[256];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += ai2 * gy[320] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += (aj2 * (gy[320] - yjyi * Iy) - 1 * gy[0]) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[1024] * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1024] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[896] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+2)*nao+(l0+0);
                            _jk = (j0+2)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[15*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[256];
                        Iy = gy[128];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[320] * prod_yz;
                        v_iy += (ai2 * gy[192] - 2 * gy[64]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += (aj2 * (gx[320] - xjxi * Ix) - 1 * gx[0]) * prod_yz;
                        v_jy += aj2 * (gy[192] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[1024] * prod_yz;
                        v_ky += ak2 * gy[896] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[1024] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[896] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+3)*nao+(l0+0);
                            _jk = (j0+3)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[19*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[19*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[576];
                        Iz = gz[0];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[640] - 1 * gy[512]) * prod_xz;
                        v_iz += ai2 * gz[64] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[640] - yjyi * Iy) - 2 * gy[320]) * prod_xz;
                        v_jz += aj2 * (gz[64] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[1344] * prod_xz;
                        v_kz += ak2 * gz[768] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1344] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[768] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+3)*nao+(l0+0);
                            _jk = (j0+3)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[23*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[23*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[512];
                        Iz = gz[128];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[576] * prod_xz;
                        v_iz += (ai2 * gz[192] - 2 * gz[64]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[576] - yjyi * Iy) - 2 * gy[256]) * prod_xz;
                        v_jz += aj2 * (gz[192] - zjzi * Iz) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[1280] * prod_xz;
                        v_kz += ak2 * gz[896] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1280] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[896] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+4)*nao+(l0+0);
                            _jk = (j0+4)*nao+(k0+0);
                            _il = (i0+3)*nao+(l0+0);
                            _ik = (i0+3)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[27*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[27*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[384];
                        Iz = gz[256];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += (ai2 * gy[448] - 2 * gy[320]) * prod_xz;
                        v_iz += ai2 * gz[320] * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += (aj2 * (gy[448] - yjyi * Iy) - 1 * gy[128]) * prod_xz;
                        v_jz += (aj2 * (gz[320] - zjzi * Iz) - 1 * gz[0]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[1152] * prod_xz;
                        v_kz += ak2 * gz[1024] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[1152] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1024] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+5)*nao+(l0+0);
                            _jk = (j0+5)*nao+(k0+0);
                            _il = (i0+1)*nao+(l0+0);
                            _ik = (i0+1)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[31*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[31*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[64];
                        Iy = gy[64];
                        Iz = gz[512];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += (ai2 * gx[128] - 1 * gx[0]) * prod_yz;
                        v_iy += (ai2 * gy[128] - 1 * gy[0]) * prod_xz;
                        v_iz += ai2 * gz[576] * prod_xy;
                        v_jx += aj2 * (gx[128] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[128] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[576] - zjzi * Iz) - 2 * gz[256]) * prod_xy;
                        v_kx += ak2 * gx[832] * prod_yz;
                        v_ky += ak2 * gy[832] * prod_xz;
                        v_kz += ak2 * gz[1280] * prod_xy;
                        v_lx += al2 * (gx[832] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[832] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1280] - zlzk * Iz) * prod_xy;
                        dd = 0.;
                        if (do_k) {
                            _jl = (j0+5)*nao+(l0+0);
                            _jk = (j0+5)*nao+(k0+0);
                            _il = (i0+5)*nao+(l0+0);
                            _ik = (i0+5)*nao+(k0+0);
                            dd = dm[_jk] * dm[_il] + dm[_jl] * dm[_ik];
                            if (jk.n_dm > 1) {
                                dd += dm[nao2+_jk] * dm[nao2+_il] + dm[nao2+_jl] * dm[nao2+_ik];
                            }
                            dd *= jk.k_factor;
                        }
                        if (do_j) {
                            _lk = (l0+0)*nao+(k0+0);
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm_cache[35*TILE2+sh_ij] * dm[_lk];
                            } else {
                                dd += jk.j_factor * dm_cache[35*TILE2+sh_ij] * (dm[_lk] + dm[nao2+_lk]);
                            }
                        }
                        Ix = gx[0];
                        Iy = gy[0];
                        Iz = gz[640];
                        prod_xy = Ix * Iy * dd;
                        prod_xz = Ix * Iz * dd;
                        prod_yz = Iy * Iz * dd;
                        v_ix += ai2 * gx[64] * prod_yz;
                        v_iy += ai2 * gy[64] * prod_xz;
                        v_iz += (ai2 * gz[704] - 2 * gz[576]) * prod_xy;
                        v_jx += aj2 * (gx[64] - xjxi * Ix) * prod_yz;
                        v_jy += aj2 * (gy[64] - yjyi * Iy) * prod_xz;
                        v_jz += (aj2 * (gz[704] - zjzi * Iz) - 2 * gz[384]) * prod_xy;
                        v_kx += ak2 * gx[768] * prod_yz;
                        v_ky += ak2 * gy[768] * prod_xz;
                        v_kz += ak2 * gz[1408] * prod_xy;
                        v_lx += al2 * (gx[768] - xlxk * Ix) * prod_yz;
                        v_ly += al2 * (gy[768] - ylyk * Iy) * prod_xz;
                        v_lz += al2 * (gz[1408] - zlzk * Iz) * prod_xy;
                        break;
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        double *ejk = jk.ejk;
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
}
__global__
void rys_ejk_ip1_2200(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        int nbas = envs.nbas;
        double *env = envs.env;
        double omega = env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                        batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip1_2200(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_ejk_ip1_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                         ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    int nroots = bounds->nroots;
    int g_size = (li + 2) * (lj + 1) * (lk + 2) * (ll + 1);
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int nfij = bounds->nfij;
    int buflen = nfij*TILE2 + ij_prims*TILE2;
    int nsq_per_block = 256;
    int gout_stride = 1;

    switch (ijkl) {
    case 156:
        nsq_per_block = 32;
        gout_stride = 8;
        break;
    case 256:
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 260:
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 280:
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 300:
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    }

    dim3 threads(nsq_per_block, gout_stride);
    buflen += nroots*2 * nsq_per_block;
    switch (ijkl) {
    case 0:
        rys_ejk_ip1_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125:
        rys_ejk_ip1_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 130:
        rys_ejk_ip1_1010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 131:
        rys_ejk_ip1_1011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 150:
        rys_ejk_ip1_1100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 155:
        rys_ejk_ip1_1110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 156:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_ejk_ip1_1111, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_ejk_ip1_1111<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 250:
        rys_ejk_ip1_2000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 255:
        rys_ejk_ip1_2010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 256:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_ejk_ip1_2011, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_ejk_ip1_2011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 260:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_ejk_ip1_2020, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_ejk_ip1_2020<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 275:
        rys_ejk_ip1_2100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 280:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_ejk_ip1_2110, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_ejk_ip1_2110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 300:
        buflen += g_size * 3 * nsq_per_block;
        cudaFuncSetAttribute(rys_ejk_ip1_2200, cudaFuncAttributeMaxDynamicSharedMemorySize, buflen*sizeof(double));
        rys_ejk_ip1_2200<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
