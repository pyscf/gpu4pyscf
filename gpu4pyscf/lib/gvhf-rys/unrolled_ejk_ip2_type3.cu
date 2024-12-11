#include "vhf.cuh"
#include "rys_roots_unrolled.cu"
#include "create_tasks_ip1.cu"
int rys_ejk_ip2_type3_unrolled_lmax = 1;
int rys_ejk_ip2_type3_unrolled_max_order = 3;


__device__ static
void _rys_ejk_ip2_type3_0000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    double *dm = jk.dm;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4;
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
        double theta_ij = ai * aj_aij;
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
        double dd;
        double Ix, Iy, Iz, prod_xy, prod_xz, prod_yz;
        double gix, giy, giz;
        double gjx, gjy, gjz;
        double gkx, gky, gkz;
        double glx, gly, glz;
        double gikx, giky, gikz;
        double gjkx, gjky, gjkz;
        double gilx, gily, gilz;
        double gjlx, gjly, gjlz;
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        
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
            double theta_kl = ak * al_akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
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
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw);
                    fac *= sqrt(theta_fac);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    }
                } else {
                    rys_roots(2, theta_rr, rw+4*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw);
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        gix = ai2 * trr_10x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double trr_01x = cpx * fac;
                        gkx = ak2 * trr_01x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gky = ak2 * trr_01y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        gikx = ai2 * trr_11x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        giky = ai2 * trr_11y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gikz = ai2 * trr_11z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gjx = aj2 * hrr_0100x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gjkx = aj2 * hrr_0110x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_10z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        glx = al2 * hrr_0001x;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gly = al2 * hrr_0001y;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        gilx = ai2 * hrr_1001x;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gily = ai2 * hrr_1001y;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gilz = ai2 * hrr_1001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gjlx = aj2 * hrr_0101x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        }
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
        int natm = envs.natm;
        double *ejk = jk.ejk;
        atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
        atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
        atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
        atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
        atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
        atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
        atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
        atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
        atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
        atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
        atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
        atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
        atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
    }
}
__global__
void rys_ejk_ip2_type3_0000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type3_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type3_1000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    double *dm = jk.dm;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4;
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
        double theta_ij = ai * aj_aij;
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
        double dd;
        double Ix, Iy, Iz, prod_xy, prod_xz, prod_yz;
        double gix, giy, giz;
        double gjx, gjy, gjz;
        double gkx, gky, gkz;
        double glx, gly, glz;
        double gikx, giky, gikz;
        double gjkx, gjky, gjkz;
        double gilx, gily, gilz;
        double gjlx, gjly, gjlz;
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        
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
            double theta_kl = ak * al_akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
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
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(2, theta_rr, rw);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw);
                    fac *= sqrt(theta_fac);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    }
                } else {
                    rys_roots(2, theta_rr, rw+4*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(2, theta_fac*theta_rr, rw);
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 2; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        gix = ai2 * trr_20x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        gkx = ak2 * trr_11x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gky = ak2 * trr_01y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        gikx = ai2 * trr_21x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        giky = ai2 * trr_11y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gikz = ai2 * trr_11z;
                        double trr_01x = cpx * fac;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gjx = aj2 * hrr_1100x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gjkx = aj2 * hrr_1110x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        glx = al2 * hrr_1001x;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gly = al2 * hrr_0001y;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        gilx = ai2 * hrr_2001x;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gily = ai2 * hrr_1001y;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gilz = ai2 * hrr_1001z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        gjlx = aj2 * hrr_1101x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gjx = aj2 * hrr_0100x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gjkx = aj2 * hrr_0110x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gjlx = aj2 * hrr_0101x;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_11y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1001y;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        }
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
        int natm = envs.natm;
        double *ejk = jk.ejk;
        atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
        atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
        atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
        atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
        atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
        atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
        atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
        atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
        atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
        atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
        atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
        atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
        atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
    }
}
__global__
void rys_ejk_ip2_type3_1000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type3_1000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type3_1010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    double *dm = jk.dm;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4;
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
        double theta_ij = ai * aj_aij;
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
        double dd;
        double Ix, Iy, Iz, prod_xy, prod_xz, prod_yz;
        double gix, giy, giz;
        double gjx, gjy, gjz;
        double gkx, gky, gkz;
        double glx, gly, glz;
        double gikx, giky, gikz;
        double gjkx, gjky, gjkz;
        double gilx, gily, gilz;
        double gjlx, gjly, gjlz;
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        
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
            double theta_kl = ak * al_akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
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
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    fac *= sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    }
                } else {
                    rys_roots(3, theta_rr, rw+6*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = 1 * Iz;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        gix = ai2 * trr_21x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double trr_01x = cpx * fac;
                        gix -= 1 * trr_01x;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        gkx = ak2 * trr_12x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gky = ak2 * trr_01y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * trr_10x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        gikx = ai2 * trr_22x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        giky = ai2 * trr_11y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gikz = ai2 * trr_11z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        gikx -= 1 * trr_02x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gjx = aj2 * hrr_1110x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * trr_10x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_1120x = trr_22x - xjxi * trr_12x;
                        gjkx = aj2 * hrr_1120x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gjkx -= 1 * (aj2 * hrr_1100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_11x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gjx = aj2 * hrr_0110x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0120x = trr_12x - xjxi * trr_02x;
                        gjkx = aj2 * hrr_0120x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * trr_11y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = trr_01y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        double hrr_0120y = trr_12y - yjyi * trr_02y;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_11y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * trr_01y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * trr_10y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        giky = ai2 * trr_22y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_02y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * trr_10y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        double hrr_1120y = trr_22y - yjyi * trr_12y;
                        gjky = aj2 * hrr_1120y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_01y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = trr_01z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * trr_11y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        gikz = ai2 * trr_12z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0110z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0120z = trr_12z - zjzi * trr_02z;
                        gjkz = aj2 * hrr_0120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_12z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0110z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = trr_11z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_11y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        gikz = ai2 * trr_22z;
                        gikz -= 1 * trr_02z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1110z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1120z = trr_22z - zjzi * trr_12z;
                        gjkz = aj2 * hrr_1120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        }
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = 1 * Iz;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        gix = ai2 * trr_21x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double trr_01x = cpx * fac;
                        gix -= 1 * trr_01x;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        glx = al2 * hrr_1011x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gly = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        gilx = ai2 * hrr_2011x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gily = ai2 * hrr_1001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gilz = ai2 * hrr_1001z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        gilx -= 1 * hrr_0011x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gjx = aj2 * hrr_1110x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        gjlx = aj2 * hrr_1111x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_11x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gjx = aj2 * hrr_0110x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        gjlx = aj2 * hrr_0111x;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_1001y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = trr_01y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        glx = al2 * hrr_1001x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        gilx = ai2 * hrr_2001x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_1001z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gjx = aj2 * hrr_1100x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        gjlx = aj2 * hrr_1101x;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_11y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * trr_01y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        gily = ai2 * hrr_2011y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0011y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gjx = aj2 * hrr_0100x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gjlx = aj2 * hrr_0101x;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        gjly = aj2 * hrr_1111y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_01y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = trr_01z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        glz = al2 * hrr_0011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1001y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gilz = ai2 * hrr_1011z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjz = aj2 * hrr_0110z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        gjlz = aj2 * hrr_0111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1011z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0110z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = trr_11z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1001y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        gilz = ai2 * hrr_2011z;
                        gilz -= 1 * hrr_0011z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjz = aj2 * hrr_1110z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        gjlz = aj2 * hrr_1111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        }
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
        int natm = envs.natm;
        double *ejk = jk.ejk;
        atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
        atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
        atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
        atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
        atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
        atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
        atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
        atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
        atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
        atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
        atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
        atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
        atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
    }
}
__global__
void rys_ejk_ip2_type3_1010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type3_1010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type3_1011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    double *dm = jk.dm;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4;
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
        double theta_ij = ai * aj_aij;
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
        double dd;
        double Ix, Iy, Iz, prod_xy, prod_xz, prod_yz;
        double gix, giy, giz;
        double gjx, gjy, gjz;
        double gkx, gky, gkz;
        double glx, gly, glz;
        double gikx, giky, gikz;
        double gjkx, gjky, gjkz;
        double gilx, gily, gilz;
        double gjlx, gjly, gjlz;
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        
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
            double theta_kl = ak * al_akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
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
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    fac *= sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    }
                } else {
                    rys_roots(3, theta_rr, rw+6*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_01x = cpx * fac;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1011x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1011x * Iy;
                        prod_xz = hrr_1011x * Iz;
                        prod_yz = 1 * Iz;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        gix = ai2 * hrr_2011x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        gix -= 1 * hrr_0011x;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double hrr_1021x = trr_13x - xlxk * trr_12x;
                        gkx = ak2 * hrr_1021x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gky = ak2 * trr_01y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gkz = ak2 * trr_01z;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        gkx -= 1 * hrr_1001x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                        double hrr_2021x = trr_23x - xlxk * trr_22x;
                        gikx = ai2 * hrr_2021x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        giky = ai2 * trr_11y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gikz = ai2 * trr_11z;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double hrr_0021x = trr_03x - xlxk * trr_02x;
                        gikx -= 1 * hrr_0021x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        gikx -= 1 * (ai2 * hrr_2001x - 1 * hrr_0001x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        gjx = aj2 * hrr_1111x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * hrr_1021x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * hrr_1001x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_1121x = hrr_2021x - xjxi * hrr_1021x;
                        gjkx = aj2 * hrr_1121x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        gjkx -= 1 * (aj2 * hrr_1101x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0011x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0011x * Iy;
                        prod_xz = hrr_0011x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1011x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        gkx = ak2 * hrr_0021x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * hrr_0001x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1021x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * hrr_1001x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        gjx = aj2 * hrr_0111x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * hrr_0021x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * hrr_0001x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0121x = hrr_1021x - xjxi * hrr_0021x;
                        gjkx = aj2 * hrr_0121x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gjkx -= 1 * (aj2 * hrr_0101x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0011x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0011x * Iy;
                        prod_xz = hrr_0011x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1011x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * hrr_0021x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * hrr_0001x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1021x;
                        giky = ai2 * trr_11y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * hrr_1001x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0111x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * hrr_0021x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * hrr_0001x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0121x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0101x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1001x * dd;
                        Iy = trr_01y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1001x * Iy;
                        prod_xz = hrr_1001x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * hrr_2001x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * hrr_0001x;
                        gkx = ak2 * hrr_1011x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_2011x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * hrr_0011x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1101x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * hrr_1011x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1111x;
                        double hrr_0120y = trr_12y - yjyi * trr_02y;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = trr_11y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * trr_01y;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * trr_10y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1011x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        giky = ai2 * trr_22y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_02y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * trr_10y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0111x;
                        double hrr_1120y = trr_22y - yjyi * trr_12y;
                        gjky = aj2 * hrr_1120y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = trr_01y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1011x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0111x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1001x * dd;
                        Iy = 1 * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_1001x * Iy;
                        prod_xz = hrr_1001x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_2001x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * hrr_0001x;
                        gkx = ak2 * hrr_1011x;
                        gky = ak2 * trr_01y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_2011x;
                        giky = ai2 * trr_11y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        gikz = ai2 * trr_12z;
                        gikx -= 1 * hrr_0011x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1101x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0110z;
                        gkx = ak2 * hrr_1011x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1111x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0120z = trr_12z - zjzi * trr_02z;
                        gjkz = aj2 * hrr_0120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = trr_10y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * 1;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1011x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_12z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0110z;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0111x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = 1 * dd;
                        Iz = trr_11z * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1011x;
                        giky = ai2 * trr_11y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        gikz = ai2 * trr_22z;
                        gikz -= 1 * trr_02z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1110z;
                        gkx = ak2 * hrr_0011x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0111x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1120z = trr_22z - zjzi * trr_12z;
                        gjkz = aj2 * hrr_1120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+1)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+1)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = hrr_0001y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_21x;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * trr_01x;
                        gkx = ak2 * trr_12x;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * trr_10x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_22x;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        giky = ai2 * hrr_1011y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * trr_02x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gjx = aj2 * hrr_1110x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * trr_10x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_1120x = trr_22x - xjxi * trr_12x;
                        gjkx = aj2 * hrr_1120x;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        gjky = aj2 * hrr_0111y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gjkx -= 1 * (aj2 * hrr_1100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+1)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+1)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_1001y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_1001y * Iz;
                        gix = ai2 * trr_11x;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        giy = ai2 * hrr_2001y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0001y;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_1011y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        giky = ai2 * hrr_2011y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * hrr_0011y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gjx = aj2 * hrr_0110x;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gjy = aj2 * hrr_1101y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_1011y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0120x = trr_12x - xjxi * trr_02x;
                        gjkx = aj2 * hrr_0120x;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        gjky = aj2 * hrr_1111y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+1)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+1)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_0001y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * hrr_1011y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_0111y;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+1)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+1)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0011y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0011y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1011y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double hrr_0021y = trr_03y - ylyk * trr_02y;
                        gky = ak2 * hrr_0021y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_0001y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double hrr_1021y = trr_13y - ylyk * trr_12y;
                        giky = ai2 * hrr_1021y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * hrr_1001y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0111y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0021y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_0001y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        double hrr_0121y = hrr_1021y - yjyi * hrr_0021y;
                        gjky = aj2 * hrr_0121y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0101y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+1)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+1)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1011y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1011y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2011y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0011y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1021y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_1001y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                        double hrr_2021y = trr_23y - ylyk * trr_22y;
                        giky = ai2 * hrr_2021y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * hrr_0021y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * hrr_2001y - 1 * hrr_0001y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1111y;
                        gjz = aj2 * hrr_0100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1021y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_1001y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        double hrr_1121y = hrr_2021y - yjyi * hrr_1021y;
                        gjky = aj2 * hrr_1121y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1101y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+1)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+1)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0011y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0011y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1011y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0021y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * hrr_0001y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_1021y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * hrr_1001y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0111y;
                        gjz = aj2 * hrr_1100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0021y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * hrr_0001y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0121y;
                        gjkz = aj2 * hrr_1110z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0101y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+1)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+1)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0001y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * hrr_1011y;
                        gikz = ai2 * trr_12z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_0110z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0111y;
                        gjkz = aj2 * hrr_0120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+1)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+1)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1001y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1001y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2001y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * hrr_0001y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1011y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_2011y;
                        gikz = ai2 * trr_12z;
                        giky -= 1 * hrr_0011y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1101y;
                        gjz = aj2 * hrr_0110z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1011y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1111y;
                        gjkz = aj2 * hrr_0120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+1)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+1)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0001y * dd;
                        Iz = trr_11z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_1011y;
                        gikz = ai2 * trr_22z;
                        gikz -= 1 * trr_02z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_1110z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0011y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0111y;
                        gjkz = aj2 * hrr_1120z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+2)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+2)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * trr_10y;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        giz = ai2 * hrr_1001z;
                        gix -= 1 * trr_01x;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * trr_01y;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gkz = ak2 * hrr_0011z;
                        gkx -= 1 * trr_10x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_22x;
                        giky = ai2 * trr_11y;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gikz = ai2 * hrr_1011z;
                        gikx -= 1 * trr_02x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjz = aj2 * hrr_0101z;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0011z;
                        gkx -= 1 * trr_10x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1120x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        gjkz = aj2 * hrr_0111z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_1100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+2)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+2)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1001z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0011z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * hrr_1011z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0101z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0011z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0111z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+2)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+2)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1001z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_10y;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        giz = ai2 * hrr_2001z;
                        giz -= 1 * hrr_0001z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1011z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * trr_11y;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        gikz = ai2 * hrr_2011z;
                        gikz -= 1 * hrr_0011z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gjz = aj2 * hrr_1101z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1011z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        gjkz = aj2 * hrr_1111z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+2)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+2)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_1001z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_0011z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * hrr_1011z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0101z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_0011z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_0111z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+2)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+2)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_11y * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * hrr_1001z;
                        giy -= 1 * trr_01y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * hrr_0011z;
                        gky -= 1 * trr_10y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_22y;
                        gikz = ai2 * hrr_1011z;
                        giky -= 1 * trr_02y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0101z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * hrr_0011z;
                        gky -= 1 * trr_10y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1120y;
                        gjkz = aj2 * hrr_0111z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+2)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+2)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_1001z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_2001z;
                        giz -= 1 * hrr_0001z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_1011z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * hrr_2011z;
                        gikz -= 1 * hrr_0011z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1101z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_1011z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_1111z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+2)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+2)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0011z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1011z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double hrr_0021z = trr_03z - zlzk * trr_02z;
                        gkz = ak2 * hrr_0021z;
                        gkz -= 1 * hrr_0001z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * trr_11y;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double hrr_1021z = trr_13z - zlzk * trr_12z;
                        gikz = ai2 * hrr_1021z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * hrr_1001z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0111z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0021z;
                        gkz -= 1 * hrr_0001z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0121z = hrr_1021z - zjzi * hrr_0021z;
                        gjkz = aj2 * hrr_0121z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0101z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+2)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+2)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0011z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1011z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0021z;
                        gkz -= 1 * hrr_0001z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * hrr_1021z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * hrr_1001z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0111z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0021z;
                        gkz -= 1 * hrr_0001z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0121z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0101z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+2)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+2)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1011z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_2011z;
                        giz -= 1 * hrr_0011z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1021z;
                        gkz -= 1 * hrr_1001z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_11y;
                        double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                        double hrr_2021z = trr_23z - zlzk * trr_22z;
                        gikz = ai2 * hrr_2021z;
                        gikz -= 1 * hrr_0021z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * hrr_2001z - 1 * hrr_0001z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1111z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1021z;
                        gkz -= 1 * hrr_1001z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1121z = hrr_2021z - zjzi * hrr_1021z;
                        gjkz = aj2 * hrr_1121z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1101z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        }
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_01x = cpx * fac;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1011x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1011x * Iy;
                        prod_xz = hrr_1011x * Iz;
                        prod_yz = 1 * Iz;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        gix = ai2 * hrr_2011x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        gix -= 1 * hrr_0011x;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double hrr_1021x = trr_13x - xlxk * trr_12x;
                        double hrr_1012x = hrr_1021x - xlxk * hrr_1011x;
                        glx = al2 * hrr_1012x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gly = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * trr_11x;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                        double hrr_2021x = trr_23x - xlxk * trr_22x;
                        double hrr_2012x = hrr_2021x - xlxk * hrr_2011x;
                        gilx = ai2 * hrr_2012x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gily = ai2 * hrr_1001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gilz = ai2 * hrr_1001z;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double hrr_0021x = trr_03x - xlxk * trr_02x;
                        double hrr_0012x = hrr_0021x - xlxk * hrr_0011x;
                        gilx -= 1 * hrr_0012x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_21x - 1 * trr_01x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        gjx = aj2 * hrr_1111x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1012x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * trr_11x;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1112x = hrr_2012x - xjxi * hrr_1012x;
                        gjlx = aj2 * hrr_1112x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gjlx -= 1 * (aj2 * hrr_1110x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0011x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0011x * Iy;
                        prod_xz = hrr_0011x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1011x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0012x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * trr_01x;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1012x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_11x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        gjx = aj2 * hrr_0111x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0012x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * trr_01x;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0112x = hrr_1012x - xjxi * hrr_0012x;
                        gjlx = aj2 * hrr_0112x;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gjlx -= 1 * (aj2 * hrr_0110x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0011x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0011x * Iy;
                        prod_xz = hrr_0011x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1011x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0012x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        glx -= 1 * trr_01x;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1012x;
                        gily = ai2 * hrr_1001y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_11x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0111x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0012x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        glx -= 1 * trr_01x;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0112x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlx -= 1 * (aj2 * hrr_0110x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1001x * dd;
                        Iy = trr_01y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1001x * Iy;
                        prod_xz = hrr_1001x * Iz;
                        prod_yz = trr_01y * Iz;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        gix = ai2 * hrr_2001x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_10z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        gix -= 1 * hrr_0001x;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        glx = al2 * hrr_1002x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * trr_10x;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double hrr_2002x = hrr_2011x - xlxk * hrr_2001x;
                        gilx = ai2 * hrr_2002x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_1001z;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        gilx -= 1 * hrr_0002x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        gjx = aj2 * hrr_1101x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1002x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * trr_10x;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1102x = hrr_2002x - xjxi * hrr_1002x;
                        gjlx = aj2 * hrr_1102x;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gjlx -= 1 * (aj2 * hrr_1100x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = trr_11y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * trr_01y;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * fac;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1002x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        gily = ai2 * hrr_2011y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0011y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_10x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gjx = aj2 * hrr_0101x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0001z;
                        glx -= 1 * fac;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0102x = hrr_1002x - xjxi * hrr_0002x;
                        gjlx = aj2 * hrr_0102x;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        gjly = aj2 * hrr_1111y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gjlx -= 1 * (aj2 * hrr_0100x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = trr_01y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1001z;
                        glx -= 1 * fac;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1002x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_10x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1001z;
                        glx -= 1 * fac;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0102x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlx -= 1 * (aj2 * hrr_0100x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1001x * dd;
                        Iy = 1 * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_1001x * Iy;
                        prod_xz = hrr_1001x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_2001x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * hrr_0001x;
                        glx = al2 * hrr_1002x;
                        gly = al2 * hrr_0001y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        glz = al2 * hrr_0011z;
                        glx -= 1 * trr_10x;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2002x;
                        gily = ai2 * hrr_1001y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gilz = ai2 * hrr_1011z;
                        gilx -= 1 * hrr_0002x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1101x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjz = aj2 * hrr_0110z;
                        glx = al2 * hrr_1002x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0011z;
                        glx -= 1 * trr_10x;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1102x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        gjlz = aj2 * hrr_0111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlx -= 1 * (aj2 * hrr_1100x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = trr_10y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0011z;
                        glx -= 1 * fac;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1002x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1011z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_10x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0110z;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0011z;
                        glx -= 1 * fac;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0102x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlx -= 1 * (aj2 * hrr_0100x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0001x * dd;
                        Iy = 1 * dd;
                        Iz = trr_11z * dd;
                        prod_xy = hrr_0001x * Iy;
                        prod_xz = hrr_0001x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1001x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1011z;
                        glx -= 1 * fac;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1002x;
                        gily = ai2 * hrr_1001y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        gilz = ai2 * hrr_2011z;
                        gilz -= 1 * hrr_0011z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilx -= 1 * (ai2 * trr_10x);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0101x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjz = aj2 * hrr_1110z;
                        glx = al2 * hrr_0002x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1011z;
                        glx -= 1 * fac;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0102x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        gjlz = aj2 * hrr_1111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlx -= 1 * (aj2 * hrr_0100x);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+1)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+1)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = hrr_0001y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * trr_01x;
                        glx = al2 * hrr_1011x;
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * 1;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2011x;
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        gily = ai2 * hrr_1002y;
                        gilz = ai2 * hrr_1001z;
                        gilx -= 1 * hrr_0011x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_10y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * 1;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1111x;
                        double hrr_0102y = hrr_1002y - yjyi * hrr_0002y;
                        gjly = aj2 * hrr_0102y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_0100y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+1)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+1)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_1001y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_1001y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * hrr_2001y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0001y;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1002y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * trr_10y;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        double hrr_2002y = hrr_2011y - ylyk * hrr_2001y;
                        gily = ai2 * hrr_2002y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0002y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_1101y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1002y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * trr_10y;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        double hrr_1102y = hrr_2002y - yjyi * hrr_1002y;
                        gjly = aj2 * hrr_1102y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_1100y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+1)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+1)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_0001y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_1001z;
                        gly -= 1 * 1;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_1002y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_10y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_1001z;
                        gly -= 1 * 1;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_0102y;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_0100y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+1)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+1)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0011y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0011y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1011y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double hrr_0021y = trr_03y - ylyk * trr_02y;
                        double hrr_0012y = hrr_0021y - ylyk * hrr_0011y;
                        gly = al2 * hrr_0012y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * trr_01y;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double hrr_1021y = trr_13y - ylyk * trr_12y;
                        double hrr_1012y = hrr_1021y - ylyk * hrr_1011y;
                        gily = ai2 * hrr_1012y;
                        gilz = ai2 * hrr_1001z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_11y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0111y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0012y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * trr_01y;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        double hrr_0112y = hrr_1012y - yjyi * hrr_0012y;
                        gjly = aj2 * hrr_0112y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_0110y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+1)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+1)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1011y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1011y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2011y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0011y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1012y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * trr_11y;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                        double hrr_2021y = trr_23y - ylyk * trr_22y;
                        double hrr_2012y = hrr_2021y - ylyk * hrr_2011y;
                        gily = ai2 * hrr_2012y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0012y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_21y - 1 * trr_01y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1111y;
                        gjz = aj2 * hrr_0100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1012y;
                        glz = al2 * hrr_0001z;
                        gly -= 1 * trr_11y;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        double hrr_1112y = hrr_2012y - yjyi * hrr_1012y;
                        gjly = aj2 * hrr_1112y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_1110y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+1)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+1)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0011y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0011y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1011y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0012y;
                        glz = al2 * hrr_1001z;
                        gly -= 1 * trr_01y;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1012y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_11y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0111y;
                        gjz = aj2 * hrr_1100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0012y;
                        glz = al2 * hrr_1001z;
                        gly -= 1 * trr_01y;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0112y;
                        gjlz = aj2 * hrr_1101z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_0110y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+1)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+1)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0001y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_0011z;
                        gly -= 1 * 1;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1002y;
                        gilz = ai2 * hrr_1011z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_10y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_0110z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_0011z;
                        gly -= 1 * 1;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0102y;
                        gjlz = aj2 * hrr_0111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_0100y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+1)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+1)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1001y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1001y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2001y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * hrr_0001y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1002y;
                        glz = al2 * hrr_0011z;
                        gly -= 1 * trr_10y;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2002y;
                        gilz = ai2 * hrr_1011z;
                        gily -= 1 * hrr_0002y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1101y;
                        gjz = aj2 * hrr_0110z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1002y;
                        glz = al2 * hrr_0011z;
                        gly -= 1 * trr_10y;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1102y;
                        gjlz = aj2 * hrr_0111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_1100y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+1];
                            dd += dm[(j0+0)*nao+l0+1] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+1];
                                dd += dm[(nao+j0+0)*nao+l0+1] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+1)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+1)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0001y * dd;
                        Iz = trr_11z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0001y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1001y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_1011z;
                        gly -= 1 * 1;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1002y;
                        gilz = ai2 * hrr_2011z;
                        gilz -= 1 * hrr_0011z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gily -= 1 * (ai2 * trr_10y);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0101y;
                        gjz = aj2 * hrr_1110z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0002y;
                        glz = al2 * hrr_1011z;
                        gly -= 1 * 1;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0102y;
                        gjlz = aj2 * hrr_1111z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjly -= 1 * (aj2 * hrr_0100y);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+2)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+2)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1001z;
                        gix -= 1 * trr_01x;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0001y;
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2011x;
                        gily = ai2 * hrr_1001y;
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        gilz = ai2 * hrr_1002z;
                        gilx -= 1 * hrr_0011x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_10z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0101z;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1111x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0102z = hrr_1002z - zjzi * hrr_0002z;
                        gjlz = aj2 * hrr_0102z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_0100z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+2)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+2)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1001z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1002z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_10z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0101z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0102z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_0100z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+2)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+2)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1001z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_2001z;
                        giz -= 1 * hrr_0001z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1002z;
                        glz -= 1 * trr_10z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_1001y;
                        double hrr_2002z = hrr_2011z - zlzk * hrr_2001z;
                        gilz = ai2 * hrr_2002z;
                        gilz -= 1 * hrr_0002z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1101z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1002z;
                        glz -= 1 * trr_10z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1102z = hrr_2002z - zjzi * hrr_1002z;
                        gjlz = aj2 * hrr_1102z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_1100z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+2)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+2)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_1001z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_1002z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_10z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0101z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_0102z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_0100z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+2)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+2)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_11y * dd;
                        Iz = hrr_0001z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * hrr_1001z;
                        giy -= 1 * trr_01y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2011y;
                        gilz = ai2 * hrr_1002z;
                        gily -= 1 * hrr_0011y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_10z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0101z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0002z;
                        glz -= 1 * wt;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1111y;
                        gjlz = aj2 * hrr_0102z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_0100z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+2)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+2)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_1001z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_2001z;
                        giz -= 1 * hrr_0001z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1002z;
                        glz -= 1 * trr_10z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_2002z;
                        gilz -= 1 * hrr_0002z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1101z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1002z;
                        glz -= 1 * trr_10z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_1102z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_1100z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+2)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+2)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0011z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1011z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double hrr_0021z = trr_03z - zlzk * trr_02z;
                        double hrr_0012z = hrr_0021z - zlzk * hrr_0011z;
                        glz = al2 * hrr_0012z;
                        glz -= 1 * trr_01z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1001y;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double hrr_1021z = trr_13z - zlzk * trr_12z;
                        double hrr_1012z = hrr_1021z - zlzk * hrr_1011z;
                        gilz = ai2 * hrr_1012z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_11z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0111z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0012z;
                        glz -= 1 * trr_01z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0112z = hrr_1012z - zjzi * hrr_0012z;
                        gjlz = aj2 * hrr_0112z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_0110z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+2)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+2)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0011z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1011z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0012z;
                        glz -= 1 * trr_01z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1012z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_11z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0111z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0012z;
                        glz -= 1 * trr_01z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0112z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_0110z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+2];
                            dd += dm[(j0+0)*nao+l0+2] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+2];
                                dd += dm[(nao+j0+0)*nao+l0+2] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+2)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+2)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1011z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_2011z;
                        giz -= 1 * hrr_0011z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1012z;
                        glz -= 1 * trr_11z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1001y;
                        double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                        double hrr_2021z = trr_23z - zlzk * trr_22z;
                        double hrr_2012z = hrr_2021z - zlzk * hrr_2011z;
                        gilz = ai2 * hrr_2012z;
                        gilz -= 1 * hrr_0012z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        gilz -= 1 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1111z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1012z;
                        glz -= 1 * trr_11z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1112z = hrr_2012z - zjzi * hrr_1012z;
                        gjlz = aj2 * hrr_1112z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        gjlz -= 1 * (aj2 * hrr_1110z);
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        }
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
        int natm = envs.natm;
        double *ejk = jk.ejk;
        atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
        atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
        atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
        atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
        atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
        atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
        atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
        atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
        atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
        atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
        atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
        atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
        atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
    }
}
__global__
void rys_ejk_ip2_type3_1011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type3_1011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type3_1100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    double *dm = jk.dm;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4;
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
        double theta_ij = ai * aj_aij;
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
        double dd;
        double Ix, Iy, Iz, prod_xy, prod_xz, prod_yz;
        double gix, giy, giz;
        double gjx, gjy, gjz;
        double gkx, gky, gkz;
        double glx, gly, glz;
        double gikx, giky, gikz;
        double gjkx, gjky, gjkz;
        double gilx, gily, gilz;
        double gjlx, gjly, gjlz;
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        
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
            double theta_kl = ak * al_akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
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
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    fac *= sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    }
                } else {
                    rys_roots(3, theta_rr, rw+6*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1100x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1100x * Iy;
                        prod_xz = hrr_1100x * Iz;
                        prod_yz = 1 * Iz;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        gix = ai2 * hrr_2100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gix -= 1 * hrr_0100x;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gkx = ak2 * hrr_1110x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gky = ak2 * trr_01y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        gikx = ai2 * hrr_2110x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        giky = ai2 * trr_11y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gikz = ai2 * trr_11z;
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gikx -= 1 * hrr_0110x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        gjx = aj2 * hrr_1200x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_10x;
                        gkx = ak2 * hrr_1110x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                        gjkx = aj2 * hrr_1210x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjkz = aj2 * hrr_0110z;
                        gjkx -= 1 * trr_11x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1110x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        gjx = aj2 * hrr_0200x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * fac;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                        gjkx = aj2 * hrr_0210x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx -= 1 * trr_01x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1110x;
                        giky = ai2 * trr_11y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gjx -= 1 * fac;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0210x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjkz = aj2 * hrr_1110z;
                        gjkx -= 1 * trr_01x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * hrr_1110y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * 1;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                        gjky = aj2 * hrr_0210y;
                        gjkz = aj2 * hrr_0110z;
                        gjky -= 1 * trr_01y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1100y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1100y * Iz;
                        gix = ai2 * trr_10x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        giy = ai2 * hrr_2100y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0100y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1110y;
                        gkz = ak2 * trr_01z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        giky = ai2 * hrr_2110y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * hrr_0110y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gjy = aj2 * hrr_1200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_10y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1110y;
                        gkz = ak2 * trr_01z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                        gjky = aj2 * hrr_1210y;
                        gjkz = aj2 * hrr_0110z;
                        gjky -= 1 * trr_11y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_11z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_1110y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_1100z;
                        gjy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_11z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0210y;
                        gjkz = aj2 * hrr_1110z;
                        gjky -= 1 * trr_01y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1100z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0110z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * trr_11y;
                        gikz = ai2 * hrr_1110z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0110z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                        gjkz = aj2 * hrr_0210z;
                        gjkz -= 1 * trr_01z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1100z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0110z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * hrr_1110z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0110z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0210z;
                        gjkz -= 1 * trr_01z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        giz = ai2 * hrr_2100z;
                        giz -= 1 * hrr_0100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1110z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_11y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        gikz = ai2 * hrr_2110z;
                        gikz -= 1 * hrr_0110z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gjz = aj2 * hrr_1200z;
                        gjz -= 1 * trr_10z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1110z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                        gjkz = aj2 * hrr_1210z;
                        gjkz -= 1 * trr_11z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        }
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1100x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1100x * Iy;
                        prod_xz = hrr_1100x * Iz;
                        prod_yz = 1 * Iz;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        gix = ai2 * hrr_2100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gix -= 1 * hrr_0100x;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        glx = al2 * hrr_1101x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gly = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double hrr_3001x = trr_31x - xlxk * trr_30x;
                        double hrr_2101x = hrr_3001x - xjxi * hrr_2001x;
                        gilx = ai2 * hrr_2101x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gily = ai2 * hrr_1001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gilz = ai2 * hrr_1001z;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gilx -= 1 * hrr_0101x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        gjx = aj2 * hrr_1200x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_10x;
                        glx = al2 * hrr_1101x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1201x = hrr_2101x - xjxi * hrr_1101x;
                        gjlx = aj2 * hrr_1201x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjlz = aj2 * hrr_0101z;
                        gjlx -= 1 * hrr_1001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1101x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        gjx = aj2 * hrr_0200x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * fac;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0201x = hrr_1101x - xjxi * hrr_0101x;
                        gjlx = aj2 * hrr_0201x;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx -= 1 * hrr_0001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1101x;
                        gily = ai2 * hrr_1001y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gjx -= 1 * fac;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0201x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gjlz = aj2 * hrr_1101z;
                        gjlx -= 1 * hrr_0001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1101y;
                        gilz = ai2 * hrr_1001z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * 1;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        double hrr_0201y = hrr_1101y - yjyi * hrr_0101y;
                        gjly = aj2 * hrr_0201y;
                        gjlz = aj2 * hrr_0101z;
                        gjly -= 1 * hrr_0001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1100y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1100y * Iz;
                        gix = ai2 * trr_10x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        giy = ai2 * hrr_2100y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0100y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1101y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_3001y = trr_31y - ylyk * trr_30y;
                        double hrr_2101y = hrr_3001y - yjyi * hrr_2001y;
                        gily = ai2 * hrr_2101y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0101y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gjy = aj2 * hrr_1200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_10y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1101y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        double hrr_1201y = hrr_2101y - yjyi * hrr_1101y;
                        gjly = aj2 * hrr_1201y;
                        gjlz = aj2 * hrr_0101z;
                        gjly -= 1 * hrr_1001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1101y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_1100z;
                        gjy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0201y;
                        gjlz = aj2 * hrr_1101z;
                        gjly -= 1 * hrr_0001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1100z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1001y;
                        gilz = ai2 * hrr_1101z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0201z = hrr_1101z - zjzi * hrr_0101z;
                        gjlz = aj2 * hrr_0201z;
                        gjlz -= 1 * hrr_0001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1100z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1101z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0201z;
                        gjlz -= 1 * hrr_0001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        giz = ai2 * hrr_2100z;
                        giz -= 1 * hrr_0100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1001y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_3001z = trr_31z - zlzk * trr_30z;
                        double hrr_2101z = hrr_3001z - zjzi * hrr_2001z;
                        gilz = ai2 * hrr_2101z;
                        gilz -= 1 * hrr_0101z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gjz = aj2 * hrr_1200z;
                        gjz -= 1 * trr_10z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1201z = hrr_2101z - zjzi * hrr_1101z;
                        gjlz = aj2 * hrr_1201z;
                        gjlz -= 1 * hrr_1001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        }
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
        int natm = envs.natm;
        double *ejk = jk.ejk;
        atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
        atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
        atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
        atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
        atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
        atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
        atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
        atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
        atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
        atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
        atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
        atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
        atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
    }
}
__global__
void rys_ejk_ip2_type3_1100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type3_1100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type3_1110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
    int do_j = jk.j_factor != 0.;
    int do_k = jk.k_factor != 0.;
    double *dm = jk.dm;
    extern __shared__ double Rpa_cicj[];
    double *rw = Rpa_cicj + iprim*jprim*TILE2*4;
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
        double theta_ij = ai * aj_aij;
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
        double dd;
        double Ix, Iy, Iz, prod_xy, prod_xz, prod_yz;
        double gix, giy, giz;
        double gjx, gjy, gjz;
        double gkx, gky, gkz;
        double glx, gly, glz;
        double gikx, giky, gikz;
        double gjkx, gjky, gjkz;
        double gilx, gily, gilz;
        double gjlx, gjly, gjlz;
        double v_ixkx = 0;
        double v_ixky = 0;
        double v_ixkz = 0;
        double v_iykx = 0;
        double v_iyky = 0;
        double v_iykz = 0;
        double v_izkx = 0;
        double v_izky = 0;
        double v_izkz = 0;
        double v_jxkx = 0;
        double v_jxky = 0;
        double v_jxkz = 0;
        double v_jykx = 0;
        double v_jyky = 0;
        double v_jykz = 0;
        double v_jzkx = 0;
        double v_jzky = 0;
        double v_jzkz = 0;
        double v_ixlx = 0;
        double v_ixly = 0;
        double v_ixlz = 0;
        double v_iylx = 0;
        double v_iyly = 0;
        double v_iylz = 0;
        double v_izlx = 0;
        double v_izly = 0;
        double v_izlz = 0;
        double v_jxlx = 0;
        double v_jxly = 0;
        double v_jxlz = 0;
        double v_jylx = 0;
        double v_jyly = 0;
        double v_jylz = 0;
        double v_jzlx = 0;
        double v_jzly = 0;
        double v_jzlz = 0;
        
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
            double theta_kl = ak * al_akl;
            double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
            double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
            double xqc = xlxk * al_akl;
            double yqc = ylyk * al_akl;
            double zqc = zlzk * al_akl;
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
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
                double xjxi = rj[0] - ri[0];
                double yjyi = rj[1] - ri[1];
                double zjzi = rj[2] - ri[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(3, theta_rr, rw);
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    fac *= sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                    }
                } else {
                    rys_roots(3, theta_rr, rw+6*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(3, theta_fac*theta_rr, rw);
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = 0; irys < 3; ++irys) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                if (task_id < ntasks) {
                    for (int irys = 0; irys < bounds.nroots; ++irys) {
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1110x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1110x * Iy;
                        prod_xz = hrr_1110x * Iz;
                        prod_yz = 1 * Iz;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        gix = ai2 * hrr_2110x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gix -= 1 * hrr_0110x;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1120x = trr_22x - xjxi * trr_12x;
                        gkx = ak2 * hrr_1120x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        gky = ak2 * trr_01y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gkz = ak2 * trr_01z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gkx -= 1 * hrr_1100x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        double hrr_2120x = trr_32x - xjxi * trr_22x;
                        gikx = ai2 * hrr_2120x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        giky = ai2 * trr_11y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gikz = ai2 * trr_11z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0120x = trr_12x - xjxi * trr_02x;
                        gikx -= 1 * hrr_0120x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gikx -= 1 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                        gjx = aj2 * hrr_1210x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_11x;
                        gkx = ak2 * hrr_1120x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * hrr_1100x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_1220x = hrr_2120x - xjxi * hrr_1120x;
                        gjkx = aj2 * hrr_1220x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjkz = aj2 * hrr_0110z;
                        gjkx -= 1 * trr_12x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        gjkx -= 1 * (aj2 * hrr_1200x - 1 * trr_10x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0110x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0110x * Iy;
                        prod_xz = hrr_0110x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1110x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        gkx = ak2 * hrr_0120x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * hrr_0100x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1120x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * hrr_1100x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                        gjx = aj2 * hrr_0210x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_01x;
                        gkx = ak2 * hrr_0120x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * hrr_0100x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        double hrr_0220x = hrr_1120x - xjxi * hrr_0120x;
                        gjkx = aj2 * hrr_0220x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx -= 1 * trr_02x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        gjkx -= 1 * (aj2 * hrr_0200x - 1 * fac);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0110x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0110x * Iy;
                        prod_xz = hrr_0110x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1110x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * hrr_0120x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * hrr_0100x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1120x;
                        giky = ai2 * trr_11y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * hrr_1100x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0210x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gjx -= 1 * trr_01x;
                        gkx = ak2 * hrr_0120x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * hrr_0100x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0220x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjkz = aj2 * hrr_1110z;
                        gjkx -= 1 * trr_02x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0200x - 1 * fac);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * trr_01x;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * trr_10x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_22x;
                        giky = ai2 * hrr_1110y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * trr_02x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * 1;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * trr_10x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1120x;
                        double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                        gjky = aj2 * hrr_0210y;
                        gjkz = aj2 * hrr_0110z;
                        gjky -= 1 * trr_01y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_1100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_1100y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_1100y * Iz;
                        gix = ai2 * trr_11x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        giy = ai2 * hrr_2100y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0100y;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_1110y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        giky = ai2 * hrr_2110y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * hrr_0110y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gjy = aj2 * hrr_1200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_10y;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_1110y;
                        gkz = ak2 * trr_01z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                        gjky = aj2 * hrr_1210y;
                        gjkz = aj2 * hrr_0110z;
                        gjky -= 1 * trr_11y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * hrr_1110y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_1100z;
                        gjy -= 1 * 1;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_11z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_0210y;
                        gjkz = aj2 * hrr_1110z;
                        gjky -= 1 * trr_01y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1100z;
                        gix -= 1 * trr_01x;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0110z;
                        gkx -= 1 * trr_10x;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_22x;
                        giky = ai2 * trr_11y;
                        gikz = ai2 * hrr_1110z;
                        gikx -= 1 * trr_02x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_20x - 1 * fac);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        gkx = ak2 * trr_12x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0110z;
                        gkx -= 1 * trr_10x;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1120x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                        gjkz = aj2 * hrr_0210z;
                        gjkz -= 1 * trr_01z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_1100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1100z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0110z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * hrr_1110z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0110z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0210z;
                        gjkz -= 1 * trr_01z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1100z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_10y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        giz = ai2 * hrr_2100z;
                        giz -= 1 * hrr_0100z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1110z;
                        gkx -= 1 * fac;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_12x;
                        giky = ai2 * trr_11y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        gikz = ai2 * hrr_2110z;
                        gikz -= 1 * hrr_0110z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikx -= 1 * (ai2 * trr_10x);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gjz = aj2 * hrr_1200z;
                        gjz -= 1 * trr_10z;
                        gkx = ak2 * trr_02x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1110z;
                        gkx -= 1 * fac;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0120x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                        gjkz = aj2 * hrr_1210z;
                        gjkz -= 1 * trr_11z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkx -= 1 * (aj2 * hrr_0100x);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1100x * dd;
                        Iy = trr_01y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1100x * Iy;
                        prod_xz = hrr_1100x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * hrr_2100x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * hrr_0100x;
                        gkx = ak2 * hrr_1110x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_2110x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * hrr_0110x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1200x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_10x;
                        gkx = ak2 * hrr_1110x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1210x;
                        double hrr_0120y = trr_12y - yjyi * trr_02y;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx -= 1 * trr_11x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_11y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * trr_01y;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * trr_10y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1110x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        giky = ai2 * trr_22y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * trr_02y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * fac;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * trr_10y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0210x;
                        double hrr_1120y = trr_22y - yjyi * trr_12y;
                        gjky = aj2 * hrr_1120y;
                        gjkz = aj2 * hrr_0110z;
                        gjkx -= 1 * trr_01x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_01y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1110x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1100z;
                        gjx -= 1 * fac;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0210x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_1110z;
                        gjkx -= 1 * trr_01x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0110y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0110y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1110y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0120y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_0100y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * hrr_1120y;
                        gikz = ai2 * trr_11z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * hrr_1100y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0210y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_01y;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0120y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_0100y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        double hrr_0220y = hrr_1120y - yjyi * hrr_0120y;
                        gjky = aj2 * hrr_0220y;
                        gjkz = aj2 * hrr_0110z;
                        gjky -= 1 * trr_02y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0200y - 1 * 1);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1110y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1110y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2110y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0110y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1120y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_1100y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        double hrr_2120y = trr_32y - yjyi * trr_22y;
                        giky = ai2 * hrr_2120y;
                        gikz = ai2 * trr_11z;
                        giky -= 1 * hrr_0120y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1210y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_11y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1120y;
                        gkz = ak2 * trr_01z;
                        gky -= 1 * hrr_1100y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        double hrr_1220y = hrr_2120y - yjyi * hrr_1120y;
                        gjky = aj2 * hrr_1220y;
                        gjkz = aj2 * hrr_0110z;
                        gjky -= 1 * trr_12y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1200y - 1 * trr_10y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0110y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0110y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1110y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0120y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * hrr_0100y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_1120y;
                        gikz = ai2 * trr_21z;
                        gikz -= 1 * trr_01z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * hrr_1100y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0210y;
                        gjz = aj2 * hrr_1100z;
                        gjy -= 1 * trr_01y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0120y;
                        gkz = ak2 * trr_11z;
                        gky -= 1 * hrr_0100y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0220y;
                        gjkz = aj2 * hrr_1110z;
                        gjky -= 1 * trr_02y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0200y - 1 * 1);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_1100z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_0110z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * hrr_1110z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_0110z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_0210z;
                        gjkz -= 1 * trr_01z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_11y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * hrr_1100z;
                        giy -= 1 * trr_01y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * hrr_0110z;
                        gky -= 1 * trr_10y;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_22y;
                        gikz = ai2 * hrr_1110z;
                        giky -= 1 * trr_02y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_20y - 1 * 1);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_12y;
                        gkz = ak2 * hrr_0110z;
                        gky -= 1 * trr_10y;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1120y;
                        gjkz = aj2 * hrr_0210z;
                        gjkz -= 1 * trr_01z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_1100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_1100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_2100z;
                        giz -= 1 * hrr_0100z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_1110z;
                        gky -= 1 * 1;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_12y;
                        gikz = ai2 * hrr_2110z;
                        gikz -= 1 * hrr_0110z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        giky -= 1 * (ai2 * trr_10y);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1200z;
                        gjz -= 1 * trr_10z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_02y;
                        gkz = ak2 * hrr_1110z;
                        gky -= 1 * 1;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0120y;
                        gjkz = aj2 * hrr_1210z;
                        gjkz -= 1 * trr_11z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjky -= 1 * (aj2 * hrr_0100y);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1100x * dd;
                        Iy = 1 * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_1100x * Iy;
                        prod_xz = hrr_1100x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_2100x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * hrr_0100x;
                        gkx = ak2 * hrr_1110x;
                        gky = ak2 * trr_01y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_2110x;
                        giky = ai2 * trr_11y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        gikz = ai2 * trr_12z;
                        gikx -= 1 * hrr_0110x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1200x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0110z;
                        gjx -= 1 * trr_10x;
                        gkx = ak2 * hrr_1110x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1210x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0120z = trr_12z - zjzi * trr_02z;
                        gjkz = aj2 * hrr_0120z;
                        gjkx -= 1 * trr_11x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_10y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * 1;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1110x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * trr_12z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0110z;
                        gjx -= 1 * fac;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0210x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0120z;
                        gjkx -= 1 * trr_01x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = 1 * dd;
                        Iz = trr_11z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * hrr_1110x;
                        giky = ai2 * trr_11y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        gikz = ai2 * trr_22z;
                        gikz -= 1 * trr_02z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1110z;
                        gjx -= 1 * fac;
                        gkx = ak2 * hrr_0110x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0210x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1120z = trr_22z - zjzi * trr_12z;
                        gjkz = aj2 * hrr_1120z;
                        gjkx -= 1 * trr_01x;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * hrr_1110y;
                        gikz = ai2 * trr_12z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_0110z;
                        gjy -= 1 * 1;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0210y;
                        gjkz = aj2 * hrr_0120z;
                        gjky -= 1 * trr_01y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1100y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1100y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2100y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * hrr_0100y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1110y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_2110y;
                        gikz = ai2 * trr_12z;
                        giky -= 1 * hrr_0110y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_10z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1200y;
                        gjz = aj2 * hrr_0110z;
                        gjy -= 1 * trr_10y;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_1110y;
                        gkz = ak2 * trr_02z;
                        gkz -= 1 * wt;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1210y;
                        gjkz = aj2 * hrr_0120z;
                        gjky -= 1 * trr_11y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_11z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * hrr_1110y;
                        gikz = ai2 * trr_22z;
                        gikz -= 1 * trr_02z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * trr_20z - 1 * wt);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_1110z;
                        gjy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * hrr_0110y;
                        gkz = ak2 * trr_12z;
                        gkz -= 1 * trr_10z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0210y;
                        gjkz = aj2 * hrr_1120z;
                        gjky -= 1 * trr_01y;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1100z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0110z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1110z;
                        gix -= 1 * fac;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0120z;
                        gkz -= 1 * hrr_0100z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_21x;
                        giky = ai2 * trr_11y;
                        gikz = ai2 * hrr_1120z;
                        gikx -= 1 * trr_01x;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * hrr_1100z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_0210z;
                        gjz -= 1 * trr_01z;
                        gkx = ak2 * trr_11x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_0120z;
                        gkz -= 1 * hrr_0100z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_1110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_0220z = hrr_1120z - zjzi * hrr_0120z;
                        gjkz = aj2 * hrr_0220z;
                        gjkz -= 1 * trr_02z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0110z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1110z;
                        giy -= 1 * 1;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0120z;
                        gkz -= 1 * hrr_0100z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_21y;
                        gikz = ai2 * hrr_1120z;
                        giky -= 1 * trr_01y;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * hrr_1100z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0210z;
                        gjz -= 1 * trr_01z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_11y;
                        gkz = ak2 * hrr_0120z;
                        gkz -= 1 * hrr_0100z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_1110y;
                        gjkz = aj2 * hrr_0220z;
                        gjkz -= 1 * trr_02z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1110z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_2110z;
                        giz -= 1 * hrr_0110z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1120z;
                        gkz -= 1 * hrr_1100z;
                        v_ixky += gix * gky * Iz;
                        v_ixkz += gix * gkz * Iy;
                        v_iykx += giy * gkx * Iz;
                        v_iykz += giy * gkz * Ix;
                        v_izkx += giz * gkx * Iy;
                        v_izky += giz * gky * Ix;
                        gikx = ai2 * trr_11x;
                        giky = ai2 * trr_11y;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        double hrr_2120z = trr_32z - zjzi * trr_22z;
                        gikz = ai2 * hrr_2120z;
                        gikz -= 1 * hrr_0120z;
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;
                        gikz -= 1 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixkx += gikx * prod_yz;
                        v_iyky += giky * prod_xz;
                        v_izkz += gikz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        gjz = aj2 * hrr_1210z;
                        gjz -= 1 * trr_11z;
                        gkx = ak2 * trr_01x;
                        gky = ak2 * trr_01y;
                        gkz = ak2 * hrr_1120z;
                        gkz -= 1 * hrr_1100z;
                        v_jxky += gjx * gky * Iz;
                        v_jxkz += gjx * gkz * Iy;
                        v_jykx += gjy * gkx * Iz;
                        v_jykz += gjy * gkz * Ix;
                        v_jzkx += gjz * gkx * Iy;
                        v_jzky += gjz * gky * Ix;
                        gjkx = aj2 * hrr_0110x;
                        gjky = aj2 * hrr_0110y;
                        double hrr_1220z = hrr_2120z - zjzi * hrr_1120z;
                        gjkz = aj2 * hrr_1220z;
                        gjkz -= 1 * trr_12z;
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;
                        gjkz -= 1 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxkx += gjkx * prod_yz;
                        v_jyky += gjky * prod_xz;
                        v_jzkz += gjkz * prod_xy;
                        }
                        {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1110x * dd;
                        Iy = 1 * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1110x * Iy;
                        prod_xz = hrr_1110x * Iz;
                        prod_yz = 1 * Iz;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        gix = ai2 * hrr_2110x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        giy = ai2 * trr_10y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        giz = ai2 * trr_10z;
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        gix -= 1 * hrr_0110x;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        glx = al2 * hrr_1111x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        gly = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        double hrr_3011x = trr_32x - xlxk * trr_31x;
                        double hrr_2111x = hrr_3011x - xjxi * hrr_2011x;
                        gilx = ai2 * hrr_2111x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gily = ai2 * hrr_1001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gilz = ai2 * hrr_1001z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        gilx -= 1 * hrr_0111x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                        gjx = aj2 * hrr_1210x;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_11x;
                        glx = al2 * hrr_1111x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1211x = hrr_2111x - xjxi * hrr_1111x;
                        gjlx = aj2 * hrr_1211x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gjlz = aj2 * hrr_0101z;
                        gjlx -= 1 * hrr_1011x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0110x * dd;
                        Iy = trr_10y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0110x * Iy;
                        prod_xz = hrr_0110x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1110x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0111x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1111x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                        gjx = aj2 * hrr_0210x;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_01x;
                        glx = al2 * hrr_0111x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0211x = hrr_1111x - xjxi * hrr_0111x;
                        gjlx = aj2 * hrr_0211x;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx -= 1 * hrr_0011x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0110x * dd;
                        Iy = 1 * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0110x * Iy;
                        prod_xz = hrr_0110x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1110x;
                        giy = ai2 * trr_10y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0111x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1111x;
                        gily = ai2 * hrr_1001y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0210x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gjz = aj2 * hrr_1100z;
                        gjx -= 1 * trr_01x;
                        glx = al2 * hrr_0111x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0211x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gjlz = aj2 * hrr_1101z;
                        gjlx -= 1 * hrr_0011x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * trr_01x;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2011x;
                        gily = ai2 * hrr_1101y;
                        gilz = ai2 * hrr_1001z;
                        gilx -= 1 * hrr_0011x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * 1;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1111x;
                        double hrr_0201y = hrr_1101y - yjyi * hrr_0101y;
                        gjly = aj2 * hrr_0201y;
                        gjlz = aj2 * hrr_0101z;
                        gjly -= 1 * hrr_0001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_1100y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_1100y * Iz;
                        gix = ai2 * trr_11x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        giy = ai2 * hrr_2100y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0100y;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1101y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_3001y = trr_31y - ylyk * trr_30y;
                        double hrr_2101y = hrr_3001y - yjyi * hrr_2001y;
                        gily = ai2 * hrr_2101y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0101y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gjy = aj2 * hrr_1200y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_10y;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1101y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        double hrr_1201y = hrr_2101y - yjyi * hrr_1101y;
                        gjly = aj2 * hrr_1201y;
                        gjlz = aj2 * hrr_0101z;
                        gjly -= 1 * hrr_1001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_1101y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_1100z;
                        gjy -= 1 * 1;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_0201y;
                        gjlz = aj2 * hrr_1101z;
                        gjly -= 1 * hrr_0001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_11x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_11x * Iy;
                        prod_xz = trr_11x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_21x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1100z;
                        gix -= 1 * trr_01x;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2011x;
                        gily = ai2 * hrr_1001y;
                        gilz = ai2 * hrr_1101z;
                        gilx -= 1 * hrr_0011x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        glx = al2 * hrr_1011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1111x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0201z = hrr_1101z - zjzi * hrr_0101z;
                        gjlz = aj2 * hrr_0201z;
                        gjlz -= 1 * hrr_0001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1100z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1101z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0201z;
                        gjlz -= 1 * hrr_0001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+0] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+0];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+0];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+0;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_01x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1100z * dd;
                        prod_xy = trr_01x * Iy;
                        prod_xz = trr_01x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_11x;
                        giy = ai2 * trr_10y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        giz = ai2 * hrr_2100z;
                        giz -= 1 * hrr_0100z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1011x;
                        gily = ai2 * hrr_1001y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_3001z = trr_31z - zlzk * trr_30z;
                        double hrr_2101z = hrr_3001z - zjzi * hrr_2001z;
                        gilz = ai2 * hrr_2101z;
                        gilz -= 1 * hrr_0101z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0110x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gjz = aj2 * hrr_1200z;
                        gjz -= 1 * trr_10z;
                        glx = al2 * hrr_0011x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0111x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1201z = hrr_2101z - zjzi * hrr_1101z;
                        gjlz = aj2 * hrr_1201z;
                        gjlz -= 1 * hrr_1001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1100x * dd;
                        Iy = trr_01y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_1100x * Iy;
                        prod_xz = hrr_1100x * Iz;
                        prod_yz = trr_01y * Iz;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        gix = ai2 * hrr_2100x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        gix -= 1 * hrr_0100x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        glx = al2 * hrr_1101x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        double hrr_3001x = trr_31x - xlxk * trr_30x;
                        double hrr_2101x = hrr_3001x - xjxi * hrr_2001x;
                        gilx = ai2 * hrr_2101x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_1001z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        gilx -= 1 * hrr_0101x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        gjx = aj2 * hrr_1200x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * trr_10x;
                        glx = al2 * hrr_1101x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_1201x = hrr_2101x - xjxi * hrr_1101x;
                        gjlx = aj2 * hrr_1201x;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx -= 1 * hrr_1001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_11y * dd;
                        Iz = wt * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * trr_01y;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1101x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        gily = ai2 * hrr_2011y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0011y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        gjx = aj2 * hrr_0200x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0100z;
                        gjx -= 1 * fac;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        double hrr_0201x = hrr_1101x - xjxi * hrr_0101x;
                        gjlx = aj2 * hrr_0201x;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        gjly = aj2 * hrr_1111y;
                        gjlz = aj2 * hrr_0101z;
                        gjlx -= 1 * hrr_0001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_01y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1101x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1100z;
                        gjx -= 1 * fac;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0201x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_1101z;
                        gjlx -= 1 * hrr_0001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0110y * dd;
                        Iz = wt * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0110y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1110y;
                        giz = ai2 * trr_10z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0111y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1111y;
                        gilz = ai2 * hrr_1001z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                        gjy = aj2 * hrr_0210y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_01y;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0111y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        double hrr_0211y = hrr_1111y - yjyi * hrr_0111y;
                        gjly = aj2 * hrr_0211y;
                        gjlz = aj2 * hrr_0101z;
                        gjly -= 1 * hrr_0011y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1110y * dd;
                        Iz = wt * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1110y * Iz;
                        gix = ai2 * trr_10x;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        giy = ai2 * hrr_2110y;
                        giz = ai2 * trr_10z;
                        giy -= 1 * hrr_0110y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1111y;
                        glz = al2 * hrr_0001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        double hrr_3011y = trr_32y - ylyk * trr_31y;
                        double hrr_2111y = hrr_3011y - yjyi * hrr_2011y;
                        gily = ai2 * hrr_2111y;
                        gilz = ai2 * hrr_1001z;
                        gily -= 1 * hrr_0111y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                        gjy = aj2 * hrr_1210y;
                        gjz = aj2 * hrr_0100z;
                        gjy -= 1 * trr_11y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1111y;
                        glz = al2 * hrr_0001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        double hrr_1211y = hrr_2111y - yjyi * hrr_1111y;
                        gjly = aj2 * hrr_1211y;
                        gjlz = aj2 * hrr_0101z;
                        gjly -= 1 * hrr_1011y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0110y * dd;
                        Iz = trr_10z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0110y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1110y;
                        giz = ai2 * trr_20z;
                        giz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0111y;
                        glz = al2 * hrr_1001z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1111y;
                        gilz = ai2 * hrr_2001z;
                        gilz -= 1 * hrr_0001z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0210y;
                        gjz = aj2 * hrr_1100z;
                        gjy -= 1 * trr_01y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0111y;
                        glz = al2 * hrr_1001z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0211y;
                        gjlz = aj2 * hrr_1101z;
                        gjly -= 1 * hrr_0011y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+1] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_1100z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_1101z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_0101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_0201z;
                        gjlz -= 1 * hrr_0001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+1] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_11y * dd;
                        Iz = hrr_0100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_11y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_21y;
                        giz = ai2 * hrr_1100z;
                        giy -= 1 * trr_01y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2011y;
                        gilz = ai2 * hrr_1101z;
                        gily -= 1 * hrr_0011y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1110y;
                        gjz = aj2 * hrr_0200z;
                        gjz -= 1 * wt;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1011y;
                        glz = al2 * hrr_0101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1111y;
                        gjlz = aj2 * hrr_0201z;
                        gjlz -= 1 * hrr_0001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+1] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+1];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+1] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+1];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+1];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+1;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_01y * dd;
                        Iz = hrr_1100z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_01y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_11y;
                        giz = ai2 * hrr_2100z;
                        giz -= 1 * hrr_0100z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1101z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1011y;
                        gilz = ai2 * hrr_2101z;
                        gilz -= 1 * hrr_0101z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0110y;
                        gjz = aj2 * hrr_1200z;
                        gjz -= 1 * trr_10z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0011y;
                        glz = al2 * hrr_1101z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0111y;
                        gjlz = aj2 * hrr_1201z;
                        gjlz -= 1 * hrr_1001z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_1100x * dd;
                        Iy = 1 * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_1100x * Iy;
                        prod_xz = hrr_1100x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_2100x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * hrr_0100x;
                        glx = al2 * hrr_1101x;
                        gly = al2 * hrr_0001y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        glz = al2 * hrr_0011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2101x;
                        gily = ai2 * hrr_1001y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gilz = ai2 * hrr_1011z;
                        gilx -= 1 * hrr_0101x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1200x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gjz = aj2 * hrr_0110z;
                        gjx -= 1 * trr_10x;
                        glx = al2 * hrr_1101x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1201x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        gjlz = aj2 * hrr_0111z;
                        gjlx -= 1 * hrr_1001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = trr_10y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1101x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1011z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0110z;
                        gjx -= 1 * fac;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0201x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0111z;
                        gjlx -= 1 * hrr_0001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+0)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+0)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+0)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+0)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+0)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = hrr_0100x * dd;
                        Iy = 1 * dd;
                        Iz = trr_11z * dd;
                        prod_xy = hrr_0100x * Iy;
                        prod_xz = hrr_0100x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * hrr_1100x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1101x;
                        gily = ai2 * hrr_1001y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        gilz = ai2 * hrr_2011z;
                        gilz -= 1 * hrr_0011z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0200x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gjz = aj2 * hrr_1110z;
                        gjx -= 1 * fac;
                        glx = al2 * hrr_0101x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0201x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        gjlz = aj2 * hrr_1111z;
                        gjlx -= 1 * hrr_0001x;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+1)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_11z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_0011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1101y;
                        gilz = ai2 * hrr_1011z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_0110z;
                        gjy -= 1 * 1;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_0011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0201y;
                        gjlz = aj2 * hrr_0111z;
                        gjly -= 1 * hrr_0001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+1)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_1100y * dd;
                        Iz = trr_01z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_1100y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_2100y;
                        giz = ai2 * trr_11z;
                        giy -= 1 * hrr_0100y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1101y;
                        glz = al2 * hrr_0011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2101y;
                        gilz = ai2 * hrr_1011z;
                        gily -= 1 * hrr_0101y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1200y;
                        gjz = aj2 * hrr_0110z;
                        gjy -= 1 * trr_10y;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1101y;
                        glz = al2 * hrr_0011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1201y;
                        gjlz = aj2 * hrr_0111z;
                        gjly -= 1 * hrr_1001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+1)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+1)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+1)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+1)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+1)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+1)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = hrr_0100y * dd;
                        Iz = trr_11z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = hrr_0100y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * hrr_1100y;
                        giz = ai2 * trr_21z;
                        giz -= 1 * trr_01z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_1011z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1101y;
                        gilz = ai2 * hrr_2011z;
                        gilz -= 1 * hrr_0011z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0200y;
                        gjz = aj2 * hrr_1110z;
                        gjy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0101y;
                        glz = al2 * hrr_1011z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0201y;
                        gjlz = aj2 * hrr_1111z;
                        gjly -= 1 * hrr_0001y;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+2] * dm[(i0+0)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+0)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+0)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+0] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+2)*nao+i0+0;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = trr_10x * dd;
                        Iy = 1 * dd;
                        Iz = hrr_0110z * dd;
                        prod_xy = trr_10x * Iy;
                        prod_xz = trr_10x * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_20x;
                        giy = ai2 * trr_10y;
                        giz = ai2 * hrr_1110z;
                        gix -= 1 * fac;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0111z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_2001x;
                        gily = ai2 * hrr_1001y;
                        gilz = ai2 * hrr_1111z;
                        gilx -= 1 * hrr_0001x;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_1100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                        gjz = aj2 * hrr_0210z;
                        gjz -= 1 * trr_01z;
                        glx = al2 * hrr_1001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_0111z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_1101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_0211z = hrr_1111z - zjzi * hrr_0111z;
                        gjlz = aj2 * hrr_0211z;
                        gjlz -= 1 * hrr_0011z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+2] * dm[(i0+1)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+1)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+1)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+1] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+2)*nao+i0+1;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = trr_10y * dd;
                        Iz = hrr_0110z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = trr_10y * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_20y;
                        giz = ai2 * hrr_1110z;
                        giy -= 1 * 1;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0111z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_2001y;
                        gilz = ai2 * hrr_1111z;
                        gily -= 1 * hrr_0001y;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_1100y;
                        gjz = aj2 * hrr_0210z;
                        gjz -= 1 * trr_01z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_1001y;
                        glz = al2 * hrr_0111z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_1101y;
                        gjlz = aj2 * hrr_0211z;
                        gjlz -= 1 * hrr_0011z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        if (do_k) {
                            dd  = dm[(j0+2)*nao+k0+2] * dm[(i0+2)*nao+l0+0];
                            dd += dm[(j0+2)*nao+l0+0] * dm[(i0+2)*nao+k0+2];
                            if (jk.n_dm > 1) {
                                dd += dm[(nao+j0+2)*nao+k0+2] * dm[(nao+i0+2)*nao+l0+0];
                                dd += dm[(nao+j0+2)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+2];
                            }
                            dd *= jk.k_factor;
                        } else {
                            dd = 0.;
                        }
                        if (do_j) {
                            if (jk.n_dm == 1) {
                                dd += jk.j_factor * dm[(j0+2)*nao+i0+2] * dm[(l0+0)*nao+k0+2];
                            } else {
                                int ji = (j0+2)*nao+i0+2;
                                int lk = (l0+0)*nao+k0+2;
                                dd += jk.j_factor * (dm[ji] + dm[nao*nao+ji]) * (dm[lk] + dm[nao*nao+lk]);
                            }
                        }
                        Ix = fac * dd;
                        Iy = 1 * dd;
                        Iz = hrr_1110z * dd;
                        prod_xy = fac * Iy;
                        prod_xz = fac * Iz;
                        prod_yz = 1 * Iz;
                        gix = ai2 * trr_10x;
                        giy = ai2 * trr_10y;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        giz = ai2 * hrr_2110z;
                        giz -= 1 * hrr_0110z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1111z;
                        v_ixly += gix * gly * Iz;
                        v_ixlz += gix * glz * Iy;
                        v_iylx += giy * glx * Iz;
                        v_iylz += giy * glz * Ix;
                        v_izlx += giz * glx * Iy;
                        v_izly += giz * gly * Ix;
                        gilx = ai2 * hrr_1001x;
                        gily = ai2 * hrr_1001y;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        double hrr_3011z = trr_32z - zlzk * trr_31z;
                        double hrr_2111z = hrr_3011z - zjzi * hrr_2011z;
                        gilz = ai2 * hrr_2111z;
                        gilz -= 1 * hrr_0111z;
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;
                        v_ixlx += gilx * prod_yz;
                        v_iyly += gily * prod_xz;
                        v_izlz += gilz * prod_xy;
                        gjx = aj2 * hrr_0100x;
                        gjy = aj2 * hrr_0100y;
                        double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                        gjz = aj2 * hrr_1210z;
                        gjz -= 1 * trr_11z;
                        glx = al2 * hrr_0001x;
                        gly = al2 * hrr_0001y;
                        glz = al2 * hrr_1111z;
                        v_jxly += gjx * gly * Iz;
                        v_jxlz += gjx * glz * Iy;
                        v_jylx += gjy * glx * Iz;
                        v_jylz += gjy * glz * Ix;
                        v_jzlx += gjz * glx * Iy;
                        v_jzly += gjz * gly * Ix;
                        gjlx = aj2 * hrr_0101x;
                        gjly = aj2 * hrr_0101y;
                        double hrr_1211z = hrr_2111z - zjzi * hrr_1111z;
                        gjlz = aj2 * hrr_1211z;
                        gjlz -= 1 * hrr_1011z;
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;
                        v_jxlx += gjlx * prod_yz;
                        v_jyly += gjly * prod_xz;
                        v_jzlz += gjlz * prod_xy;
                        }
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
        int natm = envs.natm;
        double *ejk = jk.ejk;
        atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
        atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
        atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
        atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
        atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
        atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
        atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
        atomicAdd(ejk + (ia*natm+la)*9 + 0, v_ixlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 1, v_ixly);
        atomicAdd(ejk + (ia*natm+la)*9 + 2, v_ixlz);
        atomicAdd(ejk + (ia*natm+la)*9 + 3, v_iylx);
        atomicAdd(ejk + (ia*natm+la)*9 + 4, v_iyly);
        atomicAdd(ejk + (ia*natm+la)*9 + 5, v_iylz);
        atomicAdd(ejk + (ia*natm+la)*9 + 6, v_izlx);
        atomicAdd(ejk + (ia*natm+la)*9 + 7, v_izly);
        atomicAdd(ejk + (ia*natm+la)*9 + 8, v_izlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 0, v_jxlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 1, v_jxly);
        atomicAdd(ejk + (ja*natm+la)*9 + 2, v_jxlz);
        atomicAdd(ejk + (ja*natm+la)*9 + 3, v_jylx);
        atomicAdd(ejk + (ja*natm+la)*9 + 4, v_jyly);
        atomicAdd(ejk + (ja*natm+la)*9 + 5, v_jylz);
        atomicAdd(ejk + (ja*natm+la)*9 + 6, v_jzlx);
        atomicAdd(ejk + (ja*natm+la)*9 + 7, v_jzly);
        atomicAdd(ejk + (ja*natm+la)*9 + 8, v_jzlz);
    }
}
__global__
void rys_ejk_ip2_type3_1110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type3_1110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_ejk_ip2_type3_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
                        ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int threads = scheme[0] * scheme[1];
    int nroots = bounds->nroots;
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int buflen = nroots*2 * threads + ij_prims*TILE2*4;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    switch (ijkl) {
    case 0: rys_ejk_ip2_type3_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125: rys_ejk_ip2_type3_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 130: rys_ejk_ip2_type3_1010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 131: rys_ejk_ip2_type3_1011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 150: rys_ejk_ip2_type3_1100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 155: rys_ejk_ip2_type3_1110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
