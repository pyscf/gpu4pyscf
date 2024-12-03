#include "vhf.cuh"
#include "rys_roots_unrolled.cu"
#include "create_tasks_ip1.cu"
int rys_ejk_ip2_type12_unrolled_lmax = 1;
int rys_ejk_ip2_type12_unrolled_max_order = 3;


__device__ static
void _rys_ejk_ip2_type12_0000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double g1x, g1y, g1z;
        double g2x, g2y, g2z;
        double g3x, g3y, g3z;
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        
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
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g1x = aj2 * hrr_0100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        g1y = aj2 * hrr_0100y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_10z;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        g3x = ai2 * hrr_1100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        g3y = ai2 * hrr_1100y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        g3z = ai2 * hrr_1100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g1x = al2 * hrr_0001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1y = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_01z;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g3x = ak2 * hrr_0011x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3y = ak2 * hrr_0011y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
        atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
        atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
        atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
        atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
        atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
        atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
        atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
        atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
        atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
        atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
        atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
        atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
        atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
        atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
        atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
        atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
    }
}
__global__
void rys_ejk_ip2_type12_0000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type12_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type12_1000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double g1x, g1y, g1z;
        double g2x, g2y, g2z;
        double g3x, g3y, g3z;
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        
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
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        g1x = aj2 * hrr_1100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        g1y = aj2 * hrr_0100y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * fac;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        g3x = ai2 * hrr_2100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        g3y = ai2 * hrr_1100y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g1x = al2 * hrr_1001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1y = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_01z;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_01x = cpx * fac;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        g3x = ak2 * hrr_1011x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3y = ak2 * hrr_0011y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1100x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g1x = al2 * hrr_0001x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_01z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g3x = ak2 * hrr_0011x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_11z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3z = ak2 * hrr_1011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
        atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
        atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
        atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
        atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
        atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
        atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
        atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
        atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
        atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
        atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
        atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
        atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
        atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
        atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
        atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
        atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
    }
}
__global__
void rys_ejk_ip2_type12_1000(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type12_1000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type12_1010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double g1x, g1y, g1z;
        double g2x, g2y, g2z;
        double g3x, g3y, g3z;
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        
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
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        g1x = aj2 * hrr_1110x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        g1y = aj2 * hrr_0100y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_21x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_10z;
                        double trr_01x = cpx * fac;
                        g2x -= 1 * trr_01x;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        g3x = ai2 * hrr_2110x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        g3y = ai2 * hrr_1100y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        g3x -= 1 * hrr_0110x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_31x - 3 * trr_11x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                        g3x = aj2 * (aj2 * hrr_1210x - 1 * trr_11x);
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1110x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_1100y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
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
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        g1x = aj2 * hrr_1100x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * fac;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        g3x = ai2 * hrr_2100x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1110y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_21y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * trr_01y;
                        g3x = ai2 * hrr_1100x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        g3y = ai2 * hrr_2110y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0110y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_31y - 3 * trr_11y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                        g3y = aj2 * (aj2 * hrr_1210y - 1 * trr_11y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0100y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        g1z = aj2 * hrr_0110z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_11z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        g3z = ai2 * hrr_1110z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0110z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_11z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1110z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1110z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_21z;
                        g2z -= 1 * trr_01z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        g3z = ai2 * hrr_2110z;
                        g3z -= 1 * hrr_0110z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_31z - 3 * trr_11z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                        g3z = aj2 * (aj2 * hrr_1210z - 1 * trr_11z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_01x = cpx * fac;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        g1x = al2 * hrr_1011x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1y = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_12x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_01z;
                        g2x -= 1 * trr_10x;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double hrr_1021x = trr_13x - xlxk * trr_12x;
                        g3x = ak2 * hrr_1021x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3y = ak2 * hrr_0011y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3z = ak2 * hrr_0011z;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g3x -= 1 * hrr_1001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_13x - 3 * trr_11x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_1012x = hrr_1021x - xlxk * hrr_1011x;
                        g3x = al2 * (al2 * hrr_1012x - 1 * trr_11x);
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
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
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g1x = al2 * hrr_0011x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_01z;
                        g2x -= 1 * fac;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double hrr_0021x = trr_03x - xlxk * trr_02x;
                        g3x = ak2 * hrr_0021x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0011z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0012x = hrr_0021x - xlxk * hrr_0011x;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_0001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_11z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_0011y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3z = ak2 * hrr_1011z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_1011x;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double hrr_0021y = trr_03y - ylyk * trr_02y;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        double hrr_0012y = hrr_0021y - ylyk * hrr_0011y;
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1011y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_12y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * trr_10y;
                        g3x = ak2 * hrr_0011x;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double hrr_1021y = trr_13y - ylyk * trr_12y;
                        g3y = ak2 * hrr_1021y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_1001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_13y - 3 * trr_11y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        double hrr_1012y = hrr_1021y - ylyk * hrr_1011y;
                        g3y = al2 * (al2 * hrr_1012y - 1 * trr_11y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * trr_11z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_1011z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_0011z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double hrr_0021z = trr_03z - zlzk * trr_02z;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0012z = hrr_0021z - zlzk * hrr_0011z;
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0011z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_1011z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_12z;
                        g2z -= 1 * trr_10z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double hrr_1021z = trr_13z - zlzk * trr_12z;
                        g3z = ak2 * hrr_1021z;
                        g3z -= 1 * hrr_1001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_13z - 3 * trr_11z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1012z = hrr_1021z - zlzk * hrr_1011z;
                        g3z = al2 * (al2 * hrr_1012z - 1 * trr_11z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
        atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
        atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
        atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
        atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
        atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
        atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
        atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
        atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
        atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
        atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
        atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
        atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
        atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
        atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
        atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
        atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
    }
}
__global__
void rys_ejk_ip2_type12_1010(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type12_1010(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type12_1011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double g1x, g1y, g1z;
        double g2x, g2y, g2z;
        double g3x, g3y, g3z;
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        
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
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        g1x = aj2 * hrr_1111x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        g1y = aj2 * hrr_0100y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * hrr_2011x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g2x -= 1 * hrr_0011x;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        double hrr_3011x = trr_32x - xlxk * trr_31x;
                        double hrr_2111x = hrr_3011x - xjxi * hrr_2011x;
                        g3x = ai2 * hrr_2111x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        g3y = ai2 * hrr_1100y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        g3x -= 1 * hrr_0111x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3011x - 3 * hrr_1011x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1211x = hrr_2111x - xjxi * hrr_1111x;
                        g3x = aj2 * (aj2 * hrr_1211x - 1 * hrr_1011x);
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0111x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * hrr_1011x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1111x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2011x - 1 * hrr_0011x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0211x = hrr_1111x - xjxi * hrr_0111x;
                        g3x = aj2 * (aj2 * hrr_0211x - 1 * hrr_0011x);
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0111x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * hrr_1011x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1111x;
                        g3y = ai2 * hrr_1100y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2011x - 1 * hrr_0011x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0211x - 1 * hrr_0011x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
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
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        g1x = aj2 * hrr_1101x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * hrr_2001x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * trr_10z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g2x -= 1 * hrr_0001x;
                        double hrr_3001x = trr_31x - xlxk * trr_30x;
                        double hrr_2101x = hrr_3001x - xjxi * hrr_2001x;
                        g3x = ai2 * hrr_2101x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        g3x -= 1 * hrr_0101x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3001x - 3 * hrr_1001x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1201x = hrr_2101x - xjxi * hrr_1101x;
                        g3x = aj2 * (aj2 * hrr_1201x - 1 * hrr_1001x);
                        double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0101x;
                        g1y = aj2 * hrr_1110y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * hrr_1001x;
                        g2y = ai2 * trr_21y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * trr_01y;
                        g3x = ai2 * hrr_1101x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        g3y = ai2 * hrr_2110y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0110y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2001x - 1 * hrr_0001x);
                        g3y = ai2 * (ai2 * trr_31y - 3 * trr_11y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0201x = hrr_1101x - xjxi * hrr_0101x;
                        g3x = aj2 * (aj2 * hrr_0201x - 1 * hrr_0001x);
                        double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                        g3y = aj2 * (aj2 * hrr_1210y - 1 * trr_11y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0101x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * hrr_1001x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1101x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2001x - 1 * hrr_0001x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0201x - 1 * hrr_0001x);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
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
                        g1x = aj2 * hrr_1101x;
                        g1y = aj2 * hrr_0100y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        g1z = aj2 * hrr_0110z;
                        g2x = ai2 * hrr_2001x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_11z;
                        g2x -= 1 * hrr_0001x;
                        g3x = ai2 * hrr_2101x;
                        g3y = ai2 * hrr_1100y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        g3z = ai2 * hrr_1110z;
                        g3x -= 1 * hrr_0101x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3001x - 3 * hrr_1001x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1201x - 1 * hrr_1001x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0101x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0110z;
                        g2x = ai2 * hrr_1001x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_11z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1101x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1110z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2001x - 1 * hrr_0001x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0201x - 1 * hrr_0001x);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0101x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1110z;
                        g2x = ai2 * hrr_1001x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_21z;
                        g2z -= 1 * trr_01z;
                        g3x = ai2 * hrr_1101x;
                        g3y = ai2 * hrr_1100y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        g3z = ai2 * hrr_2110z;
                        g3z -= 1 * hrr_0110z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2001x - 1 * hrr_0001x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_31z - 3 * trr_11z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0201x - 1 * hrr_0001x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                        g3z = aj2 * (aj2 * hrr_1210z - 1 * trr_11z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        g1x = aj2 * hrr_1110x;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        g1y = aj2 * hrr_0101y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_21x;
                        g2y = ai2 * hrr_1001y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * trr_01x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        g3x = ai2 * hrr_2110x;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        g3y = ai2 * hrr_1101y;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        g3x -= 1 * hrr_0110x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_31x - 3 * trr_11x);
                        g3y = ai2 * (ai2 * hrr_2001y - 1 * hrr_0001y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                        g3x = aj2 * (aj2 * hrr_1210x - 1 * trr_11x);
                        double hrr_0201y = hrr_1101y - yjyi * hrr_0101y;
                        g3y = aj2 * (aj2 * hrr_0201y - 1 * hrr_0001y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_1101y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * hrr_2001y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * hrr_0001y;
                        g3x = ai2 * hrr_1110x;
                        double hrr_3001y = trr_31y - ylyk * trr_30y;
                        double hrr_2101y = hrr_3001y - yjyi * hrr_2001y;
                        g3y = ai2 * hrr_2101y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0101y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * hrr_3001y - 3 * hrr_1001y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        double hrr_1201y = hrr_2101y - yjyi * hrr_1101y;
                        g3y = aj2 * (aj2 * hrr_1201y - 1 * hrr_1001y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_0101y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * hrr_1001y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_1101y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * hrr_2001y - 1 * hrr_0001y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_0201y - 1 * hrr_0001y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
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
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        g1x = aj2 * hrr_1100x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        g1y = aj2 * hrr_0111y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * hrr_1011y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * fac;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        g3x = ai2 * hrr_2100x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        g3y = ai2 * hrr_1111y;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * hrr_2011y - 1 * hrr_0011y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        double hrr_0211y = hrr_1111y - yjyi * hrr_0111y;
                        g3y = aj2 * (aj2 * hrr_0211y - 1 * hrr_0011y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1111y;
                        g1z = aj2 * hrr_0100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_2011y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * hrr_0011y;
                        g3x = ai2 * hrr_1100x;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        double hrr_3011y = trr_32y - ylyk * trr_31y;
                        double hrr_2111y = hrr_3011y - yjyi * hrr_2011y;
                        g3y = ai2 * hrr_2111y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0111y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_3011y - 3 * hrr_1011y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        double hrr_1211y = hrr_2111y - yjyi * hrr_1111y;
                        g3y = aj2 * (aj2 * hrr_1211y - 1 * hrr_1011y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0111y;
                        g1z = aj2 * hrr_1100z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_1011y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1111y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_2011y - 1 * hrr_0011y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0211y - 1 * hrr_0011y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0101y;
                        g1z = aj2 * hrr_0110z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * hrr_1001y;
                        g2z = ai2 * trr_11z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1101y;
                        g3z = ai2 * hrr_1110z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * hrr_2001y - 1 * hrr_0001y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0201y - 1 * hrr_0001y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1101y;
                        g1z = aj2 * hrr_0110z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_2001y;
                        g2z = ai2 * trr_11z;
                        g2y -= 1 * hrr_0001y;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2101y;
                        g3z = ai2 * hrr_1110z;
                        g3y -= 1 * hrr_0101y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_3001y - 3 * hrr_1001y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1201y - 1 * hrr_1001y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0101y;
                        g1z = aj2 * hrr_1110z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_1001y;
                        g2z = ai2 * trr_21z;
                        g2z -= 1 * trr_01z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1101y;
                        g3z = ai2 * hrr_2110z;
                        g3z -= 1 * hrr_0110z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_2001y - 1 * hrr_0001y);
                        g3z = ai2 * (ai2 * trr_31z - 3 * trr_11z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0201y - 1 * hrr_0001y);
                        g3z = aj2 * (aj2 * hrr_1210z - 1 * trr_11z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1110x;
                        g1y = aj2 * hrr_0100y;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        g1z = aj2 * hrr_0101z;
                        g2x = ai2 * trr_21x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_1001z;
                        g2x -= 1 * trr_01x;
                        g3x = ai2 * hrr_2110x;
                        g3y = ai2 * hrr_1100y;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        g3z = ai2 * hrr_1101z;
                        g3x -= 1 * hrr_0110x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_31x - 3 * trr_11x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_2001z - 1 * hrr_0001z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1210x - 1 * trr_11x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0201z = hrr_1101z - zjzi * hrr_0101z;
                        g3z = aj2 * (aj2 * hrr_0201z - 1 * hrr_0001z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0101z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * hrr_1001z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1101z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * hrr_2001z - 1 * hrr_0001z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0201z - 1 * hrr_0001z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1101z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_2001z;
                        g2z -= 1 * hrr_0001z;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_1100y;
                        double hrr_3001z = trr_31z - zlzk * trr_30z;
                        double hrr_2101z = hrr_3001z - zjzi * hrr_2001z;
                        g3z = ai2 * hrr_2101z;
                        g3z -= 1 * hrr_0101z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_3001z - 3 * hrr_1001z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1201z = hrr_2101z - zjzi * hrr_1101z;
                        g3z = aj2 * (aj2 * hrr_1201z - 1 * hrr_1001z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_0101z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * hrr_1001z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_1101z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * hrr_2001z - 1 * hrr_0001z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_0201z - 1 * hrr_0001z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1110y;
                        g1z = aj2 * hrr_0101z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_21y;
                        g2z = ai2 * hrr_1001z;
                        g2y -= 1 * trr_01y;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2110y;
                        g3z = ai2 * hrr_1101z;
                        g3y -= 1 * hrr_0110y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_31y - 3 * trr_11y);
                        g3z = ai2 * (ai2 * hrr_2001z - 1 * hrr_0001z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1210y - 1 * trr_11y);
                        g3z = aj2 * (aj2 * hrr_0201z - 1 * hrr_0001z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_1101z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * hrr_2001z;
                        g2z -= 1 * hrr_0001z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_2101z;
                        g3z -= 1 * hrr_0101z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * hrr_3001z - 3 * hrr_1001z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_1201z - 1 * hrr_1001z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0100y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        g1z = aj2 * hrr_0111z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_1011z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        g3z = ai2 * hrr_1111z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_2011z - 1 * hrr_0011z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0211z = hrr_1111z - zjzi * hrr_0111z;
                        g3z = aj2 * (aj2 * hrr_0211z - 1 * hrr_0011z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0111z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * hrr_1011z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1111z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * hrr_2011z - 1 * hrr_0011z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0211z - 1 * hrr_0011z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1111z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_2011z;
                        g2z -= 1 * hrr_0011z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        double hrr_3011z = trr_32z - zlzk * trr_31z;
                        double hrr_2111z = hrr_3011z - zjzi * hrr_2011z;
                        g3z = ai2 * hrr_2111z;
                        g3z -= 1 * hrr_0111z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_3011z - 3 * hrr_1011z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1211z = hrr_2111z - zjzi * hrr_1111z;
                        g3z = aj2 * (aj2 * hrr_1211z - 1 * hrr_1011z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double hrr_1021x = trr_13x - xlxk * trr_12x;
                        double hrr_1012x = hrr_1021x - xlxk * hrr_1011x;
                        g1x = al2 * hrr_1012x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1y = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1z = al2 * hrr_0001z;
                        g1x -= 1 * trr_11x;
                        g2x = ak2 * hrr_1021x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_01z;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g2x -= 1 * hrr_1001x;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double trr_14x = cpx * trr_13x + 3*b01 * trr_12x + 1*b00 * trr_03x;
                        double hrr_1031x = trr_14x - xlxk * trr_13x;
                        double hrr_1022x = hrr_1031x - xlxk * hrr_1021x;
                        g3x = ak2 * hrr_1022x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3y = ak2 * hrr_0011y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3z = ak2 * hrr_0011z;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        g3x -= 1 * hrr_1002x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_12x - 1 * trr_10x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_1031x - 3 * hrr_1011x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_1013x = hrr_1022x - xlxk * hrr_1012x;
                        g3x = al2 * (al2 * hrr_1013x - 3 * hrr_1011x);
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
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
                        double hrr_0021x = trr_03x - xlxk * trr_02x;
                        double hrr_0012x = hrr_0021x - xlxk * hrr_0011x;
                        g1x = al2 * hrr_0012x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0001z;
                        g1x -= 1 * trr_01x;
                        g2x = ak2 * hrr_0021x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_01z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g2x -= 1 * hrr_0001x;
                        double trr_04x = cpx * trr_03x + 3*b01 * trr_02x;
                        double hrr_0031x = trr_04x - xlxk * trr_03x;
                        double hrr_0022x = hrr_0031x - xlxk * hrr_0021x;
                        g3x = ak2 * hrr_0022x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0011z;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        g3x -= 1 * hrr_0002x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_02x - 1 * fac);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0031x - 3 * hrr_0011x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0013x = hrr_0022x - xlxk * hrr_0012x;
                        g3x = al2 * (al2 * hrr_0013x - 3 * hrr_0011x);
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
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
                        g1x = al2 * hrr_0012x;
                        g1y = al2 * hrr_0001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g1z = al2 * hrr_1001z;
                        g1x -= 1 * trr_01x;
                        g2x = ak2 * hrr_0021x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_11z;
                        g2x -= 1 * hrr_0001x;
                        g3x = ak2 * hrr_0022x;
                        g3y = ak2 * hrr_0011y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3z = ak2 * hrr_1011z;
                        g3x -= 1 * hrr_0002x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_02x - 1 * fac);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0031x - 3 * hrr_0011x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0013x - 3 * hrr_0011x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1002x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_0001z;
                        g1x -= 1 * trr_10x;
                        g2x = ak2 * hrr_1011x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_1012x;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double hrr_0021y = trr_03y - ylyk * trr_02y;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_11x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_1021x - 1 * hrr_1001x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_1003x = hrr_1012x - xlxk * hrr_1002x;
                        g3x = al2 * (al2 * hrr_1003x - 3 * hrr_1001x);
                        double hrr_0012y = hrr_0021y - ylyk * hrr_0011y;
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0002x;
                        g1y = al2 * hrr_1011y;
                        g1z = al2 * hrr_0001z;
                        g1x -= 1 * fac;
                        g2x = ak2 * hrr_0011x;
                        g2y = ak2 * trr_12y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * trr_10y;
                        g3x = ak2 * hrr_0012x;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double hrr_1021y = trr_13y - ylyk * trr_12y;
                        g3y = ak2 * hrr_1021y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_1001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_01x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0021x - 1 * hrr_0001x);
                        g3y = ak2 * (ak2 * trr_13y - 3 * trr_11y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0003x = hrr_0012x - xlxk * hrr_0002x;
                        g3x = al2 * (al2 * hrr_0003x - 3 * hrr_0001x);
                        double hrr_1012y = hrr_1021y - ylyk * hrr_1011y;
                        g3y = al2 * (al2 * hrr_1012y - 1 * trr_11y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0002x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_1001z;
                        g1x -= 1 * fac;
                        g2x = ak2 * hrr_0011x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * trr_11z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_0012x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_1011z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_01x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0021x - 1 * hrr_0001x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0003x - 3 * hrr_0001x);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1002x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_0011z;
                        g1x -= 1 * trr_10x;
                        g2x = ak2 * hrr_1011x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_1012x;
                        g3y = ak2 * hrr_0011y;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double hrr_0021z = trr_03z - zlzk * trr_02z;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_11x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_1021x - 1 * hrr_1001x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1003x - 3 * hrr_1001x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0012z = hrr_0021z - zlzk * hrr_0011z;
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0002x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0011z;
                        g1x -= 1 * fac;
                        g2x = ak2 * hrr_0011x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_0012x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_01x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0021x - 1 * hrr_0001x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0003x - 3 * hrr_0001x);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0002x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_1011z;
                        g1x -= 1 * fac;
                        g2x = ak2 * hrr_0011x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_12z;
                        g2z -= 1 * trr_10z;
                        g3x = ak2 * hrr_0012x;
                        g3y = ak2 * hrr_0011y;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double hrr_1021z = trr_13z - zlzk * trr_12z;
                        g3z = ak2 * hrr_1021z;
                        g3z -= 1 * hrr_1001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3x -= 1 * (ak2 * trr_01x);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0021x - 1 * hrr_0001x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_13z - 3 * trr_11z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0003x - 3 * hrr_0001x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1012z = hrr_1021z - zlzk * hrr_1011z;
                        g3z = al2 * (al2 * hrr_1012z - 1 * trr_11z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1011x;
                        g1y = al2 * hrr_0002y;
                        g1z = al2 * hrr_0001z;
                        g1y -= 1 * 1;
                        g2x = ak2 * trr_12x;
                        g2y = ak2 * hrr_0011y;
                        g2z = ak2 * trr_01z;
                        g2x -= 1 * trr_10x;
                        g3x = ak2 * hrr_1021x;
                        g3y = ak2 * hrr_0012y;
                        g3z = ak2 * hrr_0011z;
                        g3x -= 1 * hrr_1001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_01y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_13x - 3 * trr_11x);
                        g3y = ak2 * (ak2 * hrr_0021y - 1 * hrr_0001y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1012x - 1 * trr_11x);
                        double hrr_0003y = hrr_0012y - ylyk * hrr_0002y;
                        g3y = al2 * (al2 * hrr_0003y - 3 * hrr_0001y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_1002y;
                        g1z = al2 * hrr_0001z;
                        g1y -= 1 * trr_10y;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * hrr_1011y;
                        g2z = ak2 * trr_01z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_1012y;
                        g3z = ak2 * hrr_0011z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_11y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * hrr_1021y - 1 * hrr_1001y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        double hrr_1003y = hrr_1012y - ylyk * hrr_1002y;
                        g3y = al2 * (al2 * hrr_1003y - 3 * hrr_1001y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_0002y;
                        g1z = al2 * hrr_1001z;
                        g1y -= 1 * 1;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * hrr_0011y;
                        g2z = ak2 * trr_11z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_0012y;
                        g3z = ak2 * hrr_1011z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_01y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * hrr_0021y - 1 * hrr_0001y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_0003y - 3 * hrr_0001y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0012y;
                        g1z = al2 * hrr_0001z;
                        g1y -= 1 * trr_01y;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * hrr_0021y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * hrr_0001y;
                        g3x = ak2 * hrr_1011x;
                        double trr_04y = cpy * trr_03y + 3*b01 * trr_02y;
                        double hrr_0031y = trr_04y - ylyk * trr_03y;
                        double hrr_0022y = hrr_0031y - ylyk * hrr_0021y;
                        g3y = ak2 * hrr_0022y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_0002y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_02y - 1 * 1);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * hrr_0031y - 3 * hrr_0011y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        double hrr_0013y = hrr_0022y - ylyk * hrr_0012y;
                        g3y = al2 * (al2 * hrr_0013y - 3 * hrr_0011y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1012y;
                        g1z = al2 * hrr_0001z;
                        g1y -= 1 * trr_11y;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_1021y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * hrr_1001y;
                        g3x = ak2 * hrr_0011x;
                        double trr_14y = cpy * trr_13y + 3*b01 * trr_12y + 1*b00 * trr_03y;
                        double hrr_1031y = trr_14y - ylyk * trr_13y;
                        double hrr_1022y = hrr_1031y - ylyk * hrr_1021y;
                        g3y = ak2 * hrr_1022y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_1002y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_12y - 1 * trr_10y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_1031y - 3 * hrr_1011y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        double hrr_1013y = hrr_1022y - ylyk * hrr_1012y;
                        g3y = al2 * (al2 * hrr_1013y - 3 * hrr_1011y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0012y;
                        g1z = al2 * hrr_1001z;
                        g1y -= 1 * trr_01y;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_0021y;
                        g2z = ak2 * trr_11z;
                        g2y -= 1 * hrr_0001y;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0022y;
                        g3z = ak2 * hrr_1011z;
                        g3y -= 1 * hrr_0002y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_02y - 1 * 1);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_0031y - 3 * hrr_0011y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0013y - 3 * hrr_0011y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0002y;
                        g1z = al2 * hrr_0011z;
                        g1y -= 1 * 1;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * hrr_0011y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0012y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_01y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * hrr_0021y - 1 * hrr_0001y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0003y - 3 * hrr_0001y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1002y;
                        g1z = al2 * hrr_0011z;
                        g1y -= 1 * trr_10y;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_1011y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1012y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_11y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_1021y - 1 * hrr_1001y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1003y - 3 * hrr_1001y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0002y;
                        g1z = al2 * hrr_1011z;
                        g1y -= 1 * 1;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_0011y;
                        g2z = ak2 * trr_12z;
                        g2z -= 1 * trr_10z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0012y;
                        g3z = ak2 * hrr_1021z;
                        g3z -= 1 * hrr_1001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3y -= 1 * (ak2 * trr_01y);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_0021y - 1 * hrr_0001y);
                        g3z = ak2 * (ak2 * trr_13z - 3 * trr_11z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0003y - 3 * hrr_0001y);
                        g3z = al2 * (al2 * hrr_1012z - 1 * trr_11z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1011x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_0002z;
                        g1z -= 1 * wt;
                        g2x = ak2 * trr_12x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * hrr_0011z;
                        g2x -= 1 * trr_10x;
                        g3x = ak2 * hrr_1021x;
                        g3y = ak2 * hrr_0011y;
                        g3z = ak2 * hrr_0012z;
                        g3x -= 1 * hrr_1001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_01z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_13x - 3 * trr_11x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * hrr_0021z - 1 * hrr_0001z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1012x - 1 * trr_11x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0003z = hrr_0012z - zlzk * hrr_0002z;
                        g3z = al2 * (al2 * hrr_0003z - 3 * hrr_0001z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0002z;
                        g1z -= 1 * wt;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * hrr_0011z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0012z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_01z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * hrr_0021z - 1 * hrr_0001z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0003z - 3 * hrr_0001z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_1002z;
                        g1z -= 1 * trr_10z;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * hrr_1011z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_0011y;
                        g3z = ak2 * hrr_1012z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_11z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * hrr_1021z - 1 * hrr_1001z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1003z = hrr_1012z - zlzk * hrr_1002z;
                        g3z = al2 * (al2 * hrr_1003z - 3 * hrr_1001z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_0002z;
                        g1z -= 1 * wt;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * hrr_0011z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_0012z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_01z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * hrr_0021z - 1 * hrr_0001z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_0003z - 3 * hrr_0001z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1011y;
                        g1z = al2 * hrr_0002z;
                        g1z -= 1 * wt;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_12y;
                        g2z = ak2 * hrr_0011z;
                        g2y -= 1 * trr_10y;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1021y;
                        g3z = ak2 * hrr_0012z;
                        g3y -= 1 * hrr_1001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_01z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_13y - 3 * trr_11y);
                        g3z = ak2 * (ak2 * hrr_0021z - 1 * hrr_0001z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1012y - 1 * trr_11y);
                        g3z = al2 * (al2 * hrr_0003z - 3 * hrr_0001z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_1002z;
                        g1z -= 1 * trr_10z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * hrr_1011z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_1012z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_11z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * hrr_1021z - 1 * hrr_1001z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_1003z - 3 * hrr_1001z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_0012z;
                        g1z -= 1 * trr_01z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * hrr_0021z;
                        g2z -= 1 * hrr_0001z;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_04z = cpz * trr_03z + 3*b01 * trr_02z;
                        double hrr_0031z = trr_04z - zlzk * trr_03z;
                        double hrr_0022z = hrr_0031z - zlzk * hrr_0021z;
                        g3z = ak2 * hrr_0022z;
                        g3z -= 1 * hrr_0002z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_02z - 1 * wt);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * hrr_0031z - 3 * hrr_0011z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0013z = hrr_0022z - zlzk * hrr_0012z;
                        g3z = al2 * (al2 * hrr_0013z - 3 * hrr_0011z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0012z;
                        g1z -= 1 * trr_01z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * hrr_0021z;
                        g2z -= 1 * hrr_0001z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0022z;
                        g3z -= 1 * hrr_0002z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_02z - 1 * wt);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * hrr_0031z - 3 * hrr_0011z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0013z - 3 * hrr_0011z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_1012z;
                        g1z -= 1 * trr_11z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * hrr_1021z;
                        g2z -= 1 * hrr_1001z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_14z = cpz * trr_13z + 3*b01 * trr_12z + 1*b00 * trr_03z;
                        double hrr_1031z = trr_14z - zlzk * trr_13z;
                        double hrr_1022z = hrr_1031z - zlzk * hrr_1021z;
                        g3z = ak2 * hrr_1022z;
                        g3z -= 1 * hrr_1002z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        g3z -= 1 * (ak2 * trr_12z - 1 * trr_10z);
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * hrr_1031z - 3 * hrr_1011z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1013z = hrr_1022z - zlzk * hrr_1012z;
                        g3z = al2 * (al2 * hrr_1013z - 3 * hrr_1011z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
        atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
        atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
        atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
        atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
        atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
        atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
        atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
        atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
        atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
        atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
        atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
        atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
        atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
        atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
        atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
        atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
    }
}
__global__
void rys_ejk_ip2_type12_1011(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type12_1011(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type12_1100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double g1x, g1y, g1z;
        double g2x, g2y, g2z;
        double g3x, g3y, g3z;
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        
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
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        g1x = aj2 * hrr_1200x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        g1y = aj2 * hrr_0100y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        g1z = aj2 * hrr_0100z;
                        g1x -= 1 * trr_10x;
                        g2x = ai2 * hrr_2100x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g2x -= 1 * hrr_0100x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                        g3x = ai2 * hrr_2200x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        g3y = ai2 * hrr_1100y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        g3x -= 1 * hrr_0200x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_20x - 1 * fac);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3100x - 3 * hrr_1100x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1300x = hrr_2200x - xjxi * hrr_1200x;
                        g3x = aj2 * (aj2 * hrr_1300x - 3 * hrr_1100x);
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0200x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0100z;
                        g1x -= 1 * fac;
                        g2x = ai2 * hrr_1100x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1200x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_10x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0300x = hrr_1200x - xjxi * hrr_0200x;
                        g3x = aj2 * (aj2 * hrr_0300x - 3 * hrr_0100x);
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0200x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1100z;
                        g1x -= 1 * fac;
                        g2x = ai2 * hrr_1100x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1200x;
                        g3y = ai2 * hrr_1100y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_10x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0300x - 3 * hrr_0100x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0200y;
                        g1z = aj2 * hrr_0100z;
                        g1y -= 1 * 1;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * hrr_1100y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1200y;
                        g3z = ai2 * hrr_1100z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_10y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        double hrr_0300y = hrr_1200y - yjyi * hrr_0200y;
                        g3y = aj2 * (aj2 * hrr_0300y - 3 * hrr_0100y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1200y;
                        g1z = aj2 * hrr_0100z;
                        g1y -= 1 * trr_10y;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_2100y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * hrr_0100y;
                        g3x = ai2 * hrr_1100x;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                        g3y = ai2 * hrr_2200y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0200y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_20y - 1 * 1);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_3100y - 3 * hrr_1100y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        double hrr_1300y = hrr_2200y - yjyi * hrr_1200y;
                        g3y = aj2 * (aj2 * hrr_1300y - 3 * hrr_1100y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0200y;
                        g1z = aj2 * hrr_1100z;
                        g1y -= 1 * 1;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_1100y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1200y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_10y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0300y - 3 * hrr_0100y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_0200z;
                        g1z -= 1 * wt;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_1100z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1100y;
                        g3z = ai2 * hrr_1200z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_10z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0300z = hrr_1200z - zjzi * hrr_0200z;
                        g3z = aj2 * (aj2 * hrr_0300z - 3 * hrr_0100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0200z;
                        g1z -= 1 * wt;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * hrr_1100z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1200z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_10z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0300z - 3 * hrr_0100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1200z;
                        g1z -= 1 * trr_10z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_2100z;
                        g2z -= 1 * hrr_0100z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                        g3z = ai2 * hrr_2200z;
                        g3z -= 1 * hrr_0200z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_20z - 1 * wt);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_3100z - 3 * hrr_1100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1300z = hrr_2200z - zjzi * hrr_1200z;
                        g3z = aj2 * (aj2 * hrr_1300z - 3 * hrr_1100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        g1x = al2 * hrr_1101x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1y = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1z = al2 * hrr_0001z;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        g2x = ak2 * hrr_1110x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_01z;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        double trr_01x = cpx * fac;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        g3x = ak2 * hrr_1111x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3y = ak2 * hrr_0011y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        double hrr_1120x = trr_22x - xjxi * trr_12x;
                        g3x = ak2 * (ak2 * hrr_1120x - 1 * hrr_1100x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_2002x = hrr_2011x - xlxk * hrr_2001x;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        double hrr_1102x = hrr_2002x - xjxi * hrr_1002x;
                        g3x = al2 * (al2 * hrr_1102x - 1 * hrr_1100x);
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
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
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        g1x = al2 * hrr_0101x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0001z;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        g2x = ak2 * hrr_0110x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_01z;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        g3x = ak2 * hrr_0111x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        double hrr_0120x = trr_12x - xjxi * trr_02x;
                        g3x = ak2 * (ak2 * hrr_0120x - 1 * hrr_0100x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        double hrr_0102x = hrr_1002x - xjxi * hrr_0002x;
                        g3x = al2 * (al2 * hrr_0102x - 1 * hrr_0100x);
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
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
                        g1x = al2 * hrr_0101x;
                        g1y = al2 * hrr_0001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * hrr_0110x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_11z;
                        g3x = ak2 * hrr_0111x;
                        g3y = ak2 * hrr_0011y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3z = ak2 * hrr_1011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0120x - 1 * hrr_0100x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0102x - 1 * hrr_0100x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0100y = trr_10y - yjyi * 1;
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
                        g1x = al2 * hrr_1001x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        g1y = al2 * hrr_0101y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_11x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        g2y = ak2 * hrr_0110y;
                        g2z = ak2 * trr_01z;
                        g3x = ak2 * hrr_1011x;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        g3y = ak2 * hrr_0111y;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        double hrr_0120y = trr_12y - yjyi * trr_02y;
                        g3y = ak2 * (ak2 * hrr_0120y - 1 * hrr_0100y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        double hrr_0102y = hrr_1002y - yjyi * hrr_0002y;
                        g3y = al2 * (al2 * hrr_0102y - 1 * hrr_0100y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
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
                        g1x = al2 * hrr_0001x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        g1y = al2 * hrr_1101y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_01x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        g2y = ak2 * hrr_1110y;
                        g2z = ak2 * trr_01z;
                        g3x = ak2 * hrr_0011x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        g3y = ak2 * hrr_1111y;
                        g3z = ak2 * hrr_0011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        double hrr_1120y = trr_22y - yjyi * trr_12y;
                        g3y = ak2 * (ak2 * hrr_1120y - 1 * hrr_1100y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        double hrr_2002y = hrr_2011y - ylyk * hrr_2001y;
                        double hrr_1102y = hrr_2002y - yjyi * hrr_1002y;
                        g3y = al2 * (al2 * hrr_1102y - 1 * hrr_1100y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0101y;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_0110y;
                        g2z = ak2 * trr_11z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0111y;
                        g3z = ak2 * hrr_1011z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_0120y - 1 * hrr_0100y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0102y - 1 * hrr_0100y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0100z = trr_10z - zjzi * wt;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0001y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        g1z = al2 * hrr_0101z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_01y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        g2z = ak2 * hrr_0110z;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0011y;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        g3z = ak2 * hrr_0111z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        double hrr_0120z = trr_12z - zjzi * trr_02z;
                        g3z = ak2 * (ak2 * hrr_0120z - 1 * hrr_0100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0102z = hrr_1002z - zjzi * hrr_0002z;
                        g3z = al2 * (al2 * hrr_0102z - 1 * hrr_0100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0101z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * hrr_0110z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0111z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * hrr_0120z - 1 * hrr_0100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0102z - 1 * hrr_0100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0001y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        g1z = al2 * hrr_1101z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_01y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        g2z = ak2 * hrr_1110z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        g3z = ak2 * hrr_1111z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        double hrr_1120z = trr_22z - zjzi * trr_12z;
                        g3z = ak2 * (ak2 * hrr_1120z - 1 * hrr_1100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_2002z = hrr_2011z - zlzk * hrr_2001z;
                        double hrr_1102z = hrr_2002z - zjzi * hrr_1002z;
                        g3z = al2 * (al2 * hrr_1102z - 1 * hrr_1100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
        atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
        atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
        atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
        atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
        atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
        atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
        atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
        atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
        atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
        atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
        atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
        atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
        atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
        atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
        atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
        atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
    }
}
__global__
void rys_ejk_ip2_type12_1100(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type12_1100(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_type12_1110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
        double g1x, g1y, g1z;
        double g2x, g2y, g2z;
        double g3x, g3y, g3z;
        double v_ixx = 0;
        double v_ixy = 0;
        double v_ixz = 0;
        double v_iyy = 0;
        double v_iyz = 0;
        double v_izz = 0;
        double v_jxx = 0;
        double v_jxy = 0;
        double v_jxz = 0;
        double v_jyy = 0;
        double v_jyz = 0;
        double v_jzz = 0;
        double v_kxx = 0;
        double v_kxy = 0;
        double v_kxz = 0;
        double v_kyy = 0;
        double v_kyz = 0;
        double v_kzz = 0;
        double v_lxx = 0;
        double v_lxy = 0;
        double v_lxz = 0;
        double v_lyy = 0;
        double v_lyz = 0;
        double v_lzz = 0;
        double v1xx = 0;
        double v1xy = 0;
        double v1xz = 0;
        double v1yx = 0;
        double v1yy = 0;
        double v1yz = 0;
        double v1zx = 0;
        double v1zy = 0;
        double v1zz = 0;
        double v2xx = 0;
        double v2xy = 0;
        double v2xz = 0;
        double v2yx = 0;
        double v2yy = 0;
        double v2yz = 0;
        double v2zx = 0;
        double v2zy = 0;
        double v2zz = 0;
        
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
                        double hrr_1210x = hrr_2110x - xjxi * hrr_1110x;
                        g1x = aj2 * hrr_1210x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - yjyi * 1;
                        g1y = aj2 * hrr_0100y;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        g1z = aj2 * hrr_0100z;
                        g1x -= 1 * trr_11x;
                        g2x = ai2 * hrr_2110x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_10z;
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        g2x -= 1 * hrr_0110x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double trr_41x = cpx * trr_40x + 4*b00 * trr_30x;
                        double hrr_3110x = trr_41x - xjxi * trr_31x;
                        double hrr_2210x = hrr_3110x - xjxi * hrr_2110x;
                        g3x = ai2 * hrr_2210x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        g3y = ai2 * hrr_1100y;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0210x = hrr_1110x - xjxi * hrr_0110x;
                        g3x -= 1 * hrr_0210x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_21x - 1 * trr_01x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3110x - 3 * hrr_1110x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1310x = hrr_2210x - xjxi * hrr_1210x;
                        g3x = aj2 * (aj2 * hrr_1310x - 3 * hrr_1110x);
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0210x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0100z;
                        g1x -= 1 * trr_01x;
                        g2x = ai2 * hrr_1110x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1210x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_11x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2110x - 1 * hrr_0110x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0310x = hrr_1210x - xjxi * hrr_0210x;
                        g3x = aj2 * (aj2 * hrr_0310x - 3 * hrr_0110x);
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0210x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1100z;
                        g1x -= 1 * trr_01x;
                        g2x = ai2 * hrr_1110x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1210x;
                        g3y = ai2 * hrr_1100y;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_11x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2110x - 1 * hrr_0110x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0310x - 3 * hrr_0110x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1110x;
                        g1y = aj2 * hrr_0200y;
                        g1z = aj2 * hrr_0100z;
                        g1y -= 1 * 1;
                        g2x = ai2 * trr_21x;
                        g2y = ai2 * hrr_1100y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * trr_01x;
                        g3x = ai2 * hrr_2110x;
                        g3y = ai2 * hrr_1200y;
                        g3z = ai2 * hrr_1100z;
                        g3x -= 1 * hrr_0110x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_10y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_31x - 3 * trr_11x);
                        g3y = ai2 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1210x - 1 * trr_11x);
                        double hrr_0300y = hrr_1200y - yjyi * hrr_0200y;
                        g3y = aj2 * (aj2 * hrr_0300y - 3 * hrr_0100y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_1200y;
                        g1z = aj2 * hrr_0100z;
                        g1y -= 1 * trr_10y;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * hrr_2100y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * hrr_0100y;
                        g3x = ai2 * hrr_1110x;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                        g3y = ai2 * hrr_2200y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0200y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_20y - 1 * 1);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * hrr_3100y - 3 * hrr_1100y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        double hrr_1300y = hrr_2200y - yjyi * hrr_1200y;
                        g3y = aj2 * (aj2 * hrr_1300y - 3 * hrr_1100y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_0200y;
                        g1z = aj2 * hrr_1100z;
                        g1y -= 1 * 1;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * hrr_1100y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_1200y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_10y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_0300y - 3 * hrr_0100y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1110x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_0200z;
                        g1z -= 1 * wt;
                        g2x = ai2 * trr_21x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_1100z;
                        g2x -= 1 * trr_01x;
                        g3x = ai2 * hrr_2110x;
                        g3y = ai2 * hrr_1100y;
                        g3z = ai2 * hrr_1200z;
                        g3x -= 1 * hrr_0110x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_10z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_31x - 3 * trr_11x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1210x - 1 * trr_11x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0300z = hrr_1200z - zjzi * hrr_0200z;
                        g3z = aj2 * (aj2 * hrr_0300z - 3 * hrr_0100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0200z;
                        g1z -= 1 * wt;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * hrr_1100z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1200z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_10z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0300z - 3 * hrr_0100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0110x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1200z;
                        g1z -= 1 * trr_10z;
                        g2x = ai2 * trr_11x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_2100z;
                        g2z -= 1 * hrr_0100z;
                        g3x = ai2 * hrr_1110x;
                        g3y = ai2 * hrr_1100y;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                        g3z = ai2 * hrr_2200z;
                        g3z -= 1 * hrr_0200z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_20z - 1 * wt);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_21x - 1 * trr_01x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_3100z - 3 * hrr_1100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0210x - 1 * trr_01x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1300z = hrr_2200z - zjzi * hrr_1200z;
                        g3z = aj2 * (aj2 * hrr_1300z - 3 * hrr_1100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
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
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        g1x = aj2 * hrr_1200x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_0100z;
                        g1x -= 1 * trr_10x;
                        g2x = ai2 * hrr_2100x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g2x -= 1 * hrr_0100x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                        g3x = ai2 * hrr_2200x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_1100z;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        g3x -= 1 * hrr_0200x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_20x - 1 * fac);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3100x - 3 * hrr_1100x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_1300x = hrr_2200x - xjxi * hrr_1200x;
                        g3x = aj2 * (aj2 * hrr_1300x - 3 * hrr_1100x);
                        double hrr_0210y = hrr_1110y - yjyi * hrr_0110y;
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0200x;
                        g1y = aj2 * hrr_1110y;
                        g1z = aj2 * hrr_0100z;
                        g1x -= 1 * fac;
                        g2x = ai2 * hrr_1100x;
                        g2y = ai2 * trr_21y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * trr_01y;
                        g3x = ai2 * hrr_1200x;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        g3y = ai2 * hrr_2110y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0110y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_10x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        g3y = ai2 * (ai2 * trr_31y - 3 * trr_11y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        double hrr_0300x = hrr_1200x - xjxi * hrr_0200x;
                        g3x = aj2 * (aj2 * hrr_0300x - 3 * hrr_0100x);
                        double hrr_1210y = hrr_2110y - yjyi * hrr_1110y;
                        g3y = aj2 * (aj2 * hrr_1210y - 1 * trr_11y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0200x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_1100z;
                        g1x -= 1 * fac;
                        g2x = ai2 * hrr_1100x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1200x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_10x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0300x - 3 * hrr_0100x);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0210y;
                        g1z = aj2 * hrr_0100z;
                        g1y -= 1 * trr_01y;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * hrr_1110y;
                        g2z = ai2 * trr_10z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1210y;
                        g3z = ai2 * hrr_1100z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_11y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * hrr_2110y - 1 * hrr_0110y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        double hrr_0310y = hrr_1210y - yjyi * hrr_0210y;
                        g3y = aj2 * (aj2 * hrr_0310y - 3 * hrr_0110y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1210y;
                        g1z = aj2 * hrr_0100z;
                        g1y -= 1 * trr_11y;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_2110y;
                        g2z = ai2 * trr_10z;
                        g2y -= 1 * hrr_0110y;
                        g3x = ai2 * hrr_1100x;
                        double trr_41y = cpy * trr_40y + 4*b00 * trr_30y;
                        double hrr_3110y = trr_41y - yjyi * trr_31y;
                        double hrr_2210y = hrr_3110y - yjyi * hrr_2110y;
                        g3y = ai2 * hrr_2210y;
                        g3z = ai2 * hrr_1100z;
                        g3y -= 1 * hrr_0210y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_21y - 1 * trr_01y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_3110y - 3 * hrr_1110y);
                        g3z = ai2 * (ai2 * trr_20z - 1 * wt);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        double hrr_1310y = hrr_2210y - yjyi * hrr_1210y;
                        g3y = aj2 * (aj2 * hrr_1310y - 3 * hrr_1110y);
                        g3z = aj2 * (aj2 * hrr_0200z - 1 * wt);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0210y;
                        g1z = aj2 * hrr_1100z;
                        g1y -= 1 * trr_01y;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_1110y;
                        g2z = ai2 * trr_20z;
                        g2z -= 1 * wt;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1210y;
                        g3z = ai2 * hrr_2100z;
                        g3z -= 1 * hrr_0100z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_11y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_2110y - 1 * hrr_0110y);
                        g3z = ai2 * (ai2 * trr_30z - 3 * trr_10z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0310y - 3 * hrr_0110y);
                        g3z = aj2 * (aj2 * hrr_1200z - 1 * trr_10z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_0200z;
                        g1z -= 1 * wt;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * hrr_1100z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_1200z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_10z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_0300z - 3 * hrr_0100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1110y;
                        g1z = aj2 * hrr_0200z;
                        g1z -= 1 * wt;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_21y;
                        g2z = ai2 * hrr_1100z;
                        g2y -= 1 * trr_01y;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2110y;
                        g3z = ai2 * hrr_1200z;
                        g3y -= 1 * hrr_0110y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_10z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_31y - 3 * trr_11y);
                        g3z = ai2 * (ai2 * hrr_2100z - 1 * hrr_0100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1210y - 1 * trr_11y);
                        g3z = aj2 * (aj2 * hrr_0300z - 3 * hrr_0100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0110y;
                        g1z = aj2 * hrr_1200z;
                        g1z -= 1 * trr_10z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_11y;
                        g2z = ai2 * hrr_2100z;
                        g2z -= 1 * hrr_0100z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1110y;
                        g3z = ai2 * hrr_2200z;
                        g3z -= 1 * hrr_0200z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_20z - 1 * wt);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_21y - 1 * trr_01y);
                        g3z = ai2 * (ai2 * hrr_3100z - 3 * hrr_1100z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0210y - 1 * trr_01y);
                        g3z = aj2 * (aj2 * hrr_1300z - 3 * hrr_1100z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
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
                        g1x = aj2 * hrr_1200x;
                        g1y = aj2 * hrr_0100y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        g1z = aj2 * hrr_0110z;
                        g1x -= 1 * trr_10x;
                        g2x = ai2 * hrr_2100x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_11z;
                        g2x -= 1 * hrr_0100x;
                        g3x = ai2 * hrr_2200x;
                        g3y = ai2 * hrr_1100y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        g3z = ai2 * hrr_1110z;
                        g3x -= 1 * hrr_0200x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_20x - 1 * fac);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_3100x - 3 * hrr_1100x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1300x - 3 * hrr_1100x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0210z = hrr_1110z - zjzi * hrr_0110z;
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0200x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0110z;
                        g1x -= 1 * fac;
                        g2x = ai2 * hrr_1100x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * trr_11z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1200x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1110z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_10x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0300x - 3 * hrr_0100x);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0200x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1110z;
                        g1x -= 1 * fac;
                        g2x = ai2 * hrr_1100x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * trr_21z;
                        g2z -= 1 * trr_01z;
                        g3x = ai2 * hrr_1200x;
                        g3y = ai2 * hrr_1100y;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        g3z = ai2 * hrr_2110z;
                        g3z -= 1 * hrr_0110z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3x -= 1 * (ai2 * trr_10x);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * hrr_2100x - 1 * hrr_0100x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * trr_31z - 3 * trr_11z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0300x - 3 * hrr_0100x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1210z = hrr_2110z - zjzi * hrr_1110z;
                        g3z = aj2 * (aj2 * hrr_1210z - 1 * trr_11z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0200y;
                        g1z = aj2 * hrr_0110z;
                        g1y -= 1 * 1;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * hrr_1100y;
                        g2z = ai2 * trr_11z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1200y;
                        g3z = ai2 * hrr_1110z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_10y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0300y - 3 * hrr_0100y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1200y;
                        g1z = aj2 * hrr_0110z;
                        g1y -= 1 * trr_10y;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_2100y;
                        g2z = ai2 * trr_11z;
                        g2y -= 1 * hrr_0100y;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2200y;
                        g3z = ai2 * hrr_1110z;
                        g3y -= 1 * hrr_0200y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_20y - 1 * 1);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_3100y - 3 * hrr_1100y);
                        g3z = ai2 * (ai2 * trr_21z - 1 * trr_01z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1300y - 3 * hrr_1100y);
                        g3z = aj2 * (aj2 * hrr_0210z - 1 * trr_01z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0200y;
                        g1z = aj2 * hrr_1110z;
                        g1y -= 1 * 1;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * hrr_1100y;
                        g2z = ai2 * trr_21z;
                        g2z -= 1 * trr_01z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1200y;
                        g3z = ai2 * hrr_2110z;
                        g3z -= 1 * hrr_0110z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3y -= 1 * (ai2 * trr_10y);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * hrr_2100y - 1 * hrr_0100y);
                        g3z = ai2 * (ai2 * trr_31z - 3 * trr_11z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0300y - 3 * hrr_0100y);
                        g3z = aj2 * (aj2 * hrr_1210z - 1 * trr_11z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_1100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_0210z;
                        g1z -= 1 * trr_01z;
                        g2x = ai2 * trr_20x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_1110z;
                        g2x -= 1 * fac;
                        g3x = ai2 * hrr_2100x;
                        g3y = ai2 * hrr_1100y;
                        g3z = ai2 * hrr_1210z;
                        g3x -= 1 * hrr_0100x;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_11z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_30x - 3 * trr_10x);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_2110z - 1 * hrr_0110z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_1200x - 1 * trr_10x);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_0310z = hrr_1210z - zjzi * hrr_0210z;
                        g3z = aj2 * (aj2 * hrr_0310z - 3 * hrr_0110z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_1100y;
                        g1z = aj2 * hrr_0210z;
                        g1z -= 1 * trr_01z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_20y;
                        g2z = ai2 * hrr_1110z;
                        g2y -= 1 * 1;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_2100y;
                        g3z = ai2 * hrr_1210z;
                        g3y -= 1 * hrr_0100y;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_11z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_30y - 3 * trr_10y);
                        g3z = ai2 * (ai2 * hrr_2110z - 1 * hrr_0110z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_1200y - 1 * trr_10y);
                        g3z = aj2 * (aj2 * hrr_0310z - 3 * hrr_0110z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        g1x = aj2 * hrr_0100x;
                        g1y = aj2 * hrr_0100y;
                        g1z = aj2 * hrr_1210z;
                        g1z -= 1 * trr_11z;
                        g2x = ai2 * trr_10x;
                        g2y = ai2 * trr_10y;
                        g2z = ai2 * hrr_2110z;
                        g2z -= 1 * hrr_0110z;
                        g3x = ai2 * hrr_1100x;
                        g3y = ai2 * hrr_1100y;
                        double trr_41z = cpz * trr_40z + 4*b00 * trr_30z;
                        double hrr_3110z = trr_41z - zjzi * trr_31z;
                        double hrr_2210z = hrr_3110z - zjzi * hrr_2110z;
                        g3z = ai2 * hrr_2210z;
                        g3z -= 1 * hrr_0210z;
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        g3z -= 1 * (ai2 * trr_21z - 1 * trr_01z);
                        v1xx += g3x * prod_yz;
                        v1yy += g3y * prod_xz;
                        v1zz += g3z * prod_xy;
                        v1xy += g2x * g1y * Iz;
                        v1xz += g2x * g1z * Iy;
                        v1yx += g2y * g1x * Iz;
                        v1yz += g2y * g1z * Ix;
                        v1zx += g2z * g1x * Iy;
                        v1zy += g2z * g1y * Ix;
                        g3x = ai2 * (ai2 * trr_20x - 1 * fac);
                        g3y = ai2 * (ai2 * trr_20y - 1 * 1);
                        g3z = ai2 * (ai2 * hrr_3110z - 3 * hrr_1110z);
                        v_ixx += g3x * prod_yz;
                        v_iyy += g3y * prod_xz;
                        v_izz += g3z * prod_xy;
                        v_ixy += g2x * g2y * Iz;
                        v_ixz += g2x * g2z * Iy;
                        v_iyz += g2y * g2z * Ix;
                        g3x = aj2 * (aj2 * hrr_0200x - 1 * fac);
                        g3y = aj2 * (aj2 * hrr_0200y - 1 * 1);
                        double hrr_1310z = hrr_2210z - zjzi * hrr_1210z;
                        g3z = aj2 * (aj2 * hrr_1310z - 3 * hrr_1110z);
                        v_jxx += g3x * prod_yz;
                        v_jyy += g3y * prod_xz;
                        v_jzz += g3z * prod_xy;
                        v_jxy += g1x * g1y * Iz;
                        v_jxz += g1x * g1z * Iy;
                        v_jyz += g1y * g1z * Ix;
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
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        double trr_01x = cpx * fac;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        g1x = al2 * hrr_1111x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1y = al2 * hrr_0001y;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1z = al2 * hrr_0001z;
                        double hrr_1120x = trr_22x - xjxi * trr_12x;
                        g2x = ak2 * hrr_1120x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_01z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        g2x -= 1 * hrr_1100x;
                        double trr_23x = cpx * trr_22x + 2*b01 * trr_21x + 2*b00 * trr_12x;
                        double hrr_2021x = trr_23x - xlxk * trr_22x;
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double trr_13x = cpx * trr_12x + 2*b01 * trr_11x + 1*b00 * trr_02x;
                        double hrr_1021x = trr_13x - xlxk * trr_12x;
                        double hrr_1121x = hrr_2021x - xjxi * hrr_1021x;
                        g3x = ak2 * hrr_1121x;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3y = ak2 * hrr_0011y;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3z = ak2 * hrr_0011z;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        g3x -= 1 * hrr_1101x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        double hrr_1130x = trr_23x - xjxi * trr_13x;
                        g3x = ak2 * (ak2 * hrr_1130x - 3 * hrr_1110x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_2012x = hrr_2021x - xlxk * hrr_2011x;
                        double hrr_1012x = hrr_1021x - xlxk * hrr_1011x;
                        double hrr_1112x = hrr_2012x - xjxi * hrr_1012x;
                        g3x = al2 * (al2 * hrr_1112x - 1 * hrr_1110x);
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
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
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        g1x = al2 * hrr_0111x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0001z;
                        double hrr_0120x = trr_12x - xjxi * trr_02x;
                        g2x = ak2 * hrr_0120x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_01z;
                        double hrr_0100x = trr_10x - xjxi * fac;
                        g2x -= 1 * hrr_0100x;
                        double trr_03x = cpx * trr_02x + 2*b01 * trr_01x;
                        double hrr_0021x = trr_03x - xlxk * trr_02x;
                        double hrr_0121x = hrr_1021x - xjxi * hrr_0021x;
                        g3x = ak2 * hrr_0121x;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0011z;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        g3x -= 1 * hrr_0101x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        double hrr_0130x = trr_13x - xjxi * trr_03x;
                        g3x = ak2 * (ak2 * hrr_0130x - 3 * hrr_0110x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0012x = hrr_0021x - xlxk * hrr_0011x;
                        double hrr_0112x = hrr_1012x - xjxi * hrr_0012x;
                        g3x = al2 * (al2 * hrr_0112x - 1 * hrr_0110x);
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
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
                        g1x = al2 * hrr_0111x;
                        g1y = al2 * hrr_0001y;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * hrr_0120x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_11z;
                        g2x -= 1 * hrr_0100x;
                        g3x = ak2 * hrr_0121x;
                        g3y = ak2 * hrr_0011y;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3z = ak2 * hrr_1011z;
                        g3x -= 1 * hrr_0101x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0130x - 3 * hrr_0110x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0112x - 1 * hrr_0110x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0100y = trr_10y - yjyi * 1;
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
                        g1x = al2 * hrr_1011x;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        g1y = al2 * hrr_0101y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_12x;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        g2y = ak2 * hrr_0110y;
                        g2z = ak2 * trr_01z;
                        g2x -= 1 * trr_10x;
                        g3x = ak2 * hrr_1021x;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        g3y = ak2 * hrr_0111y;
                        g3z = ak2 * hrr_0011z;
                        g3x -= 1 * hrr_1001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_13x - 3 * trr_11x);
                        double hrr_0120y = trr_12y - yjyi * trr_02y;
                        g3y = ak2 * (ak2 * hrr_0120y - 1 * hrr_0100y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1012x - 1 * trr_11x);
                        double hrr_0102y = hrr_1002y - yjyi * hrr_0002y;
                        g3y = al2 * (al2 * hrr_0102y - 1 * hrr_0100y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
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
                        g1x = al2 * hrr_0011x;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        g1y = al2 * hrr_1101y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_02x;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        g2y = ak2 * hrr_1110y;
                        g2z = ak2 * trr_01z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        g3y = ak2 * hrr_1111y;
                        g3z = ak2 * hrr_0011z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        double hrr_1120y = trr_22y - yjyi * trr_12y;
                        g3y = ak2 * (ak2 * hrr_1120y - 1 * hrr_1100y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        double hrr_2002y = hrr_2011y - ylyk * hrr_2001y;
                        double hrr_1102y = hrr_2002y - yjyi * hrr_1002y;
                        g3y = al2 * (al2 * hrr_1102y - 1 * hrr_1100y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_0101y;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * hrr_0110y;
                        g2z = ak2 * trr_11z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_0111y;
                        g3z = ak2 * hrr_1011z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * hrr_0120y - 1 * hrr_0100y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_0102y - 1 * hrr_0100y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double hrr_0100z = trr_10z - zjzi * wt;
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
                        g1x = al2 * hrr_1011x;
                        g1y = al2 * hrr_0001y;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        g1z = al2 * hrr_0101z;
                        g2x = ak2 * trr_12x;
                        g2y = ak2 * trr_01y;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        g2z = ak2 * hrr_0110z;
                        g2x -= 1 * trr_10x;
                        g3x = ak2 * hrr_1021x;
                        g3y = ak2 * hrr_0011y;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        g3z = ak2 * hrr_0111z;
                        g3x -= 1 * hrr_1001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_13x - 3 * trr_11x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        double hrr_0120z = trr_12z - zjzi * trr_02z;
                        g3z = ak2 * (ak2 * hrr_0120z - 1 * hrr_0100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1012x - 1 * trr_11x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0102z = hrr_1002z - zjzi * hrr_0002z;
                        g3z = al2 * (al2 * hrr_0102z - 1 * hrr_0100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0101z;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * hrr_0110z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0111z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * hrr_0120z - 1 * hrr_0100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0102z - 1 * hrr_0100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
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
                        g1x = al2 * hrr_0011x;
                        g1y = al2 * hrr_0001y;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        g1z = al2 * hrr_1101z;
                        g2x = ak2 * trr_02x;
                        g2y = ak2 * trr_01y;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        g2z = ak2 * hrr_1110z;
                        g2x -= 1 * fac;
                        g3x = ak2 * hrr_0021x;
                        g3y = ak2 * hrr_0011y;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        g3z = ak2 * hrr_1111z;
                        g3x -= 1 * hrr_0001x;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_03x - 3 * trr_01x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        double hrr_1120z = trr_22z - zjzi * trr_12z;
                        g3z = ak2 * (ak2 * hrr_1120z - 1 * hrr_1100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0012x - 1 * trr_01x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_2002z = hrr_2011z - zlzk * hrr_2001z;
                        double hrr_1102z = hrr_2002z - zjzi * hrr_1002z;
                        g3z = al2 * (al2 * hrr_1102z - 1 * hrr_1100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1101x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * hrr_1110x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_1111x;
                        double trr_03y = cpy * trr_02y + 2*b01 * trr_01y;
                        double hrr_0021y = trr_03y - ylyk * trr_02y;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_1120x - 1 * hrr_1100x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_2002x = hrr_2011x - xlxk * hrr_2001x;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        double hrr_1102x = hrr_2002x - xjxi * hrr_1002x;
                        g3x = al2 * (al2 * hrr_1102x - 1 * hrr_1100x);
                        double hrr_0012y = hrr_0021y - ylyk * hrr_0011y;
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0101x;
                        g1y = al2 * hrr_1011y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * hrr_0110x;
                        g2y = ak2 * trr_12y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * trr_10y;
                        g3x = ak2 * hrr_0111x;
                        double trr_13y = cpy * trr_12y + 2*b01 * trr_11y + 1*b00 * trr_02y;
                        double hrr_1021y = trr_13y - ylyk * trr_12y;
                        g3y = ak2 * hrr_1021y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_1001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0120x - 1 * hrr_0100x);
                        g3y = ak2 * (ak2 * trr_13y - 3 * trr_11y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        double hrr_0102x = hrr_1002x - xjxi * hrr_0002x;
                        g3x = al2 * (al2 * hrr_0102x - 1 * hrr_0100x);
                        double hrr_1012y = hrr_1021y - ylyk * hrr_1011y;
                        g3y = al2 * (al2 * hrr_1012y - 1 * trr_11y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0101x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * hrr_0110x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * trr_11z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_0111x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_1011z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0120x - 1 * hrr_0100x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0102x - 1 * hrr_0100x);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0111y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * hrr_0120y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * hrr_0100y;
                        g3x = ak2 * hrr_1011x;
                        double hrr_0121y = hrr_1021y - yjyi * hrr_0021y;
                        g3y = ak2 * hrr_0121y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_0101y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        double hrr_0130y = trr_13y - yjyi * trr_03y;
                        g3y = ak2 * (ak2 * hrr_0130y - 3 * hrr_0110y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        double hrr_0112y = hrr_1012y - yjyi * hrr_0012y;
                        g3y = al2 * (al2 * hrr_0112y - 1 * hrr_0110y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1111y;
                        g1z = al2 * hrr_0001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_1120y;
                        g2z = ak2 * trr_01z;
                        g2y -= 1 * hrr_1100y;
                        g3x = ak2 * hrr_0011x;
                        double trr_23y = cpy * trr_22y + 2*b01 * trr_21y + 2*b00 * trr_12y;
                        double hrr_2021y = trr_23y - ylyk * trr_22y;
                        double hrr_1121y = hrr_2021y - yjyi * hrr_1021y;
                        g3y = ak2 * hrr_1121y;
                        g3z = ak2 * hrr_0011z;
                        g3y -= 1 * hrr_1101y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        double hrr_1130y = trr_23y - yjyi * trr_13y;
                        g3y = ak2 * (ak2 * hrr_1130y - 3 * hrr_1110y);
                        g3z = ak2 * (ak2 * trr_02z - 1 * wt);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        double hrr_2012y = hrr_2021y - ylyk * hrr_2011y;
                        double hrr_1112y = hrr_2012y - yjyi * hrr_1012y;
                        g3y = al2 * (al2 * hrr_1112y - 1 * hrr_1110y);
                        g3z = al2 * (al2 * hrr_0002z - 1 * wt);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0111y;
                        g1z = al2 * hrr_1001z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_0120y;
                        g2z = ak2 * trr_11z;
                        g2y -= 1 * hrr_0100y;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0121y;
                        g3z = ak2 * hrr_1011z;
                        g3y -= 1 * hrr_0101y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_0130y - 3 * hrr_0110y);
                        g3z = ak2 * (ak2 * trr_12z - 1 * trr_10z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0112y - 1 * hrr_0110y);
                        g3z = al2 * (al2 * hrr_1002z - 1 * trr_10z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_0101z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * hrr_0110z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_0111z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * hrr_0120z - 1 * hrr_0100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_0102z - 1 * hrr_0100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1011y;
                        g1z = al2 * hrr_0101z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_12y;
                        g2z = ak2 * hrr_0110z;
                        g2y -= 1 * trr_10y;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1021y;
                        g3z = ak2 * hrr_0111z;
                        g3y -= 1 * hrr_1001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_13y - 3 * trr_11y);
                        g3z = ak2 * (ak2 * hrr_0120z - 1 * hrr_0100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1012y - 1 * trr_11y);
                        g3z = al2 * (al2 * hrr_0102z - 1 * hrr_0100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0011y;
                        g1z = al2 * hrr_1101z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_02y;
                        g2z = ak2 * hrr_1110z;
                        g2y -= 1 * 1;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0021y;
                        g3z = ak2 * hrr_1111z;
                        g3y -= 1 * hrr_0001y;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_03y - 3 * trr_01y);
                        g3z = ak2 * (ak2 * hrr_1120z - 1 * hrr_1100z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0012y - 1 * trr_01y);
                        g3z = al2 * (al2 * hrr_1102z - 1 * hrr_1100z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1101x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_0011z;
                        g2x = ak2 * hrr_1110x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_1111x;
                        g3y = ak2 * hrr_0011y;
                        double trr_03z = cpz * trr_02z + 2*b01 * trr_01z;
                        double hrr_0021z = trr_03z - zlzk * trr_02z;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_1120x - 1 * hrr_1100x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1102x - 1 * hrr_1100x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0012z = hrr_0021z - zlzk * hrr_0011z;
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0101x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0011z;
                        g2x = ak2 * hrr_0110x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_0111x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0120x - 1 * hrr_0100x);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0102x - 1 * hrr_0100x);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0101x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_1011z;
                        g2x = ak2 * hrr_0110x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * trr_12z;
                        g2z -= 1 * trr_10z;
                        g3x = ak2 * hrr_0111x;
                        g3y = ak2 * hrr_0011y;
                        double trr_13z = cpz * trr_12z + 2*b01 * trr_11z + 1*b00 * trr_02z;
                        double hrr_1021z = trr_13z - zlzk * trr_12z;
                        g3z = ak2 * hrr_1021z;
                        g3z -= 1 * hrr_1001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * hrr_0120x - 1 * hrr_0100x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        g3z = ak2 * (ak2 * trr_13z - 3 * trr_11z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0102x - 1 * hrr_0100x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_1012z = hrr_1021z - zlzk * hrr_1011z;
                        g3z = al2 * (al2 * hrr_1012z - 1 * trr_11z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0101y;
                        g1z = al2 * hrr_0011z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * hrr_0110y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0111y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * hrr_0120y - 1 * hrr_0100y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0102y - 1 * hrr_0100y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1101y;
                        g1z = al2 * hrr_0011z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_1110y;
                        g2z = ak2 * trr_02z;
                        g2z -= 1 * wt;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1111y;
                        g3z = ak2 * hrr_0021z;
                        g3z -= 1 * hrr_0001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_1120y - 1 * hrr_1100y);
                        g3z = ak2 * (ak2 * trr_03z - 3 * trr_01z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1102y - 1 * hrr_1100y);
                        g3z = al2 * (al2 * hrr_0012z - 1 * trr_01z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0101y;
                        g1z = al2 * hrr_1011z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * hrr_0110y;
                        g2z = ak2 * trr_12z;
                        g2z -= 1 * trr_10z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0111y;
                        g3z = ak2 * hrr_1021z;
                        g3z -= 1 * hrr_1001z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * hrr_0120y - 1 * hrr_0100y);
                        g3z = ak2 * (ak2 * trr_13z - 3 * trr_11z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0102y - 1 * hrr_0100y);
                        g3z = al2 * (al2 * hrr_1012z - 1 * trr_11z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_1001x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_0111z;
                        g2x = ak2 * trr_11x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * hrr_0120z;
                        g2z -= 1 * hrr_0100z;
                        g3x = ak2 * hrr_1011x;
                        g3y = ak2 * hrr_0011y;
                        double hrr_0121z = hrr_1021z - zjzi * hrr_0021z;
                        g3z = ak2 * hrr_0121z;
                        g3z -= 1 * hrr_0101z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_12x - 1 * trr_10x);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        double hrr_0130z = trr_13z - zjzi * trr_03z;
                        g3z = ak2 * (ak2 * hrr_0130z - 3 * hrr_0110z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_1002x - 1 * trr_10x);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_0112z = hrr_1012z - zjzi * hrr_0012z;
                        g3z = al2 * (al2 * hrr_0112z - 1 * hrr_0110z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_1001y;
                        g1z = al2 * hrr_0111z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_11y;
                        g2z = ak2 * hrr_0120z;
                        g2z -= 1 * hrr_0100z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_1011y;
                        g3z = ak2 * hrr_0121z;
                        g3z -= 1 * hrr_0101z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_12y - 1 * trr_10y);
                        g3z = ak2 * (ak2 * hrr_0130z - 3 * hrr_0110z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_1002y - 1 * trr_10y);
                        g3z = al2 * (al2 * hrr_0112z - 1 * hrr_0110z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
                        g1x = al2 * hrr_0001x;
                        g1y = al2 * hrr_0001y;
                        g1z = al2 * hrr_1111z;
                        g2x = ak2 * trr_01x;
                        g2y = ak2 * trr_01y;
                        g2z = ak2 * hrr_1120z;
                        g2z -= 1 * hrr_1100z;
                        g3x = ak2 * hrr_0011x;
                        g3y = ak2 * hrr_0011y;
                        double trr_23z = cpz * trr_22z + 2*b01 * trr_21z + 2*b00 * trr_12z;
                        double hrr_2021z = trr_23z - zlzk * trr_22z;
                        double hrr_1121z = hrr_2021z - zjzi * hrr_1021z;
                        g3z = ak2 * hrr_1121z;
                        g3z -= 1 * hrr_1101z;
                        g3x *= al2;
                        g3y *= al2;
                        g3z *= al2;
                        v2xx += g3x * prod_yz;
                        v2yy += g3y * prod_xz;
                        v2zz += g3z * prod_xy;
                        v2xy += g2x * g1y * Iz;
                        v2xz += g2x * g1z * Iy;
                        v2yx += g2y * g1x * Iz;
                        v2yz += g2y * g1z * Ix;
                        v2zx += g2z * g1x * Iy;
                        v2zy += g2z * g1y * Ix;
                        g3x = ak2 * (ak2 * trr_02x - 1 * fac);
                        g3y = ak2 * (ak2 * trr_02y - 1 * 1);
                        double hrr_1130z = trr_23z - zjzi * trr_13z;
                        g3z = ak2 * (ak2 * hrr_1130z - 3 * hrr_1110z);
                        v_kxx += g3x * prod_yz;
                        v_kyy += g3y * prod_xz;
                        v_kzz += g3z * prod_xy;
                        v_kxy += g2x * g2y * Iz;
                        v_kxz += g2x * g2z * Iy;
                        v_kyz += g2y * g2z * Ix;
                        g3x = al2 * (al2 * hrr_0002x - 1 * fac);
                        g3y = al2 * (al2 * hrr_0002y - 1 * 1);
                        double hrr_2012z = hrr_2021z - zlzk * hrr_2011z;
                        double hrr_1112z = hrr_2012z - zjzi * hrr_1012z;
                        g3z = al2 * (al2 * hrr_1112z - 1 * hrr_1110z);
                        v_lxx += g3x * prod_yz;
                        v_lyy += g3y * prod_xz;
                        v_lzz += g3z * prod_xy;
                        v_lxy += g1x * g1y * Iz;
                        v_lxz += g1x * g1z * Iy;
                        v_lyz += g1y * g1z * Ix;
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
        atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
        atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
        atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
        atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
        atomicAdd(ejk + (ka*natm+la)*9 + 0, v2xx);
        atomicAdd(ejk + (ka*natm+la)*9 + 1, v2xy);
        atomicAdd(ejk + (ka*natm+la)*9 + 2, v2xz);
        atomicAdd(ejk + (ka*natm+la)*9 + 3, v2yx);
        atomicAdd(ejk + (ka*natm+la)*9 + 4, v2yy);
        atomicAdd(ejk + (ka*natm+la)*9 + 5, v2yz);
        atomicAdd(ejk + (ka*natm+la)*9 + 6, v2zx);
        atomicAdd(ejk + (ka*natm+la)*9 + 7, v2zy);
        atomicAdd(ejk + (ka*natm+la)*9 + 8, v2zz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
        atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
        atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
        atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
        atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
        atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
        atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy);
        atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy*.5);
        atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz);
        atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 0, v_lxx*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 3, v_lxy);
        atomicAdd(ejk + (la*natm+la)*9 + 4, v_lyy*.5);
        atomicAdd(ejk + (la*natm+la)*9 + 6, v_lxz);
        atomicAdd(ejk + (la*natm+la)*9 + 7, v_lyz);
        atomicAdd(ejk + (la*natm+la)*9 + 8, v_lzz*.5);
    }
}
__global__
void rys_ejk_ip2_type12_1110(RysIntEnvVars envs, JKEnergy jk, BoundsInfo bounds,
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
            _rys_ejk_ip2_type12_1110(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_ejk_ip2_type12_unrolled(RysIntEnvVars *envs, JKEnergy *jk, BoundsInfo *bounds,
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
    case 0: rys_ejk_ip2_type12_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125: rys_ejk_ip2_type12_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 130: rys_ejk_ip2_type12_1010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 131: rys_ejk_ip2_type12_1011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 150: rys_ejk_ip2_type12_1100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 155: rys_ejk_ip2_type12_1110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
