#include "vhf.cuh"
#include "rys_roots_unrolled.cu"
#include "create_tasks_ip1.cu"


__device__ static
void _rys_ejk_ip2_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    double *vj = jk.vj;
    double *vk = jk.vk;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *Rpa_cicj = dm_cache + 1 * TILE2;
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
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }

    int ij = sq_id / TILE2;
    if (ij < 1) {
        int i = ij % 1;
        int j = ij / 1;
        int sh_ij = sq_id % TILE2;
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
        int natm = envs.natm;
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
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
        double dd_jk, dd_jl, vj_dd, vk_dd;
        double g1, g2, g3, prod;
        double dm_lk_0_0 = dm[(l0+0)*nao+(k0+0)];
        if (jk.n_dm > 1) {
            int nao2 = nao * nao;
            dm_lk_0_0 += dm[nao2+(l0+0)*nao+(k0+0)];
        }
        double dm_jk_0_0 = dm[(j0+0)*nao+(k0+0)];
        double dm_jl_0_0 = dm[(j0+0)*nao+(l0+0)];
        double dm_ik_0_0 = dm[(i0+0)*nao+(k0+0)];
        double dm_il_0_0 = dm[(i0+0)*nao+(l0+0)];
        
        double vk_00xx = 0;
        double vj_00xx = 0;
        double vk_00xy = 0;
        double vj_00xy = 0;
        double vk_00xz = 0;
        double vj_00xz = 0;
        double vk_00yx = 0;
        double vj_00yx = 0;
        double vk_00yy = 0;
        double vj_00yy = 0;
        double vk_00yz = 0;
        double vj_00yz = 0;
        double vk_00zx = 0;
        double vj_00zx = 0;
        double vk_00zy = 0;
        double vj_00zy = 0;
        double vk_00zz = 0;
        double vj_00zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        g3 = ai*2 * (ai*2 * trr_20x - 1 * fac);
                        prod = g3 * 1 * wt;
                        vk_00xx += prod * vk_dd;
                        vj_00xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_00xy += prod * vk_dd;
                        vj_00xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_00xz += prod * vk_dd;
                        vj_00xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_00yx += prod * vk_dd;
                        vj_00yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        g3 = ai*2 * (ai*2 * trr_20y - 1 * 1);
                        prod = g3 * fac * wt;
                        vk_00yy += prod * vk_dd;
                        vj_00yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_00yz += prod * vk_dd;
                        vj_00yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_00zx += prod * vk_dd;
                        vj_00zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_00zy += prod * vk_dd;
                        vj_00zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        g3 = ai*2 * (ai*2 * trr_20z - 1 * wt);
                        prod = g3 * fac * 1;
                        vk_00zz += prod * vk_dd;
                        vj_00zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+ia)*9 + 0, vk_00xx);
        atomicAdd(vj + (ia*natm+ia)*9 + 0, vj_00xx);
        atomicAdd(vk + (ia*natm+ia)*9 + 1, vk_00xy);
        atomicAdd(vj + (ia*natm+ia)*9 + 1, vj_00xy);
        atomicAdd(vk + (ia*natm+ia)*9 + 2, vk_00xz);
        atomicAdd(vj + (ia*natm+ia)*9 + 2, vj_00xz);
        atomicAdd(vk + (ia*natm+ia)*9 + 3, vk_00yx);
        atomicAdd(vj + (ia*natm+ia)*9 + 3, vj_00yx);
        atomicAdd(vk + (ia*natm+ia)*9 + 4, vk_00yy);
        atomicAdd(vj + (ia*natm+ia)*9 + 4, vj_00yy);
        atomicAdd(vk + (ia*natm+ia)*9 + 5, vk_00yz);
        atomicAdd(vj + (ia*natm+ia)*9 + 5, vj_00yz);
        atomicAdd(vk + (ia*natm+ia)*9 + 6, vk_00zx);
        atomicAdd(vj + (ia*natm+ia)*9 + 6, vj_00zx);
        atomicAdd(vk + (ia*natm+ia)*9 + 7, vk_00zy);
        atomicAdd(vj + (ia*natm+ia)*9 + 7, vj_00zy);
        atomicAdd(vk + (ia*natm+ia)*9 + 8, vk_00zz);
        atomicAdd(vj + (ia*natm+ia)*9 + 8, vj_00zz);

        double vk_01xx = 0;
        double vj_01xx = 0;
        double vk_01xy = 0;
        double vj_01xy = 0;
        double vk_01xz = 0;
        double vj_01xz = 0;
        double vk_01yx = 0;
        double vj_01yx = 0;
        double vk_01yy = 0;
        double vj_01yy = 0;
        double vk_01yz = 0;
        double vj_01yz = 0;
        double vk_01zx = 0;
        double vj_01zx = 0;
        double vk_01zy = 0;
        double vj_01zy = 0;
        double vk_01zz = 0;
        double vj_01zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g3 = aj*2 * ai*2 * hrr_1100x;
                        prod = g3 * 1 * wt;
                        vk_01xx += prod * vk_dd;
                        vj_01xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_01xy += prod * vk_dd;
                        vj_01xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_01xz += prod * vk_dd;
                        vj_01xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_01yx += prod * vk_dd;
                        vj_01yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g3 = aj*2 * ai*2 * hrr_1100y;
                        prod = g3 * fac * wt;
                        vk_01yy += prod * vk_dd;
                        vj_01yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_01yz += prod * vk_dd;
                        vj_01yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_01zx += prod * vk_dd;
                        vj_01zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_01zy += prod * vk_dd;
                        vj_01zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g3 = aj*2 * ai*2 * hrr_1100z;
                        prod = g3 * fac * 1;
                        vk_01zz += prod * vk_dd;
                        vj_01zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+ja)*9 + 0, vk_01xx);
        atomicAdd(vj + (ia*natm+ja)*9 + 0, vj_01xx);
        atomicAdd(vk + (ia*natm+ja)*9 + 1, vk_01xy);
        atomicAdd(vj + (ia*natm+ja)*9 + 1, vj_01xy);
        atomicAdd(vk + (ia*natm+ja)*9 + 2, vk_01xz);
        atomicAdd(vj + (ia*natm+ja)*9 + 2, vj_01xz);
        atomicAdd(vk + (ia*natm+ja)*9 + 3, vk_01yx);
        atomicAdd(vj + (ia*natm+ja)*9 + 3, vj_01yx);
        atomicAdd(vk + (ia*natm+ja)*9 + 4, vk_01yy);
        atomicAdd(vj + (ia*natm+ja)*9 + 4, vj_01yy);
        atomicAdd(vk + (ia*natm+ja)*9 + 5, vk_01yz);
        atomicAdd(vj + (ia*natm+ja)*9 + 5, vj_01yz);
        atomicAdd(vk + (ia*natm+ja)*9 + 6, vk_01zx);
        atomicAdd(vj + (ia*natm+ja)*9 + 6, vj_01zx);
        atomicAdd(vk + (ia*natm+ja)*9 + 7, vk_01zy);
        atomicAdd(vj + (ia*natm+ja)*9 + 7, vj_01zy);
        atomicAdd(vk + (ia*natm+ja)*9 + 8, vk_01zz);
        atomicAdd(vj + (ia*natm+ja)*9 + 8, vj_01zz);

        double vk_02xx = 0;
        double vj_02xx = 0;
        double vk_02xy = 0;
        double vj_02xy = 0;
        double vk_02xz = 0;
        double vj_02xz = 0;
        double vk_02yx = 0;
        double vj_02yx = 0;
        double vk_02yy = 0;
        double vj_02yy = 0;
        double vk_02yz = 0;
        double vj_02yz = 0;
        double vk_02zx = 0;
        double vj_02zx = 0;
        double vk_02zy = 0;
        double vj_02zy = 0;
        double vk_02zz = 0;
        double vj_02zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        g3 = ak*2 * ai*2 * trr_11x;
                        prod = g3 * 1 * wt;
                        vk_02xx += prod * vk_dd;
                        vj_02xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_02xy += prod * vk_dd;
                        vj_02xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_02xz += prod * vk_dd;
                        vj_02xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g1 = ai*2 * trr_10y;
                        double trr_01x = cpx * fac;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_02yx += prod * vk_dd;
                        vj_02yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        g3 = ak*2 * ai*2 * trr_11y;
                        prod = g3 * fac * wt;
                        vk_02yy += prod * vk_dd;
                        vj_02yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_02yz += prod * vk_dd;
                        vj_02yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g1 = ai*2 * trr_10z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_02zx += prod * vk_dd;
                        vj_02zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_02zy += prod * vk_dd;
                        vj_02zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        g3 = ak*2 * ai*2 * trr_11z;
                        prod = g3 * fac * 1;
                        vk_02zz += prod * vk_dd;
                        vj_02zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+ka)*9 + 0, vk_02xx);
        atomicAdd(vj + (ia*natm+ka)*9 + 0, vj_02xx);
        atomicAdd(vk + (ia*natm+ka)*9 + 1, vk_02xy);
        atomicAdd(vj + (ia*natm+ka)*9 + 1, vj_02xy);
        atomicAdd(vk + (ia*natm+ka)*9 + 2, vk_02xz);
        atomicAdd(vj + (ia*natm+ka)*9 + 2, vj_02xz);
        atomicAdd(vk + (ia*natm+ka)*9 + 3, vk_02yx);
        atomicAdd(vj + (ia*natm+ka)*9 + 3, vj_02yx);
        atomicAdd(vk + (ia*natm+ka)*9 + 4, vk_02yy);
        atomicAdd(vj + (ia*natm+ka)*9 + 4, vj_02yy);
        atomicAdd(vk + (ia*natm+ka)*9 + 5, vk_02yz);
        atomicAdd(vj + (ia*natm+ka)*9 + 5, vj_02yz);
        atomicAdd(vk + (ia*natm+ka)*9 + 6, vk_02zx);
        atomicAdd(vj + (ia*natm+ka)*9 + 6, vj_02zx);
        atomicAdd(vk + (ia*natm+ka)*9 + 7, vk_02zy);
        atomicAdd(vj + (ia*natm+ka)*9 + 7, vj_02zy);
        atomicAdd(vk + (ia*natm+ka)*9 + 8, vk_02zz);
        atomicAdd(vj + (ia*natm+ka)*9 + 8, vj_02zz);

        double vk_03xx = 0;
        double vj_03xx = 0;
        double vk_03xy = 0;
        double vj_03xy = 0;
        double vk_03xz = 0;
        double vj_03xz = 0;
        double vk_03yx = 0;
        double vj_03yx = 0;
        double vk_03yy = 0;
        double vj_03yy = 0;
        double vk_03yz = 0;
        double vj_03yz = 0;
        double vk_03zx = 0;
        double vj_03zx = 0;
        double vk_03zy = 0;
        double vj_03zy = 0;
        double vk_03zz = 0;
        double vj_03zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g3 = al*2 * ai*2 * hrr_1001x;
                        prod = g3 * 1 * wt;
                        vk_03xx += prod * vk_dd;
                        vj_03xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_03xy += prod * vk_dd;
                        vj_03xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_03xz += prod * vk_dd;
                        vj_03xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g1 = ai*2 * trr_10y;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_03yx += prod * vk_dd;
                        vj_03yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g3 = al*2 * ai*2 * hrr_1001y;
                        prod = g3 * fac * wt;
                        vk_03yy += prod * vk_dd;
                        vj_03yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_03yz += prod * vk_dd;
                        vj_03yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g1 = ai*2 * trr_10z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_03zx += prod * vk_dd;
                        vj_03zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_03zy += prod * vk_dd;
                        vj_03zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g3 = al*2 * ai*2 * hrr_1001z;
                        prod = g3 * fac * 1;
                        vk_03zz += prod * vk_dd;
                        vj_03zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+la)*9 + 0, vk_03xx);
        atomicAdd(vj + (ia*natm+la)*9 + 0, vj_03xx);
        atomicAdd(vk + (ia*natm+la)*9 + 1, vk_03xy);
        atomicAdd(vj + (ia*natm+la)*9 + 1, vj_03xy);
        atomicAdd(vk + (ia*natm+la)*9 + 2, vk_03xz);
        atomicAdd(vj + (ia*natm+la)*9 + 2, vj_03xz);
        atomicAdd(vk + (ia*natm+la)*9 + 3, vk_03yx);
        atomicAdd(vj + (ia*natm+la)*9 + 3, vj_03yx);
        atomicAdd(vk + (ia*natm+la)*9 + 4, vk_03yy);
        atomicAdd(vj + (ia*natm+la)*9 + 4, vj_03yy);
        atomicAdd(vk + (ia*natm+la)*9 + 5, vk_03yz);
        atomicAdd(vj + (ia*natm+la)*9 + 5, vj_03yz);
        atomicAdd(vk + (ia*natm+la)*9 + 6, vk_03zx);
        atomicAdd(vj + (ia*natm+la)*9 + 6, vj_03zx);
        atomicAdd(vk + (ia*natm+la)*9 + 7, vk_03zy);
        atomicAdd(vj + (ia*natm+la)*9 + 7, vj_03zy);
        atomicAdd(vk + (ia*natm+la)*9 + 8, vk_03zz);
        atomicAdd(vj + (ia*natm+la)*9 + 8, vj_03zz);

        double vk_10xx = 0;
        double vj_10xx = 0;
        double vk_10xy = 0;
        double vj_10xy = 0;
        double vk_10xz = 0;
        double vj_10xz = 0;
        double vk_10yx = 0;
        double vj_10yx = 0;
        double vk_10yy = 0;
        double vj_10yy = 0;
        double vk_10yz = 0;
        double vj_10yz = 0;
        double vk_10zx = 0;
        double vj_10zx = 0;
        double vk_10zy = 0;
        double vj_10zy = 0;
        double vk_10zz = 0;
        double vj_10zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g3 = ai*2 * aj*2 * hrr_1100x;
                        prod = g3 * 1 * wt;
                        vk_10xx += prod * vk_dd;
                        vj_10xx += prod * vj_dd;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g1 = aj*2 * hrr_0100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_10xy += prod * vk_dd;
                        vj_10xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_10xz += prod * vk_dd;
                        vj_10xz += prod * vj_dd;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_10yx += prod * vk_dd;
                        vj_10yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g3 = ai*2 * aj*2 * hrr_1100y;
                        prod = g3 * fac * wt;
                        vk_10yy += prod * vk_dd;
                        vj_10yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_10yz += prod * vk_dd;
                        vj_10yz += prod * vj_dd;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_10zx += prod * vk_dd;
                        vj_10zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_10zy += prod * vk_dd;
                        vj_10zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g3 = ai*2 * aj*2 * hrr_1100z;
                        prod = g3 * fac * 1;
                        vk_10zz += prod * vk_dd;
                        vj_10zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+ia)*9 + 0, vk_10xx);
        atomicAdd(vj + (ja*natm+ia)*9 + 0, vj_10xx);
        atomicAdd(vk + (ja*natm+ia)*9 + 1, vk_10xy);
        atomicAdd(vj + (ja*natm+ia)*9 + 1, vj_10xy);
        atomicAdd(vk + (ja*natm+ia)*9 + 2, vk_10xz);
        atomicAdd(vj + (ja*natm+ia)*9 + 2, vj_10xz);
        atomicAdd(vk + (ja*natm+ia)*9 + 3, vk_10yx);
        atomicAdd(vj + (ja*natm+ia)*9 + 3, vj_10yx);
        atomicAdd(vk + (ja*natm+ia)*9 + 4, vk_10yy);
        atomicAdd(vj + (ja*natm+ia)*9 + 4, vj_10yy);
        atomicAdd(vk + (ja*natm+ia)*9 + 5, vk_10yz);
        atomicAdd(vj + (ja*natm+ia)*9 + 5, vj_10yz);
        atomicAdd(vk + (ja*natm+ia)*9 + 6, vk_10zx);
        atomicAdd(vj + (ja*natm+ia)*9 + 6, vj_10zx);
        atomicAdd(vk + (ja*natm+ia)*9 + 7, vk_10zy);
        atomicAdd(vj + (ja*natm+ia)*9 + 7, vj_10zy);
        atomicAdd(vk + (ja*natm+ia)*9 + 8, vk_10zz);
        atomicAdd(vj + (ja*natm+ia)*9 + 8, vj_10zz);

        double vk_11xx = 0;
        double vj_11xx = 0;
        double vk_11xy = 0;
        double vj_11xy = 0;
        double vk_11xz = 0;
        double vj_11xz = 0;
        double vk_11yx = 0;
        double vj_11yx = 0;
        double vk_11yy = 0;
        double vj_11yy = 0;
        double vk_11yz = 0;
        double vj_11yz = 0;
        double vk_11zx = 0;
        double vj_11zx = 0;
        double vk_11zy = 0;
        double vj_11zy = 0;
        double vk_11zz = 0;
        double vj_11zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        double hrr_0200x = hrr_1100x - (rj[0] - ri[0]) * hrr_0100x;
                        g3 = aj*2 * (aj*2 * hrr_0200x - 1 * fac);
                        prod = g3 * 1 * wt;
                        vk_11xx += prod * vk_dd;
                        vj_11xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_11xy += prod * vk_dd;
                        vj_11xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_11xz += prod * vk_dd;
                        vj_11xz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_11yx += prod * vk_dd;
                        vj_11yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        double hrr_0200y = hrr_1100y - (rj[1] - ri[1]) * hrr_0100y;
                        g3 = aj*2 * (aj*2 * hrr_0200y - 1 * 1);
                        prod = g3 * fac * wt;
                        vk_11yy += prod * vk_dd;
                        vj_11yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_11yz += prod * vk_dd;
                        vj_11yz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_11zx += prod * vk_dd;
                        vj_11zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_11zy += prod * vk_dd;
                        vj_11zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        double hrr_0200z = hrr_1100z - (rj[2] - ri[2]) * hrr_0100z;
                        g3 = aj*2 * (aj*2 * hrr_0200z - 1 * wt);
                        prod = g3 * fac * 1;
                        vk_11zz += prod * vk_dd;
                        vj_11zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+ja)*9 + 0, vk_11xx);
        atomicAdd(vj + (ja*natm+ja)*9 + 0, vj_11xx);
        atomicAdd(vk + (ja*natm+ja)*9 + 1, vk_11xy);
        atomicAdd(vj + (ja*natm+ja)*9 + 1, vj_11xy);
        atomicAdd(vk + (ja*natm+ja)*9 + 2, vk_11xz);
        atomicAdd(vj + (ja*natm+ja)*9 + 2, vj_11xz);
        atomicAdd(vk + (ja*natm+ja)*9 + 3, vk_11yx);
        atomicAdd(vj + (ja*natm+ja)*9 + 3, vj_11yx);
        atomicAdd(vk + (ja*natm+ja)*9 + 4, vk_11yy);
        atomicAdd(vj + (ja*natm+ja)*9 + 4, vj_11yy);
        atomicAdd(vk + (ja*natm+ja)*9 + 5, vk_11yz);
        atomicAdd(vj + (ja*natm+ja)*9 + 5, vj_11yz);
        atomicAdd(vk + (ja*natm+ja)*9 + 6, vk_11zx);
        atomicAdd(vj + (ja*natm+ja)*9 + 6, vj_11zx);
        atomicAdd(vk + (ja*natm+ja)*9 + 7, vk_11zy);
        atomicAdd(vj + (ja*natm+ja)*9 + 7, vj_11zy);
        atomicAdd(vk + (ja*natm+ja)*9 + 8, vk_11zz);
        atomicAdd(vj + (ja*natm+ja)*9 + 8, vj_11zz);

        double vk_12xx = 0;
        double vj_12xx = 0;
        double vk_12xy = 0;
        double vj_12xy = 0;
        double vk_12xz = 0;
        double vj_12xz = 0;
        double vk_12yx = 0;
        double vj_12yx = 0;
        double vk_12yy = 0;
        double vj_12yy = 0;
        double vk_12yz = 0;
        double vj_12yz = 0;
        double vk_12zx = 0;
        double vj_12zx = 0;
        double vk_12zy = 0;
        double vj_12zy = 0;
        double vk_12zz = 0;
        double vj_12zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - (rj[0] - ri[0]) * trr_01x;
                        g3 = ak*2 * aj*2 * hrr_0110x;
                        prod = g3 * 1 * wt;
                        vk_12xx += prod * vk_dd;
                        vj_12xx += prod * vj_dd;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g1 = aj*2 * hrr_0100x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_12xy += prod * vk_dd;
                        vj_12xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_12xz += prod * vk_dd;
                        vj_12xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_12yx += prod * vk_dd;
                        vj_12yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - (rj[1] - ri[1]) * trr_01y;
                        g3 = ak*2 * aj*2 * hrr_0110y;
                        prod = g3 * fac * wt;
                        vk_12yy += prod * vk_dd;
                        vj_12yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_12yz += prod * vk_dd;
                        vj_12yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_12zx += prod * vk_dd;
                        vj_12zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_12zy += prod * vk_dd;
                        vj_12zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - (rj[2] - ri[2]) * trr_01z;
                        g3 = ak*2 * aj*2 * hrr_0110z;
                        prod = g3 * fac * 1;
                        vk_12zz += prod * vk_dd;
                        vj_12zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+ka)*9 + 0, vk_12xx);
        atomicAdd(vj + (ja*natm+ka)*9 + 0, vj_12xx);
        atomicAdd(vk + (ja*natm+ka)*9 + 1, vk_12xy);
        atomicAdd(vj + (ja*natm+ka)*9 + 1, vj_12xy);
        atomicAdd(vk + (ja*natm+ka)*9 + 2, vk_12xz);
        atomicAdd(vj + (ja*natm+ka)*9 + 2, vj_12xz);
        atomicAdd(vk + (ja*natm+ka)*9 + 3, vk_12yx);
        atomicAdd(vj + (ja*natm+ka)*9 + 3, vj_12yx);
        atomicAdd(vk + (ja*natm+ka)*9 + 4, vk_12yy);
        atomicAdd(vj + (ja*natm+ka)*9 + 4, vj_12yy);
        atomicAdd(vk + (ja*natm+ka)*9 + 5, vk_12yz);
        atomicAdd(vj + (ja*natm+ka)*9 + 5, vj_12yz);
        atomicAdd(vk + (ja*natm+ka)*9 + 6, vk_12zx);
        atomicAdd(vj + (ja*natm+ka)*9 + 6, vj_12zx);
        atomicAdd(vk + (ja*natm+ka)*9 + 7, vk_12zy);
        atomicAdd(vj + (ja*natm+ka)*9 + 7, vj_12zy);
        atomicAdd(vk + (ja*natm+ka)*9 + 8, vk_12zz);
        atomicAdd(vj + (ja*natm+ka)*9 + 8, vj_12zz);

        double vk_13xx = 0;
        double vj_13xx = 0;
        double vk_13xy = 0;
        double vj_13xy = 0;
        double vk_13xz = 0;
        double vj_13xz = 0;
        double vk_13yx = 0;
        double vj_13yx = 0;
        double vk_13yy = 0;
        double vj_13yy = 0;
        double vk_13yz = 0;
        double vj_13yz = 0;
        double vk_13zx = 0;
        double vj_13zx = 0;
        double vk_13zy = 0;
        double vj_13zy = 0;
        double vk_13zz = 0;
        double vj_13zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - (rj[0] - ri[0]) * hrr_0001x;
                        g3 = al*2 * aj*2 * hrr_0101x;
                        prod = g3 * 1 * wt;
                        vk_13xx += prod * vk_dd;
                        vj_13xx += prod * vj_dd;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g1 = aj*2 * hrr_0100x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_13xy += prod * vk_dd;
                        vj_13xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_13xz += prod * vk_dd;
                        vj_13xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g1 = aj*2 * hrr_0100y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_13yx += prod * vk_dd;
                        vj_13yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        double hrr_0101y = hrr_1001y - (rj[1] - ri[1]) * hrr_0001y;
                        g3 = al*2 * aj*2 * hrr_0101y;
                        prod = g3 * fac * wt;
                        vk_13yy += prod * vk_dd;
                        vj_13yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_13yz += prod * vk_dd;
                        vj_13yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g1 = aj*2 * hrr_0100z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_13zx += prod * vk_dd;
                        vj_13zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_13zy += prod * vk_dd;
                        vj_13zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        double hrr_0101z = hrr_1001z - (rj[2] - ri[2]) * hrr_0001z;
                        g3 = al*2 * aj*2 * hrr_0101z;
                        prod = g3 * fac * 1;
                        vk_13zz += prod * vk_dd;
                        vj_13zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+la)*9 + 0, vk_13xx);
        atomicAdd(vj + (ja*natm+la)*9 + 0, vj_13xx);
        atomicAdd(vk + (ja*natm+la)*9 + 1, vk_13xy);
        atomicAdd(vj + (ja*natm+la)*9 + 1, vj_13xy);
        atomicAdd(vk + (ja*natm+la)*9 + 2, vk_13xz);
        atomicAdd(vj + (ja*natm+la)*9 + 2, vj_13xz);
        atomicAdd(vk + (ja*natm+la)*9 + 3, vk_13yx);
        atomicAdd(vj + (ja*natm+la)*9 + 3, vj_13yx);
        atomicAdd(vk + (ja*natm+la)*9 + 4, vk_13yy);
        atomicAdd(vj + (ja*natm+la)*9 + 4, vj_13yy);
        atomicAdd(vk + (ja*natm+la)*9 + 5, vk_13yz);
        atomicAdd(vj + (ja*natm+la)*9 + 5, vj_13yz);
        atomicAdd(vk + (ja*natm+la)*9 + 6, vk_13zx);
        atomicAdd(vj + (ja*natm+la)*9 + 6, vj_13zx);
        atomicAdd(vk + (ja*natm+la)*9 + 7, vk_13zy);
        atomicAdd(vj + (ja*natm+la)*9 + 7, vj_13zy);
        atomicAdd(vk + (ja*natm+la)*9 + 8, vk_13zz);
        atomicAdd(vj + (ja*natm+la)*9 + 8, vj_13zz);

        double vk_20xx = 0;
        double vj_20xx = 0;
        double vk_20xy = 0;
        double vj_20xy = 0;
        double vk_20xz = 0;
        double vj_20xz = 0;
        double vk_20yx = 0;
        double vj_20yx = 0;
        double vk_20yy = 0;
        double vj_20yy = 0;
        double vk_20yz = 0;
        double vj_20yz = 0;
        double vk_20zx = 0;
        double vj_20zx = 0;
        double vk_20zy = 0;
        double vj_20zy = 0;
        double vk_20zz = 0;
        double vj_20zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        g3 = ai*2 * ak*2 * trr_11x;
                        prod = g3 * 1 * wt;
                        vk_20xx += prod * vk_dd;
                        vj_20xx += prod * vj_dd;
                        double trr_01x = cpx * fac;
                        g1 = ak*2 * trr_01x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_20xy += prod * vk_dd;
                        vj_20xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_20xz += prod * vk_dd;
                        vj_20xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g1 = ak*2 * trr_01y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_20yx += prod * vk_dd;
                        vj_20yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        g3 = ai*2 * ak*2 * trr_11y;
                        prod = g3 * fac * wt;
                        vk_20yy += prod * vk_dd;
                        vj_20yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_20yz += prod * vk_dd;
                        vj_20yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g1 = ak*2 * trr_01z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_20zx += prod * vk_dd;
                        vj_20zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_20zy += prod * vk_dd;
                        vj_20zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        g3 = ai*2 * ak*2 * trr_11z;
                        prod = g3 * fac * 1;
                        vk_20zz += prod * vk_dd;
                        vj_20zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+ia)*9 + 0, vk_20xx);
        atomicAdd(vj + (ka*natm+ia)*9 + 0, vj_20xx);
        atomicAdd(vk + (ka*natm+ia)*9 + 1, vk_20xy);
        atomicAdd(vj + (ka*natm+ia)*9 + 1, vj_20xy);
        atomicAdd(vk + (ka*natm+ia)*9 + 2, vk_20xz);
        atomicAdd(vj + (ka*natm+ia)*9 + 2, vj_20xz);
        atomicAdd(vk + (ka*natm+ia)*9 + 3, vk_20yx);
        atomicAdd(vj + (ka*natm+ia)*9 + 3, vj_20yx);
        atomicAdd(vk + (ka*natm+ia)*9 + 4, vk_20yy);
        atomicAdd(vj + (ka*natm+ia)*9 + 4, vj_20yy);
        atomicAdd(vk + (ka*natm+ia)*9 + 5, vk_20yz);
        atomicAdd(vj + (ka*natm+ia)*9 + 5, vj_20yz);
        atomicAdd(vk + (ka*natm+ia)*9 + 6, vk_20zx);
        atomicAdd(vj + (ka*natm+ia)*9 + 6, vj_20zx);
        atomicAdd(vk + (ka*natm+ia)*9 + 7, vk_20zy);
        atomicAdd(vj + (ka*natm+ia)*9 + 7, vj_20zy);
        atomicAdd(vk + (ka*natm+ia)*9 + 8, vk_20zz);
        atomicAdd(vj + (ka*natm+ia)*9 + 8, vj_20zz);

        double vk_21xx = 0;
        double vj_21xx = 0;
        double vk_21xy = 0;
        double vj_21xy = 0;
        double vk_21xz = 0;
        double vj_21xz = 0;
        double vk_21yx = 0;
        double vj_21yx = 0;
        double vk_21yy = 0;
        double vj_21yy = 0;
        double vk_21yz = 0;
        double vj_21yz = 0;
        double vk_21zx = 0;
        double vj_21zx = 0;
        double vk_21zy = 0;
        double vj_21zy = 0;
        double vk_21zz = 0;
        double vj_21zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - (rj[0] - ri[0]) * trr_01x;
                        g3 = aj*2 * ak*2 * hrr_0110x;
                        prod = g3 * 1 * wt;
                        vk_21xx += prod * vk_dd;
                        vj_21xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_21xy += prod * vk_dd;
                        vj_21xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_21xz += prod * vk_dd;
                        vj_21xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g1 = ak*2 * trr_01y;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_21yx += prod * vk_dd;
                        vj_21yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - (rj[1] - ri[1]) * trr_01y;
                        g3 = aj*2 * ak*2 * hrr_0110y;
                        prod = g3 * fac * wt;
                        vk_21yy += prod * vk_dd;
                        vj_21yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_21yz += prod * vk_dd;
                        vj_21yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g1 = ak*2 * trr_01z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_21zx += prod * vk_dd;
                        vj_21zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_21zy += prod * vk_dd;
                        vj_21zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - (rj[2] - ri[2]) * trr_01z;
                        g3 = aj*2 * ak*2 * hrr_0110z;
                        prod = g3 * fac * 1;
                        vk_21zz += prod * vk_dd;
                        vj_21zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+ja)*9 + 0, vk_21xx);
        atomicAdd(vj + (ka*natm+ja)*9 + 0, vj_21xx);
        atomicAdd(vk + (ka*natm+ja)*9 + 1, vk_21xy);
        atomicAdd(vj + (ka*natm+ja)*9 + 1, vj_21xy);
        atomicAdd(vk + (ka*natm+ja)*9 + 2, vk_21xz);
        atomicAdd(vj + (ka*natm+ja)*9 + 2, vj_21xz);
        atomicAdd(vk + (ka*natm+ja)*9 + 3, vk_21yx);
        atomicAdd(vj + (ka*natm+ja)*9 + 3, vj_21yx);
        atomicAdd(vk + (ka*natm+ja)*9 + 4, vk_21yy);
        atomicAdd(vj + (ka*natm+ja)*9 + 4, vj_21yy);
        atomicAdd(vk + (ka*natm+ja)*9 + 5, vk_21yz);
        atomicAdd(vj + (ka*natm+ja)*9 + 5, vj_21yz);
        atomicAdd(vk + (ka*natm+ja)*9 + 6, vk_21zx);
        atomicAdd(vj + (ka*natm+ja)*9 + 6, vj_21zx);
        atomicAdd(vk + (ka*natm+ja)*9 + 7, vk_21zy);
        atomicAdd(vj + (ka*natm+ja)*9 + 7, vj_21zy);
        atomicAdd(vk + (ka*natm+ja)*9 + 8, vk_21zz);
        atomicAdd(vj + (ka*natm+ja)*9 + 8, vj_21zz);

        double vk_22xx = 0;
        double vj_22xx = 0;
        double vk_22xy = 0;
        double vj_22xy = 0;
        double vk_22xz = 0;
        double vj_22xz = 0;
        double vk_22yx = 0;
        double vj_22yx = 0;
        double vk_22yy = 0;
        double vj_22yy = 0;
        double vk_22yz = 0;
        double vj_22yz = 0;
        double vk_22zx = 0;
        double vj_22zx = 0;
        double vk_22zy = 0;
        double vj_22zy = 0;
        double vk_22zz = 0;
        double vj_22zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double trr_01x = cpx * fac;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        g3 = ak*2 * (ak*2 * trr_02x - 1 * fac);
                        prod = g3 * 1 * wt;
                        vk_22xx += prod * vk_dd;
                        vj_22xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_22xy += prod * vk_dd;
                        vj_22xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_22xz += prod * vk_dd;
                        vj_22xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_22yx += prod * vk_dd;
                        vj_22yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        g3 = ak*2 * (ak*2 * trr_02y - 1 * 1);
                        prod = g3 * fac * wt;
                        vk_22yy += prod * vk_dd;
                        vj_22yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_22yz += prod * vk_dd;
                        vj_22yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_22zx += prod * vk_dd;
                        vj_22zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_22zy += prod * vk_dd;
                        vj_22zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        g3 = ak*2 * (ak*2 * trr_02z - 1 * wt);
                        prod = g3 * fac * 1;
                        vk_22zz += prod * vk_dd;
                        vj_22zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+ka)*9 + 0, vk_22xx);
        atomicAdd(vj + (ka*natm+ka)*9 + 0, vj_22xx);
        atomicAdd(vk + (ka*natm+ka)*9 + 1, vk_22xy);
        atomicAdd(vj + (ka*natm+ka)*9 + 1, vj_22xy);
        atomicAdd(vk + (ka*natm+ka)*9 + 2, vk_22xz);
        atomicAdd(vj + (ka*natm+ka)*9 + 2, vj_22xz);
        atomicAdd(vk + (ka*natm+ka)*9 + 3, vk_22yx);
        atomicAdd(vj + (ka*natm+ka)*9 + 3, vj_22yx);
        atomicAdd(vk + (ka*natm+ka)*9 + 4, vk_22yy);
        atomicAdd(vj + (ka*natm+ka)*9 + 4, vj_22yy);
        atomicAdd(vk + (ka*natm+ka)*9 + 5, vk_22yz);
        atomicAdd(vj + (ka*natm+ka)*9 + 5, vj_22yz);
        atomicAdd(vk + (ka*natm+ka)*9 + 6, vk_22zx);
        atomicAdd(vj + (ka*natm+ka)*9 + 6, vj_22zx);
        atomicAdd(vk + (ka*natm+ka)*9 + 7, vk_22zy);
        atomicAdd(vj + (ka*natm+ka)*9 + 7, vj_22zy);
        atomicAdd(vk + (ka*natm+ka)*9 + 8, vk_22zz);
        atomicAdd(vj + (ka*natm+ka)*9 + 8, vj_22zz);

        double vk_23xx = 0;
        double vj_23xx = 0;
        double vk_23xy = 0;
        double vj_23xy = 0;
        double vk_23xz = 0;
        double vj_23xz = 0;
        double vk_23yx = 0;
        double vj_23yx = 0;
        double vk_23yy = 0;
        double vj_23yy = 0;
        double vk_23yz = 0;
        double vj_23yz = 0;
        double vk_23zx = 0;
        double vj_23zx = 0;
        double vk_23zy = 0;
        double vj_23zy = 0;
        double vk_23zz = 0;
        double vj_23zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double trr_01x = cpx * fac;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g3 = al*2 * ak*2 * hrr_0011x;
                        prod = g3 * 1 * wt;
                        vk_23xx += prod * vk_dd;
                        vj_23xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_23xy += prod * vk_dd;
                        vj_23xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_23xz += prod * vk_dd;
                        vj_23xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_23yx += prod * vk_dd;
                        vj_23yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3 = al*2 * ak*2 * hrr_0011y;
                        prod = g3 * fac * wt;
                        vk_23yy += prod * vk_dd;
                        vj_23yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_23yz += prod * vk_dd;
                        vj_23yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_23zx += prod * vk_dd;
                        vj_23zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_23zy += prod * vk_dd;
                        vj_23zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3 = al*2 * ak*2 * hrr_0011z;
                        prod = g3 * fac * 1;
                        vk_23zz += prod * vk_dd;
                        vj_23zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+la)*9 + 0, vk_23xx);
        atomicAdd(vj + (ka*natm+la)*9 + 0, vj_23xx);
        atomicAdd(vk + (ka*natm+la)*9 + 1, vk_23xy);
        atomicAdd(vj + (ka*natm+la)*9 + 1, vj_23xy);
        atomicAdd(vk + (ka*natm+la)*9 + 2, vk_23xz);
        atomicAdd(vj + (ka*natm+la)*9 + 2, vj_23xz);
        atomicAdd(vk + (ka*natm+la)*9 + 3, vk_23yx);
        atomicAdd(vj + (ka*natm+la)*9 + 3, vj_23yx);
        atomicAdd(vk + (ka*natm+la)*9 + 4, vk_23yy);
        atomicAdd(vj + (ka*natm+la)*9 + 4, vj_23yy);
        atomicAdd(vk + (ka*natm+la)*9 + 5, vk_23yz);
        atomicAdd(vj + (ka*natm+la)*9 + 5, vj_23yz);
        atomicAdd(vk + (ka*natm+la)*9 + 6, vk_23zx);
        atomicAdd(vj + (ka*natm+la)*9 + 6, vj_23zx);
        atomicAdd(vk + (ka*natm+la)*9 + 7, vk_23zy);
        atomicAdd(vj + (ka*natm+la)*9 + 7, vj_23zy);
        atomicAdd(vk + (ka*natm+la)*9 + 8, vk_23zz);
        atomicAdd(vj + (ka*natm+la)*9 + 8, vj_23zz);

        double vk_30xx = 0;
        double vj_30xx = 0;
        double vk_30xy = 0;
        double vj_30xy = 0;
        double vk_30xz = 0;
        double vj_30xz = 0;
        double vk_30yx = 0;
        double vj_30yx = 0;
        double vk_30yy = 0;
        double vj_30yy = 0;
        double vk_30yz = 0;
        double vj_30yz = 0;
        double vk_30zx = 0;
        double vj_30zx = 0;
        double vk_30zy = 0;
        double vj_30zy = 0;
        double vk_30zz = 0;
        double vj_30zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g3 = ai*2 * al*2 * hrr_1001x;
                        prod = g3 * 1 * wt;
                        vk_30xx += prod * vk_dd;
                        vj_30xx += prod * vj_dd;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g1 = al*2 * hrr_0001x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_30xy += prod * vk_dd;
                        vj_30xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_30xz += prod * vk_dd;
                        vj_30xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1 = al*2 * hrr_0001y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_30yx += prod * vk_dd;
                        vj_30yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g3 = ai*2 * al*2 * hrr_1001y;
                        prod = g3 * fac * wt;
                        vk_30yy += prod * vk_dd;
                        vj_30yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_30yz += prod * vk_dd;
                        vj_30yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1 = al*2 * hrr_0001z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_30zx += prod * vk_dd;
                        vj_30zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_30zy += prod * vk_dd;
                        vj_30zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g3 = ai*2 * al*2 * hrr_1001z;
                        prod = g3 * fac * 1;
                        vk_30zz += prod * vk_dd;
                        vj_30zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+ia)*9 + 0, vk_30xx);
        atomicAdd(vj + (la*natm+ia)*9 + 0, vj_30xx);
        atomicAdd(vk + (la*natm+ia)*9 + 1, vk_30xy);
        atomicAdd(vj + (la*natm+ia)*9 + 1, vj_30xy);
        atomicAdd(vk + (la*natm+ia)*9 + 2, vk_30xz);
        atomicAdd(vj + (la*natm+ia)*9 + 2, vj_30xz);
        atomicAdd(vk + (la*natm+ia)*9 + 3, vk_30yx);
        atomicAdd(vj + (la*natm+ia)*9 + 3, vj_30yx);
        atomicAdd(vk + (la*natm+ia)*9 + 4, vk_30yy);
        atomicAdd(vj + (la*natm+ia)*9 + 4, vj_30yy);
        atomicAdd(vk + (la*natm+ia)*9 + 5, vk_30yz);
        atomicAdd(vj + (la*natm+ia)*9 + 5, vj_30yz);
        atomicAdd(vk + (la*natm+ia)*9 + 6, vk_30zx);
        atomicAdd(vj + (la*natm+ia)*9 + 6, vj_30zx);
        atomicAdd(vk + (la*natm+ia)*9 + 7, vk_30zy);
        atomicAdd(vj + (la*natm+ia)*9 + 7, vj_30zy);
        atomicAdd(vk + (la*natm+ia)*9 + 8, vk_30zz);
        atomicAdd(vj + (la*natm+ia)*9 + 8, vj_30zz);

        double vk_31xx = 0;
        double vj_31xx = 0;
        double vk_31xy = 0;
        double vj_31xy = 0;
        double vk_31xz = 0;
        double vj_31xz = 0;
        double vk_31yx = 0;
        double vj_31yx = 0;
        double vk_31yy = 0;
        double vj_31yy = 0;
        double vk_31yz = 0;
        double vj_31yz = 0;
        double vk_31zx = 0;
        double vj_31zx = 0;
        double vk_31zy = 0;
        double vj_31zy = 0;
        double vk_31zz = 0;
        double vj_31zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b00 = .5 * rt_aa;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - (rj[0] - ri[0]) * hrr_0001x;
                        g3 = aj*2 * al*2 * hrr_0101x;
                        prod = g3 * 1 * wt;
                        vk_31xx += prod * vk_dd;
                        vj_31xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_31xy += prod * vk_dd;
                        vj_31xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_31xz += prod * vk_dd;
                        vj_31xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1 = al*2 * hrr_0001y;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_31yx += prod * vk_dd;
                        vj_31yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        double hrr_0101y = hrr_1001y - (rj[1] - ri[1]) * hrr_0001y;
                        g3 = aj*2 * al*2 * hrr_0101y;
                        prod = g3 * fac * wt;
                        vk_31yy += prod * vk_dd;
                        vj_31yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_31yz += prod * vk_dd;
                        vj_31yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1 = al*2 * hrr_0001z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_31zx += prod * vk_dd;
                        vj_31zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_31zy += prod * vk_dd;
                        vj_31zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        double hrr_0101z = hrr_1001z - (rj[2] - ri[2]) * hrr_0001z;
                        g3 = aj*2 * al*2 * hrr_0101z;
                        prod = g3 * fac * 1;
                        vk_31zz += prod * vk_dd;
                        vj_31zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+ja)*9 + 0, vk_31xx);
        atomicAdd(vj + (la*natm+ja)*9 + 0, vj_31xx);
        atomicAdd(vk + (la*natm+ja)*9 + 1, vk_31xy);
        atomicAdd(vj + (la*natm+ja)*9 + 1, vj_31xy);
        atomicAdd(vk + (la*natm+ja)*9 + 2, vk_31xz);
        atomicAdd(vj + (la*natm+ja)*9 + 2, vj_31xz);
        atomicAdd(vk + (la*natm+ja)*9 + 3, vk_31yx);
        atomicAdd(vj + (la*natm+ja)*9 + 3, vj_31yx);
        atomicAdd(vk + (la*natm+ja)*9 + 4, vk_31yy);
        atomicAdd(vj + (la*natm+ja)*9 + 4, vj_31yy);
        atomicAdd(vk + (la*natm+ja)*9 + 5, vk_31yz);
        atomicAdd(vj + (la*natm+ja)*9 + 5, vj_31yz);
        atomicAdd(vk + (la*natm+ja)*9 + 6, vk_31zx);
        atomicAdd(vj + (la*natm+ja)*9 + 6, vj_31zx);
        atomicAdd(vk + (la*natm+ja)*9 + 7, vk_31zy);
        atomicAdd(vj + (la*natm+ja)*9 + 7, vj_31zy);
        atomicAdd(vk + (la*natm+ja)*9 + 8, vk_31zz);
        atomicAdd(vj + (la*natm+ja)*9 + 8, vj_31zz);

        double vk_32xx = 0;
        double vj_32xx = 0;
        double vk_32xy = 0;
        double vj_32xy = 0;
        double vk_32xz = 0;
        double vj_32xz = 0;
        double vk_32yx = 0;
        double vj_32yx = 0;
        double vk_32yy = 0;
        double vj_32yy = 0;
        double vk_32yz = 0;
        double vj_32yz = 0;
        double vk_32zx = 0;
        double vj_32zx = 0;
        double vk_32zy = 0;
        double vj_32zy = 0;
        double vk_32zz = 0;
        double vj_32zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double trr_01x = cpx * fac;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g3 = ak*2 * al*2 * hrr_0011x;
                        prod = g3 * 1 * wt;
                        vk_32xx += prod * vk_dd;
                        vj_32xx += prod * vj_dd;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g1 = al*2 * hrr_0001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_32xy += prod * vk_dd;
                        vj_32xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_32xz += prod * vk_dd;
                        vj_32xz += prod * vj_dd;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1 = al*2 * hrr_0001y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_32yx += prod * vk_dd;
                        vj_32yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3 = ak*2 * al*2 * hrr_0011y;
                        prod = g3 * fac * wt;
                        vk_32yy += prod * vk_dd;
                        vj_32yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_32yz += prod * vk_dd;
                        vj_32yz += prod * vj_dd;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1 = al*2 * hrr_0001z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_32zx += prod * vk_dd;
                        vj_32zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_32zy += prod * vk_dd;
                        vj_32zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3 = ak*2 * al*2 * hrr_0011z;
                        prod = g3 * fac * 1;
                        vk_32zz += prod * vk_dd;
                        vj_32zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+ka)*9 + 0, vk_32xx);
        atomicAdd(vj + (la*natm+ka)*9 + 0, vj_32xx);
        atomicAdd(vk + (la*natm+ka)*9 + 1, vk_32xy);
        atomicAdd(vj + (la*natm+ka)*9 + 1, vj_32xy);
        atomicAdd(vk + (la*natm+ka)*9 + 2, vk_32xz);
        atomicAdd(vj + (la*natm+ka)*9 + 2, vj_32xz);
        atomicAdd(vk + (la*natm+ka)*9 + 3, vk_32yx);
        atomicAdd(vj + (la*natm+ka)*9 + 3, vj_32yx);
        atomicAdd(vk + (la*natm+ka)*9 + 4, vk_32yy);
        atomicAdd(vj + (la*natm+ka)*9 + 4, vj_32yy);
        atomicAdd(vk + (la*natm+ka)*9 + 5, vk_32yz);
        atomicAdd(vj + (la*natm+ka)*9 + 5, vj_32yz);
        atomicAdd(vk + (la*natm+ka)*9 + 6, vk_32zx);
        atomicAdd(vj + (la*natm+ka)*9 + 6, vj_32zx);
        atomicAdd(vk + (la*natm+ka)*9 + 7, vk_32zy);
        atomicAdd(vj + (la*natm+ka)*9 + 7, vj_32zy);
        atomicAdd(vk + (la*natm+ka)*9 + 8, vk_32zz);
        atomicAdd(vj + (la*natm+ka)*9 + 8, vj_32zz);

        double vk_33xx = 0;
        double vj_33xx = 0;
        double vk_33xy = 0;
        double vj_33xy = 0;
        double vk_33xz = 0;
        double vj_33xz = 0;
        double vk_33yx = 0;
        double vj_33yx = 0;
        double vk_33yy = 0;
        double vj_33yy = 0;
        double vk_33yz = 0;
        double vj_33yz = 0;
        double vk_33zx = 0;
        double vj_33zx = 0;
        double vk_33zy = 0;
        double vj_33zy = 0;
        double vk_33zz = 0;
        double vj_33zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double trr_01x = cpx * fac;
                        double b01 = .5/akl * (1 - rt_akl);
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        g3 = al*2 * (al*2 * hrr_0002x - 1 * fac);
                        prod = g3 * 1 * wt;
                        vk_33xx += prod * vk_dd;
                        vj_33xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_33xy += prod * vk_dd;
                        vj_33xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_33xz += prod * vk_dd;
                        vj_33xz += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_33yx += prod * vk_dd;
                        vj_33yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3 = al*2 * (al*2 * hrr_0002y - 1 * 1);
                        prod = g3 * fac * wt;
                        vk_33yy += prod * vk_dd;
                        vj_33yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_33yz += prod * vk_dd;
                        vj_33yz += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_33zx += prod * vk_dd;
                        vj_33zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_33zy += prod * vk_dd;
                        vj_33zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3 = al*2 * (al*2 * hrr_0002z - 1 * wt);
                        prod = g3 * fac * 1;
                        vk_33zz += prod * vk_dd;
                        vj_33zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+la)*9 + 0, vk_33xx);
        atomicAdd(vj + (la*natm+la)*9 + 0, vj_33xx);
        atomicAdd(vk + (la*natm+la)*9 + 1, vk_33xy);
        atomicAdd(vj + (la*natm+la)*9 + 1, vj_33xy);
        atomicAdd(vk + (la*natm+la)*9 + 2, vk_33xz);
        atomicAdd(vj + (la*natm+la)*9 + 2, vj_33xz);
        atomicAdd(vk + (la*natm+la)*9 + 3, vk_33yx);
        atomicAdd(vj + (la*natm+la)*9 + 3, vj_33yx);
        atomicAdd(vk + (la*natm+la)*9 + 4, vk_33yy);
        atomicAdd(vj + (la*natm+la)*9 + 4, vj_33yy);
        atomicAdd(vk + (la*natm+la)*9 + 5, vk_33yz);
        atomicAdd(vj + (la*natm+la)*9 + 5, vj_33yz);
        atomicAdd(vk + (la*natm+la)*9 + 6, vk_33zx);
        atomicAdd(vj + (la*natm+la)*9 + 6, vj_33zx);
        atomicAdd(vk + (la*natm+la)*9 + 7, vk_33zy);
        atomicAdd(vj + (la*natm+la)*9 + 7, vj_33zy);
        atomicAdd(vk + (la*natm+la)*9 + 8, vk_33zz);
        atomicAdd(vj + (la*natm+la)*9 + 8, vj_33zz);
    }
}
__global__
void rys_ejk_ip2_0000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip2_0000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__ static
void _rys_ejk_ip2_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    double *vj = jk.vj;
    double *vk = jk.vk;
    double *dm = jk.dm;
    extern __shared__ double dm_cache[];
    double *Rpa_cicj = dm_cache + 3 * TILE2;
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
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[sh_ij+3*TILE2] = ci[ip] * cj[jp] * Kab;
    }

    int ij = sq_id / TILE2;
    if (ij < 3) {
        int i = ij % 3;
        int j = ij / 3;
        int sh_ij = sq_id % TILE2;
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
        int natm = envs.natm;
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
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
        double dd_jk, dd_jl, vj_dd, vk_dd;
        double g1, g2, g3, prod;
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
        
        double vk_00xx = 0;
        double vj_00xx = 0;
        double vk_00xy = 0;
        double vj_00xy = 0;
        double vk_00xz = 0;
        double vj_00xz = 0;
        double vk_00yx = 0;
        double vj_00yx = 0;
        double vk_00yy = 0;
        double vj_00yy = 0;
        double vk_00yz = 0;
        double vj_00yz = 0;
        double vk_00zx = 0;
        double vj_00zx = 0;
        double vk_00zy = 0;
        double vj_00zy = 0;
        double vk_00zz = 0;
        double vj_00zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        g3 = ai*2 * (ai*2 * trr_30x - 3 * trr_10x);
                        prod = g3 * 1 * wt;
                        vk_00xx += prod * vk_dd;
                        vj_00xx += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_00xy += prod * vk_dd;
                        vj_00xy += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_00xz += prod * vk_dd;
                        vj_00xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * wt;
                        vk_00yx += prod * vk_dd;
                        vj_00yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        g3 = ai*2 * (ai*2 * trr_20y - 1 * 1);
                        prod = g3 * trr_10x * wt;
                        vk_00yy += prod * vk_dd;
                        vj_00yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10x;
                        vk_00yz += prod * vk_dd;
                        vj_00yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * 1;
                        vk_00zx += prod * vk_dd;
                        vj_00zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10x;
                        vk_00zy += prod * vk_dd;
                        vj_00zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        g3 = ai*2 * (ai*2 * trr_20z - 1 * wt);
                        prod = g3 * trr_10x * 1;
                        vk_00zz += prod * vk_dd;
                        vj_00zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * (ai*2 * trr_20x - 1 * fac);
                        prod = g3 * trr_10y * wt;
                        vk_00xx += prod * vk_dd;
                        vj_00xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * wt;
                        vk_00xy += prod * vk_dd;
                        vj_00xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10y;
                        vk_00xz += prod * vk_dd;
                        vj_00xz += prod * vj_dd;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_00yx += prod * vk_dd;
                        vj_00yx += prod * vj_dd;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        g3 = ai*2 * (ai*2 * trr_30y - 3 * trr_10y);
                        prod = g3 * fac * wt;
                        vk_00yy += prod * vk_dd;
                        vj_00yy += prod * vj_dd;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_00yz += prod * vk_dd;
                        vj_00yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10y;
                        vk_00zx += prod * vk_dd;
                        vj_00zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * fac;
                        vk_00zy += prod * vk_dd;
                        vj_00zy += prod * vj_dd;
                        g3 = ai*2 * (ai*2 * trr_20z - 1 * wt);
                        prod = g3 * fac * trr_10y;
                        vk_00zz += prod * vk_dd;
                        vj_00zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * (ai*2 * trr_20x - 1 * fac);
                        prod = g3 * 1 * trr_10z;
                        vk_00xx += prod * vk_dd;
                        vj_00xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10z;
                        vk_00xy += prod * vk_dd;
                        vj_00xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * 1;
                        vk_00xz += prod * vk_dd;
                        vj_00xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10z;
                        vk_00yx += prod * vk_dd;
                        vj_00yx += prod * vj_dd;
                        g3 = ai*2 * (ai*2 * trr_20y - 1 * 1);
                        prod = g3 * fac * trr_10z;
                        vk_00yy += prod * vk_dd;
                        vj_00yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * fac;
                        vk_00yz += prod * vk_dd;
                        vj_00yz += prod * vj_dd;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_00zx += prod * vk_dd;
                        vj_00zx += prod * vj_dd;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_00zy += prod * vk_dd;
                        vj_00zy += prod * vj_dd;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        g3 = ai*2 * (ai*2 * trr_30z - 3 * trr_10z);
                        prod = g3 * fac * 1;
                        vk_00zz += prod * vk_dd;
                        vj_00zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+ia)*9 + 0, vk_00xx);
        atomicAdd(vj + (ia*natm+ia)*9 + 0, vj_00xx);
        atomicAdd(vk + (ia*natm+ia)*9 + 1, vk_00xy);
        atomicAdd(vj + (ia*natm+ia)*9 + 1, vj_00xy);
        atomicAdd(vk + (ia*natm+ia)*9 + 2, vk_00xz);
        atomicAdd(vj + (ia*natm+ia)*9 + 2, vj_00xz);
        atomicAdd(vk + (ia*natm+ia)*9 + 3, vk_00yx);
        atomicAdd(vj + (ia*natm+ia)*9 + 3, vj_00yx);
        atomicAdd(vk + (ia*natm+ia)*9 + 4, vk_00yy);
        atomicAdd(vj + (ia*natm+ia)*9 + 4, vj_00yy);
        atomicAdd(vk + (ia*natm+ia)*9 + 5, vk_00yz);
        atomicAdd(vj + (ia*natm+ia)*9 + 5, vj_00yz);
        atomicAdd(vk + (ia*natm+ia)*9 + 6, vk_00zx);
        atomicAdd(vj + (ia*natm+ia)*9 + 6, vj_00zx);
        atomicAdd(vk + (ia*natm+ia)*9 + 7, vk_00zy);
        atomicAdd(vj + (ia*natm+ia)*9 + 7, vj_00zy);
        atomicAdd(vk + (ia*natm+ia)*9 + 8, vk_00zz);
        atomicAdd(vj + (ia*natm+ia)*9 + 8, vj_00zz);

        double vk_01xx = 0;
        double vj_01xx = 0;
        double vk_01xy = 0;
        double vj_01xy = 0;
        double vk_01xz = 0;
        double vj_01xz = 0;
        double vk_01yx = 0;
        double vj_01yx = 0;
        double vk_01yy = 0;
        double vj_01yy = 0;
        double vk_01yz = 0;
        double vj_01yz = 0;
        double vk_01zx = 0;
        double vj_01zx = 0;
        double vk_01zy = 0;
        double vj_01zy = 0;
        double vk_01zz = 0;
        double vj_01zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - (rj[0] - ri[0]) * trr_20x;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g3 = aj*2 * (ai*2 * hrr_2100x - 1 * hrr_0100x);
                        prod = g3 * 1 * wt;
                        vk_01xx += prod * vk_dd;
                        vj_01xx += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_01xy += prod * vk_dd;
                        vj_01xy += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_01xz += prod * vk_dd;
                        vj_01xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * wt;
                        vk_01yx += prod * vk_dd;
                        vj_01yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g3 = aj*2 * ai*2 * hrr_1100y;
                        prod = g3 * trr_10x * wt;
                        vk_01yy += prod * vk_dd;
                        vj_01yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10x;
                        vk_01yz += prod * vk_dd;
                        vj_01yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * 1;
                        vk_01zx += prod * vk_dd;
                        vj_01zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10x;
                        vk_01zy += prod * vk_dd;
                        vj_01zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g3 = aj*2 * ai*2 * hrr_1100z;
                        prod = g3 * trr_10x * 1;
                        vk_01zz += prod * vk_dd;
                        vj_01zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = aj*2 * ai*2 * hrr_1100x;
                        prod = g3 * trr_10y * wt;
                        vk_01xx += prod * vk_dd;
                        vj_01xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * wt;
                        vk_01xy += prod * vk_dd;
                        vj_01xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10y;
                        vk_01xz += prod * vk_dd;
                        vj_01xz += prod * vj_dd;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_01yx += prod * vk_dd;
                        vj_01yx += prod * vj_dd;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - (rj[1] - ri[1]) * trr_20y;
                        g3 = aj*2 * (ai*2 * hrr_2100y - 1 * hrr_0100y);
                        prod = g3 * fac * wt;
                        vk_01yy += prod * vk_dd;
                        vj_01yy += prod * vj_dd;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_01yz += prod * vk_dd;
                        vj_01yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10y;
                        vk_01zx += prod * vk_dd;
                        vj_01zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * fac;
                        vk_01zy += prod * vk_dd;
                        vj_01zy += prod * vj_dd;
                        g3 = aj*2 * ai*2 * hrr_1100z;
                        prod = g3 * fac * trr_10y;
                        vk_01zz += prod * vk_dd;
                        vj_01zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = aj*2 * ai*2 * hrr_1100x;
                        prod = g3 * 1 * trr_10z;
                        vk_01xx += prod * vk_dd;
                        vj_01xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10z;
                        vk_01xy += prod * vk_dd;
                        vj_01xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * 1;
                        vk_01xz += prod * vk_dd;
                        vj_01xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10z;
                        vk_01yx += prod * vk_dd;
                        vj_01yx += prod * vj_dd;
                        g3 = aj*2 * ai*2 * hrr_1100y;
                        prod = g3 * fac * trr_10z;
                        vk_01yy += prod * vk_dd;
                        vj_01yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * fac;
                        vk_01yz += prod * vk_dd;
                        vj_01yz += prod * vj_dd;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_01zx += prod * vk_dd;
                        vj_01zx += prod * vj_dd;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_01zy += prod * vk_dd;
                        vj_01zy += prod * vj_dd;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - (rj[2] - ri[2]) * trr_20z;
                        g3 = aj*2 * (ai*2 * hrr_2100z - 1 * hrr_0100z);
                        prod = g3 * fac * 1;
                        vk_01zz += prod * vk_dd;
                        vj_01zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+ja)*9 + 0, vk_01xx);
        atomicAdd(vj + (ia*natm+ja)*9 + 0, vj_01xx);
        atomicAdd(vk + (ia*natm+ja)*9 + 1, vk_01xy);
        atomicAdd(vj + (ia*natm+ja)*9 + 1, vj_01xy);
        atomicAdd(vk + (ia*natm+ja)*9 + 2, vk_01xz);
        atomicAdd(vj + (ia*natm+ja)*9 + 2, vj_01xz);
        atomicAdd(vk + (ia*natm+ja)*9 + 3, vk_01yx);
        atomicAdd(vj + (ia*natm+ja)*9 + 3, vj_01yx);
        atomicAdd(vk + (ia*natm+ja)*9 + 4, vk_01yy);
        atomicAdd(vj + (ia*natm+ja)*9 + 4, vj_01yy);
        atomicAdd(vk + (ia*natm+ja)*9 + 5, vk_01yz);
        atomicAdd(vj + (ia*natm+ja)*9 + 5, vj_01yz);
        atomicAdd(vk + (ia*natm+ja)*9 + 6, vk_01zx);
        atomicAdd(vj + (ia*natm+ja)*9 + 6, vj_01zx);
        atomicAdd(vk + (ia*natm+ja)*9 + 7, vk_01zy);
        atomicAdd(vj + (ia*natm+ja)*9 + 7, vj_01zy);
        atomicAdd(vk + (ia*natm+ja)*9 + 8, vk_01zz);
        atomicAdd(vj + (ia*natm+ja)*9 + 8, vj_01zz);

        double vk_02xx = 0;
        double vj_02xx = 0;
        double vk_02xy = 0;
        double vj_02xy = 0;
        double vk_02xz = 0;
        double vj_02xz = 0;
        double vk_02yx = 0;
        double vj_02yx = 0;
        double vk_02yy = 0;
        double vj_02yy = 0;
        double vk_02yz = 0;
        double vj_02yz = 0;
        double vk_02zx = 0;
        double vj_02zx = 0;
        double vk_02zy = 0;
        double vj_02zy = 0;
        double vk_02zz = 0;
        double vj_02zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_01x = cpx * fac;
                        g3 = ak*2 * (ai*2 * trr_21x - 1 * trr_01x);
                        prod = g3 * 1 * wt;
                        vk_02xx += prod * vk_dd;
                        vj_02xx += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_02xy += prod * vk_dd;
                        vj_02xy += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_02xz += prod * vk_dd;
                        vj_02xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g1 = ai*2 * trr_10y;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * wt;
                        vk_02yx += prod * vk_dd;
                        vj_02yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        g3 = ak*2 * ai*2 * trr_11y;
                        prod = g3 * trr_10x * wt;
                        vk_02yy += prod * vk_dd;
                        vj_02yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10x;
                        vk_02yz += prod * vk_dd;
                        vj_02yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g1 = ai*2 * trr_10z;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * 1;
                        vk_02zx += prod * vk_dd;
                        vj_02zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10x;
                        vk_02zy += prod * vk_dd;
                        vj_02zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        g3 = ak*2 * ai*2 * trr_11z;
                        prod = g3 * trr_10x * 1;
                        vk_02zz += prod * vk_dd;
                        vj_02zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ak*2 * ai*2 * trr_11x;
                        prod = g3 * trr_10y * wt;
                        vk_02xx += prod * vk_dd;
                        vj_02xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * wt;
                        vk_02xy += prod * vk_dd;
                        vj_02xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10y;
                        vk_02xz += prod * vk_dd;
                        vj_02xz += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_02yx += prod * vk_dd;
                        vj_02yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        g3 = ak*2 * (ai*2 * trr_21y - 1 * trr_01y);
                        prod = g3 * fac * wt;
                        vk_02yy += prod * vk_dd;
                        vj_02yy += prod * vj_dd;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_02yz += prod * vk_dd;
                        vj_02yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10y;
                        vk_02zx += prod * vk_dd;
                        vj_02zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * fac;
                        vk_02zy += prod * vk_dd;
                        vj_02zy += prod * vj_dd;
                        g3 = ak*2 * ai*2 * trr_11z;
                        prod = g3 * fac * trr_10y;
                        vk_02zz += prod * vk_dd;
                        vj_02zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ak*2 * ai*2 * trr_11x;
                        prod = g3 * 1 * trr_10z;
                        vk_02xx += prod * vk_dd;
                        vj_02xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10z;
                        vk_02xy += prod * vk_dd;
                        vj_02xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * 1;
                        vk_02xz += prod * vk_dd;
                        vj_02xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10z;
                        vk_02yx += prod * vk_dd;
                        vj_02yx += prod * vj_dd;
                        g3 = ak*2 * ai*2 * trr_11y;
                        prod = g3 * fac * trr_10z;
                        vk_02yy += prod * vk_dd;
                        vj_02yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * fac;
                        vk_02yz += prod * vk_dd;
                        vj_02yz += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_02zx += prod * vk_dd;
                        vj_02zx += prod * vj_dd;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_02zy += prod * vk_dd;
                        vj_02zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        g3 = ak*2 * (ai*2 * trr_21z - 1 * trr_01z);
                        prod = g3 * fac * 1;
                        vk_02zz += prod * vk_dd;
                        vj_02zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+ka)*9 + 0, vk_02xx);
        atomicAdd(vj + (ia*natm+ka)*9 + 0, vj_02xx);
        atomicAdd(vk + (ia*natm+ka)*9 + 1, vk_02xy);
        atomicAdd(vj + (ia*natm+ka)*9 + 1, vj_02xy);
        atomicAdd(vk + (ia*natm+ka)*9 + 2, vk_02xz);
        atomicAdd(vj + (ia*natm+ka)*9 + 2, vj_02xz);
        atomicAdd(vk + (ia*natm+ka)*9 + 3, vk_02yx);
        atomicAdd(vj + (ia*natm+ka)*9 + 3, vj_02yx);
        atomicAdd(vk + (ia*natm+ka)*9 + 4, vk_02yy);
        atomicAdd(vj + (ia*natm+ka)*9 + 4, vj_02yy);
        atomicAdd(vk + (ia*natm+ka)*9 + 5, vk_02yz);
        atomicAdd(vj + (ia*natm+ka)*9 + 5, vj_02yz);
        atomicAdd(vk + (ia*natm+ka)*9 + 6, vk_02zx);
        atomicAdd(vj + (ia*natm+ka)*9 + 6, vj_02zx);
        atomicAdd(vk + (ia*natm+ka)*9 + 7, vk_02zy);
        atomicAdd(vj + (ia*natm+ka)*9 + 7, vj_02zy);
        atomicAdd(vk + (ia*natm+ka)*9 + 8, vk_02zz);
        atomicAdd(vj + (ia*natm+ka)*9 + 8, vj_02zz);

        double vk_03xx = 0;
        double vj_03xx = 0;
        double vk_03xy = 0;
        double vj_03xy = 0;
        double vk_03xz = 0;
        double vj_03xz = 0;
        double vk_03yx = 0;
        double vj_03yx = 0;
        double vk_03yy = 0;
        double vj_03yy = 0;
        double vk_03yz = 0;
        double vj_03yz = 0;
        double vk_03zx = 0;
        double vj_03zx = 0;
        double vk_03zy = 0;
        double vj_03zy = 0;
        double vk_03zz = 0;
        double vj_03zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g3 = al*2 * (ai*2 * hrr_2001x - 1 * hrr_0001x);
                        prod = g3 * 1 * wt;
                        vk_03xx += prod * vk_dd;
                        vj_03xx += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_03xy += prod * vk_dd;
                        vj_03xy += prod * vj_dd;
                        g1 = ai*2 * trr_20x;
                        g1 -= 1 * fac;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_03xz += prod * vk_dd;
                        vj_03xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g1 = ai*2 * trr_10y;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * wt;
                        vk_03yx += prod * vk_dd;
                        vj_03yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g3 = al*2 * ai*2 * hrr_1001y;
                        prod = g3 * trr_10x * wt;
                        vk_03yy += prod * vk_dd;
                        vj_03yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10x;
                        vk_03yz += prod * vk_dd;
                        vj_03yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g1 = ai*2 * trr_10z;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * 1;
                        vk_03zx += prod * vk_dd;
                        vj_03zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10x;
                        vk_03zy += prod * vk_dd;
                        vj_03zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g3 = al*2 * ai*2 * hrr_1001z;
                        prod = g3 * trr_10x * 1;
                        vk_03zz += prod * vk_dd;
                        vj_03zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = al*2 * ai*2 * hrr_1001x;
                        prod = g3 * trr_10y * wt;
                        vk_03xx += prod * vk_dd;
                        vj_03xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * wt;
                        vk_03xy += prod * vk_dd;
                        vj_03xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10y;
                        vk_03xz += prod * vk_dd;
                        vj_03xz += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_03yx += prod * vk_dd;
                        vj_03yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        g3 = al*2 * (ai*2 * hrr_2001y - 1 * hrr_0001y);
                        prod = g3 * fac * wt;
                        vk_03yy += prod * vk_dd;
                        vj_03yy += prod * vj_dd;
                        g1 = ai*2 * trr_20y;
                        g1 -= 1 * 1;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_03yz += prod * vk_dd;
                        vj_03yz += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10y;
                        vk_03zx += prod * vk_dd;
                        vj_03zx += prod * vj_dd;
                        g1 = ai*2 * trr_10z;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * fac;
                        vk_03zy += prod * vk_dd;
                        vj_03zy += prod * vj_dd;
                        g3 = al*2 * ai*2 * hrr_1001z;
                        prod = g3 * fac * trr_10y;
                        vk_03zz += prod * vk_dd;
                        vj_03zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = al*2 * ai*2 * hrr_1001x;
                        prod = g3 * 1 * trr_10z;
                        vk_03xx += prod * vk_dd;
                        vj_03xx += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10z;
                        vk_03xy += prod * vk_dd;
                        vj_03xy += prod * vj_dd;
                        g1 = ai*2 * trr_10x;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * 1;
                        vk_03xz += prod * vk_dd;
                        vj_03xz += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10z;
                        vk_03yx += prod * vk_dd;
                        vj_03yx += prod * vj_dd;
                        g3 = al*2 * ai*2 * hrr_1001y;
                        prod = g3 * fac * trr_10z;
                        vk_03yy += prod * vk_dd;
                        vj_03yy += prod * vj_dd;
                        g1 = ai*2 * trr_10y;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * fac;
                        vk_03yz += prod * vk_dd;
                        vj_03yz += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_03zx += prod * vk_dd;
                        vj_03zx += prod * vj_dd;
                        g1 = ai*2 * trr_20z;
                        g1 -= 1 * wt;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_03zy += prod * vk_dd;
                        vj_03zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        g3 = al*2 * (ai*2 * hrr_2001z - 1 * hrr_0001z);
                        prod = g3 * fac * 1;
                        vk_03zz += prod * vk_dd;
                        vj_03zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ia*natm+la)*9 + 0, vk_03xx);
        atomicAdd(vj + (ia*natm+la)*9 + 0, vj_03xx);
        atomicAdd(vk + (ia*natm+la)*9 + 1, vk_03xy);
        atomicAdd(vj + (ia*natm+la)*9 + 1, vj_03xy);
        atomicAdd(vk + (ia*natm+la)*9 + 2, vk_03xz);
        atomicAdd(vj + (ia*natm+la)*9 + 2, vj_03xz);
        atomicAdd(vk + (ia*natm+la)*9 + 3, vk_03yx);
        atomicAdd(vj + (ia*natm+la)*9 + 3, vj_03yx);
        atomicAdd(vk + (ia*natm+la)*9 + 4, vk_03yy);
        atomicAdd(vj + (ia*natm+la)*9 + 4, vj_03yy);
        atomicAdd(vk + (ia*natm+la)*9 + 5, vk_03yz);
        atomicAdd(vj + (ia*natm+la)*9 + 5, vj_03yz);
        atomicAdd(vk + (ia*natm+la)*9 + 6, vk_03zx);
        atomicAdd(vj + (ia*natm+la)*9 + 6, vj_03zx);
        atomicAdd(vk + (ia*natm+la)*9 + 7, vk_03zy);
        atomicAdd(vj + (ia*natm+la)*9 + 7, vj_03zy);
        atomicAdd(vk + (ia*natm+la)*9 + 8, vk_03zz);
        atomicAdd(vj + (ia*natm+la)*9 + 8, vj_03zz);

        double vk_10xx = 0;
        double vj_10xx = 0;
        double vk_10xy = 0;
        double vj_10xy = 0;
        double vk_10xz = 0;
        double vj_10xz = 0;
        double vk_10yx = 0;
        double vj_10yx = 0;
        double vk_10yy = 0;
        double vj_10yy = 0;
        double vk_10yz = 0;
        double vj_10yz = 0;
        double vk_10zx = 0;
        double vj_10zx = 0;
        double vk_10zy = 0;
        double vj_10zy = 0;
        double vk_10zz = 0;
        double vj_10zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - (rj[0] - ri[0]) * trr_20x;
                        g3 = ai*2 * aj*2 * hrr_2100x;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g3 -= 1 * aj*2 * hrr_0100x;
                        prod = g3 * 1 * wt;
                        vk_10xx += prod * vk_dd;
                        vj_10xx += prod * vj_dd;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g1 = aj*2 * hrr_1100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_10xy += prod * vk_dd;
                        vj_10xy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_10xz += prod * vk_dd;
                        vj_10xz += prod * vj_dd;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * wt;
                        vk_10yx += prod * vk_dd;
                        vj_10yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g3 = ai*2 * aj*2 * hrr_1100y;
                        prod = g3 * trr_10x * wt;
                        vk_10yy += prod * vk_dd;
                        vj_10yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10x;
                        vk_10yz += prod * vk_dd;
                        vj_10yz += prod * vj_dd;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * 1;
                        vk_10zx += prod * vk_dd;
                        vj_10zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10x;
                        vk_10zy += prod * vk_dd;
                        vj_10zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g3 = ai*2 * aj*2 * hrr_1100z;
                        prod = g3 * trr_10x * 1;
                        vk_10zz += prod * vk_dd;
                        vj_10zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * aj*2 * hrr_1100x;
                        prod = g3 * trr_10y * wt;
                        vk_10xx += prod * vk_dd;
                        vj_10xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * wt;
                        vk_10xy += prod * vk_dd;
                        vj_10xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10y;
                        vk_10xz += prod * vk_dd;
                        vj_10xz += prod * vj_dd;
                        g1 = aj*2 * hrr_1100y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_10yx += prod * vk_dd;
                        vj_10yx += prod * vj_dd;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - (rj[1] - ri[1]) * trr_20y;
                        g3 = ai*2 * aj*2 * hrr_2100y;
                        g3 -= 1 * aj*2 * hrr_0100y;
                        prod = g3 * fac * wt;
                        vk_10yy += prod * vk_dd;
                        vj_10yy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_10yz += prod * vk_dd;
                        vj_10yz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10y;
                        vk_10zx += prod * vk_dd;
                        vj_10zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * fac;
                        vk_10zy += prod * vk_dd;
                        vj_10zy += prod * vj_dd;
                        g3 = ai*2 * aj*2 * hrr_1100z;
                        prod = g3 * fac * trr_10y;
                        vk_10zz += prod * vk_dd;
                        vj_10zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * aj*2 * hrr_1100x;
                        prod = g3 * 1 * trr_10z;
                        vk_10xx += prod * vk_dd;
                        vj_10xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10z;
                        vk_10xy += prod * vk_dd;
                        vj_10xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * 1;
                        vk_10xz += prod * vk_dd;
                        vj_10xz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10z;
                        vk_10yx += prod * vk_dd;
                        vj_10yx += prod * vj_dd;
                        g3 = ai*2 * aj*2 * hrr_1100y;
                        prod = g3 * fac * trr_10z;
                        vk_10yy += prod * vk_dd;
                        vj_10yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * fac;
                        vk_10yz += prod * vk_dd;
                        vj_10yz += prod * vj_dd;
                        g1 = aj*2 * hrr_1100z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_10zx += prod * vk_dd;
                        vj_10zx += prod * vj_dd;
                        g1 = aj*2 * hrr_1100z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_10zy += prod * vk_dd;
                        vj_10zy += prod * vj_dd;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - (rj[2] - ri[2]) * trr_20z;
                        g3 = ai*2 * aj*2 * hrr_2100z;
                        g3 -= 1 * aj*2 * hrr_0100z;
                        prod = g3 * fac * 1;
                        vk_10zz += prod * vk_dd;
                        vj_10zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+ia)*9 + 0, vk_10xx);
        atomicAdd(vj + (ja*natm+ia)*9 + 0, vj_10xx);
        atomicAdd(vk + (ja*natm+ia)*9 + 1, vk_10xy);
        atomicAdd(vj + (ja*natm+ia)*9 + 1, vj_10xy);
        atomicAdd(vk + (ja*natm+ia)*9 + 2, vk_10xz);
        atomicAdd(vj + (ja*natm+ia)*9 + 2, vj_10xz);
        atomicAdd(vk + (ja*natm+ia)*9 + 3, vk_10yx);
        atomicAdd(vj + (ja*natm+ia)*9 + 3, vj_10yx);
        atomicAdd(vk + (ja*natm+ia)*9 + 4, vk_10yy);
        atomicAdd(vj + (ja*natm+ia)*9 + 4, vj_10yy);
        atomicAdd(vk + (ja*natm+ia)*9 + 5, vk_10yz);
        atomicAdd(vj + (ja*natm+ia)*9 + 5, vj_10yz);
        atomicAdd(vk + (ja*natm+ia)*9 + 6, vk_10zx);
        atomicAdd(vj + (ja*natm+ia)*9 + 6, vj_10zx);
        atomicAdd(vk + (ja*natm+ia)*9 + 7, vk_10zy);
        atomicAdd(vj + (ja*natm+ia)*9 + 7, vj_10zy);
        atomicAdd(vk + (ja*natm+ia)*9 + 8, vk_10zz);
        atomicAdd(vj + (ja*natm+ia)*9 + 8, vj_10zz);

        double vk_11xx = 0;
        double vj_11xx = 0;
        double vk_11xy = 0;
        double vj_11xy = 0;
        double vk_11xz = 0;
        double vj_11xz = 0;
        double vk_11yx = 0;
        double vj_11yx = 0;
        double vk_11yy = 0;
        double vj_11yy = 0;
        double vk_11yz = 0;
        double vj_11yz = 0;
        double vk_11zx = 0;
        double vj_11zx = 0;
        double vk_11zy = 0;
        double vj_11zy = 0;
        double vk_11zz = 0;
        double vj_11zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - (rj[0] - ri[0]) * trr_20x;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        double hrr_1200x = hrr_2100x - (rj[0] - ri[0]) * hrr_1100x;
                        g3 = aj*2 * (aj*2 * hrr_1200x - 1 * trr_10x);
                        prod = g3 * 1 * wt;
                        vk_11xx += prod * vk_dd;
                        vj_11xx += prod * vj_dd;
                        g1 = aj*2 * hrr_1100x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_11xy += prod * vk_dd;
                        vj_11xy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_11xz += prod * vk_dd;
                        vj_11xz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * wt;
                        vk_11yx += prod * vk_dd;
                        vj_11yx += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        double hrr_0200y = hrr_1100y - (rj[1] - ri[1]) * hrr_0100y;
                        g3 = aj*2 * (aj*2 * hrr_0200y - 1 * 1);
                        prod = g3 * trr_10x * wt;
                        vk_11yy += prod * vk_dd;
                        vj_11yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10x;
                        vk_11yz += prod * vk_dd;
                        vj_11yz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * 1;
                        vk_11zx += prod * vk_dd;
                        vj_11zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10x;
                        vk_11zy += prod * vk_dd;
                        vj_11zy += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        double hrr_0200z = hrr_1100z - (rj[2] - ri[2]) * hrr_0100z;
                        g3 = aj*2 * (aj*2 * hrr_0200z - 1 * wt);
                        prod = g3 * trr_10x * 1;
                        vk_11zz += prod * vk_dd;
                        vj_11zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        double hrr_0200x = hrr_1100x - (rj[0] - ri[0]) * hrr_0100x;
                        g3 = aj*2 * (aj*2 * hrr_0200x - 1 * fac);
                        prod = g3 * trr_10y * wt;
                        vk_11xx += prod * vk_dd;
                        vj_11xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * wt;
                        vk_11xy += prod * vk_dd;
                        vj_11xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10y;
                        vk_11xz += prod * vk_dd;
                        vj_11xz += prod * vj_dd;
                        g1 = aj*2 * hrr_1100y;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_11yx += prod * vk_dd;
                        vj_11yx += prod * vj_dd;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - (rj[1] - ri[1]) * trr_20y;
                        double hrr_1200y = hrr_2100y - (rj[1] - ri[1]) * hrr_1100y;
                        g3 = aj*2 * (aj*2 * hrr_1200y - 1 * trr_10y);
                        prod = g3 * fac * wt;
                        vk_11yy += prod * vk_dd;
                        vj_11yy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_11yz += prod * vk_dd;
                        vj_11yz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10y;
                        vk_11zx += prod * vk_dd;
                        vj_11zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * fac;
                        vk_11zy += prod * vk_dd;
                        vj_11zy += prod * vj_dd;
                        g3 = aj*2 * (aj*2 * hrr_0200z - 1 * wt);
                        prod = g3 * fac * trr_10y;
                        vk_11zz += prod * vk_dd;
                        vj_11zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = aj*2 * (aj*2 * hrr_0200x - 1 * fac);
                        prod = g3 * 1 * trr_10z;
                        vk_11xx += prod * vk_dd;
                        vj_11xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10z;
                        vk_11xy += prod * vk_dd;
                        vj_11xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * 1;
                        vk_11xz += prod * vk_dd;
                        vj_11xz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10z;
                        vk_11yx += prod * vk_dd;
                        vj_11yx += prod * vj_dd;
                        g3 = aj*2 * (aj*2 * hrr_0200y - 1 * 1);
                        prod = g3 * fac * trr_10z;
                        vk_11yy += prod * vk_dd;
                        vj_11yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * fac;
                        vk_11yz += prod * vk_dd;
                        vj_11yz += prod * vj_dd;
                        g1 = aj*2 * hrr_1100z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_11zx += prod * vk_dd;
                        vj_11zx += prod * vj_dd;
                        g1 = aj*2 * hrr_1100z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_11zy += prod * vk_dd;
                        vj_11zy += prod * vj_dd;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - (rj[2] - ri[2]) * trr_20z;
                        double hrr_1200z = hrr_2100z - (rj[2] - ri[2]) * hrr_1100z;
                        g3 = aj*2 * (aj*2 * hrr_1200z - 1 * trr_10z);
                        prod = g3 * fac * 1;
                        vk_11zz += prod * vk_dd;
                        vj_11zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+ja)*9 + 0, vk_11xx);
        atomicAdd(vj + (ja*natm+ja)*9 + 0, vj_11xx);
        atomicAdd(vk + (ja*natm+ja)*9 + 1, vk_11xy);
        atomicAdd(vj + (ja*natm+ja)*9 + 1, vj_11xy);
        atomicAdd(vk + (ja*natm+ja)*9 + 2, vk_11xz);
        atomicAdd(vj + (ja*natm+ja)*9 + 2, vj_11xz);
        atomicAdd(vk + (ja*natm+ja)*9 + 3, vk_11yx);
        atomicAdd(vj + (ja*natm+ja)*9 + 3, vj_11yx);
        atomicAdd(vk + (ja*natm+ja)*9 + 4, vk_11yy);
        atomicAdd(vj + (ja*natm+ja)*9 + 4, vj_11yy);
        atomicAdd(vk + (ja*natm+ja)*9 + 5, vk_11yz);
        atomicAdd(vj + (ja*natm+ja)*9 + 5, vj_11yz);
        atomicAdd(vk + (ja*natm+ja)*9 + 6, vk_11zx);
        atomicAdd(vj + (ja*natm+ja)*9 + 6, vj_11zx);
        atomicAdd(vk + (ja*natm+ja)*9 + 7, vk_11zy);
        atomicAdd(vj + (ja*natm+ja)*9 + 7, vj_11zy);
        atomicAdd(vk + (ja*natm+ja)*9 + 8, vk_11zz);
        atomicAdd(vj + (ja*natm+ja)*9 + 8, vj_11zz);

        double vk_12xx = 0;
        double vj_12xx = 0;
        double vk_12xy = 0;
        double vj_12xy = 0;
        double vk_12xz = 0;
        double vj_12xz = 0;
        double vk_12yx = 0;
        double vj_12yx = 0;
        double vk_12yy = 0;
        double vj_12yy = 0;
        double vk_12yz = 0;
        double vj_12yz = 0;
        double vk_12zx = 0;
        double vj_12zx = 0;
        double vk_12zy = 0;
        double vj_12zy = 0;
        double vk_12zz = 0;
        double vj_12zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
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
                        double hrr_1110x = trr_21x - (rj[0] - ri[0]) * trr_11x;
                        g3 = ak*2 * aj*2 * hrr_1110x;
                        prod = g3 * 1 * wt;
                        vk_12xx += prod * vk_dd;
                        vj_12xx += prod * vj_dd;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g1 = aj*2 * hrr_1100x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_12xy += prod * vk_dd;
                        vj_12xy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_12xz += prod * vk_dd;
                        vj_12xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * wt;
                        vk_12yx += prod * vk_dd;
                        vj_12yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - (rj[1] - ri[1]) * trr_01y;
                        g3 = ak*2 * aj*2 * hrr_0110y;
                        prod = g3 * trr_10x * wt;
                        vk_12yy += prod * vk_dd;
                        vj_12yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10x;
                        vk_12yz += prod * vk_dd;
                        vj_12yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * 1;
                        vk_12zx += prod * vk_dd;
                        vj_12zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10x;
                        vk_12zy += prod * vk_dd;
                        vj_12zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - (rj[2] - ri[2]) * trr_01z;
                        g3 = ak*2 * aj*2 * hrr_0110z;
                        prod = g3 * trr_10x * 1;
                        vk_12zz += prod * vk_dd;
                        vj_12zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - (rj[0] - ri[0]) * trr_01x;
                        g3 = ak*2 * aj*2 * hrr_0110x;
                        prod = g3 * trr_10y * wt;
                        vk_12xx += prod * vk_dd;
                        vj_12xx += prod * vj_dd;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * wt;
                        vk_12xy += prod * vk_dd;
                        vj_12xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10y;
                        vk_12xz += prod * vk_dd;
                        vj_12xz += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g1 = aj*2 * hrr_1100y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_12yx += prod * vk_dd;
                        vj_12yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - (rj[1] - ri[1]) * trr_11y;
                        g3 = ak*2 * aj*2 * hrr_1110y;
                        prod = g3 * fac * wt;
                        vk_12yy += prod * vk_dd;
                        vj_12yy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_12yz += prod * vk_dd;
                        vj_12yz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10y;
                        vk_12zx += prod * vk_dd;
                        vj_12zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * fac;
                        vk_12zy += prod * vk_dd;
                        vj_12zy += prod * vj_dd;
                        g3 = ak*2 * aj*2 * hrr_0110z;
                        prod = g3 * fac * trr_10y;
                        vk_12zz += prod * vk_dd;
                        vj_12zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ak*2 * aj*2 * hrr_0110x;
                        prod = g3 * 1 * trr_10z;
                        vk_12xx += prod * vk_dd;
                        vj_12xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10z;
                        vk_12xy += prod * vk_dd;
                        vj_12xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * 1;
                        vk_12xz += prod * vk_dd;
                        vj_12xz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10z;
                        vk_12yx += prod * vk_dd;
                        vj_12yx += prod * vj_dd;
                        g3 = ak*2 * aj*2 * hrr_0110y;
                        prod = g3 * fac * trr_10z;
                        vk_12yy += prod * vk_dd;
                        vj_12yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * fac;
                        vk_12yz += prod * vk_dd;
                        vj_12yz += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g1 = aj*2 * hrr_1100z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_12zx += prod * vk_dd;
                        vj_12zx += prod * vj_dd;
                        g1 = aj*2 * hrr_1100z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_12zy += prod * vk_dd;
                        vj_12zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - (rj[2] - ri[2]) * trr_11z;
                        g3 = ak*2 * aj*2 * hrr_1110z;
                        prod = g3 * fac * 1;
                        vk_12zz += prod * vk_dd;
                        vj_12zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+ka)*9 + 0, vk_12xx);
        atomicAdd(vj + (ja*natm+ka)*9 + 0, vj_12xx);
        atomicAdd(vk + (ja*natm+ka)*9 + 1, vk_12xy);
        atomicAdd(vj + (ja*natm+ka)*9 + 1, vj_12xy);
        atomicAdd(vk + (ja*natm+ka)*9 + 2, vk_12xz);
        atomicAdd(vj + (ja*natm+ka)*9 + 2, vj_12xz);
        atomicAdd(vk + (ja*natm+ka)*9 + 3, vk_12yx);
        atomicAdd(vj + (ja*natm+ka)*9 + 3, vj_12yx);
        atomicAdd(vk + (ja*natm+ka)*9 + 4, vk_12yy);
        atomicAdd(vj + (ja*natm+ka)*9 + 4, vj_12yy);
        atomicAdd(vk + (ja*natm+ka)*9 + 5, vk_12yz);
        atomicAdd(vj + (ja*natm+ka)*9 + 5, vj_12yz);
        atomicAdd(vk + (ja*natm+ka)*9 + 6, vk_12zx);
        atomicAdd(vj + (ja*natm+ka)*9 + 6, vj_12zx);
        atomicAdd(vk + (ja*natm+ka)*9 + 7, vk_12zy);
        atomicAdd(vj + (ja*natm+ka)*9 + 7, vj_12zy);
        atomicAdd(vk + (ja*natm+ka)*9 + 8, vk_12zz);
        atomicAdd(vj + (ja*natm+ka)*9 + 8, vj_12zz);

        double vk_13xx = 0;
        double vj_13xx = 0;
        double vk_13xy = 0;
        double vj_13xy = 0;
        double vk_13xz = 0;
        double vj_13xz = 0;
        double vk_13yx = 0;
        double vj_13yx = 0;
        double vk_13yy = 0;
        double vj_13yy = 0;
        double vk_13yz = 0;
        double vj_13yz = 0;
        double vk_13zx = 0;
        double vj_13zx = 0;
        double vk_13zy = 0;
        double vj_13zy = 0;
        double vk_13zz = 0;
        double vj_13zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - (rj[0] - ri[0]) * hrr_1001x;
                        g3 = al*2 * aj*2 * hrr_1101x;
                        prod = g3 * 1 * wt;
                        vk_13xx += prod * vk_dd;
                        vj_13xx += prod * vj_dd;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g1 = aj*2 * hrr_1100x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_13xy += prod * vk_dd;
                        vj_13xy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_13xz += prod * vk_dd;
                        vj_13xz += prod * vj_dd;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g1 = aj*2 * hrr_0100y;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * wt;
                        vk_13yx += prod * vk_dd;
                        vj_13yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        double hrr_0101y = hrr_1001y - (rj[1] - ri[1]) * hrr_0001y;
                        g3 = al*2 * aj*2 * hrr_0101y;
                        prod = g3 * trr_10x * wt;
                        vk_13yy += prod * vk_dd;
                        vj_13yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10x;
                        vk_13yz += prod * vk_dd;
                        vj_13yz += prod * vj_dd;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g1 = aj*2 * hrr_0100z;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * 1;
                        vk_13zx += prod * vk_dd;
                        vj_13zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10x;
                        vk_13zy += prod * vk_dd;
                        vj_13zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        double hrr_0101z = hrr_1001z - (rj[2] - ri[2]) * hrr_0001z;
                        g3 = al*2 * aj*2 * hrr_0101z;
                        prod = g3 * trr_10x * 1;
                        vk_13zz += prod * vk_dd;
                        vj_13zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - (rj[0] - ri[0]) * hrr_0001x;
                        g3 = al*2 * aj*2 * hrr_0101x;
                        prod = g3 * trr_10y * wt;
                        vk_13xx += prod * vk_dd;
                        vj_13xx += prod * vj_dd;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g1 = aj*2 * hrr_0100x;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * wt;
                        vk_13xy += prod * vk_dd;
                        vj_13xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10y;
                        vk_13xz += prod * vk_dd;
                        vj_13xz += prod * vj_dd;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g1 = aj*2 * hrr_1100y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_13yx += prod * vk_dd;
                        vj_13yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - (rj[1] - ri[1]) * hrr_1001y;
                        g3 = al*2 * aj*2 * hrr_1101y;
                        prod = g3 * fac * wt;
                        vk_13yy += prod * vk_dd;
                        vj_13yy += prod * vj_dd;
                        g1 = aj*2 * hrr_1100y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_13yz += prod * vk_dd;
                        vj_13yz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10y;
                        vk_13zx += prod * vk_dd;
                        vj_13zx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100z;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * fac;
                        vk_13zy += prod * vk_dd;
                        vj_13zy += prod * vj_dd;
                        g3 = al*2 * aj*2 * hrr_0101z;
                        prod = g3 * fac * trr_10y;
                        vk_13zz += prod * vk_dd;
                        vj_13zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = al*2 * aj*2 * hrr_0101x;
                        prod = g3 * 1 * trr_10z;
                        vk_13xx += prod * vk_dd;
                        vj_13xx += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10z;
                        vk_13xy += prod * vk_dd;
                        vj_13xy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100x;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * 1;
                        vk_13xz += prod * vk_dd;
                        vj_13xz += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10z;
                        vk_13yx += prod * vk_dd;
                        vj_13yx += prod * vj_dd;
                        g3 = al*2 * aj*2 * hrr_0101y;
                        prod = g3 * fac * trr_10z;
                        vk_13yy += prod * vk_dd;
                        vj_13yy += prod * vj_dd;
                        g1 = aj*2 * hrr_0100y;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * fac;
                        vk_13yz += prod * vk_dd;
                        vj_13yz += prod * vj_dd;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g1 = aj*2 * hrr_1100z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_13zx += prod * vk_dd;
                        vj_13zx += prod * vj_dd;
                        g1 = aj*2 * hrr_1100z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_13zy += prod * vk_dd;
                        vj_13zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - (rj[2] - ri[2]) * hrr_1001z;
                        g3 = al*2 * aj*2 * hrr_1101z;
                        prod = g3 * fac * 1;
                        vk_13zz += prod * vk_dd;
                        vj_13zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ja*natm+la)*9 + 0, vk_13xx);
        atomicAdd(vj + (ja*natm+la)*9 + 0, vj_13xx);
        atomicAdd(vk + (ja*natm+la)*9 + 1, vk_13xy);
        atomicAdd(vj + (ja*natm+la)*9 + 1, vj_13xy);
        atomicAdd(vk + (ja*natm+la)*9 + 2, vk_13xz);
        atomicAdd(vj + (ja*natm+la)*9 + 2, vj_13xz);
        atomicAdd(vk + (ja*natm+la)*9 + 3, vk_13yx);
        atomicAdd(vj + (ja*natm+la)*9 + 3, vj_13yx);
        atomicAdd(vk + (ja*natm+la)*9 + 4, vk_13yy);
        atomicAdd(vj + (ja*natm+la)*9 + 4, vj_13yy);
        atomicAdd(vk + (ja*natm+la)*9 + 5, vk_13yz);
        atomicAdd(vj + (ja*natm+la)*9 + 5, vj_13yz);
        atomicAdd(vk + (ja*natm+la)*9 + 6, vk_13zx);
        atomicAdd(vj + (ja*natm+la)*9 + 6, vj_13zx);
        atomicAdd(vk + (ja*natm+la)*9 + 7, vk_13zy);
        atomicAdd(vj + (ja*natm+la)*9 + 7, vj_13zy);
        atomicAdd(vk + (ja*natm+la)*9 + 8, vk_13zz);
        atomicAdd(vj + (ja*natm+la)*9 + 8, vj_13zz);

        double vk_20xx = 0;
        double vj_20xx = 0;
        double vk_20xy = 0;
        double vj_20xy = 0;
        double vk_20xz = 0;
        double vj_20xz = 0;
        double vk_20yx = 0;
        double vj_20yx = 0;
        double vk_20yy = 0;
        double vj_20yy = 0;
        double vk_20yz = 0;
        double vj_20yz = 0;
        double vk_20zx = 0;
        double vj_20zx = 0;
        double vk_20zy = 0;
        double vj_20zy = 0;
        double vk_20zz = 0;
        double vj_20zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        g3 = ai*2 * ak*2 * trr_21x;
                        double trr_01x = cpx * fac;
                        g3 -= 1 * ak*2 * trr_01x;
                        prod = g3 * 1 * wt;
                        vk_20xx += prod * vk_dd;
                        vj_20xx += prod * vj_dd;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        g1 = ak*2 * trr_11x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_20xy += prod * vk_dd;
                        vj_20xy += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_20xz += prod * vk_dd;
                        vj_20xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g1 = ak*2 * trr_01y;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * wt;
                        vk_20yx += prod * vk_dd;
                        vj_20yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        g3 = ai*2 * ak*2 * trr_11y;
                        prod = g3 * trr_10x * wt;
                        vk_20yy += prod * vk_dd;
                        vj_20yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10x;
                        vk_20yz += prod * vk_dd;
                        vj_20yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g1 = ak*2 * trr_01z;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * 1;
                        vk_20zx += prod * vk_dd;
                        vj_20zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10x;
                        vk_20zy += prod * vk_dd;
                        vj_20zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        g3 = ai*2 * ak*2 * trr_11z;
                        prod = g3 * trr_10x * 1;
                        vk_20zz += prod * vk_dd;
                        vj_20zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * ak*2 * trr_11x;
                        prod = g3 * trr_10y * wt;
                        vk_20xx += prod * vk_dd;
                        vj_20xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * wt;
                        vk_20xy += prod * vk_dd;
                        vj_20xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10y;
                        vk_20xz += prod * vk_dd;
                        vj_20xz += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_20yx += prod * vk_dd;
                        vj_20yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        g3 = ai*2 * ak*2 * trr_21y;
                        g3 -= 1 * ak*2 * trr_01y;
                        prod = g3 * fac * wt;
                        vk_20yy += prod * vk_dd;
                        vj_20yy += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_20yz += prod * vk_dd;
                        vj_20yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10y;
                        vk_20zx += prod * vk_dd;
                        vj_20zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * fac;
                        vk_20zy += prod * vk_dd;
                        vj_20zy += prod * vj_dd;
                        g3 = ai*2 * ak*2 * trr_11z;
                        prod = g3 * fac * trr_10y;
                        vk_20zz += prod * vk_dd;
                        vj_20zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * ak*2 * trr_11x;
                        prod = g3 * 1 * trr_10z;
                        vk_20xx += prod * vk_dd;
                        vj_20xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10z;
                        vk_20xy += prod * vk_dd;
                        vj_20xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * 1;
                        vk_20xz += prod * vk_dd;
                        vj_20xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10z;
                        vk_20yx += prod * vk_dd;
                        vj_20yx += prod * vj_dd;
                        g3 = ai*2 * ak*2 * trr_11y;
                        prod = g3 * fac * trr_10z;
                        vk_20yy += prod * vk_dd;
                        vj_20yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * fac;
                        vk_20yz += prod * vk_dd;
                        vj_20yz += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_20zx += prod * vk_dd;
                        vj_20zx += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_20zy += prod * vk_dd;
                        vj_20zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        g3 = ai*2 * ak*2 * trr_21z;
                        g3 -= 1 * ak*2 * trr_01z;
                        prod = g3 * fac * 1;
                        vk_20zz += prod * vk_dd;
                        vj_20zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+ia)*9 + 0, vk_20xx);
        atomicAdd(vj + (ka*natm+ia)*9 + 0, vj_20xx);
        atomicAdd(vk + (ka*natm+ia)*9 + 1, vk_20xy);
        atomicAdd(vj + (ka*natm+ia)*9 + 1, vj_20xy);
        atomicAdd(vk + (ka*natm+ia)*9 + 2, vk_20xz);
        atomicAdd(vj + (ka*natm+ia)*9 + 2, vj_20xz);
        atomicAdd(vk + (ka*natm+ia)*9 + 3, vk_20yx);
        atomicAdd(vj + (ka*natm+ia)*9 + 3, vj_20yx);
        atomicAdd(vk + (ka*natm+ia)*9 + 4, vk_20yy);
        atomicAdd(vj + (ka*natm+ia)*9 + 4, vj_20yy);
        atomicAdd(vk + (ka*natm+ia)*9 + 5, vk_20yz);
        atomicAdd(vj + (ka*natm+ia)*9 + 5, vj_20yz);
        atomicAdd(vk + (ka*natm+ia)*9 + 6, vk_20zx);
        atomicAdd(vj + (ka*natm+ia)*9 + 6, vj_20zx);
        atomicAdd(vk + (ka*natm+ia)*9 + 7, vk_20zy);
        atomicAdd(vj + (ka*natm+ia)*9 + 7, vj_20zy);
        atomicAdd(vk + (ka*natm+ia)*9 + 8, vk_20zz);
        atomicAdd(vj + (ka*natm+ia)*9 + 8, vj_20zz);

        double vk_21xx = 0;
        double vj_21xx = 0;
        double vk_21xy = 0;
        double vj_21xy = 0;
        double vk_21xz = 0;
        double vj_21xz = 0;
        double vk_21yx = 0;
        double vj_21yx = 0;
        double vk_21yy = 0;
        double vj_21yy = 0;
        double vk_21yz = 0;
        double vj_21yz = 0;
        double vk_21zx = 0;
        double vj_21zx = 0;
        double vk_21zy = 0;
        double vj_21zy = 0;
        double vk_21zz = 0;
        double vj_21zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
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
                        double hrr_1110x = trr_21x - (rj[0] - ri[0]) * trr_11x;
                        g3 = aj*2 * ak*2 * hrr_1110x;
                        prod = g3 * 1 * wt;
                        vk_21xx += prod * vk_dd;
                        vj_21xx += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_21xy += prod * vk_dd;
                        vj_21xy += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_21xz += prod * vk_dd;
                        vj_21xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g1 = ak*2 * trr_01y;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * wt;
                        vk_21yx += prod * vk_dd;
                        vj_21yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_0110y = trr_11y - (rj[1] - ri[1]) * trr_01y;
                        g3 = aj*2 * ak*2 * hrr_0110y;
                        prod = g3 * trr_10x * wt;
                        vk_21yy += prod * vk_dd;
                        vj_21yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10x;
                        vk_21yz += prod * vk_dd;
                        vj_21yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g1 = ak*2 * trr_01z;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * 1;
                        vk_21zx += prod * vk_dd;
                        vj_21zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10x;
                        vk_21zy += prod * vk_dd;
                        vj_21zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_0110z = trr_11z - (rj[2] - ri[2]) * trr_01z;
                        g3 = aj*2 * ak*2 * hrr_0110z;
                        prod = g3 * trr_10x * 1;
                        vk_21zz += prod * vk_dd;
                        vj_21zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_01x = cpx * fac;
                        double hrr_0110x = trr_11x - (rj[0] - ri[0]) * trr_01x;
                        g3 = aj*2 * ak*2 * hrr_0110x;
                        prod = g3 * trr_10y * wt;
                        vk_21xx += prod * vk_dd;
                        vj_21xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * wt;
                        vk_21xy += prod * vk_dd;
                        vj_21xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10y;
                        vk_21xz += prod * vk_dd;
                        vj_21xz += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_21yx += prod * vk_dd;
                        vj_21yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - (rj[1] - ri[1]) * trr_11y;
                        g3 = aj*2 * ak*2 * hrr_1110y;
                        prod = g3 * fac * wt;
                        vk_21yy += prod * vk_dd;
                        vj_21yy += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_21yz += prod * vk_dd;
                        vj_21yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10y;
                        vk_21zx += prod * vk_dd;
                        vj_21zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * fac;
                        vk_21zy += prod * vk_dd;
                        vj_21zy += prod * vj_dd;
                        g3 = aj*2 * ak*2 * hrr_0110z;
                        prod = g3 * fac * trr_10y;
                        vk_21zz += prod * vk_dd;
                        vj_21zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = aj*2 * ak*2 * hrr_0110x;
                        prod = g3 * 1 * trr_10z;
                        vk_21xx += prod * vk_dd;
                        vj_21xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10z;
                        vk_21xy += prod * vk_dd;
                        vj_21xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * 1;
                        vk_21xz += prod * vk_dd;
                        vj_21xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10z;
                        vk_21yx += prod * vk_dd;
                        vj_21yx += prod * vj_dd;
                        g3 = aj*2 * ak*2 * hrr_0110y;
                        prod = g3 * fac * trr_10z;
                        vk_21yy += prod * vk_dd;
                        vj_21yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * fac;
                        vk_21yz += prod * vk_dd;
                        vj_21yz += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_21zx += prod * vk_dd;
                        vj_21zx += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_21zy += prod * vk_dd;
                        vj_21zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - (rj[2] - ri[2]) * trr_11z;
                        g3 = aj*2 * ak*2 * hrr_1110z;
                        prod = g3 * fac * 1;
                        vk_21zz += prod * vk_dd;
                        vj_21zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+ja)*9 + 0, vk_21xx);
        atomicAdd(vj + (ka*natm+ja)*9 + 0, vj_21xx);
        atomicAdd(vk + (ka*natm+ja)*9 + 1, vk_21xy);
        atomicAdd(vj + (ka*natm+ja)*9 + 1, vj_21xy);
        atomicAdd(vk + (ka*natm+ja)*9 + 2, vk_21xz);
        atomicAdd(vj + (ka*natm+ja)*9 + 2, vj_21xz);
        atomicAdd(vk + (ka*natm+ja)*9 + 3, vk_21yx);
        atomicAdd(vj + (ka*natm+ja)*9 + 3, vj_21yx);
        atomicAdd(vk + (ka*natm+ja)*9 + 4, vk_21yy);
        atomicAdd(vj + (ka*natm+ja)*9 + 4, vj_21yy);
        atomicAdd(vk + (ka*natm+ja)*9 + 5, vk_21yz);
        atomicAdd(vj + (ka*natm+ja)*9 + 5, vj_21yz);
        atomicAdd(vk + (ka*natm+ja)*9 + 6, vk_21zx);
        atomicAdd(vj + (ka*natm+ja)*9 + 6, vj_21zx);
        atomicAdd(vk + (ka*natm+ja)*9 + 7, vk_21zy);
        atomicAdd(vj + (ka*natm+ja)*9 + 7, vj_21zy);
        atomicAdd(vk + (ka*natm+ja)*9 + 8, vk_21zz);
        atomicAdd(vj + (ka*natm+ja)*9 + 8, vj_21zz);

        double vk_22xx = 0;
        double vj_22xx = 0;
        double vk_22xy = 0;
        double vj_22xy = 0;
        double vk_22xz = 0;
        double vj_22xz = 0;
        double vk_22yx = 0;
        double vj_22yx = 0;
        double vk_22yy = 0;
        double vj_22yy = 0;
        double vk_22yz = 0;
        double vj_22yz = 0;
        double vk_22zx = 0;
        double vj_22zx = 0;
        double vk_22zy = 0;
        double vj_22zy = 0;
        double vk_22zz = 0;
        double vj_22zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
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
                        g3 = ak*2 * (ak*2 * trr_12x - 1 * trr_10x);
                        prod = g3 * 1 * wt;
                        vk_22xx += prod * vk_dd;
                        vj_22xx += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_22xy += prod * vk_dd;
                        vj_22xy += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_22xz += prod * vk_dd;
                        vj_22xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * wt;
                        vk_22yx += prod * vk_dd;
                        vj_22yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        g3 = ak*2 * (ak*2 * trr_02y - 1 * 1);
                        prod = g3 * trr_10x * wt;
                        vk_22yy += prod * vk_dd;
                        vj_22yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10x;
                        vk_22yz += prod * vk_dd;
                        vj_22yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * 1;
                        vk_22zx += prod * vk_dd;
                        vj_22zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10x;
                        vk_22zy += prod * vk_dd;
                        vj_22zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        g3 = ak*2 * (ak*2 * trr_02z - 1 * wt);
                        prod = g3 * trr_10x * 1;
                        vk_22zz += prod * vk_dd;
                        vj_22zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        g3 = ak*2 * (ak*2 * trr_02x - 1 * fac);
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        prod = g3 * trr_10y * wt;
                        vk_22xx += prod * vk_dd;
                        vj_22xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * wt;
                        vk_22xy += prod * vk_dd;
                        vj_22xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10y;
                        vk_22xz += prod * vk_dd;
                        vj_22xz += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_22yx += prod * vk_dd;
                        vj_22yx += prod * vj_dd;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        g3 = ak*2 * (ak*2 * trr_12y - 1 * trr_10y);
                        prod = g3 * fac * wt;
                        vk_22yy += prod * vk_dd;
                        vj_22yy += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_22yz += prod * vk_dd;
                        vj_22yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10y;
                        vk_22zx += prod * vk_dd;
                        vj_22zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * fac;
                        vk_22zy += prod * vk_dd;
                        vj_22zy += prod * vj_dd;
                        g3 = ak*2 * (ak*2 * trr_02z - 1 * wt);
                        prod = g3 * fac * trr_10y;
                        vk_22zz += prod * vk_dd;
                        vj_22zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ak*2 * (ak*2 * trr_02x - 1 * fac);
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        prod = g3 * 1 * trr_10z;
                        vk_22xx += prod * vk_dd;
                        vj_22xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10z;
                        vk_22xy += prod * vk_dd;
                        vj_22xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * 1;
                        vk_22xz += prod * vk_dd;
                        vj_22xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10z;
                        vk_22yx += prod * vk_dd;
                        vj_22yx += prod * vj_dd;
                        g3 = ak*2 * (ak*2 * trr_02y - 1 * 1);
                        prod = g3 * fac * trr_10z;
                        vk_22yy += prod * vk_dd;
                        vj_22yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * fac;
                        vk_22yz += prod * vk_dd;
                        vj_22yz += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_22zx += prod * vk_dd;
                        vj_22zx += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_22zy += prod * vk_dd;
                        vj_22zy += prod * vj_dd;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        g3 = ak*2 * (ak*2 * trr_12z - 1 * trr_10z);
                        prod = g3 * fac * 1;
                        vk_22zz += prod * vk_dd;
                        vj_22zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+ka)*9 + 0, vk_22xx);
        atomicAdd(vj + (ka*natm+ka)*9 + 0, vj_22xx);
        atomicAdd(vk + (ka*natm+ka)*9 + 1, vk_22xy);
        atomicAdd(vj + (ka*natm+ka)*9 + 1, vj_22xy);
        atomicAdd(vk + (ka*natm+ka)*9 + 2, vk_22xz);
        atomicAdd(vj + (ka*natm+ka)*9 + 2, vj_22xz);
        atomicAdd(vk + (ka*natm+ka)*9 + 3, vk_22yx);
        atomicAdd(vj + (ka*natm+ka)*9 + 3, vj_22yx);
        atomicAdd(vk + (ka*natm+ka)*9 + 4, vk_22yy);
        atomicAdd(vj + (ka*natm+ka)*9 + 4, vj_22yy);
        atomicAdd(vk + (ka*natm+ka)*9 + 5, vk_22yz);
        atomicAdd(vj + (ka*natm+ka)*9 + 5, vj_22yz);
        atomicAdd(vk + (ka*natm+ka)*9 + 6, vk_22zx);
        atomicAdd(vj + (ka*natm+ka)*9 + 6, vj_22zx);
        atomicAdd(vk + (ka*natm+ka)*9 + 7, vk_22zy);
        atomicAdd(vj + (ka*natm+ka)*9 + 7, vj_22zy);
        atomicAdd(vk + (ka*natm+ka)*9 + 8, vk_22zz);
        atomicAdd(vj + (ka*natm+ka)*9 + 8, vj_22zz);

        double vk_23xx = 0;
        double vj_23xx = 0;
        double vk_23xy = 0;
        double vj_23xy = 0;
        double vk_23xz = 0;
        double vj_23xz = 0;
        double vk_23yx = 0;
        double vj_23yx = 0;
        double vk_23yy = 0;
        double vj_23yy = 0;
        double vk_23yz = 0;
        double vj_23yz = 0;
        double vk_23zx = 0;
        double vj_23zx = 0;
        double vk_23zy = 0;
        double vj_23zy = 0;
        double vk_23zz = 0;
        double vj_23zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
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
                        g3 = al*2 * ak*2 * hrr_1011x;
                        prod = g3 * 1 * wt;
                        vk_23xx += prod * vk_dd;
                        vj_23xx += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_23xy += prod * vk_dd;
                        vj_23xy += prod * vj_dd;
                        g1 = ak*2 * trr_11x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_23xz += prod * vk_dd;
                        vj_23xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * wt;
                        vk_23yx += prod * vk_dd;
                        vj_23yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3 = al*2 * ak*2 * hrr_0011y;
                        prod = g3 * trr_10x * wt;
                        vk_23yy += prod * vk_dd;
                        vj_23yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10x;
                        vk_23yz += prod * vk_dd;
                        vj_23yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * 1;
                        vk_23zx += prod * vk_dd;
                        vj_23zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10x;
                        vk_23zy += prod * vk_dd;
                        vj_23zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3 = al*2 * ak*2 * hrr_0011z;
                        prod = g3 * trr_10x * 1;
                        vk_23zz += prod * vk_dd;
                        vj_23zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g3 = al*2 * ak*2 * hrr_0011x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        prod = g3 * trr_10y * wt;
                        vk_23xx += prod * vk_dd;
                        vj_23xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * wt;
                        vk_23xy += prod * vk_dd;
                        vj_23xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10y;
                        vk_23xz += prod * vk_dd;
                        vj_23xz += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_23yx += prod * vk_dd;
                        vj_23yx += prod * vj_dd;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3 = al*2 * ak*2 * hrr_1011y;
                        prod = g3 * fac * wt;
                        vk_23yy += prod * vk_dd;
                        vj_23yy += prod * vj_dd;
                        g1 = ak*2 * trr_11y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_23yz += prod * vk_dd;
                        vj_23yz += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10y;
                        vk_23zx += prod * vk_dd;
                        vj_23zx += prod * vj_dd;
                        g1 = ak*2 * trr_01z;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * fac;
                        vk_23zy += prod * vk_dd;
                        vj_23zy += prod * vj_dd;
                        g3 = al*2 * ak*2 * hrr_0011z;
                        prod = g3 * fac * trr_10y;
                        vk_23zz += prod * vk_dd;
                        vj_23zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = al*2 * ak*2 * hrr_0011x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        prod = g3 * 1 * trr_10z;
                        vk_23xx += prod * vk_dd;
                        vj_23xx += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10z;
                        vk_23xy += prod * vk_dd;
                        vj_23xy += prod * vj_dd;
                        g1 = ak*2 * trr_01x;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * 1;
                        vk_23xz += prod * vk_dd;
                        vj_23xz += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10z;
                        vk_23yx += prod * vk_dd;
                        vj_23yx += prod * vj_dd;
                        g3 = al*2 * ak*2 * hrr_0011y;
                        prod = g3 * fac * trr_10z;
                        vk_23yy += prod * vk_dd;
                        vj_23yy += prod * vj_dd;
                        g1 = ak*2 * trr_01y;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * fac;
                        vk_23yz += prod * vk_dd;
                        vj_23yz += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_23zx += prod * vk_dd;
                        vj_23zx += prod * vj_dd;
                        g1 = ak*2 * trr_11z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_23zy += prod * vk_dd;
                        vj_23zy += prod * vj_dd;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3 = al*2 * ak*2 * hrr_1011z;
                        prod = g3 * fac * 1;
                        vk_23zz += prod * vk_dd;
                        vj_23zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (ka*natm+la)*9 + 0, vk_23xx);
        atomicAdd(vj + (ka*natm+la)*9 + 0, vj_23xx);
        atomicAdd(vk + (ka*natm+la)*9 + 1, vk_23xy);
        atomicAdd(vj + (ka*natm+la)*9 + 1, vj_23xy);
        atomicAdd(vk + (ka*natm+la)*9 + 2, vk_23xz);
        atomicAdd(vj + (ka*natm+la)*9 + 2, vj_23xz);
        atomicAdd(vk + (ka*natm+la)*9 + 3, vk_23yx);
        atomicAdd(vj + (ka*natm+la)*9 + 3, vj_23yx);
        atomicAdd(vk + (ka*natm+la)*9 + 4, vk_23yy);
        atomicAdd(vj + (ka*natm+la)*9 + 4, vj_23yy);
        atomicAdd(vk + (ka*natm+la)*9 + 5, vk_23yz);
        atomicAdd(vj + (ka*natm+la)*9 + 5, vj_23yz);
        atomicAdd(vk + (ka*natm+la)*9 + 6, vk_23zx);
        atomicAdd(vj + (ka*natm+la)*9 + 6, vj_23zx);
        atomicAdd(vk + (ka*natm+la)*9 + 7, vk_23zy);
        atomicAdd(vj + (ka*natm+la)*9 + 7, vj_23zy);
        atomicAdd(vk + (ka*natm+la)*9 + 8, vk_23zz);
        atomicAdd(vj + (ka*natm+la)*9 + 8, vj_23zz);

        double vk_30xx = 0;
        double vj_30xx = 0;
        double vk_30xy = 0;
        double vj_30xy = 0;
        double vk_30xz = 0;
        double vj_30xz = 0;
        double vk_30yx = 0;
        double vj_30yx = 0;
        double vk_30yy = 0;
        double vj_30yy = 0;
        double vk_30yz = 0;
        double vj_30yz = 0;
        double vk_30zx = 0;
        double vj_30zx = 0;
        double vk_30zy = 0;
        double vj_30zy = 0;
        double vk_30zz = 0;
        double vj_30zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        g3 = ai*2 * al*2 * hrr_2001x;
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g3 -= 1 * al*2 * hrr_0001x;
                        prod = g3 * 1 * wt;
                        vk_30xx += prod * vk_dd;
                        vj_30xx += prod * vj_dd;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g1 = al*2 * hrr_1001x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * wt;
                        vk_30xy += prod * vk_dd;
                        vj_30xy += prod * vj_dd;
                        g1 = al*2 * hrr_1001x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * 1;
                        vk_30xz += prod * vk_dd;
                        vj_30xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1 = al*2 * hrr_0001y;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * wt;
                        vk_30yx += prod * vk_dd;
                        vj_30yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g3 = ai*2 * al*2 * hrr_1001y;
                        prod = g3 * trr_10x * wt;
                        vk_30yy += prod * vk_dd;
                        vj_30yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10x;
                        vk_30yz += prod * vk_dd;
                        vj_30yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1 = al*2 * hrr_0001z;
                        g2 = ai*2 * trr_20x;
                        g2 -= 1 * fac;
                        prod = g1 * g2 * 1;
                        vk_30zx += prod * vk_dd;
                        vj_30zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10x;
                        vk_30zy += prod * vk_dd;
                        vj_30zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g3 = ai*2 * al*2 * hrr_1001z;
                        prod = g3 * trr_10x * 1;
                        vk_30zz += prod * vk_dd;
                        vj_30zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * al*2 * hrr_1001x;
                        prod = g3 * trr_10y * wt;
                        vk_30xx += prod * vk_dd;
                        vj_30xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * wt;
                        vk_30xy += prod * vk_dd;
                        vj_30xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * trr_10y;
                        vk_30xz += prod * vk_dd;
                        vj_30xz += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * wt;
                        vk_30yx += prod * vk_dd;
                        vj_30yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        g3 = ai*2 * al*2 * hrr_2001y;
                        g3 -= 1 * al*2 * hrr_0001y;
                        prod = g3 * fac * wt;
                        vk_30yy += prod * vk_dd;
                        vj_30yy += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        g2 = ai*2 * trr_10z;
                        prod = g1 * g2 * fac;
                        vk_30yz += prod * vk_dd;
                        vj_30yz += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10y;
                        vk_30zx += prod * vk_dd;
                        vj_30zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ai*2 * trr_20y;
                        g2 -= 1 * 1;
                        prod = g1 * g2 * fac;
                        vk_30zy += prod * vk_dd;
                        vj_30zy += prod * vj_dd;
                        g3 = ai*2 * al*2 * hrr_1001z;
                        prod = g3 * fac * trr_10y;
                        vk_30zz += prod * vk_dd;
                        vj_30zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ai*2 * al*2 * hrr_1001x;
                        prod = g3 * 1 * trr_10z;
                        vk_30xx += prod * vk_dd;
                        vj_30xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * trr_10z;
                        vk_30xy += prod * vk_dd;
                        vj_30xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * 1;
                        vk_30xz += prod * vk_dd;
                        vj_30xz += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * trr_10z;
                        vk_30yx += prod * vk_dd;
                        vj_30yx += prod * vj_dd;
                        g3 = ai*2 * al*2 * hrr_1001y;
                        prod = g3 * fac * trr_10z;
                        vk_30yy += prod * vk_dd;
                        vj_30yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ai*2 * trr_20z;
                        g2 -= 1 * wt;
                        prod = g1 * g2 * fac;
                        vk_30yz += prod * vk_dd;
                        vj_30yz += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = ai*2 * trr_10x;
                        prod = g1 * g2 * 1;
                        vk_30zx += prod * vk_dd;
                        vj_30zx += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = ai*2 * trr_10y;
                        prod = g1 * g2 * fac;
                        vk_30zy += prod * vk_dd;
                        vj_30zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        g3 = ai*2 * al*2 * hrr_2001z;
                        g3 -= 1 * al*2 * hrr_0001z;
                        prod = g3 * fac * 1;
                        vk_30zz += prod * vk_dd;
                        vj_30zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+ia)*9 + 0, vk_30xx);
        atomicAdd(vj + (la*natm+ia)*9 + 0, vj_30xx);
        atomicAdd(vk + (la*natm+ia)*9 + 1, vk_30xy);
        atomicAdd(vj + (la*natm+ia)*9 + 1, vj_30xy);
        atomicAdd(vk + (la*natm+ia)*9 + 2, vk_30xz);
        atomicAdd(vj + (la*natm+ia)*9 + 2, vj_30xz);
        atomicAdd(vk + (la*natm+ia)*9 + 3, vk_30yx);
        atomicAdd(vj + (la*natm+ia)*9 + 3, vj_30yx);
        atomicAdd(vk + (la*natm+ia)*9 + 4, vk_30yy);
        atomicAdd(vj + (la*natm+ia)*9 + 4, vj_30yy);
        atomicAdd(vk + (la*natm+ia)*9 + 5, vk_30yz);
        atomicAdd(vj + (la*natm+ia)*9 + 5, vj_30yz);
        atomicAdd(vk + (la*natm+ia)*9 + 6, vk_30zx);
        atomicAdd(vj + (la*natm+ia)*9 + 6, vj_30zx);
        atomicAdd(vk + (la*natm+ia)*9 + 7, vk_30zy);
        atomicAdd(vj + (la*natm+ia)*9 + 7, vj_30zy);
        atomicAdd(vk + (la*natm+ia)*9 + 8, vk_30zz);
        atomicAdd(vj + (la*natm+ia)*9 + 8, vj_30zz);

        double vk_31xx = 0;
        double vj_31xx = 0;
        double vk_31xy = 0;
        double vj_31xy = 0;
        double vk_31xz = 0;
        double vj_31xz = 0;
        double vk_31yx = 0;
        double vj_31yx = 0;
        double vk_31yy = 0;
        double vj_31yy = 0;
        double vk_31yz = 0;
        double vj_31yz = 0;
        double vk_31zx = 0;
        double vj_31zx = 0;
        double vk_31zy = 0;
        double vj_31zy = 0;
        double vk_31zz = 0;
        double vj_31zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double rt_akl = rt_aa * aij;
                        double cpx = xqc + xpq*rt_akl;
                        double rt_aij = rt_aa * akl;
                        double c0x = Rpa[0*TILE2+sh_ij] - xpq*rt_aij;
                        double trr_10x = c0x * fac;
                        double b10 = .5/aij * (1 - rt_aij);
                        double trr_20x = c0x * trr_10x + 1*b10 * fac;
                        double b00 = .5 * rt_aa;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double trr_11x = cpx * trr_10x + 1*b00 * fac;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - (rj[0] - ri[0]) * hrr_1001x;
                        g3 = aj*2 * al*2 * hrr_1101x;
                        prod = g3 * 1 * wt;
                        vk_31xx += prod * vk_dd;
                        vj_31xx += prod * vj_dd;
                        g1 = al*2 * hrr_1001x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        double hrr_0100y = trr_10y - (rj[1] - ri[1]) * 1;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * wt;
                        vk_31xy += prod * vk_dd;
                        vj_31xy += prod * vj_dd;
                        g1 = al*2 * hrr_1001x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        double hrr_0100z = trr_10z - (rj[2] - ri[2]) * wt;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * 1;
                        vk_31xz += prod * vk_dd;
                        vj_31xz += prod * vj_dd;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1 = al*2 * hrr_0001y;
                        double hrr_1100x = trr_20x - (rj[0] - ri[0]) * trr_10x;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * wt;
                        vk_31yx += prod * vk_dd;
                        vj_31yx += prod * vj_dd;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        double hrr_0101y = hrr_1001y - (rj[1] - ri[1]) * hrr_0001y;
                        g3 = aj*2 * al*2 * hrr_0101y;
                        prod = g3 * trr_10x * wt;
                        vk_31yy += prod * vk_dd;
                        vj_31yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10x;
                        vk_31yz += prod * vk_dd;
                        vj_31yz += prod * vj_dd;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1 = al*2 * hrr_0001z;
                        g2 = aj*2 * hrr_1100x;
                        prod = g1 * g2 * 1;
                        vk_31zx += prod * vk_dd;
                        vj_31zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10x;
                        vk_31zy += prod * vk_dd;
                        vj_31zy += prod * vj_dd;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        double hrr_0101z = hrr_1001z - (rj[2] - ri[2]) * hrr_0001z;
                        g3 = aj*2 * al*2 * hrr_0101z;
                        prod = g3 * trr_10x * 1;
                        vk_31zz += prod * vk_dd;
                        vj_31zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_01x = cpx * fac;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0101x = hrr_1001x - (rj[0] - ri[0]) * hrr_0001x;
                        g3 = aj*2 * al*2 * hrr_0101x;
                        prod = g3 * trr_10y * wt;
                        vk_31xx += prod * vk_dd;
                        vj_31xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_20y = c0y * trr_10y + 1*b10 * 1;
                        double hrr_1100y = trr_20y - (rj[1] - ri[1]) * trr_10y;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * wt;
                        vk_31xy += prod * vk_dd;
                        vj_31xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * trr_10y;
                        vk_31xz += prod * vk_dd;
                        vj_31xz += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        double hrr_0100x = trr_10x - (rj[0] - ri[0]) * fac;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * wt;
                        vk_31yx += prod * vk_dd;
                        vj_31yx += prod * vj_dd;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - (rj[1] - ri[1]) * hrr_1001y;
                        g3 = aj*2 * al*2 * hrr_1101y;
                        prod = g3 * fac * wt;
                        vk_31yy += prod * vk_dd;
                        vj_31yy += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        g2 = aj*2 * hrr_0100z;
                        prod = g1 * g2 * fac;
                        vk_31yz += prod * vk_dd;
                        vj_31yz += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10y;
                        vk_31zx += prod * vk_dd;
                        vj_31zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = aj*2 * hrr_1100y;
                        prod = g1 * g2 * fac;
                        vk_31zy += prod * vk_dd;
                        vj_31zy += prod * vj_dd;
                        g3 = aj*2 * al*2 * hrr_0101z;
                        prod = g3 * fac * trr_10y;
                        vk_31zz += prod * vk_dd;
                        vj_31zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = aj*2 * al*2 * hrr_0101x;
                        prod = g3 * 1 * trr_10z;
                        vk_31xx += prod * vk_dd;
                        vj_31xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * trr_10z;
                        vk_31xy += prod * vk_dd;
                        vj_31xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - (rj[2] - ri[2]) * trr_10z;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * 1;
                        vk_31xz += prod * vk_dd;
                        vj_31xz += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * trr_10z;
                        vk_31yx += prod * vk_dd;
                        vj_31yx += prod * vj_dd;
                        g3 = aj*2 * al*2 * hrr_0101y;
                        prod = g3 * fac * trr_10z;
                        vk_31yy += prod * vk_dd;
                        vj_31yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = aj*2 * hrr_1100z;
                        prod = g1 * g2 * fac;
                        vk_31yz += prod * vk_dd;
                        vj_31yz += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = aj*2 * hrr_0100x;
                        prod = g1 * g2 * 1;
                        vk_31zx += prod * vk_dd;
                        vj_31zx += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = aj*2 * hrr_0100y;
                        prod = g1 * g2 * fac;
                        vk_31zy += prod * vk_dd;
                        vj_31zy += prod * vj_dd;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - (rj[2] - ri[2]) * hrr_1001z;
                        g3 = aj*2 * al*2 * hrr_1101z;
                        prod = g3 * fac * 1;
                        vk_31zz += prod * vk_dd;
                        vj_31zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+ja)*9 + 0, vk_31xx);
        atomicAdd(vj + (la*natm+ja)*9 + 0, vj_31xx);
        atomicAdd(vk + (la*natm+ja)*9 + 1, vk_31xy);
        atomicAdd(vj + (la*natm+ja)*9 + 1, vj_31xy);
        atomicAdd(vk + (la*natm+ja)*9 + 2, vk_31xz);
        atomicAdd(vj + (la*natm+ja)*9 + 2, vj_31xz);
        atomicAdd(vk + (la*natm+ja)*9 + 3, vk_31yx);
        atomicAdd(vj + (la*natm+ja)*9 + 3, vj_31yx);
        atomicAdd(vk + (la*natm+ja)*9 + 4, vk_31yy);
        atomicAdd(vj + (la*natm+ja)*9 + 4, vj_31yy);
        atomicAdd(vk + (la*natm+ja)*9 + 5, vk_31yz);
        atomicAdd(vj + (la*natm+ja)*9 + 5, vj_31yz);
        atomicAdd(vk + (la*natm+ja)*9 + 6, vk_31zx);
        atomicAdd(vj + (la*natm+ja)*9 + 6, vj_31zx);
        atomicAdd(vk + (la*natm+ja)*9 + 7, vk_31zy);
        atomicAdd(vj + (la*natm+ja)*9 + 7, vj_31zy);
        atomicAdd(vk + (la*natm+ja)*9 + 8, vk_31zz);
        atomicAdd(vj + (la*natm+ja)*9 + 8, vj_31zz);

        double vk_32xx = 0;
        double vj_32xx = 0;
        double vk_32xy = 0;
        double vj_32xy = 0;
        double vk_32xz = 0;
        double vj_32xz = 0;
        double vk_32yx = 0;
        double vj_32yx = 0;
        double vk_32yy = 0;
        double vj_32yy = 0;
        double vk_32yz = 0;
        double vj_32yz = 0;
        double vk_32zx = 0;
        double vj_32zx = 0;
        double vk_32zy = 0;
        double vj_32zy = 0;
        double vk_32zz = 0;
        double vj_32zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
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
                        g3 = ak*2 * al*2 * hrr_1011x;
                        prod = g3 * 1 * wt;
                        vk_32xx += prod * vk_dd;
                        vj_32xx += prod * vj_dd;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        g1 = al*2 * hrr_1001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * wt;
                        vk_32xy += prod * vk_dd;
                        vj_32xy += prod * vj_dd;
                        g1 = al*2 * hrr_1001x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * 1;
                        vk_32xz += prod * vk_dd;
                        vj_32xz += prod * vj_dd;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g1 = al*2 * hrr_0001y;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * wt;
                        vk_32yx += prod * vk_dd;
                        vj_32yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        g3 = ak*2 * al*2 * hrr_0011y;
                        prod = g3 * trr_10x * wt;
                        vk_32yy += prod * vk_dd;
                        vj_32yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10x;
                        vk_32yz += prod * vk_dd;
                        vj_32yz += prod * vj_dd;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g1 = al*2 * hrr_0001z;
                        g2 = ak*2 * trr_11x;
                        prod = g1 * g2 * 1;
                        vk_32zx += prod * vk_dd;
                        vj_32zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10x;
                        vk_32zy += prod * vk_dd;
                        vj_32zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        g3 = ak*2 * al*2 * hrr_0011z;
                        prod = g3 * trr_10x * 1;
                        vk_32zz += prod * vk_dd;
                        vj_32zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        g3 = ak*2 * al*2 * hrr_0011x;
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        prod = g3 * trr_10y * wt;
                        vk_32xx += prod * vk_dd;
                        vj_32xx += prod * vj_dd;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        g1 = al*2 * hrr_0001x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * wt;
                        vk_32xy += prod * vk_dd;
                        vj_32xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * trr_10y;
                        vk_32xz += prod * vk_dd;
                        vj_32xz += prod * vj_dd;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g1 = al*2 * hrr_1001y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * wt;
                        vk_32yx += prod * vk_dd;
                        vj_32yx += prod * vj_dd;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        g3 = ak*2 * al*2 * hrr_1011y;
                        prod = g3 * fac * wt;
                        vk_32yy += prod * vk_dd;
                        vj_32yy += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        g2 = ak*2 * trr_01z;
                        prod = g1 * g2 * fac;
                        vk_32yz += prod * vk_dd;
                        vj_32yz += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10y;
                        vk_32zx += prod * vk_dd;
                        vj_32zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = ak*2 * trr_11y;
                        prod = g1 * g2 * fac;
                        vk_32zy += prod * vk_dd;
                        vj_32zy += prod * vj_dd;
                        g3 = ak*2 * al*2 * hrr_0011z;
                        prod = g3 * fac * trr_10y;
                        vk_32zz += prod * vk_dd;
                        vj_32zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = ak*2 * al*2 * hrr_0011x;
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        prod = g3 * 1 * trr_10z;
                        vk_32xx += prod * vk_dd;
                        vj_32xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * trr_10z;
                        vk_32xy += prod * vk_dd;
                        vj_32xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * 1;
                        vk_32xz += prod * vk_dd;
                        vj_32xz += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * trr_10z;
                        vk_32yx += prod * vk_dd;
                        vj_32yx += prod * vj_dd;
                        g3 = ak*2 * al*2 * hrr_0011y;
                        prod = g3 * fac * trr_10z;
                        vk_32yy += prod * vk_dd;
                        vj_32yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = ak*2 * trr_11z;
                        prod = g1 * g2 * fac;
                        vk_32yz += prod * vk_dd;
                        vj_32yz += prod * vj_dd;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g1 = al*2 * hrr_1001z;
                        g2 = ak*2 * trr_01x;
                        prod = g1 * g2 * 1;
                        vk_32zx += prod * vk_dd;
                        vj_32zx += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = ak*2 * trr_01y;
                        prod = g1 * g2 * fac;
                        vk_32zy += prod * vk_dd;
                        vj_32zy += prod * vj_dd;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        g3 = ak*2 * al*2 * hrr_1011z;
                        prod = g3 * fac * 1;
                        vk_32zz += prod * vk_dd;
                        vj_32zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+ka)*9 + 0, vk_32xx);
        atomicAdd(vj + (la*natm+ka)*9 + 0, vj_32xx);
        atomicAdd(vk + (la*natm+ka)*9 + 1, vk_32xy);
        atomicAdd(vj + (la*natm+ka)*9 + 1, vj_32xy);
        atomicAdd(vk + (la*natm+ka)*9 + 2, vk_32xz);
        atomicAdd(vj + (la*natm+ka)*9 + 2, vj_32xz);
        atomicAdd(vk + (la*natm+ka)*9 + 3, vk_32yx);
        atomicAdd(vj + (la*natm+ka)*9 + 3, vj_32yx);
        atomicAdd(vk + (la*natm+ka)*9 + 4, vk_32yy);
        atomicAdd(vj + (la*natm+ka)*9 + 4, vj_32yy);
        atomicAdd(vk + (la*natm+ka)*9 + 5, vk_32yz);
        atomicAdd(vj + (la*natm+ka)*9 + 5, vj_32yz);
        atomicAdd(vk + (la*natm+ka)*9 + 6, vk_32zx);
        atomicAdd(vj + (la*natm+ka)*9 + 6, vj_32zx);
        atomicAdd(vk + (la*natm+ka)*9 + 7, vk_32zy);
        atomicAdd(vj + (la*natm+ka)*9 + 7, vj_32zy);
        atomicAdd(vk + (la*natm+ka)*9 + 8, vk_32zz);
        atomicAdd(vj + (la*natm+ka)*9 + 8, vj_32zz);

        double vk_33xx = 0;
        double vj_33xx = 0;
        double vk_33xy = 0;
        double vj_33xy = 0;
        double vk_33xz = 0;
        double vj_33xz = 0;
        double vk_33yx = 0;
        double vj_33yx = 0;
        double vk_33yy = 0;
        double vj_33yy = 0;
        double vk_33yz = 0;
        double vj_33yz = 0;
        double vk_33zx = 0;
        double vj_33zx = 0;
        double vk_33zy = 0;
        double vj_33zy = 0;
        double vk_33zz = 0;
        double vj_33zz = 0;
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
            double xkl = rk[0] + xqc;
            double ykl = rk[1] + yqc;
            double zkl = rk[2] + zqc;
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
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                double theta = aij * akl / (aij + akl);
                double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                rys_roots(2, theta*rr, rw);
                __syncthreads();
                if (task_id < ntasks) {
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[sq_id + (2*irys+1)*nsq_per_block];
                        double rt = rw[sq_id +  2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_0_0;
                            dd_jl = dm_jl_0_0 * dm_ik_0_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+0)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+0)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[0*TILE2+sh_ij] * dm_lk_0_0;
                        }
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
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1002x = hrr_1011x - xlxk * hrr_1001x;
                        g3 = al*2 * (al*2 * hrr_1002x - 1 * trr_10x);
                        prod = g3 * 1 * wt;
                        vk_33xx += prod * vk_dd;
                        vj_33xx += prod * vj_dd;
                        g1 = al*2 * hrr_1001x;
                        double cpy = yqc + ypq*rt_akl;
                        double trr_01y = cpy * 1;
                        double hrr_0001y = trr_01y - ylyk * 1;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * wt;
                        vk_33xy += prod * vk_dd;
                        vj_33xy += prod * vj_dd;
                        g1 = al*2 * hrr_1001x;
                        double cpz = zqc + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * 1;
                        vk_33xz += prod * vk_dd;
                        vj_33xz += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * wt;
                        vk_33yx += prod * vk_dd;
                        vj_33yx += prod * vj_dd;
                        double trr_02y = cpy * trr_01y + 1*b01 * 1;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        double hrr_0002y = hrr_0011y - ylyk * hrr_0001y;
                        g3 = al*2 * (al*2 * hrr_0002y - 1 * 1);
                        prod = g3 * trr_10x * wt;
                        vk_33yy += prod * vk_dd;
                        vj_33yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10x;
                        vk_33yz += prod * vk_dd;
                        vj_33yz += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = al*2 * hrr_1001x;
                        prod = g1 * g2 * 1;
                        vk_33zx += prod * vk_dd;
                        vj_33zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10x;
                        vk_33zy += prod * vk_dd;
                        vj_33zy += prod * vj_dd;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        double hrr_0002z = hrr_0011z - zlzk * hrr_0001z;
                        g3 = al*2 * (al*2 * hrr_0002z - 1 * wt);
                        prod = g3 * trr_10x * 1;
                        vk_33zz += prod * vk_dd;
                        vj_33zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_1_0;
                            dd_jl = dm_jl_0_0 * dm_ik_1_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+1)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+1)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[1*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        double trr_02x = cpx * trr_01x + 1*b01 * fac;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0001x = trr_01x - xlxk * fac;
                        double hrr_0002x = hrr_0011x - xlxk * hrr_0001x;
                        g3 = al*2 * (al*2 * hrr_0002x - 1 * fac);
                        double c0y = Rpa[1*TILE2+sh_ij] - ypq*rt_aij;
                        double trr_10y = c0y * 1;
                        prod = g3 * trr_10y * wt;
                        vk_33xx += prod * vk_dd;
                        vj_33xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_11y = cpy * trr_10y + 1*b00 * 1;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * wt;
                        vk_33xy += prod * vk_dd;
                        vj_33xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * trr_10y;
                        vk_33xz += prod * vk_dd;
                        vj_33xz += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * wt;
                        vk_33yx += prod * vk_dd;
                        vj_33yx += prod * vj_dd;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        double hrr_1002y = hrr_1011y - ylyk * hrr_1001y;
                        g3 = al*2 * (al*2 * hrr_1002y - 1 * trr_10y);
                        prod = g3 * fac * wt;
                        vk_33yy += prod * vk_dd;
                        vj_33yy += prod * vj_dd;
                        g1 = al*2 * hrr_1001y;
                        g2 = al*2 * hrr_0001z;
                        prod = g1 * g2 * fac;
                        vk_33yz += prod * vk_dd;
                        vj_33yz += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10y;
                        vk_33zx += prod * vk_dd;
                        vj_33zx += prod * vj_dd;
                        g1 = al*2 * hrr_0001z;
                        g2 = al*2 * hrr_1001y;
                        prod = g1 * g2 * fac;
                        vk_33zy += prod * vk_dd;
                        vj_33zy += prod * vj_dd;
                        g3 = al*2 * (al*2 * hrr_0002z - 1 * wt);
                        prod = g3 * fac * trr_10y;
                        vk_33zz += prod * vk_dd;
                        vj_33zz += prod * vj_dd;
                        if (vk != NULL) {
                            dd_jk = dm_jk_0_0 * dm_il_2_0;
                            dd_jl = dm_jl_0_0 * dm_ik_2_0;
                            vk_dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                dd_jk = dm[(nao+j0+0)*nao+k0+0] * dm[(nao+i0+2)*nao+l0+0];
                                dd_jl = dm[(nao+j0+0)*nao+l0+0] * dm[(nao+i0+2)*nao+k0+0];
                                vk_dd += dd_jk + dd_jl;
                            }
                        }
                        if (vj != NULL) {
                            vj_dd = dm_cache[2*TILE2+sh_ij] * dm_lk_0_0;
                        }
                        g3 = al*2 * (al*2 * hrr_0002x - 1 * fac);
                        double c0z = Rpa[2*TILE2+sh_ij] - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        prod = g3 * 1 * trr_10z;
                        vk_33xx += prod * vk_dd;
                        vj_33xx += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * trr_10z;
                        vk_33xy += prod * vk_dd;
                        vj_33xy += prod * vj_dd;
                        g1 = al*2 * hrr_0001x;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * 1;
                        vk_33xz += prod * vk_dd;
                        vj_33xz += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * trr_10z;
                        vk_33yx += prod * vk_dd;
                        vj_33yx += prod * vj_dd;
                        g3 = al*2 * (al*2 * hrr_0002y - 1 * 1);
                        prod = g3 * fac * trr_10z;
                        vk_33yy += prod * vk_dd;
                        vj_33yy += prod * vj_dd;
                        g1 = al*2 * hrr_0001y;
                        g2 = al*2 * hrr_1001z;
                        prod = g1 * g2 * fac;
                        vk_33yz += prod * vk_dd;
                        vj_33yz += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = al*2 * hrr_0001x;
                        prod = g1 * g2 * 1;
                        vk_33zx += prod * vk_dd;
                        vj_33zx += prod * vj_dd;
                        g1 = al*2 * hrr_1001z;
                        g2 = al*2 * hrr_0001y;
                        prod = g1 * g2 * fac;
                        vk_33zy += prod * vk_dd;
                        vj_33zy += prod * vj_dd;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        double hrr_1002z = hrr_1011z - zlzk * hrr_1001z;
                        g3 = al*2 * (al*2 * hrr_1002z - 1 * trr_10z);
                        prod = g3 * fac * 1;
                        vk_33zz += prod * vk_dd;
                        vj_33zz += prod * vj_dd;
                    }
                }
            }
        }
        atomicAdd(vk + (la*natm+la)*9 + 0, vk_33xx);
        atomicAdd(vj + (la*natm+la)*9 + 0, vj_33xx);
        atomicAdd(vk + (la*natm+la)*9 + 1, vk_33xy);
        atomicAdd(vj + (la*natm+la)*9 + 1, vj_33xy);
        atomicAdd(vk + (la*natm+la)*9 + 2, vk_33xz);
        atomicAdd(vj + (la*natm+la)*9 + 2, vj_33xz);
        atomicAdd(vk + (la*natm+la)*9 + 3, vk_33yx);
        atomicAdd(vj + (la*natm+la)*9 + 3, vj_33yx);
        atomicAdd(vk + (la*natm+la)*9 + 4, vk_33yy);
        atomicAdd(vj + (la*natm+la)*9 + 4, vj_33yy);
        atomicAdd(vk + (la*natm+la)*9 + 5, vk_33yz);
        atomicAdd(vj + (la*natm+la)*9 + 5, vj_33yz);
        atomicAdd(vk + (la*natm+la)*9 + 6, vk_33zx);
        atomicAdd(vj + (la*natm+la)*9 + 6, vj_33zx);
        atomicAdd(vk + (la*natm+la)*9 + 7, vk_33zy);
        atomicAdd(vj + (la*natm+la)*9 + 7, vj_33zy);
        atomicAdd(vk + (la*natm+la)*9 + 8, vk_33zz);
        atomicAdd(vj + (la*natm+la)*9 + 8, vj_33zz);
    }
}
__global__
void rys_ejk_ip2_1000(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            int tile_ij = bounds.tile_ij_mapping[batch_ij];
            int nbas_tiles = nbas / TILE;
            int tile_i = tile_ij / nbas_tiles;
            int tile_j = tile_ij % nbas_tiles;
            int ish0 = tile_i * TILE;
            int jsh0 = tile_j * TILE;
            _rys_ejk_ip2_1000(envs, jk, bounds, shl_quartet_idx, ntasks, ish0, jsh0);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

int rys_ejk_ip2_unrolled(RysIntEnvVars *envs, JKMatrix *jk, BoundsInfo *bounds,
                         ShellQuartet *pool, uint32_t *batch_head, int *scheme, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int threads = scheme[0] * scheme[1];
    int nroots = (li + lj + lk + ll + 2) / 2 + 1;
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int ij_prims = iprim * jprim;
    int nfi = (li + 1) * (li + 2) / 2;
    int nfj = (lj + 1) * (lj + 2) / 2;
    int buflen = nroots*2 * threads + nfi*nfj*TILE2 + ij_prims*TILE2*4;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    switch (ijkl) {
    case 0: rys_ejk_ip2_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    case 125: rys_ejk_ip2_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *jk, *bounds, pool, batch_head); break;
    default: return 0;
    }
    return 1;
}
