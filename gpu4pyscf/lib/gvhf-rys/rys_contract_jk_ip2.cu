#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks_ip1.cu"

__device__
static void rys_ejk_ip2_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                                ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int nfi = bounds.nfi;
    int nfk = bounds.nfk;
    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj + 2;
    int lkl = lk + ll + 2;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 2);
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
    int *idy_kl = idx_kl + nfkl;
    int *idz_kl = idy_kl + nfkl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    double *env = envs.env;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double *dm = jk.dm;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3], Rpq[3];
    int at1_at2 = gout_id % 16;
    int at1 = at1_at2 % 4;
    int at2 = at1_at2 / 4;
    int nx_at1, ny_at1, nz_at1, stride_at1;
    int nx_at2, ny_at2, nz_at2, stride_at2;
    switch (at1) {
    case 0: stride_at1 =          nsq_per_block; break;
    case 1: stride_at1 = stride_j*nsq_per_block; break;
    case 2: stride_at1 = stride_k*nsq_per_block; break;
    case 3: stride_at1 = stride_l*nsq_per_block; break;
    }
    switch (at2) {
    case 0: stride_at2 =          nsq_per_block; break;
    case 1: stride_at2 = stride_j*nsq_per_block; break;
    case 2: stride_at2 = stride_k*nsq_per_block; break;
    case 3: stride_at2 = stride_l*nsq_per_block; break;
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
        //int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
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
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];

        int stride_assoc;
        double x1x2_assoc, y1y2_assoc, z1z2_assoc;
        if (at1 == at2) {
            switch (at1) {
            case 0: stride_assoc = stride_j*nsq_per_block;
                    x1x2_assoc = ri[0] - rj[0];
                    y1y2_assoc = ri[1] - rj[1];
                    z1z2_assoc = ri[2] - rj[2];
                    break;

            case 1: stride_assoc = nsq_per_block;
                    x1x2_assoc = rj[0] - ri[0];
                    y1y2_assoc = rj[1] - ri[1];
                    z1z2_assoc = rj[2] - ri[2];
                    break;

            case 2: stride_assoc = stride_l*nsq_per_block;
                    x1x2_assoc = rk[0] - rl[0];
                    y1y2_assoc = rk[1] - rl[1];
                    z1z2_assoc = rk[2] - rl[2];
                    break;

            case 3: stride_assoc = stride_k*nsq_per_block;
                    x1x2_assoc = rl[0] - rk[0];
                    y1y2_assoc = rl[1] - rk[1];
                    z1z2_assoc = rl[2] - rk[2];
                    break;
            }
        }
        double vj_xx = 0;
        double vj_xy = 0;
        double vj_xz = 0;
        double vj_yx = 0;
        double vj_yy = 0;
        double vj_yz = 0;
        double vj_zx = 0;
        double vj_zy = 0;
        double vj_zz = 0;
        double vk_xx = 0;
        double vk_xy = 0;
        double vk_xz = 0;
        double vk_yx = 0;
        double vk_yy = 0;
        double vk_yz = 0;
        double vk_zx = 0;
        double vk_zy = 0;
        double vk_zz = 0;
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xjxi = rj[0] - ri[0];
            double yjyi = rj[1] - ri[1];
            double zjzi = rj[2] - ri[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xjxi * aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yjyi * aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zjzi * aj_aij;
            double theta_ij = ai * aj / aij;
            double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double ak2 = ak * 2;
            double al2 = al * 2;
            double al_akl = al / akl;
            double xlxk = rl[0] - rk[0];
            double ylyk = rl[1] - rk[1];
            double zlzk = rl[2] - rk[2];
            Rqc[0] = xlxk * al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ylyk * al_akl;
            Rqc[2] = zlzk * al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = ck[kp] * cl[lp] * Kcd;
                g[sq_id] = ckcl;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = expi[ip];
                double aj = expj[jp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double *Rpa = Rpa_cicj + ijp*4*nsq_per_block;
                double xij = ri[0] + Rpa[sq_id+0*nsq_per_block];
                double yij = ri[1] + Rpa[sq_id+1*nsq_per_block];
                double zij = ri[2] + Rpa[sq_id+2*nsq_per_block];
                double xkl = rk[0] + Rqc[0];
                double ykl = rk[1] + Rqc[1];
                double zkl = rk[2] + Rqc[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                Rpq[0] = xpq;
                Rpq[1] = ypq;
                Rpq[2] = zpq;
                __syncthreads();
                if (gout_id == 0) {
                    double cicj = Rpa[sq_id+3*nsq_per_block];
                    g[sq_id + g_size * nsq_per_block] = cicj / (aij*akl*sqrt(aij+akl));
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * akl / (aij + akl);
                double theta_rr = theta * rr;
                if (omega == 0) {
                    rys_roots(nroots, theta_rr, rw);
                } else {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                }
                double a2_at1, a2_at2;
                switch (at1) {
                case 0: a2_at1 = ai2; break;
                case 1: a2_at1 = aj2; break;
                case 2: a2_at1 = ak2; break;
                case 3: a2_at1 = al2; break;
                }
                switch (at2) {
                case 0: a2_at2 = ai2; break;
                case 1: a2_at2 = aj2; break;
                case 2: a2_at2 = ak2; break;
                case 3: a2_at2 = al2; break;
                }
                double s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        g[sq_id + 2*g_size*nsq_per_block] = rw[sq_id+(irys*2+1)*nsq_per_block];
                    }
                    double rt = rw[sq_id + irys*2*nsq_per_block];
                    double rt_aa = rt / (aij + akl);
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b00 = .5 * rt_aa;
                    double b10 = .5/aij * (1 - rt_aij);
                    double b01 = .5/akl * (1 - rt_akl);

                    __syncthreads();
                    // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                    for (int n = gout_id; n < 3; n += gout_stride) {
                        double *_gx = g + n * g_size * nsq_per_block;
                        int ir = sq_id + n * nsq_per_block;
                        double c0x = Rpa[ir] - rt_aij * Rpq[n];
                        s0x = _gx[sq_id];
                        s1x = c0x * s0x;
                        _gx[sq_id + nsq_per_block] = s1x;
                        for (int i = 1; i < lij; ++i) {
                            s2x = c0x * s1x + i * b10 * s0x;
                            _gx[sq_id + (i+1)*nsq_per_block] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }

                    int lij3 = (lij+1)*3;
                    for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                        __syncthreads();
                        int i = n / 3; //for i in range(lij+1):
                        int _ix = n % 3;
                        double *_gx = g + (i + _ix * g_size) * nsq_per_block;
                        double cpx = Rqc[_ix] + rt_akl * Rpq[_ix];
                        //for i in range(lij+1):
                        //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                        if (n < lij3) {
                            s0x = _gx[sq_id];
                            s1x = cpx * s0x;
                            if (i > 0) {
                                s1x += i * b00 * _gx[sq_id-nsq_per_block];
                            }
                            _gx[sq_id + stride_k*nsq_per_block] = s1x;
                        }

                        //for k in range(1, lkl):
                        //    for i in range(lij+1):
                        //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                        for (int k = 1; k < lkl; ++k) {
                            __syncthreads();
                            if (n < lij3) {
                                s2x = cpx*s1x + k*b01*s0x;
                                if (i > 0) {
                                    s2x += i * b00 * _gx[sq_id + (k*stride_k-1)*nsq_per_block];
                                }
                                _gx[sq_id + (k*stride_k+stride_k)*nsq_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    // hrr
                    // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                    // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                    __syncthreads();
                    if (task_id < ntasks) {
                        int lkl3 = (lkl+1)*3;
                        for (int m = gout_id; m < lkl3; m += gout_stride) {
                            int k = m / 3;
                            int _ix = m % 3;
                            double xixj = ri[_ix] - rj[_ix];
                            double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                            for (int j = 0; j <= lj; ++j) {
                                int ij = (lij-j) + j*stride_j;
                                s1x = _gx[sq_id + ij*nsq_per_block];
                                for (--ij; ij >= j*stride_j; --ij) {
                                    s0x = _gx[sq_id + ij*nsq_per_block];
                                    _gx[sq_id + (ij+stride_j)*nsq_per_block] = xixj * s0x + s1x;
                                    s1x = s0x;
                                }
                            }
                        }
                    }
                    __syncthreads();
                    if (task_id < ntasks) {
                        for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                            int i = n / 3;
                            int _ix = n % 3;
                            double xkxl = rk[_ix] - rl[_ix];
                            double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                            for (int l = 0; l <= ll; ++l) {
                                int kl = (lkl-l)*stride_k + l*stride_l;
                                s1x = _gx[sq_id + kl*nsq_per_block];
                                for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                                    s0x = _gx[sq_id + kl*nsq_per_block];
                                    _gx[sq_id + (kl+stride_l)*nsq_per_block] = xkxl * s0x + s1x;
                                    s1x = s0x;
                                }
                            }
                        }
                    }

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id/16; n < nfij*nfkl; n+=gout_stride/16) {
                        int kl = n / nfij;
                        int ij = n % nfij;
                        if (kl >= nfkl) break;
                        int ijx = idx_ij[ij];
                        int ijy = idy_ij[ij];
                        int ijz = idz_ij[ij];
                        int klx = idx_kl[kl];
                        int kly = idy_kl[kl];
                        int klz = idz_kl[kl];
                        int ix = ijx % (li + 1);
                        int jx = ijx / (li + 1);
                        int iy = ijy % (li + 1);
                        int jy = ijy / (li + 1);
                        int iz = ijz % (li + 1);
                        int jz = ijz / (li + 1);
                        int kx = klx % (lk + 1);
                        int lx = klx / (lk + 1);
                        int ky = kly % (lk + 1);
                        int ly = kly / (lk + 1);
                        int kz = klz % (lk + 1);
                        int lz = klz / (lk + 1);
                        switch (at1) {
                        case 0: nx_at1 = ix; ny_at1 = iy; nz_at1 = iz; break;
                        case 1: nx_at1 = jx; ny_at1 = jy; nz_at1 = jz; break;
                        case 2: nx_at1 = kx; ny_at1 = ky; nz_at1 = kz; break;
                        case 3: nx_at1 = lx; ny_at1 = ly; nz_at1 = lz; break;
                        }
                        switch (at2) {
                        case 0: nx_at2 = ix; ny_at2 = iy; nz_at2 = iz; break;
                        case 1: nx_at2 = jx; ny_at2 = jy; nz_at2 = jz; break;
                        case 2: nx_at2 = kx; ny_at2 = ky; nz_at2 = kz; break;
                        case 3: nx_at2 = lx; ny_at2 = ly; nz_at2 = lz; break;
                        }
                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double g1x = a2_at1 * gx[addrx+stride_at1];
                        double g1y = a2_at1 * gy[addry+stride_at1];
                        double g1z = a2_at1 * gz[addrz+stride_at1];
                        if (nx_at1 > 0) { g1x -= nx_at1 * gx[addrx-stride_at1]; }
                        if (ny_at1 > 0) { g1y -= ny_at1 * gy[addry-stride_at1]; }
                        if (nz_at1 > 0) { g1z -= nz_at1 * gz[addrz-stride_at1]; }

                        double g2x = a2_at2 * gx[addrx+stride_at2];
                        double g2y = a2_at2 * gy[addry+stride_at2];
                        double g2z = a2_at2 * gz[addrz+stride_at2];
                        if (nx_at2 > 0) { g2x -= nx_at2 * gx[addrx-stride_at2]; }
                        if (ny_at2 > 0) { g2y -= ny_at2 * gy[addry-stride_at2]; }
                        if (nz_at2 > 0) { g2z -= nz_at2 * gz[addrz-stride_at2]; }

                        double g3x, g3y, g3z;
                        if (at1 == at2) {
                            double _gx_inc2 = gx[addrx+stride_at1+stride_assoc] - gx[addrx+stride_at1] * x1x2_assoc;
                            double _gy_inc2 = gy[addry+stride_at1+stride_assoc] - gy[addry+stride_at1] * y1y2_assoc;
                            double _gz_inc2 = gz[addrz+stride_at1+stride_assoc] - gz[addrz+stride_at1] * z1z2_assoc;
                            g3x = a2_at1 * (a2_at1 * _gx_inc2 - (2*nx_at1+1) * gx[addrx]);
                            g3y = a2_at1 * (a2_at1 * _gy_inc2 - (2*ny_at1+1) * gy[addry]);
                            g3z = a2_at1 * (a2_at1 * _gz_inc2 - (2*nz_at1+1) * gz[addrz]);
                            if (nx_at1 > 1) { g3x += nx_at1*(nx_at1-1) * gx[addrx-stride_at1*2]; }
                            if (ny_at1 > 1) { g3y += ny_at1*(ny_at1-1) * gy[addry-stride_at1*2]; }
                            if (nz_at1 > 1) { g3z += nz_at1*(nz_at1-1) * gz[addrz-stride_at1*2]; }
                        } else {
                            g3x = a2_at1 * gx[addrx+stride_at1+stride_at2];
                            g3y = a2_at1 * gy[addry+stride_at1+stride_at2];
                            g3z = a2_at1 * gz[addrz+stride_at1+stride_at2];
                            if (nx_at1 > 0) { g3x -= nx_at1 * gx[addrx-stride_at1+stride_at2]; }
                            if (ny_at1 > 0) { g3y -= ny_at1 * gy[addry-stride_at1+stride_at2]; }
                            if (nz_at1 > 0) { g3z -= nz_at1 * gz[addrz-stride_at1+stride_at2]; }
                            g3x *= a2_at2;
                            g3y *= a2_at2;
                            g3z *= a2_at2;

                            if (nx_at2 > 0) {
                                double fx = a2_at1 * gx[addrx+stride_at1-stride_at2];
                                if (nx_at1 > 0) { fx -= nx_at1 * gx[addrx-stride_at1-stride_at2]; }
                                g3x -= nx_at2 * fx;
                            }
                            if (ny_at2 > 0) {
                                double fy = a2_at1 * gy[addry+stride_at1-stride_at2];
                                if (ny_at1 > 0) { fy -= ny_at1 * gy[addry-stride_at1-stride_at2]; }
                                g3y -= ny_at2 * fy;
                            }
                            if (nz_at2 > 0) {
                                double fz = a2_at1 * gz[addrz+stride_at1-stride_at2];
                                if (nz_at1 > 0) { fz -= nz_at1 * gz[addrz-stride_at1-stride_at2]; }
                                g3z -= nz_at2 * fz;
                            }
                        }
                        double gout_xx = g3x * gy[addry] * gz[addrz];
                        double gout_yy = g3y * gx[addrx] * gz[addrz];
                        double gout_zz = g3z * gx[addrx] * gy[addry];
                        double gout_xy = g2x * g1y * gz[addrz];
                        double gout_xz = g2x * g1z * gy[addry];
                        double gout_yx = g2y * g1x * gz[addrz];
                        double gout_yz = g2y * g1z * gx[addrx];
                        double gout_zx = g2z * g1x * gy[addry];
                        double gout_zy = g2z * g1y * gx[addrx];

                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;
                        if (vk != NULL) {
                            int _jl = _j*nao+_l;
                            int _jk = _j*nao+_k;
                            int _il = _i*nao+_l;
                            int _ik = _i*nao+_k;
                            double dd_jk = dm[_jk] * dm[_il];
                            double dd_jl = dm[_jl] * dm[_ik];
                            double dd = dd_jk + dd_jl;
                            if (jk.n_dm > 1) {
                                int nao2 = nao * nao;
                                double dd_jk = dm[nao2+_jk] * dm[nao2+_il];
                                double dd_jl = dm[nao2+_jl] * dm[nao2+_ik];
                                dd += dd_jk + dd_jl;
                            }
                            vk_xx += gout_xx * dd;
                            vk_yy += gout_yy * dd;
                            vk_zz += gout_zz * dd;
                            vk_xy += gout_xy * dd;
                            vk_xz += gout_xz * dd;
                            vk_yx += gout_yx * dd;
                            vk_yz += gout_yz * dd;
                            vk_zx += gout_zx * dd;
                            vk_zy += gout_zy * dd;
                        }
                        if (vj != NULL) {
                            int _ji = _j*nao+_i;
                            int _lk = _l*nao+_k;
                            double dd;
                            if (jk.n_dm == 1) {
                                dd = dm[_ji] * dm[_lk];
                            } else {
                                int nao2 = nao * nao;
                                dd = (dm[_ji] + dm[nao2+_ji]) * (dm[_lk] + dm[nao2+_lk]);
                            }
                            vj_xx += gout_xx * dd;
                            vj_yy += gout_yy * dd;
                            vj_zz += gout_zz * dd;
                            vj_xy += gout_xy * dd;
                            vj_xz += gout_xz * dd;
                            vj_yx += gout_yx * dd;
                            vj_yz += gout_yz * dd;
                            vj_zx += gout_zx * dd;
                            vj_zy += gout_zy * dd;
                        }
                    }
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        switch (at1) {
        case 0: nx_at1 = ia; break;
        case 1: nx_at1 = ja; break;
        case 2: nx_at1 = ka; break;
        case 3: nx_at1 = la; break;
        }
        switch (at2) {
        case 0: nx_at2 = ia; break;
        case 1: nx_at2 = ja; break;
        case 2: nx_at2 = ka; break;
        case 3: nx_at2 = la; break;
        }
        int natm = envs.natm;
        if (vk != NULL) {
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 0, vk_xx);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 1, vk_xy);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 2, vk_xz);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 3, vk_yx);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 4, vk_yy);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 5, vk_yz);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 6, vk_zx);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 7, vk_zy);
            atomicAdd(vk + (nx_at2*natm+nx_at1)*9 + 8, vk_zz);
        }
        if (vj != NULL) {
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 0, vj_xx);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 1, vj_xy);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 2, vj_xz);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 3, vj_yx);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 4, vj_yy);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 5, vj_yz);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 6, vj_zx);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 7, vj_zy);
            atomicAdd(vj + (nx_at2*natm+nx_at1)*9 + 8, vj_zz);
        }
    }
}

__global__
void rys_ejk_ip2_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_ejk_ip2_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
