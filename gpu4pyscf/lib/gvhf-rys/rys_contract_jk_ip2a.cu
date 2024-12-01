#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks_ip1.cu"
#include "create_tasks_ip2.cu"

//type 1: (d^2i j | k l)
//type 2: (di dj | k l)
//type 3: (di j | dk l)

__device__
static void rys_ejk_ip2_type1_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int g_stride_i =          nsq_per_block;
    int g_stride_j = stride_j*nsq_per_block;
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
    double Rqc[3], Rpq[3], rirj[3], rkrl[3];

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
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        rkrl[0] = rk[0] - rl[0];
        rkrl[1] = rk[1] - rl[1];
        rkrl[2] = rk[2] - rl[2];
        double vj_ixx = 0;
        double vj_ixy = 0;
        double vj_ixz = 0;
        double vj_iyy = 0;
        double vj_iyz = 0;
        double vj_izz = 0;
        double vj_jxx = 0;
        double vj_jxy = 0;
        double vj_jxz = 0;
        double vj_jyy = 0;
        double vj_jyz = 0;
        double vj_jzz = 0;
        double vk_ixx = 0;
        double vk_ixy = 0;
        double vk_ixz = 0;
        double vk_iyy = 0;
        double vk_iyz = 0;
        double vk_izz = 0;
        double vk_jxx = 0;
        double vk_jxy = 0;
        double vk_jxz = 0;
        double vk_jyy = 0;
        double vk_jyz = 0;
        double vk_jzz = 0;
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xixj = rirj[0];
            double yiyj = rirj[1];
            double zizj = rirj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xkxl = rkrl[0];
            double ykyl = rkrl[1];
            double zkzl = rkrl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(_nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
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

                    if (lkl > 0) {
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
                            double xixj = rirj[_ix];
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
                    if (ll > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xkxl = rkrl[_ix];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
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
                    }

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
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
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;

                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double g1x, g1y, g1z;
                        double g3x, g3y, g3z;
                        double _gx_inc2, _gy_inc2, _gz_inc2;

                        g1x = ai2 * gx[addrx+g_stride_i];
                        g1y = ai2 * gy[addry+g_stride_i];
                        g1z = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { g1x -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { g1y -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { g1z -= iz * gz[addrz-g_stride_i]; }
                        double xixj = rirj[0];
                        double yiyj = rirj[1];
                        double zizj = rirj[2];
                        _gx_inc2 = gx[addrx+g_stride_i+g_stride_j] - gx[addrx+g_stride_i] * xixj;
                        _gy_inc2 = gy[addry+g_stride_i+g_stride_j] - gy[addry+g_stride_i] * yiyj;
                        _gz_inc2 = gz[addrz+g_stride_i+g_stride_j] - gz[addrz+g_stride_i] * zizj;
                        g3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * gx[addrx]);
                        g3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * gy[addry]);
                        g3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * gz[addrz]);
                        if (ix > 1) { g3x += ix*(ix-1) * gx[addrx-g_stride_i*2]; }
                        if (iy > 1) { g3y += iy*(iy-1) * gy[addry-g_stride_i*2]; }
                        if (iz > 1) { g3z += iz*(iz-1) * gz[addrz-g_stride_i*2]; }

                        double gout_ixx = g3x * gy[addry] * gz[addrz];
                        double gout_iyy = g3y * gx[addrx] * gz[addrz];
                        double gout_izz = g3z * gx[addrx] * gy[addry];
                        double gout_ixy = g1x * g1y * gz[addrz];
                        double gout_ixz = g1x * g1z * gy[addry];
                        double gout_iyz = g1y * g1z * gx[addrx];

                        g1x = aj2 * gx[addrx+g_stride_j];
                        g1y = aj2 * gy[addry+g_stride_j];
                        g1z = aj2 * gz[addrz+g_stride_j];
                        if (jx > 0) { g1x -= jx * gx[addrx-g_stride_j]; }
                        if (jy > 0) { g1y -= jy * gy[addry-g_stride_j]; }
                        if (jz > 0) { g1z -= jz * gz[addrz-g_stride_j]; }
                        _gx_inc2 = gx[addrx+g_stride_i+g_stride_j] + gx[addrx+g_stride_j] * xixj;
                        _gy_inc2 = gy[addry+g_stride_i+g_stride_j] + gy[addry+g_stride_j] * yiyj;
                        _gz_inc2 = gz[addrz+g_stride_i+g_stride_j] + gz[addrz+g_stride_j] * zizj;
                        g3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * gx[addrx]);
                        g3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * gy[addry]);
                        g3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * gz[addrz]);
                        if (jx > 1) { g3x += jx*(jx-1) * gx[addrx-g_stride_j*2]; }
                        if (jy > 1) { g3y += jy*(jy-1) * gy[addry-g_stride_j*2]; }
                        if (jz > 1) { g3z += jz*(jz-1) * gz[addrz-g_stride_j*2]; }

                        double gout_jxx = g3x * gy[addry] * gz[addrz];
                        double gout_jyy = g3y * gx[addrx] * gz[addrz];
                        double gout_jzz = g3z * gx[addrx] * gy[addry];
                        double gout_jxy = g1x * g1y * gz[addrz];
                        double gout_jxz = g1x * g1z * gy[addry];
                        double gout_jyz = g1y * g1z * gx[addrx];

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
                            vk_ixx += gout_ixx * dd;
                            vk_iyy += gout_iyy * dd;
                            vk_izz += gout_izz * dd;
                            vk_ixy += gout_ixy * dd;
                            vk_ixz += gout_ixz * dd;
                            vk_iyz += gout_iyz * dd;
                            vk_jxx += gout_jxx * dd;
                            vk_jyy += gout_jyy * dd;
                            vk_jzz += gout_jzz * dd;
                            vk_jxy += gout_jxy * dd;
                            vk_jxz += gout_jxz * dd;
                            vk_jyz += gout_jyz * dd;
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
                            vj_ixx += gout_ixx * dd;
                            vj_iyy += gout_iyy * dd;
                            vj_izz += gout_izz * dd;
                            vj_ixy += gout_ixy * dd;
                            vj_ixz += gout_ixz * dd;
                            vj_iyz += gout_iyz * dd;
                            vj_jxx += gout_jxx * dd;
                            vj_jyy += gout_jyy * dd;
                            vj_jzz += gout_jzz * dd;
                            vj_jxy += gout_jxy * dd;
                            vj_jxz += gout_jxz * dd;
                            vj_jyz += gout_jyz * dd;
                        }
                    }
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int natm = envs.natm;
        int t_id = sq_id + gout_id * nsq_per_block;
        int threads = nsq_per_block * gout_stride;
        if (vj != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vj_ixx;
            reduce[t_id+1 *threads] = vj_ixy;
            reduce[t_id+2 *threads] = vj_iyy;
            reduce[t_id+3 *threads] = vj_ixz;
            reduce[t_id+4 *threads] = vj_iyz;
            reduce[t_id+5 *threads] = vj_izz;
            reduce[t_id+6 *threads] = vj_jxx;
            reduce[t_id+7 *threads] = vj_jxy;
            reduce[t_id+8 *threads] = vj_jyy;
            reduce[t_id+9 *threads] = vj_jxz;
            reduce[t_id+10*threads] = vj_jyz;
            reduce[t_id+11*threads] = vj_jzz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 12; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vj+(ia*natm+ia)*9+0, reduce[sq_id+0 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+3, reduce[sq_id+1 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+4, reduce[sq_id+2 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+6, reduce[sq_id+3 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+7, reduce[sq_id+4 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+8, reduce[sq_id+5 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+0, reduce[sq_id+6 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+3, reduce[sq_id+7 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+4, reduce[sq_id+8 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+6, reduce[sq_id+9 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+7, reduce[sq_id+10*threads]);
                atomicAdd(vj+(ja*natm+ja)*9+8, reduce[sq_id+11*threads]);
            }
        }
        if (vk != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vk_ixx;
            reduce[t_id+1 *threads] = vk_ixy;
            reduce[t_id+2 *threads] = vk_iyy;
            reduce[t_id+3 *threads] = vk_ixz;
            reduce[t_id+4 *threads] = vk_iyz;
            reduce[t_id+5 *threads] = vk_izz;
            reduce[t_id+6 *threads] = vk_jxx;
            reduce[t_id+7 *threads] = vk_jxy;
            reduce[t_id+8 *threads] = vk_jyy;
            reduce[t_id+9 *threads] = vk_jxz;
            reduce[t_id+10*threads] = vk_jyz;
            reduce[t_id+11*threads] = vk_jzz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 12; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vk+(ia*natm+ia)*9+0, reduce[sq_id+0 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+3, reduce[sq_id+1 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+4, reduce[sq_id+2 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+6, reduce[sq_id+3 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+7, reduce[sq_id+4 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+8, reduce[sq_id+5 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+0, reduce[sq_id+6 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+3, reduce[sq_id+7 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+4, reduce[sq_id+8 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+6, reduce[sq_id+9 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+7, reduce[sq_id+10*threads]);
                atomicAdd(vk+(ja*natm+ja)*9+8, reduce[sq_id+11*threads]);
            }
        }
    }
}

__device__
static void rys_ejk_ip2_type2_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int g_stride_i =          nsq_per_block;
    int g_stride_j = stride_j*nsq_per_block;
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
    double Rqc[3], Rpq[3], rirj[3], rkrl[3];

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
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        rkrl[0] = rk[0] - rl[0];
        rkrl[1] = rk[1] - rl[1];
        rkrl[2] = rk[2] - rl[2];
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
            double xixj = rirj[0];
            double yiyj = rirj[1];
            double zizj = rirj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xkxl = rkrl[0];
            double ykyl = rkrl[1];
            double zkzl = rkrl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(_nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
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

                    if (lkl > 0) {
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
                            double xixj = rirj[_ix];
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
                    if (ll > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xkxl = rkrl[_ix];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
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
                    }

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
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
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;

                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;

                        double g1x = aj2 * gx[addrx+g_stride_j];
                        double g1y = aj2 * gy[addry+g_stride_j];
                        double g1z = aj2 * gz[addrz+g_stride_j];
                        if (jx > 0) { g1x -= jx * gx[addrx-g_stride_j]; }
                        if (jy > 0) { g1y -= jy * gy[addry-g_stride_j]; }
                        if (jz > 0) { g1z -= jz * gz[addrz-g_stride_j]; }

                        double g2x = ai2 * gx[addrx+g_stride_i];
                        double g2y = ai2 * gy[addry+g_stride_i];
                        double g2z = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { g2x -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { g2y -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { g2z -= iz * gz[addrz-g_stride_i]; }

                        double g3x, g3y, g3z;
                        g3x = ai2 * gx[addrx+g_stride_i+g_stride_j];
                        g3y = ai2 * gy[addry+g_stride_i+g_stride_j];
                        g3z = ai2 * gz[addrz+g_stride_i+g_stride_j];
                        if (ix > 0) { g3x -= ix * gx[addrx-g_stride_i+g_stride_j]; }
                        if (iy > 0) { g3y -= iy * gy[addry-g_stride_i+g_stride_j]; }
                        if (iz > 0) { g3z -= iz * gz[addrz-g_stride_i+g_stride_j]; }
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;

                        if (jx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_j];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_j]; }
                            g3x -= jx * fx;
                        }
                        if (jy > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_j];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_j]; }
                            g3y -= jy * fy;
                        }
                        if (jz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_j];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_j]; }
                            g3z -= jz * fz;
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
        int natm = envs.natm;
        int t_id = sq_id + gout_id * nsq_per_block;
        int threads = nsq_per_block * gout_stride;
        if (vj != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vj_xx;
            reduce[t_id+1 *threads] = vj_xy;
            reduce[t_id+2 *threads] = vj_xz;
            reduce[t_id+3 *threads] = vj_yx;
            reduce[t_id+4 *threads] = vj_yy;
            reduce[t_id+5 *threads] = vj_yz;
            reduce[t_id+6 *threads] = vj_zx;
            reduce[t_id+7 *threads] = vj_zy;
            reduce[t_id+8 *threads] = vj_zz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vj + (ia*natm+ja)*9 + 0, reduce[sq_id+0 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 1, reduce[sq_id+1 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 2, reduce[sq_id+2 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 3, reduce[sq_id+3 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 4, reduce[sq_id+4 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 5, reduce[sq_id+5 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 6, reduce[sq_id+6 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 7, reduce[sq_id+7 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 8, reduce[sq_id+8 *threads]);
            }
        }
        if (vk != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vk_xx;
            reduce[t_id+1 *threads] = vk_xy;
            reduce[t_id+2 *threads] = vk_xz;
            reduce[t_id+3 *threads] = vk_yx;
            reduce[t_id+4 *threads] = vk_yy;
            reduce[t_id+5 *threads] = vk_yz;
            reduce[t_id+6 *threads] = vk_zx;
            reduce[t_id+7 *threads] = vk_zy;
            reduce[t_id+8 *threads] = vk_zz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vk + (ia*natm+ja)*9 + 0, reduce[sq_id+0 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 1, reduce[sq_id+1 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 2, reduce[sq_id+2 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 3, reduce[sq_id+3 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 4, reduce[sq_id+4 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 5, reduce[sq_id+5 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 6, reduce[sq_id+6 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 7, reduce[sq_id+7 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 8, reduce[sq_id+8 *threads]);
            }
        }
    }
}

__device__
static void rys_ejk_ip2_type12_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int g_stride_i =          nsq_per_block;
    int g_stride_j = stride_j*nsq_per_block;
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
    double Rqc[3], Rpq[3], rirj[3], rkrl[3];

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
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        rkrl[0] = rk[0] - rl[0];
        rkrl[1] = rk[1] - rl[1];
        rkrl[2] = rk[2] - rl[2];
        double vj_ixx = 0;
        double vj_ixy = 0;
        double vj_ixz = 0;
        double vj_iyy = 0;
        double vj_iyz = 0;
        double vj_izz = 0;
        double vj_jxx = 0;
        double vj_jxy = 0;
        double vj_jxz = 0;
        double vj_jyy = 0;
        double vj_jyz = 0;
        double vj_jzz = 0;
        double vk_ixx = 0;
        double vk_ixy = 0;
        double vk_ixz = 0;
        double vk_iyy = 0;
        double vk_iyz = 0;
        double vk_izz = 0;
        double vk_jxx = 0;
        double vk_jxy = 0;
        double vk_jxz = 0;
        double vk_jyy = 0;
        double vk_jyz = 0;
        double vk_jzz = 0;

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
            double xixj = rirj[0];
            double yiyj = rirj[1];
            double zizj = rirj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double al_akl = al / akl;
            double xkxl = rkrl[0];
            double ykyl = rkrl[1];
            double zkzl = rkrl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(_nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
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

                    if (lkl > 0) {
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
                            double xixj = rirj[_ix];
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
                    if (ll > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xkxl = rkrl[_ix];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
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
                    }

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
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
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;

                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double g1x, g1y, g1z;
                        double g2x, g2y, g2z;
                        double g3x, g3y, g3z;
                        double _gx_inc2, _gy_inc2, _gz_inc2;
                        double xixj = rirj[0];
                        double yiyj = rirj[1];
                        double zizj = rirj[2];

                        g1x = aj2 * gx[addrx+g_stride_j];
                        g1y = aj2 * gy[addry+g_stride_j];
                        g1z = aj2 * gz[addrz+g_stride_j];
                        if (jx > 0) { g1x -= jx * gx[addrx-g_stride_j]; }
                        if (jy > 0) { g1y -= jy * gy[addry-g_stride_j]; }
                        if (jz > 0) { g1z -= jz * gz[addrz-g_stride_j]; }
                        _gx_inc2 = gx[addrx+g_stride_i+g_stride_j] + gx[addrx+g_stride_j] * xixj;
                        _gy_inc2 = gy[addry+g_stride_i+g_stride_j] + gy[addry+g_stride_j] * yiyj;
                        _gz_inc2 = gz[addrz+g_stride_i+g_stride_j] + gz[addrz+g_stride_j] * zizj;
                        g3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * gx[addrx]);
                        g3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * gy[addry]);
                        g3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * gz[addrz]);
                        if (jx > 1) { g3x += jx*(jx-1) * gx[addrx-g_stride_j*2]; }
                        if (jy > 1) { g3y += jy*(jy-1) * gy[addry-g_stride_j*2]; }
                        if (jz > 1) { g3z += jz*(jz-1) * gz[addrz-g_stride_j*2]; }
                        double gout_jxx = g3x * gy[addry] * gz[addrz];
                        double gout_jyy = g3y * gx[addrx] * gz[addrz];
                        double gout_jzz = g3z * gx[addrx] * gy[addry];
                        double gout_jxy = g1x * g1y * gz[addrz];
                        double gout_jxz = g1x * g1z * gy[addry];
                        double gout_jyz = g1y * g1z * gx[addrx];

                        g2x = ai2 * gx[addrx+g_stride_i];
                        g2y = ai2 * gy[addry+g_stride_i];
                        g2z = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { g2x -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { g2y -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { g2z -= iz * gz[addrz-g_stride_i]; }
                        _gx_inc2 = gx[addrx+g_stride_i+g_stride_j] - gx[addrx+g_stride_i] * xixj;
                        _gy_inc2 = gy[addry+g_stride_i+g_stride_j] - gy[addry+g_stride_i] * yiyj;
                        _gz_inc2 = gz[addrz+g_stride_i+g_stride_j] - gz[addrz+g_stride_i] * zizj;
                        g3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * gx[addrx]);
                        g3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * gy[addry]);
                        g3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * gz[addrz]);
                        if (ix > 1) { g3x += ix*(ix-1) * gx[addrx-g_stride_i*2]; }
                        if (iy > 1) { g3y += iy*(iy-1) * gy[addry-g_stride_i*2]; }
                        if (iz > 1) { g3z += iz*(iz-1) * gz[addrz-g_stride_i*2]; }
                        double gout_ixx = g3x * gy[addry] * gz[addrz];
                        double gout_iyy = g3y * gx[addrx] * gz[addrz];
                        double gout_izz = g3z * gx[addrx] * gy[addry];
                        double gout_ixy = g2x * g2y * gz[addrz];
                        double gout_ixz = g2x * g2z * gy[addry];
                        double gout_iyz = g2y * g2z * gx[addrx];

                        g3x = ai2 * gx[addrx+g_stride_i+g_stride_j];
                        g3y = ai2 * gy[addry+g_stride_i+g_stride_j];
                        g3z = ai2 * gz[addrz+g_stride_i+g_stride_j];
                        if (ix > 0) { g3x -= ix * gx[addrx-g_stride_i+g_stride_j]; }
                        if (iy > 0) { g3y -= iy * gy[addry-g_stride_i+g_stride_j]; }
                        if (iz > 0) { g3z -= iz * gz[addrz-g_stride_i+g_stride_j]; }
                        g3x *= aj2;
                        g3y *= aj2;
                        g3z *= aj2;
                        if (jx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_j];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_j]; }
                            g3x -= jx * fx;
                        }
                        if (jy > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_j];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_j]; }
                            g3y -= jy * fy;
                        }
                        if (jz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_j];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_j]; }
                            g3z -= jz * fz;
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
                            vk_ixx += gout_ixx * dd;
                            vk_iyy += gout_iyy * dd;
                            vk_izz += gout_izz * dd;
                            vk_ixy += gout_ixy * dd;
                            vk_ixz += gout_ixz * dd;
                            vk_iyz += gout_iyz * dd;
                            vk_jxx += gout_jxx * dd;
                            vk_jyy += gout_jyy * dd;
                            vk_jzz += gout_jzz * dd;
                            vk_jxy += gout_jxy * dd;
                            vk_jxz += gout_jxz * dd;
                            vk_jyz += gout_jyz * dd;
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
                            vj_ixx += gout_ixx * dd;
                            vj_iyy += gout_iyy * dd;
                            vj_izz += gout_izz * dd;
                            vj_ixy += gout_ixy * dd;
                            vj_ixz += gout_ixz * dd;
                            vj_iyz += gout_iyz * dd;
                            vj_jxx += gout_jxx * dd;
                            vj_jyy += gout_jyy * dd;
                            vj_jzz += gout_jzz * dd;
                            vj_jxy += gout_jxy * dd;
                            vj_jxz += gout_jxz * dd;
                            vj_jyz += gout_jyz * dd;
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
        int natm = envs.natm;
        int t_id = sq_id + gout_id * nsq_per_block;
        int threads = nsq_per_block * gout_stride;
        if (vj != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vj_xx;
            reduce[t_id+1 *threads] = vj_xy;
            reduce[t_id+2 *threads] = vj_xz;
            reduce[t_id+3 *threads] = vj_yx;
            reduce[t_id+4 *threads] = vj_yy;
            reduce[t_id+5 *threads] = vj_yz;
            reduce[t_id+6 *threads] = vj_zx;
            reduce[t_id+7 *threads] = vj_zy;
            reduce[t_id+8 *threads] = vj_zz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vj + (ia*natm+ja)*9 + 0, reduce[sq_id+0 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 1, reduce[sq_id+1 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 2, reduce[sq_id+2 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 3, reduce[sq_id+3 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 4, reduce[sq_id+4 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 5, reduce[sq_id+5 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 6, reduce[sq_id+6 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 7, reduce[sq_id+7 *threads]);
                atomicAdd(vj + (ia*natm+ja)*9 + 8, reduce[sq_id+8 *threads]);
            }

            __syncthreads();
            reduce[t_id+0 *threads] = vj_ixx;
            reduce[t_id+1 *threads] = vj_ixy;
            reduce[t_id+2 *threads] = vj_iyy;
            reduce[t_id+3 *threads] = vj_ixz;
            reduce[t_id+4 *threads] = vj_iyz;
            reduce[t_id+5 *threads] = vj_izz;
            reduce[t_id+6 *threads] = vj_jxx;
            reduce[t_id+7 *threads] = vj_jxy;
            reduce[t_id+8 *threads] = vj_jyy;
            reduce[t_id+9 *threads] = vj_jxz;
            reduce[t_id+10*threads] = vj_jyz;
            reduce[t_id+11*threads] = vj_jzz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 12; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vj+(ia*natm+ia)*9+0, reduce[sq_id+0 *threads]*.5);
                atomicAdd(vj+(ia*natm+ia)*9+3, reduce[sq_id+1 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+4, reduce[sq_id+2 *threads]*.5);
                atomicAdd(vj+(ia*natm+ia)*9+6, reduce[sq_id+3 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+7, reduce[sq_id+4 *threads]);
                atomicAdd(vj+(ia*natm+ia)*9+8, reduce[sq_id+5 *threads]*.5);
                atomicAdd(vj+(ja*natm+ja)*9+0, reduce[sq_id+6 *threads]*.5);
                atomicAdd(vj+(ja*natm+ja)*9+3, reduce[sq_id+7 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+4, reduce[sq_id+8 *threads]*.5);
                atomicAdd(vj+(ja*natm+ja)*9+6, reduce[sq_id+9 *threads]);
                atomicAdd(vj+(ja*natm+ja)*9+7, reduce[sq_id+10*threads]);
                atomicAdd(vj+(ja*natm+ja)*9+8, reduce[sq_id+11*threads]*.5);
            }
        }
        if (vk != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vk_xx;
            reduce[t_id+1 *threads] = vk_xy;
            reduce[t_id+2 *threads] = vk_xz;
            reduce[t_id+3 *threads] = vk_yx;
            reduce[t_id+4 *threads] = vk_yy;
            reduce[t_id+5 *threads] = vk_yz;
            reduce[t_id+6 *threads] = vk_zx;
            reduce[t_id+7 *threads] = vk_zy;
            reduce[t_id+8 *threads] = vk_zz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vk + (ia*natm+ja)*9 + 0, reduce[sq_id+0 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 1, reduce[sq_id+1 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 2, reduce[sq_id+2 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 3, reduce[sq_id+3 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 4, reduce[sq_id+4 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 5, reduce[sq_id+5 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 6, reduce[sq_id+6 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 7, reduce[sq_id+7 *threads]);
                atomicAdd(vk + (ia*natm+ja)*9 + 8, reduce[sq_id+8 *threads]);
            }

            __syncthreads();
            reduce[t_id+0 *threads] = vk_ixx;
            reduce[t_id+1 *threads] = vk_ixy;
            reduce[t_id+2 *threads] = vk_iyy;
            reduce[t_id+3 *threads] = vk_ixz;
            reduce[t_id+4 *threads] = vk_iyz;
            reduce[t_id+5 *threads] = vk_izz;
            reduce[t_id+6 *threads] = vk_jxx;
            reduce[t_id+7 *threads] = vk_jxy;
            reduce[t_id+8 *threads] = vk_jyy;
            reduce[t_id+9 *threads] = vk_jxz;
            reduce[t_id+10*threads] = vk_jyz;
            reduce[t_id+11*threads] = vk_jzz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 12; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vk+(ia*natm+ia)*9+0, reduce[sq_id+0 *threads]*.5);
                atomicAdd(vk+(ia*natm+ia)*9+3, reduce[sq_id+1 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+4, reduce[sq_id+2 *threads]*.5);
                atomicAdd(vk+(ia*natm+ia)*9+6, reduce[sq_id+3 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+7, reduce[sq_id+4 *threads]);
                atomicAdd(vk+(ia*natm+ia)*9+8, reduce[sq_id+5 *threads]*.5);
                atomicAdd(vk+(ja*natm+ja)*9+0, reduce[sq_id+6 *threads]*.5);
                atomicAdd(vk+(ja*natm+ja)*9+3, reduce[sq_id+7 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+4, reduce[sq_id+8 *threads]*.5);
                atomicAdd(vk+(ja*natm+ja)*9+6, reduce[sq_id+9 *threads]);
                atomicAdd(vk+(ja*natm+ja)*9+7, reduce[sq_id+10*threads]);
                atomicAdd(vk+(ja*natm+ja)*9+8, reduce[sq_id+11*threads]*.5);
            }
        }
    }
}

__device__
static void rys_ejk_ip2_type3_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int lij = li + lj + 1;
    int lkl = lk + ll + 1;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_stride_i =          nsq_per_block;
    int g_stride_k = stride_k*nsq_per_block;
    int g_size = stride_l * (ll + 1);
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
    double Rqc[3], Rpq[3], rirj[3], rkrl[3];

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
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        rkrl[0] = rk[0] - rl[0];
        rkrl[1] = rk[1] - rl[1];
        rkrl[2] = rk[2] - rl[2];
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
            double xixj = rirj[0];
            double yiyj = rirj[1];
            double zizj = rirj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
            Rpa[sq_id+3*nsq_per_block] = fac_sym * ci[ip] * cj[jp] * Kab;
        }
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            double ak = expk[kp];
            double al = expl[lp];
            double akl = ak + al;
            double ak2 = ak * 2;
            double al_akl = al / akl;
            double xkxl = rkrl[0];
            double ykyl = rkrl[1];
            double zkzl = rkrl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(_nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
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
                    if (lj > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            int lkl3 = (lkl+1)*3;
                            for (int m = gout_id; m < lkl3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                double xixj = rirj[_ix];
                                double *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                                for (int j = 0; j < lj; ++j) {
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
                    }
                    if (ll > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                double xkxl = rkrl[_ix];
                                double *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
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
                    }

                    __syncthreads();
                    if (task_id >= ntasks) {
                        continue;
                    }
                    double *gx = g;
                    double *gy = gx + nsq_per_block * g_size;
                    double *gz = gy + nsq_per_block * g_size;
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
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
                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;

                        double g1x = ak2 * gx[addrx+g_stride_k];
                        double g1y = ak2 * gy[addry+g_stride_k];
                        double g1z = ak2 * gz[addrz+g_stride_k];
                        if (kx > 0) { g1x -= kx * gx[addrx-g_stride_k]; }
                        if (ky > 0) { g1y -= ky * gy[addry-g_stride_k]; }
                        if (kz > 0) { g1z -= kz * gz[addrz-g_stride_k]; }

                        double g2x = ai2 * gx[addrx+g_stride_i];
                        double g2y = ai2 * gy[addry+g_stride_i];
                        double g2z = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { g2x -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { g2y -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { g2z -= iz * gz[addrz-g_stride_i]; }

                        double g3x, g3y, g3z;
                        g3x = ai2 * gx[addrx+g_stride_i+g_stride_k];
                        g3y = ai2 * gy[addry+g_stride_i+g_stride_k];
                        g3z = ai2 * gz[addrz+g_stride_i+g_stride_k];
                        if (ix > 0) { g3x -= ix * gx[addrx-g_stride_i+g_stride_k]; }
                        if (iy > 0) { g3y -= iy * gy[addry-g_stride_i+g_stride_k]; }
                        if (iz > 0) { g3z -= iz * gz[addrz-g_stride_i+g_stride_k]; }
                        g3x *= ak2;
                        g3y *= ak2;
                        g3z *= ak2;

                        if (kx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_k];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_k]; }
                            g3x -= kx * fx;
                        }
                        if (ky > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_k];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_k]; }
                            g3y -= ky * fy;
                        }
                        if (kz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_k];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_k]; }
                            g3z -= kz * fz;
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
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int natm = envs.natm;
        int t_id = sq_id + gout_id * nsq_per_block;
        int threads = nsq_per_block * gout_stride;
        if (vj != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vj_xx;
            reduce[t_id+1 *threads] = vj_xy;
            reduce[t_id+2 *threads] = vj_xz;
            reduce[t_id+3 *threads] = vj_yx;
            reduce[t_id+4 *threads] = vj_yy;
            reduce[t_id+5 *threads] = vj_yz;
            reduce[t_id+6 *threads] = vj_zx;
            reduce[t_id+7 *threads] = vj_zy;
            reduce[t_id+8 *threads] = vj_zz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vj + (ia*natm+ka)*9 + 0, reduce[sq_id+0 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 1, reduce[sq_id+1 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 2, reduce[sq_id+2 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 3, reduce[sq_id+3 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 4, reduce[sq_id+4 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 5, reduce[sq_id+5 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 6, reduce[sq_id+6 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 7, reduce[sq_id+7 *threads]);
                atomicAdd(vj + (ia*natm+ka)*9 + 8, reduce[sq_id+8 *threads]);
            }
        }
        if (vk != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vk_xx;
            reduce[t_id+1 *threads] = vk_xy;
            reduce[t_id+2 *threads] = vk_xz;
            reduce[t_id+3 *threads] = vk_yx;
            reduce[t_id+4 *threads] = vk_yy;
            reduce[t_id+5 *threads] = vk_yz;
            reduce[t_id+6 *threads] = vk_zx;
            reduce[t_id+7 *threads] = vk_zy;
            reduce[t_id+8 *threads] = vk_zz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0 && task_id < ntasks) {
                atomicAdd(vk + (ia*natm+ka)*9 + 0, reduce[sq_id+0 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 1, reduce[sq_id+1 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 2, reduce[sq_id+2 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 3, reduce[sq_id+3 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 4, reduce[sq_id+4 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 5, reduce[sq_id+5 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 6, reduce[sq_id+6 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 7, reduce[sq_id+7 *threads]);
                atomicAdd(vk + (ia*natm+ka)*9 + 8, reduce[sq_id+8 *threads]);
            }
        }
    }
}

__device__
static void rys_ej_ip2_type3_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int g_stride_i =          nsq_per_block;
    int g_stride_j = stride_j*nsq_per_block;
    int g_stride_k = stride_k*nsq_per_block;
    int g_stride_l = stride_l*nsq_per_block;
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
    double *dm = jk.dm;
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3], Rpq[3], rirj[3], rkrl[3];

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
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        rkrl[0] = rk[0] - rl[0];
        rkrl[1] = rk[1] - rl[1];
        rkrl[2] = rk[2] - rl[2];
        double vj_ik_xx = 0;
        double vj_ik_xy = 0;
        double vj_ik_xz = 0;
        double vj_ik_yx = 0;
        double vj_ik_yy = 0;
        double vj_ik_yz = 0;
        double vj_ik_zx = 0;
        double vj_ik_zy = 0;
        double vj_ik_zz = 0;
        double vj_jk_xx = 0;
        double vj_jk_xy = 0;
        double vj_jk_xz = 0;
        double vj_jk_yx = 0;
        double vj_jk_yy = 0;
        double vj_jk_yz = 0;
        double vj_jk_zx = 0;
        double vj_jk_zy = 0;
        double vj_jk_zz = 0;
        double vj_il_xx = 0;
        double vj_il_xy = 0;
        double vj_il_xz = 0;
        double vj_il_yx = 0;
        double vj_il_yy = 0;
        double vj_il_yz = 0;
        double vj_il_zx = 0;
        double vj_il_zy = 0;
        double vj_il_zz = 0;
        double vj_jl_xx = 0;
        double vj_jl_xy = 0;
        double vj_jl_xz = 0;
        double vj_jl_yx = 0;
        double vj_jl_yy = 0;
        double vj_jl_yz = 0;
        double vj_jl_zx = 0;
        double vj_jl_zy = 0;
        double vj_jl_zz = 0;
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            double ai = expi[ip];
            double aj = expj[jp];
            double aij = ai + aj;
            double aj_aij = aj / aij;
            double xixj = rirj[0];
            double yiyj = rirj[1];
            double zizj = rirj[2];
            double *Rpa = Rpa_cicj + ij*4*nsq_per_block;
            Rpa[sq_id+0*nsq_per_block] = xixj * -aj_aij;
            Rpa[sq_id+1*nsq_per_block] = yiyj * -aj_aij;
            Rpa[sq_id+2*nsq_per_block] = zizj * -aj_aij;
            double theta_ij = ai * aj_aij;
            double Kab = exp(-theta_ij * (xixj*xixj+yiyj*yiyj+zizj*zizj));
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
            double xkxl = rkrl[0];
            double ykyl = rkrl[1];
            double zkzl = rkrl[2];
            Rqc[0] = xkxl * -al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ykyl * -al_akl;
            Rqc[2] = zkzl * -al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xkxl*xkxl+ykyl*ykyl+zkzl*zkzl));
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
                } else if (omega > 0) {
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = sqrt(theta_fac);
                    for (int irys = gout_id; irys < nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
                } else {
                    int _nroots = nroots/2;
                    rys_roots(_nroots, theta_rr, rw+nroots*nsq_per_block);
                    double theta_fac = omega * omega / (omega * omega + theta);
                    rys_roots(_nroots, theta_fac*theta_rr, rw);
                    __syncthreads();
                    double sqrt_theta_fac = -sqrt(theta_fac);
                    for (int irys = gout_id; irys < _nroots; irys+=gout_stride) {
                        rw[sq_id+ irys*2   *nsq_per_block] *= theta_fac;
                        rw[sq_id+(irys*2+1)*nsq_per_block] *= sqrt_theta_fac;
                    }
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
                            double xixj = rirj[_ix];
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
                            double xkxl = rkrl[_ix];
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
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
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

                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;
                        int _ji = _j*nao+_i;
                        int _lk = _l*nao+_k;
                        double dd;
                        if (jk.n_dm == 1) {
                            dd = dm[_ji] * dm[_lk];
                        } else {
                            int nao2 = nao * nao;
                            dd = (dm[_ji] + dm[nao2+_ji]) * (dm[_lk] + dm[nao2+_lk]);
                        }

                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double prod_yz = gy[addry] * gz[addrz] * dd;
                        double prod_xz = gx[addrx] * gz[addrz] * dd;
                        double prod_xy = gx[addrx] * gy[addry] * dd;
                        double Ix = gx[addrx] * dd;
                        double Iy = gy[addry] * dd;
                        double Iz = gz[addrz] * dd;

                        double gix, giy, giz;
                        double gjx, gjy, gjz;
                        double gkx, gky, gkz;
                        double glx, gly, glz;
                        double gikx, giky, gikz;
                        double gjkx, gjky, gjkz;
                        double gilx, gily, gilz;
                        double gjlx, gjly, gjlz;
                        gix = ai2 * gx[addrx+g_stride_i];
                        giy = ai2 * gy[addry+g_stride_i];
                        giz = ai2 * gz[addrz+g_stride_i];
                        if (ix > 0) { gix -= ix * gx[addrx-g_stride_i]; }
                        if (iy > 0) { giy -= iy * gy[addry-g_stride_i]; }
                        if (iz > 0) { giz -= iz * gz[addrz-g_stride_i]; }

                        gjx = aj2 * gx[addrx+g_stride_j];
                        gjy = aj2 * gy[addry+g_stride_j];
                        gjz = aj2 * gz[addrz+g_stride_j];
                        if (jx > 0) { gjx -= jx * gx[addrx-g_stride_j]; }
                        if (jy > 0) { gjy -= jy * gy[addry-g_stride_j]; }
                        if (jz > 0) { gjz -= jz * gz[addrz-g_stride_j]; }

                        gkx = ak2 * gx[addrx+g_stride_k];
                        gky = ak2 * gy[addry+g_stride_k];
                        gkz = ak2 * gz[addrz+g_stride_k];
                        if (kx > 0) { gkx -= kx * gx[addrx-g_stride_k]; }
                        if (ky > 0) { gky -= ky * gy[addry-g_stride_k]; }
                        if (kz > 0) { gkz -= kz * gz[addrz-g_stride_k]; }

                        glx = al2 * gx[addrx+g_stride_l];
                        gly = al2 * gy[addry+g_stride_l];
                        glz = al2 * gz[addrz+g_stride_l];
                        if (lx > 0) { glx -= lx * gx[addrx-g_stride_l]; }
                        if (ly > 0) { gly -= ly * gy[addry-g_stride_l]; }
                        if (lz > 0) { glz -= lz * gz[addrz-g_stride_l]; }

                        gikx = ai2 * gx[addrx+g_stride_i+g_stride_k];
                        giky = ai2 * gy[addry+g_stride_i+g_stride_k];
                        gikz = ai2 * gz[addrz+g_stride_i+g_stride_k];
                        if (ix > 0) { gikx -= ix * gx[addrx-g_stride_i+g_stride_k]; }
                        if (iy > 0) { giky -= iy * gy[addry-g_stride_i+g_stride_k]; }
                        if (iz > 0) { gikz -= iz * gz[addrz-g_stride_i+g_stride_k]; }
                        gikx *= ak2;
                        giky *= ak2;
                        gikz *= ak2;

                        gjkx = aj2 * gx[addrx+g_stride_j+g_stride_k];
                        gjky = aj2 * gy[addry+g_stride_j+g_stride_k];
                        gjkz = aj2 * gz[addrz+g_stride_j+g_stride_k];
                        if (jx > 0) { gjkx -= jx * gx[addrx-g_stride_j+g_stride_k]; }
                        if (jy > 0) { gjky -= jy * gy[addry-g_stride_j+g_stride_k]; }
                        if (jz > 0) { gjkz -= jz * gz[addrz-g_stride_j+g_stride_k]; }
                        gjkx *= ak2;
                        gjky *= ak2;
                        gjkz *= ak2;

                        if (kx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_k];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_k]; }
                            gikx -= kx * fx;
                            fx = aj2 * gx[addrx+g_stride_j-g_stride_k];
                            if (jx > 0) { fx -= jx * gx[addrx-g_stride_j-g_stride_k]; }
                            gjkx -= kx * fx;
                        }
                        if (ky > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_k];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_k]; }
                            giky -= ky * fy;
                            fy = aj2 * gy[addry+g_stride_j-g_stride_k];
                            if (jy > 0) { fy -= jy * gy[addry-g_stride_j-g_stride_k]; }
                            gjky -= ky * fy;
                        }
                        if (kz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_k];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_k]; }
                            gikz -= kz * fz;
                            fz = aj2 * gz[addrz+g_stride_j-g_stride_k];
                            if (jz > 0) { fz -= jz * gz[addrz-g_stride_j-g_stride_k]; }
                            gjkz -= kz * fz;
                        }
                        vj_ik_xx += gikx * prod_yz;
                        vj_ik_yy += giky * prod_xz;
                        vj_ik_zz += gikz * prod_xy;
                        vj_ik_xy += gix * gky * Iz;
                        vj_ik_xz += gix * gkz * Iy;
                        vj_ik_yx += giy * gkx * Iz;
                        vj_ik_yz += giy * gkz * Ix;
                        vj_ik_zx += giz * gkx * Iy;
                        vj_ik_zy += giz * gky * Ix;
                        vj_jk_xx += gjkx * prod_yz;
                        vj_jk_yy += gjky * prod_xz;
                        vj_jk_zz += gjkz * prod_xy;
                        vj_jk_xy += gjx * gky * Iz;
                        vj_jk_xz += gjx * gkz * Iy;
                        vj_jk_yx += gjy * gkx * Iz;
                        vj_jk_yz += gjy * gkz * Ix;
                        vj_jk_zx += gjz * gkx * Iy;
                        vj_jk_zy += gjz * gky * Ix;

                        gilx = ai2 * gx[addrx+g_stride_i+g_stride_l];
                        gily = ai2 * gy[addry+g_stride_i+g_stride_l];
                        gilz = ai2 * gz[addrz+g_stride_i+g_stride_l];
                        if (ix > 0) { gilx -= ix * gx[addrx-g_stride_i+g_stride_l]; }
                        if (iy > 0) { gily -= iy * gy[addry-g_stride_i+g_stride_l]; }
                        if (iz > 0) { gilz -= iz * gz[addrz-g_stride_i+g_stride_l]; }
                        gilx *= al2;
                        gily *= al2;
                        gilz *= al2;

                        gjlx = aj2 * gx[addrx+g_stride_j+g_stride_l];
                        gjly = aj2 * gy[addry+g_stride_j+g_stride_l];
                        gjlz = aj2 * gz[addrz+g_stride_j+g_stride_l];
                        if (jx > 0) { gjlx -= jx * gx[addrx-g_stride_j+g_stride_l]; }
                        if (jy > 0) { gjly -= jy * gy[addry-g_stride_j+g_stride_l]; }
                        if (jz > 0) { gjlz -= jz * gz[addrz-g_stride_j+g_stride_l]; }
                        gjlx *= al2;
                        gjly *= al2;
                        gjlz *= al2;

                        if (lx > 0) {
                            double fx = ai2 * gx[addrx+g_stride_i-g_stride_l];
                            if (ix > 0) { fx -= ix * gx[addrx-g_stride_i-g_stride_l]; }
                            gilx -= lx * fx;
                            fx = aj2 * gx[addrx+g_stride_j-g_stride_l];
                            if (jx > 0) { fx -= jx * gx[addrx-g_stride_j-g_stride_l]; }
                            gjlx -= lx * fx;
                        }
                        if (ly > 0) {
                            double fy = ai2 * gy[addry+g_stride_i-g_stride_l];
                            if (iy > 0) { fy -= iy * gy[addry-g_stride_i-g_stride_l]; }
                            gily -= ly * fy;
                            fy = aj2 * gy[addry+g_stride_j-g_stride_l];
                            if (jy > 0) { fy -= jy * gy[addry-g_stride_j-g_stride_l]; }
                            gjly -= ly * fy;
                        }
                        if (lz > 0) {
                            double fz = ai2 * gz[addrz+g_stride_i-g_stride_l];
                            if (iz > 0) { fz -= iz * gz[addrz-g_stride_i-g_stride_l]; }
                            gilz -= lz * fz;
                            fz = aj2 * gz[addrz+g_stride_j-g_stride_l];
                            if (jz > 0) { fz -= jz * gz[addrz-g_stride_j-g_stride_l]; }
                            gjlz -= lz * fz;
                        }
                        vj_il_xx += gilx * prod_yz;
                        vj_il_yy += gily * prod_xz;
                        vj_il_zz += gilz * prod_xy;
                        vj_il_xy += gix * gly * Iz;
                        vj_il_xz += gix * glz * Iy;
                        vj_il_yx += giy * glx * Iz;
                        vj_il_yz += giy * glz * Ix;
                        vj_il_zx += giz * glx * Iy;
                        vj_il_zy += giz * gly * Ix;
                        vj_jl_xx += gjlx * prod_yz;
                        vj_jl_yy += gjly * prod_xz;
                        vj_jl_zz += gjlz * prod_xy;
                        vj_jl_xy += gjx * gly * Iz;
                        vj_jl_xz += gjx * glz * Iy;
                        vj_jl_yx += gjy * glx * Iz;
                        vj_jl_yz += gjy * glz * Ix;
                        vj_jl_zx += gjz * glx * Iy;
                        vj_jl_zy += gjz * gly * Ix;
                    }
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        int natm = envs.natm;
        atomicAdd(vj + (ia*natm+ka)*9 + 0, vj_ik_xx);
        atomicAdd(vj + (ia*natm+ka)*9 + 1, vj_ik_xy);
        atomicAdd(vj + (ia*natm+ka)*9 + 2, vj_ik_xz);
        atomicAdd(vj + (ia*natm+ka)*9 + 3, vj_ik_yx);
        atomicAdd(vj + (ia*natm+ka)*9 + 4, vj_ik_yy);
        atomicAdd(vj + (ia*natm+ka)*9 + 5, vj_ik_yz);
        atomicAdd(vj + (ia*natm+ka)*9 + 6, vj_ik_zx);
        atomicAdd(vj + (ia*natm+ka)*9 + 7, vj_ik_zy);
        atomicAdd(vj + (ia*natm+ka)*9 + 8, vj_ik_zz);
        atomicAdd(vj + (ja*natm+ka)*9 + 0, vj_jk_xx);
        atomicAdd(vj + (ja*natm+ka)*9 + 1, vj_jk_xy);
        atomicAdd(vj + (ja*natm+ka)*9 + 2, vj_jk_xz);
        atomicAdd(vj + (ja*natm+ka)*9 + 3, vj_jk_yx);
        atomicAdd(vj + (ja*natm+ka)*9 + 4, vj_jk_yy);
        atomicAdd(vj + (ja*natm+ka)*9 + 5, vj_jk_yz);
        atomicAdd(vj + (ja*natm+ka)*9 + 6, vj_jk_zx);
        atomicAdd(vj + (ja*natm+ka)*9 + 7, vj_jk_zy);
        atomicAdd(vj + (ja*natm+ka)*9 + 8, vj_jk_zz);
        atomicAdd(vj + (ia*natm+la)*9 + 0, vj_il_xx);
        atomicAdd(vj + (ia*natm+la)*9 + 1, vj_il_xy);
        atomicAdd(vj + (ia*natm+la)*9 + 2, vj_il_xz);
        atomicAdd(vj + (ia*natm+la)*9 + 3, vj_il_yx);
        atomicAdd(vj + (ia*natm+la)*9 + 4, vj_il_yy);
        atomicAdd(vj + (ia*natm+la)*9 + 5, vj_il_yz);
        atomicAdd(vj + (ia*natm+la)*9 + 6, vj_il_zx);
        atomicAdd(vj + (ia*natm+la)*9 + 7, vj_il_zy);
        atomicAdd(vj + (ia*natm+la)*9 + 8, vj_il_zz);
        atomicAdd(vj + (ja*natm+la)*9 + 0, vj_jl_xx);
        atomicAdd(vj + (ja*natm+la)*9 + 1, vj_jl_xy);
        atomicAdd(vj + (ja*natm+la)*9 + 2, vj_jl_xz);
        atomicAdd(vj + (ja*natm+la)*9 + 3, vj_jl_yx);
        atomicAdd(vj + (ja*natm+la)*9 + 4, vj_jl_yy);
        atomicAdd(vj + (ja*natm+la)*9 + 5, vj_jl_yz);
        atomicAdd(vj + (ja*natm+la)*9 + 6, vj_jl_zx);
        atomicAdd(vj + (ja*natm+la)*9 + 7, vj_jl_zy);
        atomicAdd(vj + (ja*natm+la)*9 + 8, vj_jl_zz);
    }
}

__global__
void rys_ejk_ip2_type1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_ip2_type2_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_ejk_ip2_type1_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__global__
void rys_ejk_ip2_type2_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_ip2_type2_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_ejk_ip2_type2_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__global__
void rys_ejk_ip2_type12_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_ip2_type2_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_ejk_ip2_type12_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__global__
void rys_ejk_ip2_type3_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
        int ntasks = _fill_ejk_ip2_type3_tasks(shl_quartet_idx, envs, jk, bounds,
                                     batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_ejk_ip2_type3_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__global__
void rys_ej_ip2_type3_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            rys_ej_ip2_type3_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
