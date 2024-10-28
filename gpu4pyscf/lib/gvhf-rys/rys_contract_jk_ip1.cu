#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks_ip1.cu"
#include "create_tasks.cu"

#define GWIDTH_IP1 18

__device__
static void rys_jk_ip1_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                               ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int t_id = sq_id + gout_id * nsq_per_block;
    int threads = nsq_per_block * gout_stride;
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
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
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
    double omega = env[PTR_RANGE_OMEGA];
    extern __shared__ double rw[];
    double *g = rw + nsq_per_block * nroots*2;
    double *Rpa_cicj = g + nsq_per_block * g_size*3;
    double Rqc[3];
    double goutx[GWIDTH_IP1];
    double gouty[GWIDTH_IP1];
    double goutz[GWIDTH_IP1];

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
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = expi[ip];
        double aj = expj[jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double xjxi = rj[0] - ri[0];
        double yjyi = rj[1] - ri[1];
        double zjzi = rj[2] - ri[2];
        double *Rpa = Rpa_cicj + ij*12;
        double xpa = xjxi * aj_aij;
        double ypa = yjyi * aj_aij;
        double zpa = zjzi * aj_aij;
        Rpa[0] = xpa;
        Rpa[1] = ypa;
        Rpa[2] = zpa;
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xjxi*xjxi+yjyi*yjyi+zjzi*zjzi));
        Rpa[3] = ci[ip] * cj[jp] * Kab;
        Rpa[4] = ri[0] + xpa;
        Rpa[5] = ri[1] + ypa;
        Rpa[6] = ri[2] + zpa;
        Rpa[7] = -xjxi;
        Rpa[8] = -yjyi;
        Rpa[9] = -zjzi;
        Rpa[10] = aij;
        Rpa[11] = ai * 2;
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
        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GWIDTH_IP1) {
#pragma unroll
        for (int n = 0; n < GWIDTH_IP1; ++n) { goutx[n] = 0; gouty[n] = 0; goutz[n] = 0; }

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
            Rqc[0] = xlxk * al_akl; // (ak*xk+al*xl)/akl
            Rqc[1] = ylyk * al_akl;
            Rqc[2] = zlzk * al_akl;
            __syncthreads();
            if (gout_id == 0) {
                double theta_kl = ak * al / akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * ck[kp] * cl[lp] * Kcd;
                g[sq_id] = ckcl;
            }
            int ijprim = iprim * jprim;
            for (int ijp = 0; ijp < ijprim; ++ijp) {
                double *Rpa = Rpa_cicj + ijp*12;
                double *rij = Rpa + 4;
                double *rirj = Rpa + 7;
                double xij = rij[0];
                double yij = rij[1];
                double zij = rij[2];
                double xkl = rk[0] + Rqc[0];
                double ykl = rk[1] + Rqc[1];
                double zkl = rk[2] + Rqc[2];
                double xpq = xij - xkl;
                double ypq = yij - ykl;
                double zpq = zij - zkl;
                __syncthreads();
                double aij = Rpa[10];
                if (gout_id == 0) {
                    double cicj = Rpa[3];
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
                    for (int irys = 0; irys < nroots; ++irys) {
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
                    double aij = Rpa[10];
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
                        double xij = rij[n];
                        double xkl = rk[n] + Rqc[n];
                        double xpq = xij - xkl;
                        double c0x = Rpa[n] - rt_aij * xpq;
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
                            double xij = rij[_ix];
                            double xkl = rk[_ix] + Rqc[_ix];
                            double xpq = xij - xkl;
                            double cpx = Rqc[_ix] + rt_akl * xpq;
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
                                double xkxl = rk[_ix] - rl[_ix];
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
#pragma unroll
                    for (int n = 0; n < GWIDTH_IP1; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
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
                        double ai2 = Rpa[11];
                        double fx = ai2 * gx[addrx+nsq_per_block];
                        double fy = ai2 * gy[addry+nsq_per_block];
                        double fz = ai2 * gz[addrz+nsq_per_block];
                        if (ix > 0) { fx -= ix * gx[addrx-nsq_per_block]; }
                        if (iy > 0) { fy -= iy * gy[addry-nsq_per_block]; }
                        if (iz > 0) { fz -= iz * gz[addrz-nsq_per_block]; }
                        goutx[n] += fx * gy[addry] * gz[addrz];
                        gouty[n] += fy * gx[addrx] * gz[addrz];
                        goutz[n] += fz * gx[addrx] * gy[addry];
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
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
#pragma unroll
        for (int n = 0; n < GWIDTH_IP1; ++n) {
            int ijkl = (gout_start + n*gout_stride+gout_id);
            int kl = ijkl / nfij;
            int ij = ijkl % nfij;
            if (kl >= nfkl) break;
            double sx = goutx[n];
            double sy = gouty[n];
            double sz = goutz[n];
            int i = ij % nfi;
            int j = ij / nfi;
            int k = kl % nfk;
            int l = kl / nfk;
            int _i = i + i0;
            int _j = j + j0;
            int _k = k + k0;
            int _l = l + l0;
            if (do_j) {
                int _ji = _j*nao+_i;
                int _lk = _l*nao+_k;
                atomicAdd(vj_x+_lk, sx * dm[_ji]);
                atomicAdd(vj_y+_lk, sy * dm[_ji]);
                atomicAdd(vj_z+_lk, sz * dm[_ji]);
                atomicAdd(vj_x+_ji, sx * dm[_lk]);
                atomicAdd(vj_y+_ji, sy * dm[_lk]);
                atomicAdd(vj_z+_ji, sz * dm[_lk]);
            }
            if (do_k) {
                int _jl = _j*nao+_l;
                int _jk = _j*nao+_k;
                int _il = _i*nao+_l;
                int _ik = _i*nao+_k;
                atomicAdd(vk_x+_ik, sx * dm[_jl]);
                atomicAdd(vk_y+_ik, sy * dm[_jl]);
                atomicAdd(vk_z+_ik, sz * dm[_jl]);
                atomicAdd(vk_x+_il, sx * dm[_jk]);
                atomicAdd(vk_y+_il, sy * dm[_jk]);
                atomicAdd(vk_z+_il, sz * dm[_jk]);
                atomicAdd(vk_x+_jk, sx * dm[_il]);
                atomicAdd(vk_y+_jk, sy * dm[_il]);
                atomicAdd(vk_z+_jk, sz * dm[_il]);
                atomicAdd(vk_x+_jl, sx * dm[_ik]);
                atomicAdd(vk_y+_jl, sy * dm[_ik]);
                atomicAdd(vk_z+_jl, sz * dm[_ik]);
            }
        }
    } }
}

__global__
void rys_jk_ip1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                       ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH1;
    __shared__ int batch_id;
    if (t_id == 0) {
        batch_id = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.npairs_kl + QUEUE_DEPTH1 - 1) / QUEUE_DEPTH1;
    int nbatches = bounds.npairs_ij * nbatches_kl;
    while (batch_id < nbatches) {
        int batch_ij = batch_id / nbatches_kl;
        int batch_kl = batch_id % nbatches_kl;
        int ntasks = _fill_jk_tasks_s2kl(shl_quartet_idx, envs, jk, bounds,
                                         batch_ij, batch_kl);
        if (ntasks > 0) {
            rys_jk_ip1_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

__device__
static void rys_ejk_ip1_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
    int nfj = nfij/nfi;
    int nfl = nfkl/nfk;
    int *idx_i = c_g_pair_idx + c_g_pair_offsets[li*LMAX1];
    int *idy_i = idx_i + nfi;
    int *idz_i = idy_i + nfi;
    int *idx_j = c_g_pair_idx + c_g_pair_offsets[lj*LMAX1];
    int *idy_j = idx_j + nfj;
    int *idz_j = idy_j + nfj;
    int *idx_k = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1];
    int *idy_k = idx_k + nfk;
    int *idz_k = idy_k + nfk;
    int *idx_l = c_g_pair_idx + c_g_pair_offsets[ll*LMAX1];
    int *idy_l = idx_l + nfl;
    int *idz_l = idy_l + nfl;
    int i_1 = nsq_per_block;
    int j_1 = stride_j*nsq_per_block;
    int k_1 = stride_k*nsq_per_block;
    int l_1 = stride_l*nsq_per_block;
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
        double vj_ix = 0;
        double vj_iy = 0;
        double vj_iz = 0;
        double vj_jx = 0;
        double vj_jy = 0;
        double vj_jz = 0;
        double vj_kx = 0;
        double vj_ky = 0;
        double vj_kz = 0;
        double vj_lx = 0;
        double vj_ly = 0;
        double vj_lz = 0;
        double vk_ix = 0;
        double vk_iy = 0;
        double vk_iz = 0;
        double vk_jx = 0;
        double vk_jy = 0;
        double vk_jz = 0;
        double vk_kx = 0;
        double vk_ky = 0;
        double vk_kz = 0;
        double vk_lx = 0;
        double vk_ly = 0;
        double vk_lz = 0;
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
                    for (int n = gout_id; n < nfij*nfkl; n+=gout_stride) {
                        int kl = n / nfij;
                        int ij = n % nfij;
                        if (kl >= nfkl) break;
                        int i = ij % nfi;
                        int j = ij / nfi;
                        int k = kl % nfk;
                        int l = kl / nfk;
                        int ix = idx_i[i];
                        int iy = idy_i[i];
                        int iz = idz_i[i];
                        int jx = idx_j[j];
                        int jy = idy_j[j];
                        int jz = idz_j[j];
                        int kx = idx_k[k];
                        int ky = idy_k[k];
                        int kz = idz_k[k];
                        int lx = idx_l[l];
                        int ly = idy_l[l];
                        int lz = idz_l[l];
                        int _i = i + i0;
                        int _j = j + j0;
                        int _k = k + k0;
                        int _l = l + l0;
                        int _jl = _j*nao+_l;
                        int _jk = _j*nao+_k;
                        int _il = _i*nao+_l;
                        int _ik = _i*nao+_k;
                        int _ji = _j*nao+_i;
                        int _lk = _l*nao+_k;
                        double dd_jk = dm[_jk] * dm[_il];
                        double dd_jl = dm[_jl] * dm[_ik];
                        double dd_k = dd_jk + dd_jl;
                        double dd_j = dm[_ji] * dm[_lk];
                        if (jk.n_dm > 1) {
                            int nao2 = nao * nao;
                            double dd_jk = dm[nao2+_jk] * dm[nao2+_il];
                            double dd_jl = dm[nao2+_jl] * dm[nao2+_ik];
                            dd_k += dd_jk + dd_jl;
                            dd_j = (dm[_ji] + dm[nao2+_ji]) * (dm[_lk] + dm[nao2+_lk]);
                        }
                        int addrx = sq_id + (ix + jx*stride_j + kx*stride_k + lx*stride_l) * nsq_per_block;
                        int addry = sq_id + (iy + jy*stride_j + ky*stride_k + ly*stride_l) * nsq_per_block;
                        int addrz = sq_id + (iz + jz*stride_j + kz*stride_k + lz*stride_l) * nsq_per_block;
                        double prod_xy = gx[addrx] * gy[addry];
                        double prod_xz = gx[addrx] * gz[addrz];
                        double prod_yz = gy[addry] * gz[addrz];
                        double fix = ai2 * gx[addrx+i_1]; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } fix *= prod_yz; vk_ix += fix * dd_k; vj_ix += fix * dd_j;
                        double fiy = ai2 * gy[addry+i_1]; if (iy > 0) { fiy -= iy * gy[addry-i_1]; } fiy *= prod_xz; vk_iy += fiy * dd_k; vj_iy += fiy * dd_j;
                        double fiz = ai2 * gz[addrz+i_1]; if (iz > 0) { fiz -= iz * gz[addrz-i_1]; } fiz *= prod_xy; vk_iz += fiz * dd_k; vj_iz += fiz * dd_j;
                        double fjx = aj2 * gx[addrx+j_1]; if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } fjx *= prod_yz; vk_jx += fjx * dd_k; vj_jx += fjx * dd_j;
                        double fjy = aj2 * gy[addry+j_1]; if (jy > 0) { fjy -= jy * gy[addry-j_1]; } fjy *= prod_xz; vk_jy += fjy * dd_k; vj_jy += fjy * dd_j;
                        double fjz = aj2 * gz[addrz+j_1]; if (jz > 0) { fjz -= jz * gz[addrz-j_1]; } fjz *= prod_xy; vk_jz += fjz * dd_k; vj_jz += fjz * dd_j;
                        double fkx = ak2 * gx[addrx+k_1]; if (kx > 0) { fkx -= kx * gx[addrx-k_1]; } fkx *= prod_yz; vk_kx += fkx * dd_k; vj_kx += fkx * dd_j;
                        double fky = ak2 * gy[addry+k_1]; if (ky > 0) { fky -= ky * gy[addry-k_1]; } fky *= prod_xz; vk_ky += fky * dd_k; vj_ky += fky * dd_j;
                        double fkz = ak2 * gz[addrz+k_1]; if (kz > 0) { fkz -= kz * gz[addrz-k_1]; } fkz *= prod_xy; vk_kz += fkz * dd_k; vj_kz += fkz * dd_j;
                        double flx = al2 * gx[addrx+l_1]; if (lx > 0) { flx -= lx * gx[addrx-l_1]; } flx *= prod_yz; vk_lx += flx * dd_k; vj_lx += flx * dd_j;
                        double fly = al2 * gy[addry+l_1]; if (ly > 0) { fly -= ly * gy[addry-l_1]; } fly *= prod_xz; vk_ly += fly * dd_k; vj_ly += fly * dd_j;
                        double flz = al2 * gz[addrz+l_1]; if (lz > 0) { flz -= lz * gz[addrz-l_1]; } flz *= prod_xy; vk_lz += flz * dd_k; vj_lz += flz * dd_j;
                    }
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        int t_id = sq_id + gout_id * nsq_per_block;
        int threads = nsq_per_block * gout_stride;
        if (vj != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vj_ix;
            reduce[t_id+1 *threads] = vj_iy;
            reduce[t_id+2 *threads] = vj_iz;
            reduce[t_id+3 *threads] = vj_jx;
            reduce[t_id+4 *threads] = vj_jy;
            reduce[t_id+5 *threads] = vj_jz;
            reduce[t_id+6 *threads] = vj_kx;
            reduce[t_id+7 *threads] = vj_ky;
            reduce[t_id+8 *threads] = vj_kz;
            reduce[t_id+9 *threads] = vj_lx;
            reduce[t_id+10*threads] = vj_ly;
            reduce[t_id+11*threads] = vj_lz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 12; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0) {
                atomicAdd(vj+ia*3+0, reduce[sq_id+0 *threads]);
                atomicAdd(vj+ia*3+1, reduce[sq_id+1 *threads]);
                atomicAdd(vj+ia*3+2, reduce[sq_id+2 *threads]);
                atomicAdd(vj+ja*3+0, reduce[sq_id+3 *threads]);
                atomicAdd(vj+ja*3+1, reduce[sq_id+4 *threads]);
                atomicAdd(vj+ja*3+2, reduce[sq_id+5 *threads]);
                atomicAdd(vj+ka*3+0, reduce[sq_id+6 *threads]);
                atomicAdd(vj+ka*3+1, reduce[sq_id+7 *threads]);
                atomicAdd(vj+ka*3+2, reduce[sq_id+8 *threads]);
                atomicAdd(vj+la*3+0, reduce[sq_id+9 *threads]);
                atomicAdd(vj+la*3+1, reduce[sq_id+10*threads]);
                atomicAdd(vj+la*3+2, reduce[sq_id+11*threads]);
            }
        }
        if (vk != NULL) {
            __syncthreads();
            double *reduce = rw;
            reduce[t_id+0 *threads] = vk_ix;
            reduce[t_id+1 *threads] = vk_iy;
            reduce[t_id+2 *threads] = vk_iz;
            reduce[t_id+3 *threads] = vk_jx;
            reduce[t_id+4 *threads] = vk_jy;
            reduce[t_id+5 *threads] = vk_jz;
            reduce[t_id+6 *threads] = vk_kx;
            reduce[t_id+7 *threads] = vk_ky;
            reduce[t_id+8 *threads] = vk_kz;
            reduce[t_id+9 *threads] = vk_lx;
            reduce[t_id+10*threads] = vk_ly;
            reduce[t_id+11*threads] = vk_lz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
#pragma unroll
                    for (int n = 0; n < 12; ++n) {
                        reduce[n*threads + t_id] += reduce[n*threads + t_id +i*nsq_per_block];
                    }
                }
            }
            if (gout_id == 0) {
                atomicAdd(vk+ia*3+0, reduce[sq_id+0 *threads]);
                atomicAdd(vk+ia*3+1, reduce[sq_id+1 *threads]);
                atomicAdd(vk+ia*3+2, reduce[sq_id+2 *threads]);
                atomicAdd(vk+ja*3+0, reduce[sq_id+3 *threads]);
                atomicAdd(vk+ja*3+1, reduce[sq_id+4 *threads]);
                atomicAdd(vk+ja*3+2, reduce[sq_id+5 *threads]);
                atomicAdd(vk+ka*3+0, reduce[sq_id+6 *threads]);
                atomicAdd(vk+ka*3+1, reduce[sq_id+7 *threads]);
                atomicAdd(vk+ka*3+2, reduce[sq_id+8 *threads]);
                atomicAdd(vk+la*3+0, reduce[sq_id+9 *threads]);
                atomicAdd(vk+la*3+1, reduce[sq_id+10*threads]);
                atomicAdd(vk+la*3+2, reduce[sq_id+11*threads]);
            }
        }
    }
}

__global__
void rys_ejk_ip1_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
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
            rys_ejk_ip1_general(envs, jk, bounds, shl_quartet_idx, ntasks);
        }
        if (t_id == 0) {
            batch_id = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}
