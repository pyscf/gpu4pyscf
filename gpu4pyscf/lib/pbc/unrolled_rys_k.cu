#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "create_tasks.cu"


__global__ static
void rys_k_0000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 4;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[1];
#pragma unroll
            for (int n = 0; n < 1; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(2, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        gout[0] += 1 * fac * wt;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        double vjl_00 = gout[0]*dm_ik_00;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 4;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[3];
#pragma unroll
            for (int n = 0; n < 3; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(2, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 2; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        gout[0] += trr_10x * fac * wt;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += 1 * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += 1 * fac * trr_10z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1010(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[9];
#pragma unroll
            for (int n = 0; n < 9; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        gout[0] += trr_11x * fac * wt;
                        double trr_01x = cpx * 1;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_01x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_01x * fac * trr_10z;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[3] += trr_10x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[4] += 1 * trr_11y * wt;
                        gout[5] += 1 * trr_01y * trr_10z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[6] += trr_10x * fac * trr_01z;
                        gout[7] += 1 * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[8] += 1 * fac * trr_11z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm_jk[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm_jk[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_01 + gout[6]*dm_jk_02;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_01 + gout[7]*dm_jk_02;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_01 + gout[8]*dm_jk_02;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[3]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[6]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[4]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[7]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[5]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[8]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+2), vik_22);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm_ik[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[3] * dm_ik_01;
                        double dm_ik_02 = dm_ik[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[6] * dm_ik_02;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm_ik[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[4] * dm_ik_11;
                        double dm_ik_12 = dm_ik[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[7] * dm_ik_12;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm_ik[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[5] * dm_ik_21;
                        double dm_ik_22 = dm_ik[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[8] * dm_ik_22;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1011(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[27];
#pragma unroll
            for (int n = 0; n < 27; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        gout[0] += hrr_1011x * fac * wt;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_0011x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_0011x * fac * trr_10z;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[3] += hrr_1001x * trr_01y * wt;
                        double hrr_0001x = trr_01x - xlxk * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[4] += hrr_0001x * trr_11y * wt;
                        gout[5] += hrr_0001x * trr_01y * trr_10z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[6] += hrr_1001x * fac * trr_01z;
                        gout[7] += hrr_0001x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[8] += hrr_0001x * fac * trr_11z;
                        double hrr_0001y = trr_01y - ylyk * fac;
                        gout[9] += trr_11x * hrr_0001y * wt;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gout[10] += trr_01x * hrr_1001y * wt;
                        gout[11] += trr_01x * hrr_0001y * trr_10z;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gout[12] += trr_10x * hrr_0011y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gout[13] += 1 * hrr_1011y * wt;
                        gout[14] += 1 * hrr_0011y * trr_10z;
                        gout[15] += trr_10x * hrr_0001y * trr_01z;
                        gout[16] += 1 * hrr_1001y * trr_01z;
                        gout[17] += 1 * hrr_0001y * trr_11z;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        gout[18] += trr_11x * fac * hrr_0001z;
                        gout[19] += trr_01x * trr_10y * hrr_0001z;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gout[20] += trr_01x * fac * hrr_1001z;
                        gout[21] += trr_10x * trr_01y * hrr_0001z;
                        gout[22] += 1 * trr_11y * hrr_0001z;
                        gout[23] += 1 * trr_01y * hrr_1001z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gout[24] += trr_10x * fac * hrr_0011z;
                        gout[25] += 1 * trr_10y * hrr_0011z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gout[26] += 1 * fac * hrr_1011z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm_jk[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm_jk[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_01 + gout[6]*dm_jk_02;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_01 = gout[9]*dm_jk_00 + gout[12]*dm_jk_01 + gout[15]*dm_jk_02;
                    atomicAdd(v_il+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[18]*dm_jk_00 + gout[21]*dm_jk_01 + gout[24]*dm_jk_02;
                    atomicAdd(v_il+(i0+0)*nao+(l0+2), vil_02);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_01 + gout[7]*dm_jk_02;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_11 = gout[10]*dm_jk_00 + gout[13]*dm_jk_01 + gout[16]*dm_jk_02;
                    atomicAdd(v_il+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[19]*dm_jk_00 + gout[22]*dm_jk_01 + gout[25]*dm_jk_02;
                    atomicAdd(v_il+(i0+1)*nao+(l0+2), vil_12);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_01 + gout[8]*dm_jk_02;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[11]*dm_jk_00 + gout[14]*dm_jk_01 + gout[17]*dm_jk_02;
                    atomicAdd(v_il+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[20]*dm_jk_00 + gout[23]*dm_jk_01 + gout[26]*dm_jk_02;
                    atomicAdd(v_il+(i0+2)*nao+(l0+2), vil_22);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm_jl[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm_jl[(j0+0)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[9]*dm_jl_01 + gout[18]*dm_jl_02;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[3]*dm_jl_00 + gout[12]*dm_jl_01 + gout[21]*dm_jl_02;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[6]*dm_jl_00 + gout[15]*dm_jl_01 + gout[24]*dm_jl_02;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[10]*dm_jl_01 + gout[19]*dm_jl_02;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[4]*dm_jl_00 + gout[13]*dm_jl_01 + gout[22]*dm_jl_02;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[7]*dm_jl_00 + gout[16]*dm_jl_01 + gout[25]*dm_jl_02;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[11]*dm_jl_01 + gout[20]*dm_jl_02;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[5]*dm_jl_00 + gout[14]*dm_jl_01 + gout[23]*dm_jl_02;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[8]*dm_jl_00 + gout[17]*dm_jl_01 + gout[26]*dm_jl_02;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+2), vik_22);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double vjl_01 = 0;
                        double vjl_02 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_01 += gout[9] * dm_ik_00;
                        vjl_02 += gout[18] * dm_ik_00;
                        double dm_ik_01 = dm_ik[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[3] * dm_ik_01;
                        vjl_01 += gout[12] * dm_ik_01;
                        vjl_02 += gout[21] * dm_ik_01;
                        double dm_ik_02 = dm_ik[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[6] * dm_ik_02;
                        vjl_01 += gout[15] * dm_ik_02;
                        vjl_02 += gout[24] * dm_ik_02;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_01 += gout[10] * dm_ik_10;
                        vjl_02 += gout[19] * dm_ik_10;
                        double dm_ik_11 = dm_ik[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[4] * dm_ik_11;
                        vjl_01 += gout[13] * dm_ik_11;
                        vjl_02 += gout[22] * dm_ik_11;
                        double dm_ik_12 = dm_ik[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[7] * dm_ik_12;
                        vjl_01 += gout[16] * dm_ik_12;
                        vjl_02 += gout[25] * dm_ik_12;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_01 += gout[11] * dm_ik_20;
                        vjl_02 += gout[20] * dm_ik_20;
                        double dm_ik_21 = dm_ik[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[5] * dm_ik_21;
                        vjl_01 += gout[14] * dm_ik_21;
                        vjl_02 += gout[23] * dm_ik_21;
                        double dm_ik_22 = dm_ik[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[8] * dm_ik_22;
                        vjl_01 += gout[17] * dm_ik_22;
                        vjl_02 += gout[26] * dm_ik_22;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(v_jl+(j0+0)*nao+(l0+1), vjl_01);
                        atomicAdd(v_jl+(j0+0)*nao+(l0+2), vjl_02);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[3] * dm_il_00;
                        vjk_02 += gout[6] * dm_il_00;
                        double dm_il_01 = dm_il[(i0+0)*nao+(l0+1)];
                        vjk_00 += gout[9] * dm_il_01;
                        vjk_01 += gout[12] * dm_il_01;
                        vjk_02 += gout[15] * dm_il_01;
                        double dm_il_02 = dm_il[(i0+0)*nao+(l0+2)];
                        vjk_00 += gout[18] * dm_il_02;
                        vjk_01 += gout[21] * dm_il_02;
                        vjk_02 += gout[24] * dm_il_02;
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[4] * dm_il_10;
                        vjk_02 += gout[7] * dm_il_10;
                        double dm_il_11 = dm_il[(i0+1)*nao+(l0+1)];
                        vjk_00 += gout[10] * dm_il_11;
                        vjk_01 += gout[13] * dm_il_11;
                        vjk_02 += gout[16] * dm_il_11;
                        double dm_il_12 = dm_il[(i0+1)*nao+(l0+2)];
                        vjk_00 += gout[19] * dm_il_12;
                        vjk_01 += gout[22] * dm_il_12;
                        vjk_02 += gout[25] * dm_il_12;
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[5] * dm_il_20;
                        vjk_02 += gout[8] * dm_il_20;
                        double dm_il_21 = dm_il[(i0+2)*nao+(l0+1)];
                        vjk_00 += gout[11] * dm_il_21;
                        vjk_01 += gout[14] * dm_il_21;
                        vjk_02 += gout[17] * dm_il_21;
                        double dm_il_22 = dm_il[(i0+2)*nao+(l0+2)];
                        vjk_00 += gout[20] * dm_il_22;
                        vjk_01 += gout[23] * dm_il_22;
                        vjk_02 += gout[26] * dm_il_22;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(v_jk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(v_jk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1100(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[9];
#pragma unroll
            for (int n = 0; n < 9; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        gout[0] += hrr_1100x * fac * wt;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_0100x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_0100x * fac * trr_10z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[3] += trr_10x * hrr_0100y * wt;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[4] += 1 * hrr_1100y * wt;
                        gout[5] += 1 * hrr_0100y * trr_10z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[6] += trr_10x * fac * hrr_0100z;
                        gout[7] += 1 * trr_10y * hrr_0100z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[8] += 1 * fac * hrr_1100z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm_jk[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm_jk[(j0+2)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_10 + gout[6]*dm_jk_20;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_10 + gout[7]*dm_jk_20;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_10 + gout[8]*dm_jk_20;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm_jl[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm_jl[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_10 + gout[6]*dm_jl_20;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[4]*dm_jl_10 + gout[7]*dm_jl_20;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[5]*dm_jl_10 + gout[8]*dm_jl_20;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        double vjl_00 = gout[0]*dm_ik_00 + gout[1]*dm_ik_10 + gout[2]*dm_ik_20;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        double vjl_10 = gout[3]*dm_ik_00 + gout[4]*dm_ik_10 + gout[5]*dm_ik_20;
                        atomicAdd(v_jl+(j0+1)*nao+(l0+0), vjl_10);
                        double vjl_20 = gout[6]*dm_ik_00 + gout[7]*dm_ik_10 + gout[8]*dm_ik_20;
                        atomicAdd(v_jl+(j0+2)*nao+(l0+0), vjl_20);
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_10 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20;
                        atomicAdd(v_jk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_20 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20;
                        atomicAdd(v_jk+(j0+2)*nao+(k0+0), vjk_20);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1110(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[27];
#pragma unroll
            for (int n = 0; n < 27; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        gout[0] += hrr_1110x * fac * wt;
                        double trr_01x = cpx * 1;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_0110x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_0110x * fac * trr_10z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[3] += trr_11x * hrr_0100y * wt;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[4] += trr_01x * hrr_1100y * wt;
                        gout[5] += trr_01x * hrr_0100y * trr_10z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[6] += trr_11x * fac * hrr_0100z;
                        gout[7] += trr_01x * trr_10y * hrr_0100z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[8] += trr_01x * fac * hrr_1100z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[9] += hrr_1100x * trr_01y * wt;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[10] += hrr_0100x * trr_11y * wt;
                        gout[11] += hrr_0100x * trr_01y * trr_10z;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gout[12] += trr_10x * hrr_0110y * wt;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gout[13] += 1 * hrr_1110y * wt;
                        gout[14] += 1 * hrr_0110y * trr_10z;
                        gout[15] += trr_10x * trr_01y * hrr_0100z;
                        gout[16] += 1 * trr_11y * hrr_0100z;
                        gout[17] += 1 * trr_01y * hrr_1100z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[18] += hrr_1100x * fac * trr_01z;
                        gout[19] += hrr_0100x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[20] += hrr_0100x * fac * trr_11z;
                        gout[21] += trr_10x * hrr_0100y * trr_01z;
                        gout[22] += 1 * hrr_1100y * trr_01z;
                        gout[23] += 1 * hrr_0100y * trr_11z;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gout[24] += trr_10x * fac * hrr_0110z;
                        gout[25] += 1 * trr_10y * hrr_0110z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gout[26] += 1 * fac * hrr_1110z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double vil_00 = 0;
                    double vil_10 = 0;
                    double vil_20 = 0;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    vil_00 += gout[0] * dm_jk_00;
                    vil_10 += gout[1] * dm_jk_00;
                    vil_20 += gout[2] * dm_jk_00;
                    double dm_jk_01 = dm_jk[(j0+0)*nao+(k0+1)];
                    vil_00 += gout[9] * dm_jk_01;
                    vil_10 += gout[10] * dm_jk_01;
                    vil_20 += gout[11] * dm_jk_01;
                    double dm_jk_02 = dm_jk[(j0+0)*nao+(k0+2)];
                    vil_00 += gout[18] * dm_jk_02;
                    vil_10 += gout[19] * dm_jk_02;
                    vil_20 += gout[20] * dm_jk_02;
                    double dm_jk_10 = dm_jk[(j0+1)*nao+(k0+0)];
                    vil_00 += gout[3] * dm_jk_10;
                    vil_10 += gout[4] * dm_jk_10;
                    vil_20 += gout[5] * dm_jk_10;
                    double dm_jk_11 = dm_jk[(j0+1)*nao+(k0+1)];
                    vil_00 += gout[12] * dm_jk_11;
                    vil_10 += gout[13] * dm_jk_11;
                    vil_20 += gout[14] * dm_jk_11;
                    double dm_jk_12 = dm_jk[(j0+1)*nao+(k0+2)];
                    vil_00 += gout[21] * dm_jk_12;
                    vil_10 += gout[22] * dm_jk_12;
                    vil_20 += gout[23] * dm_jk_12;
                    double dm_jk_20 = dm_jk[(j0+2)*nao+(k0+0)];
                    vil_00 += gout[6] * dm_jk_20;
                    vil_10 += gout[7] * dm_jk_20;
                    vil_20 += gout[8] * dm_jk_20;
                    double dm_jk_21 = dm_jk[(j0+2)*nao+(k0+1)];
                    vil_00 += gout[15] * dm_jk_21;
                    vil_10 += gout[16] * dm_jk_21;
                    vil_20 += gout[17] * dm_jk_21;
                    double dm_jk_22 = dm_jk[(j0+2)*nao+(k0+2)];
                    vil_00 += gout[24] * dm_jk_22;
                    vil_10 += gout[25] * dm_jk_22;
                    vil_20 += gout[26] * dm_jk_22;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm_jl[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm_jl[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_10 + gout[6]*dm_jl_20;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[9]*dm_jl_00 + gout[12]*dm_jl_10 + gout[15]*dm_jl_20;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[18]*dm_jl_00 + gout[21]*dm_jl_10 + gout[24]*dm_jl_20;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[4]*dm_jl_10 + gout[7]*dm_jl_20;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[10]*dm_jl_00 + gout[13]*dm_jl_10 + gout[16]*dm_jl_20;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[19]*dm_jl_00 + gout[22]*dm_jl_10 + gout[25]*dm_jl_20;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[5]*dm_jl_10 + gout[8]*dm_jl_20;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[11]*dm_jl_00 + gout[14]*dm_jl_10 + gout[17]*dm_jl_20;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[20]*dm_jl_00 + gout[23]*dm_jl_10 + gout[26]*dm_jl_20;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+2), vik_22);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[3] * dm_ik_00;
                        vjl_20 += gout[6] * dm_ik_00;
                        double dm_ik_01 = dm_ik[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[9] * dm_ik_01;
                        vjl_10 += gout[12] * dm_ik_01;
                        vjl_20 += gout[15] * dm_ik_01;
                        double dm_ik_02 = dm_ik[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[18] * dm_ik_02;
                        vjl_10 += gout[21] * dm_ik_02;
                        vjl_20 += gout[24] * dm_ik_02;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[4] * dm_ik_10;
                        vjl_20 += gout[7] * dm_ik_10;
                        double dm_ik_11 = dm_ik[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_11;
                        vjl_10 += gout[13] * dm_ik_11;
                        vjl_20 += gout[16] * dm_ik_11;
                        double dm_ik_12 = dm_ik[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[19] * dm_ik_12;
                        vjl_10 += gout[22] * dm_ik_12;
                        vjl_20 += gout[25] * dm_ik_12;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[5] * dm_ik_20;
                        vjl_20 += gout[8] * dm_ik_20;
                        double dm_ik_21 = dm_ik[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_21;
                        vjl_10 += gout[14] * dm_ik_21;
                        vjl_20 += gout[17] * dm_ik_21;
                        double dm_ik_22 = dm_ik[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[20] * dm_ik_22;
                        vjl_10 += gout[23] * dm_ik_22;
                        vjl_20 += gout[26] * dm_ik_22;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(v_jl+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(v_jl+(j0+2)*nao+(l0+0), vjl_20);
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[9]*dm_il_00 + gout[10]*dm_il_10 + gout[11]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[18]*dm_il_00 + gout[19]*dm_il_10 + gout[20]*dm_il_20;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+2), vjk_02);
                        double vjk_10 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20;
                        atomicAdd(v_jk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_11 = gout[12]*dm_il_00 + gout[13]*dm_il_10 + gout[14]*dm_il_20;
                        atomicAdd(v_jk+(j0+1)*nao+(k0+1), vjk_11);
                        double vjk_12 = gout[21]*dm_il_00 + gout[22]*dm_il_10 + gout[23]*dm_il_20;
                        atomicAdd(v_jk+(j0+1)*nao+(k0+2), vjk_12);
                        double vjk_20 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20;
                        atomicAdd(v_jk+(j0+2)*nao+(k0+0), vjk_20);
                        double vjk_21 = gout[15]*dm_il_00 + gout[16]*dm_il_10 + gout[17]*dm_il_20;
                        atomicAdd(v_jk+(j0+2)*nao+(k0+1), vjk_21);
                        double vjk_22 = gout[24]*dm_il_00 + gout[25]*dm_il_10 + gout[26]*dm_il_20;
                        atomicAdd(v_jk+(j0+2)*nao+(k0+2), vjk_22);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[6];
#pragma unroll
            for (int n = 0; n < 6; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        gout[0] += trr_20x * fac * wt;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_10x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_10x * fac * trr_10z;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += 1 * trr_20y * wt;
                        gout[4] += 1 * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += 1 * fac * trr_20z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00;
                    atomicAdd(v_il+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00;
                    atomicAdd(v_il+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00;
                    atomicAdd(v_il+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(v_ik+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(v_ik+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(v_ik+(i0+5)*nao+(k0+0), vik_50);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_30 = dm_ik[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_40 = dm_ik[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_50 = dm_ik[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        double dm_il_30 = dm_il[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        double dm_il_40 = dm_il[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        double dm_il_50 = dm_il[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2010(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[18];
#pragma unroll
            for (int n = 0; n < 18; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        gout[0] += trr_21x * fac * wt;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_11x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_11x * fac * trr_10z;
                        double trr_01x = cpx * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += trr_01x * trr_20y * wt;
                        gout[4] += trr_01x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += trr_01x * fac * trr_20z;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[6] += trr_20x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[7] += trr_10x * trr_11y * wt;
                        gout[8] += trr_10x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout[9] += 1 * trr_21y * wt;
                        gout[10] += 1 * trr_11y * trr_10z;
                        gout[11] += 1 * trr_01y * trr_20z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[12] += trr_20x * fac * trr_01z;
                        gout[13] += trr_10x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[14] += trr_10x * fac * trr_11z;
                        gout[15] += 1 * trr_20y * trr_01z;
                        gout[16] += 1 * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout[17] += 1 * fac * trr_21z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm_jk[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm_jk[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_01 + gout[12]*dm_jk_02;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_01 + gout[13]*dm_jk_02;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_01 + gout[14]*dm_jk_02;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_01 + gout[15]*dm_jk_02;
                    atomicAdd(v_il+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_01 + gout[16]*dm_jk_02;
                    atomicAdd(v_il+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_01 + gout[17]*dm_jk_02;
                    atomicAdd(v_il+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[6]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[12]*dm_jl_00;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[7]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[13]*dm_jl_00;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[8]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[14]*dm_jl_00;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+2), vik_22);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(v_ik+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[9]*dm_jl_00;
                    atomicAdd(v_ik+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[15]*dm_jl_00;
                    atomicAdd(v_ik+(i0+3)*nao+(k0+2), vik_32);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(v_ik+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[10]*dm_jl_00;
                    atomicAdd(v_ik+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[16]*dm_jl_00;
                    atomicAdd(v_ik+(i0+4)*nao+(k0+2), vik_42);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(v_ik+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_00;
                    atomicAdd(v_ik+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[17]*dm_jl_00;
                    atomicAdd(v_ik+(i0+5)*nao+(k0+2), vik_52);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm_ik[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[6] * dm_ik_01;
                        double dm_ik_02 = dm_ik[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[12] * dm_ik_02;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm_ik[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[7] * dm_ik_11;
                        double dm_ik_12 = dm_ik[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[13] * dm_ik_12;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm_ik[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[8] * dm_ik_21;
                        double dm_ik_22 = dm_ik[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[14] * dm_ik_22;
                        double dm_ik_30 = dm_ik[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_31 = dm_ik[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[9] * dm_ik_31;
                        double dm_ik_32 = dm_ik[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[15] * dm_ik_32;
                        double dm_ik_40 = dm_ik[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_41 = dm_ik[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_41;
                        double dm_ik_42 = dm_ik[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[16] * dm_ik_42;
                        double dm_ik_50 = dm_ik[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        double dm_ik_51 = dm_ik[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_51;
                        double dm_ik_52 = dm_ik[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[17] * dm_ik_52;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[6] * dm_il_00;
                        vjk_02 += gout[12] * dm_il_00;
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[7] * dm_il_10;
                        vjk_02 += gout[13] * dm_il_10;
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[8] * dm_il_20;
                        vjk_02 += gout[14] * dm_il_20;
                        double dm_il_30 = dm_il[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_01 += gout[9] * dm_il_30;
                        vjk_02 += gout[15] * dm_il_30;
                        double dm_il_40 = dm_il[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_01 += gout[10] * dm_il_40;
                        vjk_02 += gout[16] * dm_il_40;
                        double dm_il_50 = dm_il[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_01 += gout[11] * dm_il_50;
                        vjk_02 += gout[17] * dm_il_50;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(v_jk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(v_jk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2100(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                int *bas_mask_idx, int *Ts_ij_lookup,
                int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                float *q_cond_ij, float *q_cond_kl,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                int64_t *pool, int *head)
{
    int sq_id = threadIdx.x;
    int t_id = sq_id;
    int nsq_per_block = blockDim.x;
    int threads = nsq_per_block;

    extern __shared__ double shared_memory[];
    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * 8;

    int64_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;
    __shared__ int ntasks, pair_ij, pair_kl0;
while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    __shared__ int ish, jsh, cell_j, ish_cell0, jsh_cell0, i0, j0;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    int *bas = envs.bas;
    double *env = envs.env;
    if (t_id == 0) {
        int64_t bas_ij = pair_ij_mapping[pair_ij];
        ish = bas_ij / NBAS_MAX;
        jsh = bas_ij % NBAS_MAX;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int *ao_loc = envs.ao_loc;
        int _ish = bas_mask_idx[ish];
        int _jsh = bas_mask_idx[jsh];
        ish_cell0 = _ish % nbas_cell0;
        jsh_cell0 = _jsh % nbas_cell0;
        cell_j = _jsh / nbas_cell0;
        i0 = ao_loc[ish_cell0];
        j0 = ao_loc[jsh_cell0];
    }
    if (t_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[t_id] = env[ri_ptr+t_id];
        rjri[t_id] = env[rj_ptr+t_id] - ri[t_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }

    if (t_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                          pair_kl_mapping, bas_mask_idx, Ts_ij_lookup, nimgs, nbas_cell0,
                          q_cond_ij, q_cond_kl, s_cond_ij, s_cond_kl, diffuse_exps,
                          kmat, envs, bounds);
        if (ntasks == 0) continue;
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            int64_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / NBAS_MAX;
            int lsh = bas_kl % NBAS_MAX;
            int _ksh = bas_mask_idx[ksh];
            int cell_k = _ksh / nbas_cell0;
            int ksh_cell0 = _ksh % nbas_cell0;
            int _lsh = bas_mask_idx[lsh];
            int cell_l = _lsh / nbas_cell0;
            int lsh_cell0 = _lsh % nbas_cell0;
            double fac_sym = PI_FAC;
            if (task_id < ntasks) {
                if (ksh_cell0 == lsh_cell0) fac_sym *= .5;
                if (ish_cell0 == ksh_cell0 && jsh_cell0 == lsh_cell0) fac_sym *= .5;
            } else {
                fac_sym = 0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[18];
#pragma unroll
            for (int n = 0; n < 18; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }
                    __syncthreads();
                    double cicj = cicj_cache[ijp];
                    double fac = cicj * ckcl / (aij*akl*sqrt(aij+akl));
                    double xpa = rjri[0] * aj_aij;
                    double ypa = rjri[1] * aj_aij;
                    double zpa = rjri[2] * aj_aij;
                    double xij = ri[0] + xpa;
                    double yij = ri[1] + ypa;
                    double zij = ri[2] + zpa;
                    double xqc = xlxk * al_akl; // (ak*xk+al*xl)/akl
                    double yqc = ylyk * al_akl;
                    double zqc = zlzk * al_akl;
                    double xkl = env[rk+0] + xqc;
                    double ykl = env[rk+1] + yqc;
                    double zkl = env[rk+2] + zqc;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    rys_roots_for_k(4, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) continue;
                    for (int irys = 0; irys < 4; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double aij = aij_cache[0];
                        double aj_aij = aij_cache[1];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        gout[0] += hrr_2100x * fac * wt;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_1100x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_1100x * fac * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += hrr_0100x * trr_20y * wt;
                        gout[4] += hrr_0100x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += hrr_0100x * fac * trr_20z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[6] += trr_20x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[7] += trr_10x * hrr_1100y * wt;
                        gout[8] += trr_10x * hrr_0100y * trr_10z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout[9] += 1 * hrr_2100y * wt;
                        gout[10] += 1 * hrr_1100y * trr_10z;
                        gout[11] += 1 * hrr_0100y * trr_20z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[12] += trr_20x * fac * hrr_0100z;
                        gout[13] += trr_10x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[14] += trr_10x * fac * hrr_1100z;
                        gout[15] += 1 * trr_20y * hrr_0100z;
                        gout[16] += 1 * trr_10y * hrr_1100z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout[17] += 1 * fac * hrr_2100z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int k0 = ao_loc[ksh_cell0];
                int l0 = ao_loc[lsh_cell0];
                size_t nao2 = (size_t)nao * nao;
                size_t dm_size = nao2 * nimgs_uniq_pair;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *vk = kmat.vk + i_dm * dm_size;
                    double *dm = kmat.dm + i_dm * dm_size;
                    double *dm_jk = dm + Ts_ij_lookup[cell_j+cell_k*nimgs] * nao2;
                    double *dm_jl = dm + Ts_ij_lookup[cell_j+cell_l*nimgs] * nao2;
                    double *v_il = vk + Ts_ij_lookup[cell_l] * nao2;
                    double *v_ik = vk + Ts_ij_lookup[cell_k] * nao2;
                    double dm_jk_00 = dm_jk[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm_jk[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm_jk[(j0+2)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_10 + gout[12]*dm_jk_20;
                    atomicAdd(v_il+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_10 + gout[13]*dm_jk_20;
                    atomicAdd(v_il+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_10 + gout[14]*dm_jk_20;
                    atomicAdd(v_il+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_10 + gout[15]*dm_jk_20;
                    atomicAdd(v_il+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_10 + gout[16]*dm_jk_20;
                    atomicAdd(v_il+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_10 + gout[17]*dm_jk_20;
                    atomicAdd(v_il+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm_jl[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm_jl[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm_jl[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[6]*dm_jl_10 + gout[12]*dm_jl_20;
                    atomicAdd(v_ik+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[7]*dm_jl_10 + gout[13]*dm_jl_20;
                    atomicAdd(v_ik+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[8]*dm_jl_10 + gout[14]*dm_jl_20;
                    atomicAdd(v_ik+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00 + gout[9]*dm_jl_10 + gout[15]*dm_jl_20;
                    atomicAdd(v_ik+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00 + gout[10]*dm_jl_10 + gout[16]*dm_jl_20;
                    atomicAdd(v_ik+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00 + gout[11]*dm_jl_10 + gout[17]*dm_jl_20;
                    atomicAdd(v_ik+(i0+5)*nao+(k0+0), vik_50);
                    if (ish_cell0 != jsh_cell0) {
                        double *dm_ik = dm + Ts_ij_lookup[cell_k*nimgs] * nao2;
                        double *dm_il = dm + Ts_ij_lookup[cell_l*nimgs] * nao2;
                        double *v_jl = vk + Ts_ij_lookup[cell_j*nimgs+cell_l] * nao2;
                        double *v_jk = vk + Ts_ij_lookup[cell_j*nimgs+cell_k] * nao2;
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double dm_ik_00 = dm_ik[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[6] * dm_ik_00;
                        vjl_20 += gout[12] * dm_ik_00;
                        double dm_ik_10 = dm_ik[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[7] * dm_ik_10;
                        vjl_20 += gout[13] * dm_ik_10;
                        double dm_ik_20 = dm_ik[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[8] * dm_ik_20;
                        vjl_20 += gout[14] * dm_ik_20;
                        double dm_ik_30 = dm_ik[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        vjl_10 += gout[9] * dm_ik_30;
                        vjl_20 += gout[15] * dm_ik_30;
                        double dm_ik_40 = dm_ik[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        vjl_10 += gout[10] * dm_ik_40;
                        vjl_20 += gout[16] * dm_ik_40;
                        double dm_ik_50 = dm_ik[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        vjl_10 += gout[11] * dm_ik_50;
                        vjl_20 += gout[17] * dm_ik_50;
                        atomicAdd(v_jl+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(v_jl+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(v_jl+(j0+2)*nao+(l0+0), vjl_20);
                        double vjk_00 = 0;
                        double vjk_10 = 0;
                        double vjk_20 = 0;
                        double dm_il_00 = dm_il[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_10 += gout[6] * dm_il_00;
                        vjk_20 += gout[12] * dm_il_00;
                        double dm_il_10 = dm_il[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_10 += gout[7] * dm_il_10;
                        vjk_20 += gout[13] * dm_il_10;
                        double dm_il_20 = dm_il[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_10 += gout[8] * dm_il_20;
                        vjk_20 += gout[14] * dm_il_20;
                        double dm_il_30 = dm_il[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_10 += gout[9] * dm_il_30;
                        vjk_20 += gout[15] * dm_il_30;
                        double dm_il_40 = dm_il[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_10 += gout[10] * dm_il_40;
                        vjk_20 += gout[16] * dm_il_40;
                        double dm_il_50 = dm_il[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_10 += gout[11] * dm_il_50;
                        vjk_20 += gout[17] * dm_il_50;
                        atomicAdd(v_jk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(v_jk+(j0+1)*nao+(k0+0), vjk_10);
                        atomicAdd(v_jk+(j0+2)*nao+(k0+0), vjk_20);
                    }
                }
            }
        }
    }
}
}

int PBCrys_k_unrolled(RysIntEnvVars *envs, JKMatrix *kmat, BoundsInfo *bounds,
                    int64_t *pair_ij_mapping, int64_t *pair_kl_mapping,
                    int *bas_mask_idx, int *Ts_ij_lookup,
                    int nimgs, int nimgs_uniq_pair, int nbas_cell0, int nao,
                    float *q_cond_ij, float *q_cond_kl,
                    float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                    int64_t *pool, int *head, int workers)
{
    int li = bounds->li;
    int lj = bounds->lj;
    int lk = bounds->lk;
    int ll = bounds->ll;
    int ijkl = li*125 + lj*25 + lk*5 + ll;
    int nroots = bounds->nroots;
    int nsq_per_block = 256;
    int gout_stride = 1;

    switch (ijkl) {
    case 0: // (0, 0, 0, 0)
        adjust_threads(rys_k_0000, nsq_per_block);
        break;
    case 125: // (1, 0, 0, 0)
        adjust_threads(rys_k_1000, nsq_per_block);
        break;
    case 130: // (1, 0, 1, 0)
        adjust_threads(rys_k_1010, nsq_per_block);
        break;
    case 131: // (1, 0, 1, 1)
        adjust_threads(rys_k_1011, nsq_per_block);
        break;
    case 150: // (1, 1, 0, 0)
        adjust_threads(rys_k_1100, nsq_per_block);
        break;
    case 155: // (1, 1, 1, 0)
        adjust_threads(rys_k_1110, nsq_per_block);
        break;
    case 250: // (2, 0, 0, 0)
        adjust_threads(rys_k_2000, nsq_per_block);
        break;
    case 255: // (2, 0, 1, 0)
        adjust_threads(rys_k_2010, nsq_per_block);
        break;
    case 275: // (2, 1, 0, 0)
        adjust_threads(rys_k_2100, nsq_per_block);
        break;
    }

    dim3 threads(nsq_per_block, gout_stride);
    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int buflen = nroots*2 * nsq_per_block + iprim*jprim;
    switch (ijkl) {
    case 0: // (0, 0, 0, 0)
        rys_k_0000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 125: // (1, 0, 0, 0)
        rys_k_1000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 130: // (1, 0, 1, 0)
        rys_k_1010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 131: // (1, 0, 1, 1)
        rys_k_1011<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 150: // (1, 1, 0, 0)
        rys_k_1100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 155: // (1, 1, 1, 0)
        rys_k_1110<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 250: // (2, 0, 0, 0)
        rys_k_2000<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 255: // (2, 0, 1, 0)
        rys_k_2010<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 275: // (2, 1, 0, 0)
        rys_k_2100<<<workers, threads, buflen*sizeof(double)>>>(*envs, *kmat, *bounds,
            pair_ij_mapping, pair_kl_mapping, bas_mask_idx, Ts_ij_lookup,
            nimgs, nimgs_uniq_pair, nbas_cell0, nao, q_cond_ij, q_cond_kl,
            s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    default: return 0;
    }
    return 1;
}
