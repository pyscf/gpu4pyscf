#include <cuda.h>
#include <cuda_runtime.h>
#include "vhf.cuh"
#include "rys_roots_for_k.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "create_tasks.cu"


__global__ static
void rys_k_0000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        gout[0] += 1 * fac * wt;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    if (ish != jsh) {
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        double vjl_00 = gout[0]*dm_ik_00;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1010(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_01 + gout[6]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_01 + gout[7]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_01 + gout[8]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[6]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[7]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[8]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[3] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[6] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[4] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[7] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[5] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[8] * dm_ik_22;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1011(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_01 + gout[6]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_01 = gout[9]*dm_jk_00 + gout[12]*dm_jk_01 + gout[15]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[18]*dm_jk_00 + gout[21]*dm_jk_01 + gout[24]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_01 + gout[7]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_11 = gout[10]*dm_jk_00 + gout[13]*dm_jk_01 + gout[16]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[19]*dm_jk_00 + gout[22]*dm_jk_01 + gout[25]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_01 + gout[8]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[11]*dm_jk_00 + gout[14]*dm_jk_01 + gout[17]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[20]*dm_jk_00 + gout[23]*dm_jk_01 + gout[26]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[9]*dm_jl_01 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[3]*dm_jl_00 + gout[12]*dm_jl_01 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[6]*dm_jl_00 + gout[15]*dm_jl_01 + gout[24]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[10]*dm_jl_01 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[4]*dm_jl_00 + gout[13]*dm_jl_01 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[7]*dm_jl_00 + gout[16]*dm_jl_01 + gout[25]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[11]*dm_jl_01 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[5]*dm_jl_00 + gout[14]*dm_jl_01 + gout[23]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[8]*dm_jl_00 + gout[17]*dm_jl_01 + gout[26]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_01 = 0;
                        double vjl_02 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_01 += gout[9] * dm_ik_00;
                        vjl_02 += gout[18] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[3] * dm_ik_01;
                        vjl_01 += gout[12] * dm_ik_01;
                        vjl_02 += gout[21] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[6] * dm_ik_02;
                        vjl_01 += gout[15] * dm_ik_02;
                        vjl_02 += gout[24] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_01 += gout[10] * dm_ik_10;
                        vjl_02 += gout[19] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[4] * dm_ik_11;
                        vjl_01 += gout[13] * dm_ik_11;
                        vjl_02 += gout[22] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[7] * dm_ik_12;
                        vjl_01 += gout[16] * dm_ik_12;
                        vjl_02 += gout[25] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_01 += gout[11] * dm_ik_20;
                        vjl_02 += gout[20] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[5] * dm_ik_21;
                        vjl_01 += gout[14] * dm_ik_21;
                        vjl_02 += gout[23] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[8] * dm_ik_22;
                        vjl_01 += gout[17] * dm_ik_22;
                        vjl_02 += gout[26] * dm_ik_22;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+0)*nao+(l0+1), vjl_01);
                        atomicAdd(vk+(j0+0)*nao+(l0+2), vjl_02);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[3] * dm_il_00;
                        vjk_02 += gout[6] * dm_il_00;
                        double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                        vjk_00 += gout[9] * dm_il_01;
                        vjk_01 += gout[12] * dm_il_01;
                        vjk_02 += gout[15] * dm_il_01;
                        double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                        vjk_00 += gout[18] * dm_il_02;
                        vjk_01 += gout[21] * dm_il_02;
                        vjk_02 += gout[24] * dm_il_02;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[4] * dm_il_10;
                        vjk_02 += gout[7] * dm_il_10;
                        double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                        vjk_00 += gout[10] * dm_il_11;
                        vjk_01 += gout[13] * dm_il_11;
                        vjk_02 += gout[16] * dm_il_11;
                        double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                        vjk_00 += gout[19] * dm_il_12;
                        vjk_01 += gout[22] * dm_il_12;
                        vjk_02 += gout[25] * dm_il_12;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[5] * dm_il_20;
                        vjk_02 += gout[8] * dm_il_20;
                        double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                        vjk_00 += gout[11] * dm_il_21;
                        vjk_01 += gout[14] * dm_il_21;
                        vjk_02 += gout[17] * dm_il_21;
                        double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                        vjk_00 += gout[20] * dm_il_22;
                        vjk_01 += gout[23] * dm_il_22;
                        vjk_02 += gout[26] * dm_il_22;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1100(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_10 + gout[6]*dm_jk_20;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_10 + gout[7]*dm_jk_20;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_10 + gout[8]*dm_jk_20;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_10 + gout[6]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[4]*dm_jl_10 + gout[7]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[5]*dm_jl_10 + gout[8]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    if (ish != jsh) {
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        double vjl_00 = gout[0]*dm_ik_00 + gout[1]*dm_ik_10 + gout[2]*dm_ik_20;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjl_10 = gout[3]*dm_ik_00 + gout[4]*dm_ik_10 + gout[5]*dm_ik_20;
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        double vjl_20 = gout[6]*dm_ik_00 + gout[7]*dm_ik_10 + gout[8]*dm_ik_20;
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_10 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20;
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_20 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20;
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1110(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double vil_00 = 0;
                    double vil_10 = 0;
                    double vil_20 = 0;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_00 += gout[0] * dm_jk_00;
                    vil_10 += gout[1] * dm_jk_00;
                    vil_20 += gout[2] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_00 += gout[9] * dm_jk_01;
                    vil_10 += gout[10] * dm_jk_01;
                    vil_20 += gout[11] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_00 += gout[18] * dm_jk_02;
                    vil_10 += gout[19] * dm_jk_02;
                    vil_20 += gout[20] * dm_jk_02;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_00 += gout[3] * dm_jk_10;
                    vil_10 += gout[4] * dm_jk_10;
                    vil_20 += gout[5] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_00 += gout[12] * dm_jk_11;
                    vil_10 += gout[13] * dm_jk_11;
                    vil_20 += gout[14] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_00 += gout[21] * dm_jk_12;
                    vil_10 += gout[22] * dm_jk_12;
                    vil_20 += gout[23] * dm_jk_12;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_00 += gout[6] * dm_jk_20;
                    vil_10 += gout[7] * dm_jk_20;
                    vil_20 += gout[8] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_00 += gout[15] * dm_jk_21;
                    vil_10 += gout[16] * dm_jk_21;
                    vil_20 += gout[17] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_00 += gout[24] * dm_jk_22;
                    vil_10 += gout[25] * dm_jk_22;
                    vil_20 += gout[26] * dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_10 + gout[6]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[9]*dm_jl_00 + gout[12]*dm_jl_10 + gout[15]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[18]*dm_jl_00 + gout[21]*dm_jl_10 + gout[24]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[4]*dm_jl_10 + gout[7]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[10]*dm_jl_00 + gout[13]*dm_jl_10 + gout[16]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[19]*dm_jl_00 + gout[22]*dm_jl_10 + gout[25]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[5]*dm_jl_10 + gout[8]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[11]*dm_jl_00 + gout[14]*dm_jl_10 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[20]*dm_jl_00 + gout[23]*dm_jl_10 + gout[26]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[3] * dm_ik_00;
                        vjl_20 += gout[6] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[9] * dm_ik_01;
                        vjl_10 += gout[12] * dm_ik_01;
                        vjl_20 += gout[15] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[18] * dm_ik_02;
                        vjl_10 += gout[21] * dm_ik_02;
                        vjl_20 += gout[24] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[4] * dm_ik_10;
                        vjl_20 += gout[7] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_11;
                        vjl_10 += gout[13] * dm_ik_11;
                        vjl_20 += gout[16] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[19] * dm_ik_12;
                        vjl_10 += gout[22] * dm_ik_12;
                        vjl_20 += gout[25] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[5] * dm_ik_20;
                        vjl_20 += gout[8] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_21;
                        vjl_10 += gout[14] * dm_ik_21;
                        vjl_20 += gout[17] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[20] * dm_ik_22;
                        vjl_10 += gout[23] * dm_ik_22;
                        vjl_20 += gout[26] * dm_ik_22;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[9]*dm_il_00 + gout[10]*dm_il_10 + gout[11]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[18]*dm_il_00 + gout[19]*dm_il_10 + gout[20]*dm_il_20;
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                        double vjk_10 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20;
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_11 = gout[12]*dm_il_00 + gout[13]*dm_il_10 + gout[14]*dm_il_20;
                        atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                        double vjk_12 = gout[21]*dm_il_00 + gout[22]*dm_il_10 + gout[23]*dm_il_20;
                        atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                        double vjk_20 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20;
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                        double vjk_21 = gout[15]*dm_il_00 + gout[16]*dm_il_10 + gout[17]*dm_il_20;
                        atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                        double vjk_22 = gout[24]*dm_il_00 + gout[25]*dm_il_10 + gout[26]*dm_il_20;
                        atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_1111(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[81];
#pragma unroll
            for (int n = 0; n < 81; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                        gout[0] += hrr_1111x * fac * wt;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_0111x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_0111x * fac * trr_10z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[3] += hrr_1011x * hrr_0100y * wt;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[4] += hrr_0011x * hrr_1100y * wt;
                        gout[5] += hrr_0011x * hrr_0100y * trr_10z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[6] += hrr_1011x * fac * hrr_0100z;
                        gout[7] += hrr_0011x * trr_10y * hrr_0100z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[8] += hrr_0011x * fac * hrr_1100z;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[9] += hrr_1101x * trr_01y * wt;
                        double hrr_0001x = trr_01x - xlxk * 1;
                        double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[10] += hrr_0101x * trr_11y * wt;
                        gout[11] += hrr_0101x * trr_01y * trr_10z;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gout[12] += hrr_1001x * hrr_0110y * wt;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gout[13] += hrr_0001x * hrr_1110y * wt;
                        gout[14] += hrr_0001x * hrr_0110y * trr_10z;
                        gout[15] += hrr_1001x * trr_01y * hrr_0100z;
                        gout[16] += hrr_0001x * trr_11y * hrr_0100z;
                        gout[17] += hrr_0001x * trr_01y * hrr_1100z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[18] += hrr_1101x * fac * trr_01z;
                        gout[19] += hrr_0101x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[20] += hrr_0101x * fac * trr_11z;
                        gout[21] += hrr_1001x * hrr_0100y * trr_01z;
                        gout[22] += hrr_0001x * hrr_1100y * trr_01z;
                        gout[23] += hrr_0001x * hrr_0100y * trr_11z;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gout[24] += hrr_1001x * fac * hrr_0110z;
                        gout[25] += hrr_0001x * trr_10y * hrr_0110z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gout[26] += hrr_0001x * fac * hrr_1110z;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        double hrr_0001y = trr_01y - ylyk * fac;
                        gout[27] += hrr_1110x * hrr_0001y * wt;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gout[28] += hrr_0110x * hrr_1001y * wt;
                        gout[29] += hrr_0110x * hrr_0001y * trr_10z;
                        double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                        gout[30] += trr_11x * hrr_0101y * wt;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                        gout[31] += trr_01x * hrr_1101y * wt;
                        gout[32] += trr_01x * hrr_0101y * trr_10z;
                        gout[33] += trr_11x * hrr_0001y * hrr_0100z;
                        gout[34] += trr_01x * hrr_1001y * hrr_0100z;
                        gout[35] += trr_01x * hrr_0001y * hrr_1100z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gout[36] += hrr_1100x * hrr_0011y * wt;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gout[37] += hrr_0100x * hrr_1011y * wt;
                        gout[38] += hrr_0100x * hrr_0011y * trr_10z;
                        double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                        gout[39] += trr_10x * hrr_0111y * wt;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                        gout[40] += 1 * hrr_1111y * wt;
                        gout[41] += 1 * hrr_0111y * trr_10z;
                        gout[42] += trr_10x * hrr_0011y * hrr_0100z;
                        gout[43] += 1 * hrr_1011y * hrr_0100z;
                        gout[44] += 1 * hrr_0011y * hrr_1100z;
                        gout[45] += hrr_1100x * hrr_0001y * trr_01z;
                        gout[46] += hrr_0100x * hrr_1001y * trr_01z;
                        gout[47] += hrr_0100x * hrr_0001y * trr_11z;
                        gout[48] += trr_10x * hrr_0101y * trr_01z;
                        gout[49] += 1 * hrr_1101y * trr_01z;
                        gout[50] += 1 * hrr_0101y * trr_11z;
                        gout[51] += trr_10x * hrr_0001y * hrr_0110z;
                        gout[52] += 1 * hrr_1001y * hrr_0110z;
                        gout[53] += 1 * hrr_0001y * hrr_1110z;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        gout[54] += hrr_1110x * fac * hrr_0001z;
                        gout[55] += hrr_0110x * trr_10y * hrr_0001z;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gout[56] += hrr_0110x * fac * hrr_1001z;
                        gout[57] += trr_11x * hrr_0100y * hrr_0001z;
                        gout[58] += trr_01x * hrr_1100y * hrr_0001z;
                        gout[59] += trr_01x * hrr_0100y * hrr_1001z;
                        double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                        gout[60] += trr_11x * fac * hrr_0101z;
                        gout[61] += trr_01x * trr_10y * hrr_0101z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                        gout[62] += trr_01x * fac * hrr_1101z;
                        gout[63] += hrr_1100x * trr_01y * hrr_0001z;
                        gout[64] += hrr_0100x * trr_11y * hrr_0001z;
                        gout[65] += hrr_0100x * trr_01y * hrr_1001z;
                        gout[66] += trr_10x * hrr_0110y * hrr_0001z;
                        gout[67] += 1 * hrr_1110y * hrr_0001z;
                        gout[68] += 1 * hrr_0110y * hrr_1001z;
                        gout[69] += trr_10x * trr_01y * hrr_0101z;
                        gout[70] += 1 * trr_11y * hrr_0101z;
                        gout[71] += 1 * trr_01y * hrr_1101z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gout[72] += hrr_1100x * fac * hrr_0011z;
                        gout[73] += hrr_0100x * trr_10y * hrr_0011z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gout[74] += hrr_0100x * fac * hrr_1011z;
                        gout[75] += trr_10x * hrr_0100y * hrr_0011z;
                        gout[76] += 1 * hrr_1100y * hrr_0011z;
                        gout[77] += 1 * hrr_0100y * hrr_1011z;
                        double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                        gout[78] += trr_10x * fac * hrr_0111z;
                        gout[79] += 1 * trr_10y * hrr_0111z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                        gout[80] += 1 * fac * hrr_1111z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_10 + gout[6]*dm_jk_20 + gout[9]*dm_jk_01 + gout[12]*dm_jk_11 + gout[15]*dm_jk_21 + gout[18]*dm_jk_02 + gout[21]*dm_jk_12 + gout[24]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_01 = gout[27]*dm_jk_00 + gout[30]*dm_jk_10 + gout[33]*dm_jk_20 + gout[36]*dm_jk_01 + gout[39]*dm_jk_11 + gout[42]*dm_jk_21 + gout[45]*dm_jk_02 + gout[48]*dm_jk_12 + gout[51]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[54]*dm_jk_00 + gout[57]*dm_jk_10 + gout[60]*dm_jk_20 + gout[63]*dm_jk_01 + gout[66]*dm_jk_11 + gout[69]*dm_jk_21 + gout[72]*dm_jk_02 + gout[75]*dm_jk_12 + gout[78]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double vil_10 = gout[1]*dm_jk_00 + gout[4]*dm_jk_10 + gout[7]*dm_jk_20 + gout[10]*dm_jk_01 + gout[13]*dm_jk_11 + gout[16]*dm_jk_21 + gout[19]*dm_jk_02 + gout[22]*dm_jk_12 + gout[25]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_11 = gout[28]*dm_jk_00 + gout[31]*dm_jk_10 + gout[34]*dm_jk_20 + gout[37]*dm_jk_01 + gout[40]*dm_jk_11 + gout[43]*dm_jk_21 + gout[46]*dm_jk_02 + gout[49]*dm_jk_12 + gout[52]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[55]*dm_jk_00 + gout[58]*dm_jk_10 + gout[61]*dm_jk_20 + gout[64]*dm_jk_01 + gout[67]*dm_jk_11 + gout[70]*dm_jk_21 + gout[73]*dm_jk_02 + gout[76]*dm_jk_12 + gout[79]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double vil_20 = gout[2]*dm_jk_00 + gout[5]*dm_jk_10 + gout[8]*dm_jk_20 + gout[11]*dm_jk_01 + gout[14]*dm_jk_11 + gout[17]*dm_jk_21 + gout[20]*dm_jk_02 + gout[23]*dm_jk_12 + gout[26]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[29]*dm_jk_00 + gout[32]*dm_jk_10 + gout[35]*dm_jk_20 + gout[38]*dm_jk_01 + gout[41]*dm_jk_11 + gout[44]*dm_jk_21 + gout[47]*dm_jk_02 + gout[50]*dm_jk_12 + gout[53]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[56]*dm_jk_00 + gout[59]*dm_jk_10 + gout[62]*dm_jk_20 + gout[65]*dm_jk_01 + gout[68]*dm_jk_11 + gout[71]*dm_jk_21 + gout[74]*dm_jk_02 + gout[77]*dm_jk_12 + gout[80]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_10 + gout[6]*dm_jl_20 + gout[27]*dm_jl_01 + gout[30]*dm_jl_11 + gout[33]*dm_jl_21 + gout[54]*dm_jl_02 + gout[57]*dm_jl_12 + gout[60]*dm_jl_22;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[9]*dm_jl_00 + gout[12]*dm_jl_10 + gout[15]*dm_jl_20 + gout[36]*dm_jl_01 + gout[39]*dm_jl_11 + gout[42]*dm_jl_21 + gout[63]*dm_jl_02 + gout[66]*dm_jl_12 + gout[69]*dm_jl_22;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[18]*dm_jl_00 + gout[21]*dm_jl_10 + gout[24]*dm_jl_20 + gout[45]*dm_jl_01 + gout[48]*dm_jl_11 + gout[51]*dm_jl_21 + gout[72]*dm_jl_02 + gout[75]*dm_jl_12 + gout[78]*dm_jl_22;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[4]*dm_jl_10 + gout[7]*dm_jl_20 + gout[28]*dm_jl_01 + gout[31]*dm_jl_11 + gout[34]*dm_jl_21 + gout[55]*dm_jl_02 + gout[58]*dm_jl_12 + gout[61]*dm_jl_22;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[10]*dm_jl_00 + gout[13]*dm_jl_10 + gout[16]*dm_jl_20 + gout[37]*dm_jl_01 + gout[40]*dm_jl_11 + gout[43]*dm_jl_21 + gout[64]*dm_jl_02 + gout[67]*dm_jl_12 + gout[70]*dm_jl_22;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[19]*dm_jl_00 + gout[22]*dm_jl_10 + gout[25]*dm_jl_20 + gout[46]*dm_jl_01 + gout[49]*dm_jl_11 + gout[52]*dm_jl_21 + gout[73]*dm_jl_02 + gout[76]*dm_jl_12 + gout[79]*dm_jl_22;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[5]*dm_jl_10 + gout[8]*dm_jl_20 + gout[29]*dm_jl_01 + gout[32]*dm_jl_11 + gout[35]*dm_jl_21 + gout[56]*dm_jl_02 + gout[59]*dm_jl_12 + gout[62]*dm_jl_22;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[11]*dm_jl_00 + gout[14]*dm_jl_10 + gout[17]*dm_jl_20 + gout[38]*dm_jl_01 + gout[41]*dm_jl_11 + gout[44]*dm_jl_21 + gout[65]*dm_jl_02 + gout[68]*dm_jl_12 + gout[71]*dm_jl_22;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[20]*dm_jl_00 + gout[23]*dm_jl_10 + gout[26]*dm_jl_20 + gout[47]*dm_jl_01 + gout[50]*dm_jl_11 + gout[53]*dm_jl_21 + gout[74]*dm_jl_02 + gout[77]*dm_jl_12 + gout[80]*dm_jl_22;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    if (ish != jsh) {
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        double vjl_00 = gout[0]*dm_ik_00 + gout[1]*dm_ik_10 + gout[2]*dm_ik_20 + gout[9]*dm_ik_01 + gout[10]*dm_ik_11 + gout[11]*dm_ik_21 + gout[18]*dm_ik_02 + gout[19]*dm_ik_12 + gout[20]*dm_ik_22;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjl_01 = gout[27]*dm_ik_00 + gout[28]*dm_ik_10 + gout[29]*dm_ik_20 + gout[36]*dm_ik_01 + gout[37]*dm_ik_11 + gout[38]*dm_ik_21 + gout[45]*dm_ik_02 + gout[46]*dm_ik_12 + gout[47]*dm_ik_22;
                        atomicAdd(vk+(j0+0)*nao+(l0+1), vjl_01);
                        double vjl_02 = gout[54]*dm_ik_00 + gout[55]*dm_ik_10 + gout[56]*dm_ik_20 + gout[63]*dm_ik_01 + gout[64]*dm_ik_11 + gout[65]*dm_ik_21 + gout[72]*dm_ik_02 + gout[73]*dm_ik_12 + gout[74]*dm_ik_22;
                        atomicAdd(vk+(j0+0)*nao+(l0+2), vjl_02);
                        double vjl_10 = gout[3]*dm_ik_00 + gout[4]*dm_ik_10 + gout[5]*dm_ik_20 + gout[12]*dm_ik_01 + gout[13]*dm_ik_11 + gout[14]*dm_ik_21 + gout[21]*dm_ik_02 + gout[22]*dm_ik_12 + gout[23]*dm_ik_22;
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        double vjl_11 = gout[30]*dm_ik_00 + gout[31]*dm_ik_10 + gout[32]*dm_ik_20 + gout[39]*dm_ik_01 + gout[40]*dm_ik_11 + gout[41]*dm_ik_21 + gout[48]*dm_ik_02 + gout[49]*dm_ik_12 + gout[50]*dm_ik_22;
                        atomicAdd(vk+(j0+1)*nao+(l0+1), vjl_11);
                        double vjl_12 = gout[57]*dm_ik_00 + gout[58]*dm_ik_10 + gout[59]*dm_ik_20 + gout[66]*dm_ik_01 + gout[67]*dm_ik_11 + gout[68]*dm_ik_21 + gout[75]*dm_ik_02 + gout[76]*dm_ik_12 + gout[77]*dm_ik_22;
                        atomicAdd(vk+(j0+1)*nao+(l0+2), vjl_12);
                        double vjl_20 = gout[6]*dm_ik_00 + gout[7]*dm_ik_10 + gout[8]*dm_ik_20 + gout[15]*dm_ik_01 + gout[16]*dm_ik_11 + gout[17]*dm_ik_21 + gout[24]*dm_ik_02 + gout[25]*dm_ik_12 + gout[26]*dm_ik_22;
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double vjl_21 = gout[33]*dm_ik_00 + gout[34]*dm_ik_10 + gout[35]*dm_ik_20 + gout[42]*dm_ik_01 + gout[43]*dm_ik_11 + gout[44]*dm_ik_21 + gout[51]*dm_ik_02 + gout[52]*dm_ik_12 + gout[53]*dm_ik_22;
                        atomicAdd(vk+(j0+2)*nao+(l0+1), vjl_21);
                        double vjl_22 = gout[60]*dm_ik_00 + gout[61]*dm_ik_10 + gout[62]*dm_ik_20 + gout[69]*dm_ik_01 + gout[70]*dm_ik_11 + gout[71]*dm_ik_21 + gout[78]*dm_ik_02 + gout[79]*dm_ik_12 + gout[80]*dm_ik_22;
                        atomicAdd(vk+(j0+2)*nao+(l0+2), vjl_22);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                        double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                        double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                        double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                        double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                        double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20 + gout[27]*dm_il_01 + gout[28]*dm_il_11 + gout[29]*dm_il_21 + gout[54]*dm_il_02 + gout[55]*dm_il_12 + gout[56]*dm_il_22;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[9]*dm_il_00 + gout[10]*dm_il_10 + gout[11]*dm_il_20 + gout[36]*dm_il_01 + gout[37]*dm_il_11 + gout[38]*dm_il_21 + gout[63]*dm_il_02 + gout[64]*dm_il_12 + gout[65]*dm_il_22;
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[18]*dm_il_00 + gout[19]*dm_il_10 + gout[20]*dm_il_20 + gout[45]*dm_il_01 + gout[46]*dm_il_11 + gout[47]*dm_il_21 + gout[72]*dm_il_02 + gout[73]*dm_il_12 + gout[74]*dm_il_22;
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                        double vjk_10 = gout[3]*dm_il_00 + gout[4]*dm_il_10 + gout[5]*dm_il_20 + gout[30]*dm_il_01 + gout[31]*dm_il_11 + gout[32]*dm_il_21 + gout[57]*dm_il_02 + gout[58]*dm_il_12 + gout[59]*dm_il_22;
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_11 = gout[12]*dm_il_00 + gout[13]*dm_il_10 + gout[14]*dm_il_20 + gout[39]*dm_il_01 + gout[40]*dm_il_11 + gout[41]*dm_il_21 + gout[66]*dm_il_02 + gout[67]*dm_il_12 + gout[68]*dm_il_22;
                        atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                        double vjk_12 = gout[21]*dm_il_00 + gout[22]*dm_il_10 + gout[23]*dm_il_20 + gout[48]*dm_il_01 + gout[49]*dm_il_11 + gout[50]*dm_il_21 + gout[75]*dm_il_02 + gout[76]*dm_il_12 + gout[77]*dm_il_22;
                        atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                        double vjk_20 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20 + gout[33]*dm_il_01 + gout[34]*dm_il_11 + gout[35]*dm_il_21 + gout[60]*dm_il_02 + gout[61]*dm_il_12 + gout[62]*dm_il_22;
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                        double vjk_21 = gout[15]*dm_il_00 + gout[16]*dm_il_10 + gout[17]*dm_il_20 + gout[42]*dm_il_01 + gout[43]*dm_il_11 + gout[44]*dm_il_21 + gout[69]*dm_il_02 + gout[70]*dm_il_12 + gout[71]*dm_il_22;
                        atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                        double vjk_22 = gout[24]*dm_il_00 + gout[25]*dm_il_10 + gout[26]*dm_il_20 + gout[51]*dm_il_01 + gout[52]*dm_il_11 + gout[53]*dm_il_21 + gout[78]*dm_il_02 + gout[79]*dm_il_12 + gout[80]*dm_il_22;
                        atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2010(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_01 + gout[12]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_01 + gout[13]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_01 + gout[14]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_01 + gout[15]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_01 + gout[16]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_01 + gout[17]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[6]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[12]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[7]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[13]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[8]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[14]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[9]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[15]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[10]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[16]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[17]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[6] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[12] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[7] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[13] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[8] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[14] * dm_ik_22;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[9] * dm_ik_31;
                        double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[15] * dm_ik_32;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_41;
                        double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[16] * dm_ik_42;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_51;
                        double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[17] * dm_ik_52;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[6] * dm_il_00;
                        vjk_02 += gout[12] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[7] * dm_il_10;
                        vjk_02 += gout[13] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[8] * dm_il_20;
                        vjk_02 += gout[14] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_01 += gout[9] * dm_il_30;
                        vjk_02 += gout[15] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_01 += gout[10] * dm_il_40;
                        vjk_02 += gout[16] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_01 += gout[11] * dm_il_50;
                        vjk_02 += gout[17] * dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2011(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[54];
#pragma unroll
            for (int n = 0; n < 54; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double hrr_2011x = trr_22x - xlxk * trr_21x;
                        gout[0] += hrr_2011x * fac * wt;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double hrr_1011x = trr_12x - xlxk * trr_11x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_1011x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_1011x * fac * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double hrr_0011x = trr_02x - xlxk * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += hrr_0011x * trr_20y * wt;
                        gout[4] += hrr_0011x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += hrr_0011x * fac * trr_20z;
                        double hrr_2001x = trr_21x - xlxk * trr_20x;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[6] += hrr_2001x * trr_01y * wt;
                        double hrr_1001x = trr_11x - xlxk * trr_10x;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[7] += hrr_1001x * trr_11y * wt;
                        gout[8] += hrr_1001x * trr_01y * trr_10z;
                        double hrr_0001x = trr_01x - xlxk * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout[9] += hrr_0001x * trr_21y * wt;
                        gout[10] += hrr_0001x * trr_11y * trr_10z;
                        gout[11] += hrr_0001x * trr_01y * trr_20z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[12] += hrr_2001x * fac * trr_01z;
                        gout[13] += hrr_1001x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[14] += hrr_1001x * fac * trr_11z;
                        gout[15] += hrr_0001x * trr_20y * trr_01z;
                        gout[16] += hrr_0001x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout[17] += hrr_0001x * fac * trr_21z;
                        double hrr_0001y = trr_01y - ylyk * fac;
                        gout[18] += trr_21x * hrr_0001y * wt;
                        double hrr_1001y = trr_11y - ylyk * trr_10y;
                        gout[19] += trr_11x * hrr_1001y * wt;
                        gout[20] += trr_11x * hrr_0001y * trr_10z;
                        double hrr_2001y = trr_21y - ylyk * trr_20y;
                        gout[21] += trr_01x * hrr_2001y * wt;
                        gout[22] += trr_01x * hrr_1001y * trr_10z;
                        gout[23] += trr_01x * hrr_0001y * trr_20z;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac;
                        double hrr_0011y = trr_02y - ylyk * trr_01y;
                        gout[24] += trr_20x * hrr_0011y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        double hrr_1011y = trr_12y - ylyk * trr_11y;
                        gout[25] += trr_10x * hrr_1011y * wt;
                        gout[26] += trr_10x * hrr_0011y * trr_10z;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        double hrr_2011y = trr_22y - ylyk * trr_21y;
                        gout[27] += 1 * hrr_2011y * wt;
                        gout[28] += 1 * hrr_1011y * trr_10z;
                        gout[29] += 1 * hrr_0011y * trr_20z;
                        gout[30] += trr_20x * hrr_0001y * trr_01z;
                        gout[31] += trr_10x * hrr_1001y * trr_01z;
                        gout[32] += trr_10x * hrr_0001y * trr_11z;
                        gout[33] += 1 * hrr_2001y * trr_01z;
                        gout[34] += 1 * hrr_1001y * trr_11z;
                        gout[35] += 1 * hrr_0001y * trr_21z;
                        double hrr_0001z = trr_01z - zlzk * wt;
                        gout[36] += trr_21x * fac * hrr_0001z;
                        gout[37] += trr_11x * trr_10y * hrr_0001z;
                        double hrr_1001z = trr_11z - zlzk * trr_10z;
                        gout[38] += trr_11x * fac * hrr_1001z;
                        gout[39] += trr_01x * trr_20y * hrr_0001z;
                        gout[40] += trr_01x * trr_10y * hrr_1001z;
                        double hrr_2001z = trr_21z - zlzk * trr_20z;
                        gout[41] += trr_01x * fac * hrr_2001z;
                        gout[42] += trr_20x * trr_01y * hrr_0001z;
                        gout[43] += trr_10x * trr_11y * hrr_0001z;
                        gout[44] += trr_10x * trr_01y * hrr_1001z;
                        gout[45] += 1 * trr_21y * hrr_0001z;
                        gout[46] += 1 * trr_11y * hrr_1001z;
                        gout[47] += 1 * trr_01y * hrr_2001z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        double hrr_0011z = trr_02z - zlzk * trr_01z;
                        gout[48] += trr_20x * fac * hrr_0011z;
                        gout[49] += trr_10x * trr_10y * hrr_0011z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        double hrr_1011z = trr_12z - zlzk * trr_11z;
                        gout[50] += trr_10x * fac * hrr_1011z;
                        gout[51] += 1 * trr_20y * hrr_0011z;
                        gout[52] += 1 * trr_10y * hrr_1011z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        double hrr_2011z = trr_22z - zlzk * trr_21z;
                        gout[53] += 1 * fac * hrr_2011z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_01 + gout[12]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_01 = gout[18]*dm_jk_00 + gout[24]*dm_jk_01 + gout[30]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[36]*dm_jk_00 + gout[42]*dm_jk_01 + gout[48]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_01 + gout[13]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_11 = gout[19]*dm_jk_00 + gout[25]*dm_jk_01 + gout[31]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[37]*dm_jk_00 + gout[43]*dm_jk_01 + gout[49]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_01 + gout[14]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[20]*dm_jk_00 + gout[26]*dm_jk_01 + gout[32]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[38]*dm_jk_00 + gout[44]*dm_jk_01 + gout[50]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_01 + gout[15]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[21]*dm_jk_00 + gout[27]*dm_jk_01 + gout[33]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[39]*dm_jk_00 + gout[45]*dm_jk_01 + gout[51]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_01 + gout[16]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[22]*dm_jk_00 + gout[28]*dm_jk_01 + gout[34]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[40]*dm_jk_00 + gout[46]*dm_jk_01 + gout[52]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_01 + gout[17]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[23]*dm_jk_00 + gout[29]*dm_jk_01 + gout[35]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[41]*dm_jk_00 + gout[47]*dm_jk_01 + gout[53]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[18]*dm_jl_01 + gout[36]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[6]*dm_jl_00 + gout[24]*dm_jl_01 + gout[42]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[12]*dm_jl_00 + gout[30]*dm_jl_01 + gout[48]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[19]*dm_jl_01 + gout[37]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[7]*dm_jl_00 + gout[25]*dm_jl_01 + gout[43]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[13]*dm_jl_00 + gout[31]*dm_jl_01 + gout[49]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[20]*dm_jl_01 + gout[38]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[8]*dm_jl_00 + gout[26]*dm_jl_01 + gout[44]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[14]*dm_jl_00 + gout[32]*dm_jl_01 + gout[50]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_30 = gout[3]*dm_jl_00 + gout[21]*dm_jl_01 + gout[39]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[9]*dm_jl_00 + gout[27]*dm_jl_01 + gout[45]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[15]*dm_jl_00 + gout[33]*dm_jl_01 + gout[51]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_40 = gout[4]*dm_jl_00 + gout[22]*dm_jl_01 + gout[40]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[10]*dm_jl_00 + gout[28]*dm_jl_01 + gout[46]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[16]*dm_jl_00 + gout[34]*dm_jl_01 + gout[52]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_50 = gout[5]*dm_jl_00 + gout[23]*dm_jl_01 + gout[41]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_00 + gout[29]*dm_jl_01 + gout[47]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[17]*dm_jl_00 + gout[35]*dm_jl_01 + gout[53]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_01 = 0;
                        double vjl_02 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_01 += gout[18] * dm_ik_00;
                        vjl_02 += gout[36] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[6] * dm_ik_01;
                        vjl_01 += gout[24] * dm_ik_01;
                        vjl_02 += gout[42] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[12] * dm_ik_02;
                        vjl_01 += gout[30] * dm_ik_02;
                        vjl_02 += gout[48] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_01 += gout[19] * dm_ik_10;
                        vjl_02 += gout[37] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[7] * dm_ik_11;
                        vjl_01 += gout[25] * dm_ik_11;
                        vjl_02 += gout[43] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[13] * dm_ik_12;
                        vjl_01 += gout[31] * dm_ik_12;
                        vjl_02 += gout[49] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_01 += gout[20] * dm_ik_20;
                        vjl_02 += gout[38] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[8] * dm_ik_21;
                        vjl_01 += gout[26] * dm_ik_21;
                        vjl_02 += gout[44] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[14] * dm_ik_22;
                        vjl_01 += gout[32] * dm_ik_22;
                        vjl_02 += gout[50] * dm_ik_22;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        vjl_01 += gout[21] * dm_ik_30;
                        vjl_02 += gout[39] * dm_ik_30;
                        double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[9] * dm_ik_31;
                        vjl_01 += gout[27] * dm_ik_31;
                        vjl_02 += gout[45] * dm_ik_31;
                        double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[15] * dm_ik_32;
                        vjl_01 += gout[33] * dm_ik_32;
                        vjl_02 += gout[51] * dm_ik_32;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        vjl_01 += gout[22] * dm_ik_40;
                        vjl_02 += gout[40] * dm_ik_40;
                        double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_41;
                        vjl_01 += gout[28] * dm_ik_41;
                        vjl_02 += gout[46] * dm_ik_41;
                        double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[16] * dm_ik_42;
                        vjl_01 += gout[34] * dm_ik_42;
                        vjl_02 += gout[52] * dm_ik_42;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        vjl_01 += gout[23] * dm_ik_50;
                        vjl_02 += gout[41] * dm_ik_50;
                        double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_51;
                        vjl_01 += gout[29] * dm_ik_51;
                        vjl_02 += gout[47] * dm_ik_51;
                        double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[17] * dm_ik_52;
                        vjl_01 += gout[35] * dm_ik_52;
                        vjl_02 += gout[53] * dm_ik_52;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+0)*nao+(l0+1), vjl_01);
                        atomicAdd(vk+(j0+0)*nao+(l0+2), vjl_02);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[6] * dm_il_00;
                        vjk_02 += gout[12] * dm_il_00;
                        double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                        vjk_00 += gout[18] * dm_il_01;
                        vjk_01 += gout[24] * dm_il_01;
                        vjk_02 += gout[30] * dm_il_01;
                        double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                        vjk_00 += gout[36] * dm_il_02;
                        vjk_01 += gout[42] * dm_il_02;
                        vjk_02 += gout[48] * dm_il_02;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[7] * dm_il_10;
                        vjk_02 += gout[13] * dm_il_10;
                        double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                        vjk_00 += gout[19] * dm_il_11;
                        vjk_01 += gout[25] * dm_il_11;
                        vjk_02 += gout[31] * dm_il_11;
                        double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                        vjk_00 += gout[37] * dm_il_12;
                        vjk_01 += gout[43] * dm_il_12;
                        vjk_02 += gout[49] * dm_il_12;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[8] * dm_il_20;
                        vjk_02 += gout[14] * dm_il_20;
                        double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                        vjk_00 += gout[20] * dm_il_21;
                        vjk_01 += gout[26] * dm_il_21;
                        vjk_02 += gout[32] * dm_il_21;
                        double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                        vjk_00 += gout[38] * dm_il_22;
                        vjk_01 += gout[44] * dm_il_22;
                        vjk_02 += gout[50] * dm_il_22;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_01 += gout[9] * dm_il_30;
                        vjk_02 += gout[15] * dm_il_30;
                        double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                        vjk_00 += gout[21] * dm_il_31;
                        vjk_01 += gout[27] * dm_il_31;
                        vjk_02 += gout[33] * dm_il_31;
                        double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                        vjk_00 += gout[39] * dm_il_32;
                        vjk_01 += gout[45] * dm_il_32;
                        vjk_02 += gout[51] * dm_il_32;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_01 += gout[10] * dm_il_40;
                        vjk_02 += gout[16] * dm_il_40;
                        double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                        vjk_00 += gout[22] * dm_il_41;
                        vjk_01 += gout[28] * dm_il_41;
                        vjk_02 += gout[34] * dm_il_41;
                        double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                        vjk_00 += gout[40] * dm_il_42;
                        vjk_01 += gout[46] * dm_il_42;
                        vjk_02 += gout[52] * dm_il_42;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_01 += gout[11] * dm_il_50;
                        vjk_02 += gout[17] * dm_il_50;
                        double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                        vjk_00 += gout[23] * dm_il_51;
                        vjk_01 += gout[29] * dm_il_51;
                        vjk_02 += gout[35] * dm_il_51;
                        double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                        vjk_00 += gout[41] * dm_il_52;
                        vjk_01 += gout[47] * dm_il_52;
                        vjk_02 += gout[53] * dm_il_52;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2020(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[36];
#pragma unroll
            for (int n = 0; n < 36; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        gout[0] += trr_22x * fac * wt;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_12x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_12x * fac * trr_10z;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += trr_02x * trr_20y * wt;
                        gout[4] += trr_02x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += trr_02x * fac * trr_20z;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[6] += trr_21x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[7] += trr_11x * trr_11y * wt;
                        gout[8] += trr_11x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout[9] += trr_01x * trr_21y * wt;
                        gout[10] += trr_01x * trr_11y * trr_10z;
                        gout[11] += trr_01x * trr_01y * trr_20z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[12] += trr_21x * fac * trr_01z;
                        gout[13] += trr_11x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[14] += trr_11x * fac * trr_11z;
                        gout[15] += trr_01x * trr_20y * trr_01z;
                        gout[16] += trr_01x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout[17] += trr_01x * fac * trr_21z;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac;
                        gout[18] += trr_20x * trr_02y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        gout[19] += trr_10x * trr_12y * wt;
                        gout[20] += trr_10x * trr_02y * trr_10z;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        gout[21] += 1 * trr_22y * wt;
                        gout[22] += 1 * trr_12y * trr_10z;
                        gout[23] += 1 * trr_02y * trr_20z;
                        gout[24] += trr_20x * trr_01y * trr_01z;
                        gout[25] += trr_10x * trr_11y * trr_01z;
                        gout[26] += trr_10x * trr_01y * trr_11z;
                        gout[27] += 1 * trr_21y * trr_01z;
                        gout[28] += 1 * trr_11y * trr_11z;
                        gout[29] += 1 * trr_01y * trr_21z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        gout[30] += trr_20x * fac * trr_02z;
                        gout[31] += trr_10x * trr_10y * trr_02z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        gout[32] += trr_10x * fac * trr_12z;
                        gout[33] += 1 * trr_20y * trr_02z;
                        gout[34] += 1 * trr_10y * trr_12z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        gout[35] += 1 * fac * trr_22z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_01 + gout[12]*dm_jk_02 + gout[18]*dm_jk_03 + gout[24]*dm_jk_04 + gout[30]*dm_jk_05;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_01 + gout[13]*dm_jk_02 + gout[19]*dm_jk_03 + gout[25]*dm_jk_04 + gout[31]*dm_jk_05;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_01 + gout[14]*dm_jk_02 + gout[20]*dm_jk_03 + gout[26]*dm_jk_04 + gout[32]*dm_jk_05;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_01 + gout[15]*dm_jk_02 + gout[21]*dm_jk_03 + gout[27]*dm_jk_04 + gout[33]*dm_jk_05;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_01 + gout[16]*dm_jk_02 + gout[22]*dm_jk_03 + gout[28]*dm_jk_04 + gout[34]*dm_jk_05;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_01 + gout[17]*dm_jk_02 + gout[23]*dm_jk_03 + gout[29]*dm_jk_04 + gout[35]*dm_jk_05;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[6]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[12]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_03 = gout[18]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+3), vik_03);
                    double vik_04 = gout[24]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+4), vik_04);
                    double vik_05 = gout[30]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+5), vik_05);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[7]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[13]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_13 = gout[19]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+3), vik_13);
                    double vik_14 = gout[25]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+4), vik_14);
                    double vik_15 = gout[31]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+5), vik_15);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[8]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[14]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_23 = gout[20]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+3), vik_23);
                    double vik_24 = gout[26]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+4), vik_24);
                    double vik_25 = gout[32]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+5), vik_25);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[9]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[15]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_33 = gout[21]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+3), vik_33);
                    double vik_34 = gout[27]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+4), vik_34);
                    double vik_35 = gout[33]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+5), vik_35);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[10]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[16]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_43 = gout[22]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+3), vik_43);
                    double vik_44 = gout[28]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+4), vik_44);
                    double vik_45 = gout[34]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+5), vik_45);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[17]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_53 = gout[23]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+3), vik_53);
                    double vik_54 = gout[29]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+4), vik_54);
                    double vik_55 = gout[35]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+5), vik_55);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[6] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[12] * dm_ik_02;
                        double dm_ik_03 = dm[(i0+0)*nao+(k0+3)];
                        vjl_00 += gout[18] * dm_ik_03;
                        double dm_ik_04 = dm[(i0+0)*nao+(k0+4)];
                        vjl_00 += gout[24] * dm_ik_04;
                        double dm_ik_05 = dm[(i0+0)*nao+(k0+5)];
                        vjl_00 += gout[30] * dm_ik_05;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[7] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[13] * dm_ik_12;
                        double dm_ik_13 = dm[(i0+1)*nao+(k0+3)];
                        vjl_00 += gout[19] * dm_ik_13;
                        double dm_ik_14 = dm[(i0+1)*nao+(k0+4)];
                        vjl_00 += gout[25] * dm_ik_14;
                        double dm_ik_15 = dm[(i0+1)*nao+(k0+5)];
                        vjl_00 += gout[31] * dm_ik_15;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[8] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[14] * dm_ik_22;
                        double dm_ik_23 = dm[(i0+2)*nao+(k0+3)];
                        vjl_00 += gout[20] * dm_ik_23;
                        double dm_ik_24 = dm[(i0+2)*nao+(k0+4)];
                        vjl_00 += gout[26] * dm_ik_24;
                        double dm_ik_25 = dm[(i0+2)*nao+(k0+5)];
                        vjl_00 += gout[32] * dm_ik_25;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[9] * dm_ik_31;
                        double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[15] * dm_ik_32;
                        double dm_ik_33 = dm[(i0+3)*nao+(k0+3)];
                        vjl_00 += gout[21] * dm_ik_33;
                        double dm_ik_34 = dm[(i0+3)*nao+(k0+4)];
                        vjl_00 += gout[27] * dm_ik_34;
                        double dm_ik_35 = dm[(i0+3)*nao+(k0+5)];
                        vjl_00 += gout[33] * dm_ik_35;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_41;
                        double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[16] * dm_ik_42;
                        double dm_ik_43 = dm[(i0+4)*nao+(k0+3)];
                        vjl_00 += gout[22] * dm_ik_43;
                        double dm_ik_44 = dm[(i0+4)*nao+(k0+4)];
                        vjl_00 += gout[28] * dm_ik_44;
                        double dm_ik_45 = dm[(i0+4)*nao+(k0+5)];
                        vjl_00 += gout[34] * dm_ik_45;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_51;
                        double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[17] * dm_ik_52;
                        double dm_ik_53 = dm[(i0+5)*nao+(k0+3)];
                        vjl_00 += gout[23] * dm_ik_53;
                        double dm_ik_54 = dm[(i0+5)*nao+(k0+4)];
                        vjl_00 += gout[29] * dm_ik_54;
                        double dm_ik_55 = dm[(i0+5)*nao+(k0+5)];
                        vjl_00 += gout[35] * dm_ik_55;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20 + gout[3]*dm_il_30 + gout[4]*dm_il_40 + gout[5]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20 + gout[9]*dm_il_30 + gout[10]*dm_il_40 + gout[11]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[12]*dm_il_00 + gout[13]*dm_il_10 + gout[14]*dm_il_20 + gout[15]*dm_il_30 + gout[16]*dm_il_40 + gout[17]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                        double vjk_03 = gout[18]*dm_il_00 + gout[19]*dm_il_10 + gout[20]*dm_il_20 + gout[21]*dm_il_30 + gout[22]*dm_il_40 + gout[23]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                        double vjk_04 = gout[24]*dm_il_00 + gout[25]*dm_il_10 + gout[26]*dm_il_20 + gout[27]*dm_il_30 + gout[28]*dm_il_40 + gout[29]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                        double vjk_05 = gout[30]*dm_il_00 + gout[31]*dm_il_10 + gout[32]*dm_il_20 + gout[33]*dm_il_30 + gout[34]*dm_il_40 + gout[35]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2021(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int gout_id = item.get_local_id(0);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif
    int t_id = 64 * gout_id + sq_id;
    constexpr int threads = 256;
    constexpr int nsq_per_block = 64;
    constexpr int g_size = 18;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+8);

while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
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
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0] = xlxk;
                rlrk[64] = ylyk;
                rlrk[128] = zlzk;
            }
            double gout[27];

            #pragma unroll
            for (int n = 0; n < 27; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                    int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
                    int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                    int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0];
                    double ylyk = rlrk[64];
                    double zlzk = rlrk[128];
                    double theta_kl = ak * al_akl;
                    double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = PI_FAC;
                    if (task_id < ntasks) {
                        if (ksh == lsh) fac_sym *= .5;
                        if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                    } else {
                        fac_sym = 0;
                    }
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + rjri[0] * aj_aij;
                    double yij = ri[1] + rjri[1] * aj_aij;
                    double zij = ri[2] + rjri[2] * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double s0, s1, s2;
                        double rt = rw[irys*128];
                        double aij = aij_cache[0];
                        double rt_aa = rt / (aij + akl);
                        double akl = akl_cache[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        for (int n = gout_id; n < 3; n += 4) {
                            if (n == 2) {
                                gx[2304] = rw[irys*128+64];
                            }
                            double *_gx = gx + n * 1152;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aij_cache[1];
                            double c0x = Rpa - rt_aij * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[64] = s1;
                            s2 = c0x * s1 + 1 * b10 * s0;
                            _gx[128] = s2;
                            double xlxk = rlrk[n*64];
                            double Rqc = xlxk * akl_cache[64];
                            double cpx = Rqc + rt_akl * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[192] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            _gx[384] = s2;
                            s0 = s1;
                            s1 = s2;
                            s2 = cpx*s1 + 2 * b01 *s0;
                            _gx[576] = s2;
                            s0 = _gx[64];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[256] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 1 * b00 * _gx[192];
                            _gx[448] = s2;
                            s0 = s1;
                            s1 = s2;
                            s2 = cpx*s1 + 2 * b01 *s0;
                            s2 += 1 * b00 * _gx[384];
                            _gx[640] = s2;
                            s0 = _gx[128];
                            s1 = cpx * s0;
                            s1 += 2 * b00 * _gx[64];
                            _gx[320] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 2 * b00 * _gx[256];
                            _gx[512] = s2;
                            s0 = s1;
                            s1 = s2;
                            s2 = cpx*s1 + 2 * b01 *s0;
                            s2 += 2 * b00 * _gx[448];
                            _gx[704] = s2;
                            s1 = _gx[576];
                            s0 = _gx[384];
                            _gx[960] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[192];
                            _gx[768] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[0];
                            _gx[576] = s1 - xlxk * s0;
                            s1 = _gx[640];
                            s0 = _gx[448];
                            _gx[1024] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[256];
                            _gx[832] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[64];
                            _gx[640] = s1 - xlxk * s0;
                            s1 = _gx[704];
                            s0 = _gx[512];
                            _gx[1088] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[320];
                            _gx[896] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[128];
                            _gx[704] = s1 - xlxk * s0;
                        }
                        __syncthreads();
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[1088] * gx[1152] * gx[2304];
                        gout[1] += gx[960] * gx[1216] * gx[2368];
                        gout[2] += gx[832] * gx[1344] * gx[2368];
                        gout[3] += gx[896] * gx[1152] * gx[2496];
                        gout[4] += gx[768] * gx[1216] * gx[2560];
                        gout[5] += gx[640] * gx[1536] * gx[2368];
                        gout[6] += gx[704] * gx[1344] * gx[2496];
                        gout[7] += gx[576] * gx[1408] * gx[2560];
                        gout[8] += gx[640] * gx[1152] * gx[2752];
                        gout[9] += gx[512] * gx[1728] * gx[2304];
                        gout[10] += gx[384] * gx[1792] * gx[2368];
                        gout[11] += gx[256] * gx[1920] * gx[2368];
                        gout[12] += gx[320] * gx[1728] * gx[2496];
                        gout[13] += gx[192] * gx[1792] * gx[2560];
                        gout[14] += gx[64] * gx[2112] * gx[2368];
                        gout[15] += gx[128] * gx[1920] * gx[2496];
                        gout[16] += gx[0] * gx[1984] * gx[2560];
                        gout[17] += gx[64] * gx[1728] * gx[2752];
                        gout[18] += gx[512] * gx[1152] * gx[2880];
                        gout[19] += gx[384] * gx[1216] * gx[2944];
                        gout[20] += gx[256] * gx[1344] * gx[2944];
                        gout[21] += gx[320] * gx[1152] * gx[3072];
                        gout[22] += gx[192] * gx[1216] * gx[3136];
                        gout[23] += gx[64] * gx[1536] * gx[2944];
                        gout[24] += gx[128] * gx[1344] * gx[3072];
                        gout[25] += gx[0] * gx[1408] * gx[3136];
                        gout[26] += gx[64] * gx[1152] * gx[3328];
                        break;
                        case 1:
                        gout[0] += gx[1024] * gx[1216] * gx[2304];
                        gout[1] += gx[960] * gx[1152] * gx[2432];
                        gout[2] += gx[768] * gx[1472] * gx[2304];
                        gout[3] += gx[832] * gx[1216] * gx[2496];
                        gout[4] += gx[768] * gx[1152] * gx[2624];
                        gout[5] += gx[576] * gx[1664] * gx[2304];
                        gout[6] += gx[640] * gx[1408] * gx[2496];
                        gout[7] += gx[576] * gx[1344] * gx[2624];
                        gout[8] += gx[576] * gx[1280] * gx[2688];
                        gout[9] += gx[448] * gx[1792] * gx[2304];
                        gout[10] += gx[384] * gx[1728] * gx[2432];
                        gout[11] += gx[192] * gx[2048] * gx[2304];
                        gout[12] += gx[256] * gx[1792] * gx[2496];
                        gout[13] += gx[192] * gx[1728] * gx[2624];
                        gout[14] += gx[0] * gx[2240] * gx[2304];
                        gout[15] += gx[64] * gx[1984] * gx[2496];
                        gout[16] += gx[0] * gx[1920] * gx[2624];
                        gout[17] += gx[0] * gx[1856] * gx[2688];
                        gout[18] += gx[448] * gx[1216] * gx[2880];
                        gout[19] += gx[384] * gx[1152] * gx[3008];
                        gout[20] += gx[192] * gx[1472] * gx[2880];
                        gout[21] += gx[256] * gx[1216] * gx[3072];
                        gout[22] += gx[192] * gx[1152] * gx[3200];
                        gout[23] += gx[0] * gx[1664] * gx[2880];
                        gout[24] += gx[64] * gx[1408] * gx[3072];
                        gout[25] += gx[0] * gx[1344] * gx[3200];
                        gout[26] += gx[0] * gx[1280] * gx[3264];
                        break;
                        case 2:
                        gout[0] += gx[1024] * gx[1152] * gx[2368];
                        gout[1] += gx[896] * gx[1344] * gx[2304];
                        gout[2] += gx[768] * gx[1408] * gx[2368];
                        gout[3] += gx[832] * gx[1152] * gx[2560];
                        gout[4] += gx[704] * gx[1536] * gx[2304];
                        gout[5] += gx[576] * gx[1600] * gx[2368];
                        gout[6] += gx[640] * gx[1344] * gx[2560];
                        gout[7] += gx[704] * gx[1152] * gx[2688];
                        gout[8] += gx[576] * gx[1216] * gx[2752];
                        gout[9] += gx[448] * gx[1728] * gx[2368];
                        gout[10] += gx[320] * gx[1920] * gx[2304];
                        gout[11] += gx[192] * gx[1984] * gx[2368];
                        gout[12] += gx[256] * gx[1728] * gx[2560];
                        gout[13] += gx[128] * gx[2112] * gx[2304];
                        gout[14] += gx[0] * gx[2176] * gx[2368];
                        gout[15] += gx[64] * gx[1920] * gx[2560];
                        gout[16] += gx[128] * gx[1728] * gx[2688];
                        gout[17] += gx[0] * gx[1792] * gx[2752];
                        gout[18] += gx[448] * gx[1152] * gx[2944];
                        gout[19] += gx[320] * gx[1344] * gx[2880];
                        gout[20] += gx[192] * gx[1408] * gx[2944];
                        gout[21] += gx[256] * gx[1152] * gx[3136];
                        gout[22] += gx[128] * gx[1536] * gx[2880];
                        gout[23] += gx[0] * gx[1600] * gx[2944];
                        gout[24] += gx[64] * gx[1344] * gx[3136];
                        gout[25] += gx[128] * gx[1152] * gx[3264];
                        gout[26] += gx[0] * gx[1216] * gx[3328];
                        break;
                        case 3:
                        gout[0] += gx[960] * gx[1280] * gx[2304];
                        gout[1] += gx[832] * gx[1408] * gx[2304];
                        gout[2] += gx[768] * gx[1344] * gx[2432];
                        gout[3] += gx[768] * gx[1280] * gx[2496];
                        gout[4] += gx[640] * gx[1600] * gx[2304];
                        gout[5] += gx[576] * gx[1536] * gx[2432];
                        gout[6] += gx[576] * gx[1472] * gx[2496];
                        gout[7] += gx[640] * gx[1216] * gx[2688];
                        gout[8] += gx[576] * gx[1152] * gx[2816];
                        gout[9] += gx[384] * gx[1856] * gx[2304];
                        gout[10] += gx[256] * gx[1984] * gx[2304];
                        gout[11] += gx[192] * gx[1920] * gx[2432];
                        gout[12] += gx[192] * gx[1856] * gx[2496];
                        gout[13] += gx[64] * gx[2176] * gx[2304];
                        gout[14] += gx[0] * gx[2112] * gx[2432];
                        gout[15] += gx[0] * gx[2048] * gx[2496];
                        gout[16] += gx[64] * gx[1792] * gx[2688];
                        gout[17] += gx[0] * gx[1728] * gx[2816];
                        gout[18] += gx[384] * gx[1280] * gx[2880];
                        gout[19] += gx[256] * gx[1408] * gx[2880];
                        gout[20] += gx[192] * gx[1344] * gx[3008];
                        gout[21] += gx[192] * gx[1280] * gx[3072];
                        gout[22] += gx[64] * gx[1600] * gx[2880];
                        gout[23] += gx[0] * gx[1536] * gx[3008];
                        gout[24] += gx[0] * gx[1472] * gx[3072];
                        gout[25] += gx[64] * gx[1216] * gx[3264];
                        gout[26] += gx[0] * gx[1152] * gx[3392];
                        break;
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao * nao;
                    double *vk = kmat.vk + i_dm * nao * nao;
                    switch (gout_id) {
                    case 0: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_02 + gout[6]*dm_jk_04;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_01 = gout[9]*dm_jk_00 + gout[12]*dm_jk_02 + gout[15]*dm_jk_04;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[18]*dm_jk_00 + gout[21]*dm_jk_02 + gout[24]*dm_jk_04;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    double vil_20 = gout[2]*dm_jk_01 + gout[5]*dm_jk_03 + gout[8]*dm_jk_05;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[11]*dm_jk_01 + gout[14]*dm_jk_03 + gout[17]*dm_jk_05;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[20]*dm_jk_01 + gout[23]*dm_jk_03 + gout[26]*dm_jk_05;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[1]*dm_jk_00 + gout[4]*dm_jk_02 + gout[7]*dm_jk_04;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[10]*dm_jk_00 + gout[13]*dm_jk_02 + gout[16]*dm_jk_04;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[19]*dm_jk_00 + gout[22]*dm_jk_02 + gout[25]*dm_jk_04;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    break; }
                    case 1: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    double vil_10 = gout[0]*dm_jk_00 + gout[3]*dm_jk_02 + gout[6]*dm_jk_04;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_11 = gout[9]*dm_jk_00 + gout[12]*dm_jk_02 + gout[15]*dm_jk_04;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[18]*dm_jk_00 + gout[21]*dm_jk_02 + gout[24]*dm_jk_04;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    double vil_30 = gout[2]*dm_jk_01 + gout[5]*dm_jk_03 + gout[8]*dm_jk_05;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[11]*dm_jk_01 + gout[14]*dm_jk_03 + gout[17]*dm_jk_05;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[20]*dm_jk_01 + gout[23]*dm_jk_03 + gout[26]*dm_jk_05;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[1]*dm_jk_00 + gout[4]*dm_jk_02 + gout[7]*dm_jk_04;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[10]*dm_jk_00 + gout[13]*dm_jk_02 + gout[16]*dm_jk_04;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[19]*dm_jk_00 + gout[22]*dm_jk_02 + gout[25]*dm_jk_04;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    break; }
                    case 2: {
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    double vil_00 = gout[1]*dm_jk_01 + gout[4]*dm_jk_03 + gout[7]*dm_jk_05;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_01 = gout[10]*dm_jk_01 + gout[13]*dm_jk_03 + gout[16]*dm_jk_05;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[19]*dm_jk_01 + gout[22]*dm_jk_03 + gout[25]*dm_jk_05;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    double vil_20 = gout[0]*dm_jk_00 + gout[3]*dm_jk_02 + gout[6]*dm_jk_04;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[9]*dm_jk_00 + gout[12]*dm_jk_02 + gout[15]*dm_jk_04;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[18]*dm_jk_00 + gout[21]*dm_jk_02 + gout[24]*dm_jk_04;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[2]*dm_jk_01 + gout[5]*dm_jk_03 + gout[8]*dm_jk_05;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[11]*dm_jk_01 + gout[14]*dm_jk_03 + gout[17]*dm_jk_05;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[20]*dm_jk_01 + gout[23]*dm_jk_03 + gout[26]*dm_jk_05;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    break; }
                    case 3: {
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    double vil_10 = gout[1]*dm_jk_01 + gout[4]*dm_jk_03 + gout[7]*dm_jk_05;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_11 = gout[10]*dm_jk_01 + gout[13]*dm_jk_03 + gout[16]*dm_jk_05;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[19]*dm_jk_01 + gout[22]*dm_jk_03 + gout[25]*dm_jk_05;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    double vil_30 = gout[0]*dm_jk_00 + gout[3]*dm_jk_02 + gout[6]*dm_jk_04;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[9]*dm_jk_00 + gout[12]*dm_jk_02 + gout[15]*dm_jk_04;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[18]*dm_jk_00 + gout[21]*dm_jk_02 + gout[24]*dm_jk_04;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[2]*dm_jk_01 + gout[5]*dm_jk_03 + gout[8]*dm_jk_05;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[11]*dm_jk_01 + gout[14]*dm_jk_03 + gout[17]*dm_jk_05;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[20]*dm_jk_01 + gout[23]*dm_jk_03 + gout[26]*dm_jk_05;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    break; }
                    }
                    switch (gout_id) {
                    case 0: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[9]*dm_jl_01 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_02 = gout[3]*dm_jl_00 + gout[12]*dm_jl_01 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_04 = gout[6]*dm_jl_00 + gout[15]*dm_jl_01 + gout[24]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+4), vik_04);
                    double vik_21 = gout[2]*dm_jl_00 + gout[11]*dm_jl_01 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_23 = gout[5]*dm_jl_00 + gout[14]*dm_jl_01 + gout[23]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+3), vik_23);
                    double vik_25 = gout[8]*dm_jl_00 + gout[17]*dm_jl_01 + gout[26]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+5), vik_25);
                    double vik_40 = gout[1]*dm_jl_00 + gout[10]*dm_jl_01 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_42 = gout[4]*dm_jl_00 + gout[13]*dm_jl_01 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_44 = gout[7]*dm_jl_00 + gout[16]*dm_jl_01 + gout[25]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+4), vik_44);
                    break; }
                    case 1: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_10 = gout[0]*dm_jl_00 + gout[9]*dm_jl_01 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_12 = gout[3]*dm_jl_00 + gout[12]*dm_jl_01 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_14 = gout[6]*dm_jl_00 + gout[15]*dm_jl_01 + gout[24]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+4), vik_14);
                    double vik_31 = gout[2]*dm_jl_00 + gout[11]*dm_jl_01 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_33 = gout[5]*dm_jl_00 + gout[14]*dm_jl_01 + gout[23]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+3), vik_33);
                    double vik_35 = gout[8]*dm_jl_00 + gout[17]*dm_jl_01 + gout[26]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+5), vik_35);
                    double vik_50 = gout[1]*dm_jl_00 + gout[10]*dm_jl_01 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_52 = gout[4]*dm_jl_00 + gout[13]*dm_jl_01 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_54 = gout[7]*dm_jl_00 + gout[16]*dm_jl_01 + gout[25]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+4), vik_54);
                    break; }
                    case 2: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_01 = gout[1]*dm_jl_00 + gout[10]*dm_jl_01 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_03 = gout[4]*dm_jl_00 + gout[13]*dm_jl_01 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+3), vik_03);
                    double vik_05 = gout[7]*dm_jl_00 + gout[16]*dm_jl_01 + gout[25]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+5), vik_05);
                    double vik_20 = gout[0]*dm_jl_00 + gout[9]*dm_jl_01 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_22 = gout[3]*dm_jl_00 + gout[12]*dm_jl_01 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_24 = gout[6]*dm_jl_00 + gout[15]*dm_jl_01 + gout[24]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+4), vik_24);
                    double vik_41 = gout[2]*dm_jl_00 + gout[11]*dm_jl_01 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_43 = gout[5]*dm_jl_00 + gout[14]*dm_jl_01 + gout[23]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+3), vik_43);
                    double vik_45 = gout[8]*dm_jl_00 + gout[17]*dm_jl_01 + gout[26]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+5), vik_45);
                    break; }
                    case 3: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_11 = gout[1]*dm_jl_00 + gout[10]*dm_jl_01 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_13 = gout[4]*dm_jl_00 + gout[13]*dm_jl_01 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+3), vik_13);
                    double vik_15 = gout[7]*dm_jl_00 + gout[16]*dm_jl_01 + gout[25]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+5), vik_15);
                    double vik_30 = gout[0]*dm_jl_00 + gout[9]*dm_jl_01 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_32 = gout[3]*dm_jl_00 + gout[12]*dm_jl_01 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_34 = gout[6]*dm_jl_00 + gout[15]*dm_jl_01 + gout[24]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+4), vik_34);
                    double vik_51 = gout[2]*dm_jl_00 + gout[11]*dm_jl_01 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_53 = gout[5]*dm_jl_00 + gout[14]*dm_jl_01 + gout[23]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+3), vik_53);
                    double vik_55 = gout[8]*dm_jl_00 + gout[17]*dm_jl_01 + gout[26]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+5), vik_55);
                    break; }
                    }
                    if (ish != jsh) {
                    double vjl_00 = 0;
                    double vjl_01 = 0;
                    double vjl_02 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_00;
                    vjl_01 += gout[9] * dm_ik_00;
                    vjl_02 += gout[18] * dm_ik_00;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_00 += gout[3] * dm_ik_02;
                    vjl_01 += gout[12] * dm_ik_02;
                    vjl_02 += gout[21] * dm_ik_02;
                    double dm_ik_04 = dm[(i0+0)*nao+(k0+4)];
                    vjl_00 += gout[6] * dm_ik_04;
                    vjl_01 += gout[15] * dm_ik_04;
                    vjl_02 += gout[24] * dm_ik_04;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_21;
                    vjl_01 += gout[11] * dm_ik_21;
                    vjl_02 += gout[20] * dm_ik_21;
                    double dm_ik_23 = dm[(i0+2)*nao+(k0+3)];
                    vjl_00 += gout[5] * dm_ik_23;
                    vjl_01 += gout[14] * dm_ik_23;
                    vjl_02 += gout[23] * dm_ik_23;
                    double dm_ik_25 = dm[(i0+2)*nao+(k0+5)];
                    vjl_00 += gout[8] * dm_ik_25;
                    vjl_01 += gout[17] * dm_ik_25;
                    vjl_02 += gout[26] * dm_ik_25;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_40;
                    vjl_01 += gout[10] * dm_ik_40;
                    vjl_02 += gout[19] * dm_ik_40;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_00 += gout[4] * dm_ik_42;
                    vjl_01 += gout[13] * dm_ik_42;
                    vjl_02 += gout[22] * dm_ik_42;
                    double dm_ik_44 = dm[(i0+4)*nao+(k0+4)];
                    vjl_00 += gout[7] * dm_ik_44;
                    vjl_01 += gout[16] * dm_ik_44;
                    vjl_02 += gout[25] * dm_ik_44;
                    break; }
                    case 1: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_10;
                    vjl_01 += gout[9] * dm_ik_10;
                    vjl_02 += gout[18] * dm_ik_10;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_00 += gout[3] * dm_ik_12;
                    vjl_01 += gout[12] * dm_ik_12;
                    vjl_02 += gout[21] * dm_ik_12;
                    double dm_ik_14 = dm[(i0+1)*nao+(k0+4)];
                    vjl_00 += gout[6] * dm_ik_14;
                    vjl_01 += gout[15] * dm_ik_14;
                    vjl_02 += gout[24] * dm_ik_14;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_31;
                    vjl_01 += gout[11] * dm_ik_31;
                    vjl_02 += gout[20] * dm_ik_31;
                    double dm_ik_33 = dm[(i0+3)*nao+(k0+3)];
                    vjl_00 += gout[5] * dm_ik_33;
                    vjl_01 += gout[14] * dm_ik_33;
                    vjl_02 += gout[23] * dm_ik_33;
                    double dm_ik_35 = dm[(i0+3)*nao+(k0+5)];
                    vjl_00 += gout[8] * dm_ik_35;
                    vjl_01 += gout[17] * dm_ik_35;
                    vjl_02 += gout[26] * dm_ik_35;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_50;
                    vjl_01 += gout[10] * dm_ik_50;
                    vjl_02 += gout[19] * dm_ik_50;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_00 += gout[4] * dm_ik_52;
                    vjl_01 += gout[13] * dm_ik_52;
                    vjl_02 += gout[22] * dm_ik_52;
                    double dm_ik_54 = dm[(i0+5)*nao+(k0+4)];
                    vjl_00 += gout[7] * dm_ik_54;
                    vjl_01 += gout[16] * dm_ik_54;
                    vjl_02 += gout[25] * dm_ik_54;
                    break; }
                    case 2: {
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_00 += gout[1] * dm_ik_01;
                    vjl_01 += gout[10] * dm_ik_01;
                    vjl_02 += gout[19] * dm_ik_01;
                    double dm_ik_03 = dm[(i0+0)*nao+(k0+3)];
                    vjl_00 += gout[4] * dm_ik_03;
                    vjl_01 += gout[13] * dm_ik_03;
                    vjl_02 += gout[22] * dm_ik_03;
                    double dm_ik_05 = dm[(i0+0)*nao+(k0+5)];
                    vjl_00 += gout[7] * dm_ik_05;
                    vjl_01 += gout[16] * dm_ik_05;
                    vjl_02 += gout[25] * dm_ik_05;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_20;
                    vjl_01 += gout[9] * dm_ik_20;
                    vjl_02 += gout[18] * dm_ik_20;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_00 += gout[3] * dm_ik_22;
                    vjl_01 += gout[12] * dm_ik_22;
                    vjl_02 += gout[21] * dm_ik_22;
                    double dm_ik_24 = dm[(i0+2)*nao+(k0+4)];
                    vjl_00 += gout[6] * dm_ik_24;
                    vjl_01 += gout[15] * dm_ik_24;
                    vjl_02 += gout[24] * dm_ik_24;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_41;
                    vjl_01 += gout[11] * dm_ik_41;
                    vjl_02 += gout[20] * dm_ik_41;
                    double dm_ik_43 = dm[(i0+4)*nao+(k0+3)];
                    vjl_00 += gout[5] * dm_ik_43;
                    vjl_01 += gout[14] * dm_ik_43;
                    vjl_02 += gout[23] * dm_ik_43;
                    double dm_ik_45 = dm[(i0+4)*nao+(k0+5)];
                    vjl_00 += gout[8] * dm_ik_45;
                    vjl_01 += gout[17] * dm_ik_45;
                    vjl_02 += gout[26] * dm_ik_45;
                    break; }
                    case 3: {
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_00 += gout[1] * dm_ik_11;
                    vjl_01 += gout[10] * dm_ik_11;
                    vjl_02 += gout[19] * dm_ik_11;
                    double dm_ik_13 = dm[(i0+1)*nao+(k0+3)];
                    vjl_00 += gout[4] * dm_ik_13;
                    vjl_01 += gout[13] * dm_ik_13;
                    vjl_02 += gout[22] * dm_ik_13;
                    double dm_ik_15 = dm[(i0+1)*nao+(k0+5)];
                    vjl_00 += gout[7] * dm_ik_15;
                    vjl_01 += gout[16] * dm_ik_15;
                    vjl_02 += gout[25] * dm_ik_15;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_30;
                    vjl_01 += gout[9] * dm_ik_30;
                    vjl_02 += gout[18] * dm_ik_30;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_00 += gout[3] * dm_ik_32;
                    vjl_01 += gout[12] * dm_ik_32;
                    vjl_02 += gout[21] * dm_ik_32;
                    double dm_ik_34 = dm[(i0+3)*nao+(k0+4)];
                    vjl_00 += gout[6] * dm_ik_34;
                    vjl_01 += gout[15] * dm_ik_34;
                    vjl_02 += gout[24] * dm_ik_34;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_51;
                    vjl_01 += gout[11] * dm_ik_51;
                    vjl_02 += gout[20] * dm_ik_51;
                    double dm_ik_53 = dm[(i0+5)*nao+(k0+3)];
                    vjl_00 += gout[5] * dm_ik_53;
                    vjl_01 += gout[14] * dm_ik_53;
                    vjl_02 += gout[23] * dm_ik_53;
                    double dm_ik_55 = dm[(i0+5)*nao+(k0+5)];
                    vjl_00 += gout[8] * dm_ik_55;
                    vjl_01 += gout[17] * dm_ik_55;
                    vjl_02 += gout[26] * dm_ik_55;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                    atomicAdd(vk+(j0+0)*nao+(l0+1), vjl_01);
                    atomicAdd(vk+(j0+0)*nao+(l0+2), vjl_02);
                    double vjk_00 = 0;
                    double vjk_01 = 0;
                    double vjk_02 = 0;
                    double vjk_03 = 0;
                    double vjk_04 = 0;
                    double vjk_05 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_00;
                    vjk_02 += gout[3] * dm_il_00;
                    vjk_04 += gout[6] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_01;
                    vjk_02 += gout[12] * dm_il_01;
                    vjk_04 += gout[15] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_00 += gout[18] * dm_il_02;
                    vjk_02 += gout[21] * dm_il_02;
                    vjk_04 += gout[24] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_20;
                    vjk_03 += gout[5] * dm_il_20;
                    vjk_05 += gout[8] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_21;
                    vjk_03 += gout[14] * dm_il_21;
                    vjk_05 += gout[17] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_01 += gout[20] * dm_il_22;
                    vjk_03 += gout[23] * dm_il_22;
                    vjk_05 += gout[26] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_40;
                    vjk_02 += gout[4] * dm_il_40;
                    vjk_04 += gout[7] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_00 += gout[10] * dm_il_41;
                    vjk_02 += gout[13] * dm_il_41;
                    vjk_04 += gout[16] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_00 += gout[19] * dm_il_42;
                    vjk_02 += gout[22] * dm_il_42;
                    vjk_04 += gout[25] * dm_il_42;
                    break; }
                    case 1: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_10;
                    vjk_02 += gout[3] * dm_il_10;
                    vjk_04 += gout[6] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_11;
                    vjk_02 += gout[12] * dm_il_11;
                    vjk_04 += gout[15] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_00 += gout[18] * dm_il_12;
                    vjk_02 += gout[21] * dm_il_12;
                    vjk_04 += gout[24] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_30;
                    vjk_03 += gout[5] * dm_il_30;
                    vjk_05 += gout[8] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_31;
                    vjk_03 += gout[14] * dm_il_31;
                    vjk_05 += gout[17] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_01 += gout[20] * dm_il_32;
                    vjk_03 += gout[23] * dm_il_32;
                    vjk_05 += gout[26] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_50;
                    vjk_02 += gout[4] * dm_il_50;
                    vjk_04 += gout[7] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_00 += gout[10] * dm_il_51;
                    vjk_02 += gout[13] * dm_il_51;
                    vjk_04 += gout[16] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_00 += gout[19] * dm_il_52;
                    vjk_02 += gout[22] * dm_il_52;
                    vjk_04 += gout[25] * dm_il_52;
                    break; }
                    case 2: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_01 += gout[1] * dm_il_00;
                    vjk_03 += gout[4] * dm_il_00;
                    vjk_05 += gout[7] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_01 += gout[10] * dm_il_01;
                    vjk_03 += gout[13] * dm_il_01;
                    vjk_05 += gout[16] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_01 += gout[19] * dm_il_02;
                    vjk_03 += gout[22] * dm_il_02;
                    vjk_05 += gout[25] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_20;
                    vjk_02 += gout[3] * dm_il_20;
                    vjk_04 += gout[6] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_21;
                    vjk_02 += gout[12] * dm_il_21;
                    vjk_04 += gout[15] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_00 += gout[18] * dm_il_22;
                    vjk_02 += gout[21] * dm_il_22;
                    vjk_04 += gout[24] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_40;
                    vjk_03 += gout[5] * dm_il_40;
                    vjk_05 += gout[8] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_41;
                    vjk_03 += gout[14] * dm_il_41;
                    vjk_05 += gout[17] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_01 += gout[20] * dm_il_42;
                    vjk_03 += gout[23] * dm_il_42;
                    vjk_05 += gout[26] * dm_il_42;
                    break; }
                    case 3: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_01 += gout[1] * dm_il_10;
                    vjk_03 += gout[4] * dm_il_10;
                    vjk_05 += gout[7] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_01 += gout[10] * dm_il_11;
                    vjk_03 += gout[13] * dm_il_11;
                    vjk_05 += gout[16] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_01 += gout[19] * dm_il_12;
                    vjk_03 += gout[22] * dm_il_12;
                    vjk_05 += gout[25] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_30;
                    vjk_02 += gout[3] * dm_il_30;
                    vjk_04 += gout[6] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_31;
                    vjk_02 += gout[12] * dm_il_31;
                    vjk_04 += gout[15] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_00 += gout[18] * dm_il_32;
                    vjk_02 += gout[21] * dm_il_32;
                    vjk_04 += gout[24] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_50;
                    vjk_03 += gout[5] * dm_il_50;
                    vjk_05 += gout[8] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_51;
                    vjk_03 += gout[14] * dm_il_51;
                    vjk_05 += gout[17] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_01 += gout[20] * dm_il_52;
                    vjk_03 += gout[23] * dm_il_52;
                    vjk_05 += gout[26] * dm_il_52;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                    atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                    atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2100(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
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
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_10 + gout[12]*dm_jk_20;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_10 + gout[13]*dm_jk_20;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_10 + gout[14]*dm_jk_20;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_10 + gout[15]*dm_jk_20;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_10 + gout[16]*dm_jk_20;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_10 + gout[17]*dm_jk_20;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[6]*dm_jl_10 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[7]*dm_jl_10 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[8]*dm_jl_10 + gout[14]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00 + gout[9]*dm_jl_10 + gout[15]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00 + gout[10]*dm_jl_10 + gout[16]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00 + gout[11]*dm_jl_10 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[6] * dm_ik_00;
                        vjl_20 += gout[12] * dm_ik_00;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[7] * dm_ik_10;
                        vjl_20 += gout[13] * dm_ik_10;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[8] * dm_ik_20;
                        vjl_20 += gout[14] * dm_ik_20;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        vjl_10 += gout[9] * dm_ik_30;
                        vjl_20 += gout[15] * dm_ik_30;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        vjl_10 += gout[10] * dm_ik_40;
                        vjl_20 += gout[16] * dm_ik_40;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        vjl_10 += gout[11] * dm_ik_50;
                        vjl_20 += gout[17] * dm_ik_50;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double vjk_00 = 0;
                        double vjk_10 = 0;
                        double vjk_20 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_10 += gout[6] * dm_il_00;
                        vjk_20 += gout[12] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_10 += gout[7] * dm_il_10;
                        vjk_20 += gout[13] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_10 += gout[8] * dm_il_20;
                        vjk_20 += gout[14] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_10 += gout[9] * dm_il_30;
                        vjk_20 += gout[15] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_10 += gout[10] * dm_il_40;
                        vjk_20 += gout[16] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_10 += gout[11] * dm_il_50;
                        vjk_20 += gout[17] * dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2110(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[54];
#pragma unroll
            for (int n = 0; n < 54; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double hrr_2110x = trr_31x - xjxi * trr_21x;
                        gout[0] += hrr_2110x * fac * wt;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double hrr_1110x = trr_21x - xjxi * trr_11x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_1110x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_1110x * fac * trr_10z;
                        double trr_01x = cpx * 1;
                        double hrr_0110x = trr_11x - xjxi * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += hrr_0110x * trr_20y * wt;
                        gout[4] += hrr_0110x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += hrr_0110x * fac * trr_20z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[6] += trr_21x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[7] += trr_11x * hrr_1100y * wt;
                        gout[8] += trr_11x * hrr_0100y * trr_10z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout[9] += trr_01x * hrr_2100y * wt;
                        gout[10] += trr_01x * hrr_1100y * trr_10z;
                        gout[11] += trr_01x * hrr_0100y * trr_20z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[12] += trr_21x * fac * hrr_0100z;
                        gout[13] += trr_11x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[14] += trr_11x * fac * hrr_1100z;
                        gout[15] += trr_01x * trr_20y * hrr_0100z;
                        gout[16] += trr_01x * trr_10y * hrr_1100z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout[17] += trr_01x * fac * hrr_2100z;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[18] += hrr_2100x * trr_01y * wt;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[19] += hrr_1100x * trr_11y * wt;
                        gout[20] += hrr_1100x * trr_01y * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout[21] += hrr_0100x * trr_21y * wt;
                        gout[22] += hrr_0100x * trr_11y * trr_10z;
                        gout[23] += hrr_0100x * trr_01y * trr_20z;
                        double hrr_0110y = trr_11y - yjyi * trr_01y;
                        gout[24] += trr_20x * hrr_0110y * wt;
                        double hrr_1110y = trr_21y - yjyi * trr_11y;
                        gout[25] += trr_10x * hrr_1110y * wt;
                        gout[26] += trr_10x * hrr_0110y * trr_10z;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        double hrr_2110y = trr_31y - yjyi * trr_21y;
                        gout[27] += 1 * hrr_2110y * wt;
                        gout[28] += 1 * hrr_1110y * trr_10z;
                        gout[29] += 1 * hrr_0110y * trr_20z;
                        gout[30] += trr_20x * trr_01y * hrr_0100z;
                        gout[31] += trr_10x * trr_11y * hrr_0100z;
                        gout[32] += trr_10x * trr_01y * hrr_1100z;
                        gout[33] += 1 * trr_21y * hrr_0100z;
                        gout[34] += 1 * trr_11y * hrr_1100z;
                        gout[35] += 1 * trr_01y * hrr_2100z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[36] += hrr_2100x * fac * trr_01z;
                        gout[37] += hrr_1100x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[38] += hrr_1100x * fac * trr_11z;
                        gout[39] += hrr_0100x * trr_20y * trr_01z;
                        gout[40] += hrr_0100x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout[41] += hrr_0100x * fac * trr_21z;
                        gout[42] += trr_20x * hrr_0100y * trr_01z;
                        gout[43] += trr_10x * hrr_1100y * trr_01z;
                        gout[44] += trr_10x * hrr_0100y * trr_11z;
                        gout[45] += 1 * hrr_2100y * trr_01z;
                        gout[46] += 1 * hrr_1100y * trr_11z;
                        gout[47] += 1 * hrr_0100y * trr_21z;
                        double hrr_0110z = trr_11z - zjzi * trr_01z;
                        gout[48] += trr_20x * fac * hrr_0110z;
                        gout[49] += trr_10x * trr_10y * hrr_0110z;
                        double hrr_1110z = trr_21z - zjzi * trr_11z;
                        gout[50] += trr_10x * fac * hrr_1110z;
                        gout[51] += 1 * trr_20y * hrr_0110z;
                        gout[52] += 1 * trr_10y * hrr_1110z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        double hrr_2110z = trr_31z - zjzi * trr_21z;
                        gout[53] += 1 * fac * hrr_2110z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double vil_00 = 0;
                    double vil_10 = 0;
                    double vil_20 = 0;
                    double vil_30 = 0;
                    double vil_40 = 0;
                    double vil_50 = 0;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_00 += gout[0] * dm_jk_00;
                    vil_10 += gout[1] * dm_jk_00;
                    vil_20 += gout[2] * dm_jk_00;
                    vil_30 += gout[3] * dm_jk_00;
                    vil_40 += gout[4] * dm_jk_00;
                    vil_50 += gout[5] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_00 += gout[18] * dm_jk_01;
                    vil_10 += gout[19] * dm_jk_01;
                    vil_20 += gout[20] * dm_jk_01;
                    vil_30 += gout[21] * dm_jk_01;
                    vil_40 += gout[22] * dm_jk_01;
                    vil_50 += gout[23] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_00 += gout[36] * dm_jk_02;
                    vil_10 += gout[37] * dm_jk_02;
                    vil_20 += gout[38] * dm_jk_02;
                    vil_30 += gout[39] * dm_jk_02;
                    vil_40 += gout[40] * dm_jk_02;
                    vil_50 += gout[41] * dm_jk_02;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_00 += gout[6] * dm_jk_10;
                    vil_10 += gout[7] * dm_jk_10;
                    vil_20 += gout[8] * dm_jk_10;
                    vil_30 += gout[9] * dm_jk_10;
                    vil_40 += gout[10] * dm_jk_10;
                    vil_50 += gout[11] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_00 += gout[24] * dm_jk_11;
                    vil_10 += gout[25] * dm_jk_11;
                    vil_20 += gout[26] * dm_jk_11;
                    vil_30 += gout[27] * dm_jk_11;
                    vil_40 += gout[28] * dm_jk_11;
                    vil_50 += gout[29] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_00 += gout[42] * dm_jk_12;
                    vil_10 += gout[43] * dm_jk_12;
                    vil_20 += gout[44] * dm_jk_12;
                    vil_30 += gout[45] * dm_jk_12;
                    vil_40 += gout[46] * dm_jk_12;
                    vil_50 += gout[47] * dm_jk_12;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_00 += gout[12] * dm_jk_20;
                    vil_10 += gout[13] * dm_jk_20;
                    vil_20 += gout[14] * dm_jk_20;
                    vil_30 += gout[15] * dm_jk_20;
                    vil_40 += gout[16] * dm_jk_20;
                    vil_50 += gout[17] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_00 += gout[30] * dm_jk_21;
                    vil_10 += gout[31] * dm_jk_21;
                    vil_20 += gout[32] * dm_jk_21;
                    vil_30 += gout[33] * dm_jk_21;
                    vil_40 += gout[34] * dm_jk_21;
                    vil_50 += gout[35] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_00 += gout[48] * dm_jk_22;
                    vil_10 += gout[49] * dm_jk_22;
                    vil_20 += gout[50] * dm_jk_22;
                    vil_30 += gout[51] * dm_jk_22;
                    vil_40 += gout[52] * dm_jk_22;
                    vil_50 += gout[53] * dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[6]*dm_jl_10 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[18]*dm_jl_00 + gout[24]*dm_jl_10 + gout[30]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[36]*dm_jl_00 + gout[42]*dm_jl_10 + gout[48]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00 + gout[7]*dm_jl_10 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[19]*dm_jl_00 + gout[25]*dm_jl_10 + gout[31]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[37]*dm_jl_00 + gout[43]*dm_jl_10 + gout[49]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00 + gout[8]*dm_jl_10 + gout[14]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[20]*dm_jl_00 + gout[26]*dm_jl_10 + gout[32]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[38]*dm_jl_00 + gout[44]*dm_jl_10 + gout[50]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_30 = gout[3]*dm_jl_00 + gout[9]*dm_jl_10 + gout[15]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[21]*dm_jl_00 + gout[27]*dm_jl_10 + gout[33]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[39]*dm_jl_00 + gout[45]*dm_jl_10 + gout[51]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_40 = gout[4]*dm_jl_00 + gout[10]*dm_jl_10 + gout[16]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[22]*dm_jl_00 + gout[28]*dm_jl_10 + gout[34]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[40]*dm_jl_00 + gout[46]*dm_jl_10 + gout[52]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_50 = gout[5]*dm_jl_00 + gout[11]*dm_jl_10 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[23]*dm_jl_00 + gout[29]*dm_jl_10 + gout[35]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[41]*dm_jl_00 + gout[47]*dm_jl_10 + gout[53]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[6] * dm_ik_00;
                        vjl_20 += gout[12] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[18] * dm_ik_01;
                        vjl_10 += gout[24] * dm_ik_01;
                        vjl_20 += gout[30] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[36] * dm_ik_02;
                        vjl_10 += gout[42] * dm_ik_02;
                        vjl_20 += gout[48] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[7] * dm_ik_10;
                        vjl_20 += gout[13] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[19] * dm_ik_11;
                        vjl_10 += gout[25] * dm_ik_11;
                        vjl_20 += gout[31] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[37] * dm_ik_12;
                        vjl_10 += gout[43] * dm_ik_12;
                        vjl_20 += gout[49] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[8] * dm_ik_20;
                        vjl_20 += gout[14] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[20] * dm_ik_21;
                        vjl_10 += gout[26] * dm_ik_21;
                        vjl_20 += gout[32] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[38] * dm_ik_22;
                        vjl_10 += gout[44] * dm_ik_22;
                        vjl_20 += gout[50] * dm_ik_22;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        vjl_10 += gout[9] * dm_ik_30;
                        vjl_20 += gout[15] * dm_ik_30;
                        double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[21] * dm_ik_31;
                        vjl_10 += gout[27] * dm_ik_31;
                        vjl_20 += gout[33] * dm_ik_31;
                        double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[39] * dm_ik_32;
                        vjl_10 += gout[45] * dm_ik_32;
                        vjl_20 += gout[51] * dm_ik_32;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        vjl_10 += gout[10] * dm_ik_40;
                        vjl_20 += gout[16] * dm_ik_40;
                        double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[22] * dm_ik_41;
                        vjl_10 += gout[28] * dm_ik_41;
                        vjl_20 += gout[34] * dm_ik_41;
                        double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[40] * dm_ik_42;
                        vjl_10 += gout[46] * dm_ik_42;
                        vjl_20 += gout[52] * dm_ik_42;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        vjl_10 += gout[11] * dm_ik_50;
                        vjl_20 += gout[17] * dm_ik_50;
                        double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[23] * dm_ik_51;
                        vjl_10 += gout[29] * dm_ik_51;
                        vjl_20 += gout[35] * dm_ik_51;
                        double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[41] * dm_ik_52;
                        vjl_10 += gout[47] * dm_ik_52;
                        vjl_20 += gout[53] * dm_ik_52;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20 + gout[3]*dm_il_30 + gout[4]*dm_il_40 + gout[5]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_01 = gout[18]*dm_il_00 + gout[19]*dm_il_10 + gout[20]*dm_il_20 + gout[21]*dm_il_30 + gout[22]*dm_il_40 + gout[23]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        double vjk_02 = gout[36]*dm_il_00 + gout[37]*dm_il_10 + gout[38]*dm_il_20 + gout[39]*dm_il_30 + gout[40]*dm_il_40 + gout[41]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                        double vjk_10 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20 + gout[9]*dm_il_30 + gout[10]*dm_il_40 + gout[11]*dm_il_50;
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_11 = gout[24]*dm_il_00 + gout[25]*dm_il_10 + gout[26]*dm_il_20 + gout[27]*dm_il_30 + gout[28]*dm_il_40 + gout[29]*dm_il_50;
                        atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                        double vjk_12 = gout[42]*dm_il_00 + gout[43]*dm_il_10 + gout[44]*dm_il_20 + gout[45]*dm_il_30 + gout[46]*dm_il_40 + gout[47]*dm_il_50;
                        atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                        double vjk_20 = gout[12]*dm_il_00 + gout[13]*dm_il_10 + gout[14]*dm_il_20 + gout[15]*dm_il_30 + gout[16]*dm_il_40 + gout[17]*dm_il_50;
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                        double vjk_21 = gout[30]*dm_il_00 + gout[31]*dm_il_10 + gout[32]*dm_il_20 + gout[33]*dm_il_30 + gout[34]*dm_il_40 + gout[35]*dm_il_50;
                        atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                        double vjk_22 = gout[48]*dm_il_00 + gout[49]*dm_il_10 + gout[50]*dm_il_20 + gout[51]*dm_il_30 + gout[52]*dm_il_40 + gout[53]*dm_il_50;
                        atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2111(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int gout_id = item.get_local_id(0);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif
    int t_id = 32 * gout_id + sq_id;
    constexpr int threads = 256;
    constexpr int nsq_per_block = 32;
    constexpr int g_size = 24;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+8);

while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
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
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0] = xlxk;
                rlrk[32] = ylyk;
                rlrk[64] = zlzk;
            }
            double gout[21];

            #pragma unroll
            for (int n = 0; n < 21; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                    int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
                    int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                    int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0];
                    double ylyk = rlrk[32];
                    double zlzk = rlrk[64];
                    double theta_kl = ak * al_akl;
                    double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = PI_FAC;
                    if (task_id < ntasks) {
                        if (ksh == lsh) fac_sym *= .5;
                        if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                    } else {
                        fac_sym = 0;
                    }
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + rjri[0] * aj_aij;
                    double yij = ri[1] + rjri[1] * aj_aij;
                    double zij = ri[2] + rjri[2] * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double s0, s1, s2;
                        double rt = rw[irys*64];
                        double aij = aij_cache[0];
                        double rt_aa = rt / (aij + akl);
                        double akl = akl_cache[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        for (int n = gout_id; n < 3; n += 8) {
                            if (n == 2) {
                                gx[1536] = rw[irys*64+32];
                            }
                            double *_gx = gx + n * 768;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aij_cache[1];
                            double c0x = Rpa - rt_aij * Rpq[n*32];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[32] = s1;
                            s2 = c0x * s1 + 1 * b10 * s0;
                            _gx[64] = s2;
                            s0 = s1;
                            s1 = s2;
                            s2 = c0x * s1 + 2 * b10 * s0;
                            _gx[96] = s2;
                            double xlxk = rlrk[n*32];
                            double Rqc = xlxk * akl_cache[32];
                            double cpx = Rqc + rt_akl * Rpq[n*32];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[192] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            _gx[384] = s2;
                            s0 = _gx[32];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[224] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 1 * b00 * _gx[192];
                            _gx[416] = s2;
                            s0 = _gx[64];
                            s1 = cpx * s0;
                            s1 += 2 * b00 * _gx[32];
                            _gx[256] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 2 * b00 * _gx[224];
                            _gx[448] = s2;
                            s0 = _gx[96];
                            s1 = cpx * s0;
                            s1 += 3 * b00 * _gx[64];
                            _gx[288] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 3 * b00 * _gx[256];
                            _gx[480] = s2;
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
                            s1 = _gx[384];
                            s0 = _gx[192];
                            _gx[576] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[0];
                            _gx[384] = s1 - xlxk * s0;
                            s1 = _gx[416];
                            s0 = _gx[224];
                            _gx[608] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[32];
                            _gx[416] = s1 - xlxk * s0;
                            s1 = _gx[448];
                            s0 = _gx[256];
                            _gx[640] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[64];
                            _gx[448] = s1 - xlxk * s0;
                            s1 = _gx[480];
                            s0 = _gx[288];
                            _gx[672] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[96];
                            _gx[480] = s1 - xlxk * s0;
                            s1 = _gx[512];
                            s0 = _gx[320];
                            _gx[704] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[128];
                            _gx[512] = s1 - xlxk * s0;
                            s1 = _gx[544];
                            s0 = _gx[352];
                            _gx[736] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[160];
                            _gx[544] = s1 - xlxk * s0;
                        }
                        __syncthreads();
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[736] * gx[768] * gx[1536];
                        gout[1] += gx[608] * gx[864] * gx[1568];
                        gout[2] += gx[576] * gx[800] * gx[1664];
                        gout[3] += gx[448] * gx[1056] * gx[1536];
                        gout[4] += gx[416] * gx[960] * gx[1664];
                        gout[5] += gx[480] * gx[800] * gx[1760];
                        gout[6] += gx[448] * gx[768] * gx[1824];
                        gout[7] += gx[320] * gx[1152] * gx[1568];
                        gout[8] += gx[192] * gx[1280] * gx[1568];
                        gout[9] += gx[160] * gx[1344] * gx[1536];
                        gout[10] += gx[32] * gx[1440] * gx[1568];
                        gout[11] += gx[0] * gx[1376] * gx[1664];
                        gout[12] += gx[64] * gx[1248] * gx[1728];
                        gout[13] += gx[32] * gx[1152] * gx[1856];
                        gout[14] += gx[288] * gx[800] * gx[1952];
                        gout[15] += gx[256] * gx[768] * gx[2016];
                        gout[16] += gx[128] * gx[960] * gx[1952];
                        gout[17] += gx[0] * gx[1088] * gx[1952];
                        gout[18] += gx[160] * gx[768] * gx[2112];
                        gout[19] += gx[32] * gx[864] * gx[2144];
                        gout[20] += gx[0] * gx[800] * gx[2240];
                        break;
                        case 1:
                        gout[0] += gx[704] * gx[800] * gx[1536];
                        gout[1] += gx[576] * gx[928] * gx[1536];
                        gout[2] += gx[576] * gx[768] * gx[1696];
                        gout[3] += gx[416] * gx[1088] * gx[1536];
                        gout[4] += gx[384] * gx[1024] * gx[1632];
                        gout[5] += gx[480] * gx[768] * gx[1792];
                        gout[6] += gx[416] * gx[800] * gx[1824];
                        gout[7] += gx[288] * gx[1216] * gx[1536];
                        gout[8] += gx[192] * gx[1248] * gx[1600];
                        gout[9] += gx[128] * gx[1376] * gx[1536];
                        gout[10] += gx[0] * gx[1504] * gx[1536];
                        gout[11] += gx[0] * gx[1344] * gx[1696];
                        gout[12] += gx[32] * gx[1280] * gx[1728];
                        gout[13] += gx[0] * gx[1216] * gx[1824];
                        gout[14] += gx[288] * gx[768] * gx[1984];
                        gout[15] += gx[224] * gx[800] * gx[2016];
                        gout[16] += gx[96] * gx[1024] * gx[1920];
                        gout[17] += gx[0] * gx[1056] * gx[1984];
                        gout[18] += gx[128] * gx[800] * gx[2112];
                        gout[19] += gx[0] * gx[928] * gx[2112];
                        gout[20] += gx[0] * gx[768] * gx[2272];
                        break;
                        case 2:
                        gout[0] += gx[704] * gx[768] * gx[1568];
                        gout[1] += gx[576] * gx[896] * gx[1568];
                        gout[2] += gx[544] * gx[960] * gx[1536];
                        gout[3] += gx[416] * gx[1056] * gx[1568];
                        gout[4] += gx[384] * gx[992] * gx[1664];
                        gout[5] += gx[448] * gx[864] * gx[1728];
                        gout[6] += gx[416] * gx[768] * gx[1856];
                        gout[7] += gx[288] * gx[1184] * gx[1568];
                        gout[8] += gx[256] * gx[1152] * gx[1632];
                        gout[9] += gx[128] * gx[1344] * gx[1568];
                        gout[10] += gx[0] * gx[1472] * gx[1568];
                        gout[11] += gx[160] * gx[1152] * gx[1728];
                        gout[12] += gx[32] * gx[1248] * gx[1760];
                        gout[13] += gx[0] * gx[1184] * gx[1856];
                        gout[14] += gx[256] * gx[864] * gx[1920];
                        gout[15] += gx[224] * gx[768] * gx[2048];
                        gout[16] += gx[96] * gx[992] * gx[1952];
                        gout[17] += gx[64] * gx[960] * gx[2016];
                        gout[18] += gx[128] * gx[768] * gx[2144];
                        gout[19] += gx[0] * gx[896] * gx[2144];
                        break;
                        case 3:
                        gout[0] += gx[672] * gx[832] * gx[1536];
                        gout[1] += gx[576] * gx[864] * gx[1600];
                        gout[2] += gx[512] * gx[992] * gx[1536];
                        gout[3] += gx[384] * gx[1120] * gx[1536];
                        gout[4] += gx[384] * gx[960] * gx[1696];
                        gout[5] += gx[416] * gx[896] * gx[1728];
                        gout[6] += gx[384] * gx[832] * gx[1824];
                        gout[7] += gx[288] * gx[1152] * gx[1600];
                        gout[8] += gx[224] * gx[1184] * gx[1632];
                        gout[9] += gx[96] * gx[1408] * gx[1536];
                        gout[10] += gx[0] * gx[1440] * gx[1600];
                        gout[11] += gx[128] * gx[1184] * gx[1728];
                        gout[12] += gx[0] * gx[1312] * gx[1728];
                        gout[13] += gx[0] * gx[1152] * gx[1888];
                        gout[14] += gx[224] * gx[896] * gx[1920];
                        gout[15] += gx[192] * gx[832] * gx[2016];
                        gout[16] += gx[96] * gx[960] * gx[1984];
                        gout[17] += gx[32] * gx[992] * gx[2016];
                        gout[18] += gx[96] * gx[832] * gx[2112];
                        gout[19] += gx[0] * gx[864] * gx[2176];
                        break;
                        case 4:
                        gout[0] += gx[672] * gx[800] * gx[1568];
                        gout[1] += gx[640] * gx[768] * gx[1632];
                        gout[2] += gx[512] * gx[960] * gx[1568];
                        gout[3] += gx[384] * gx[1088] * gx[1568];
                        gout[4] += gx[544] * gx[768] * gx[1728];
                        gout[5] += gx[416] * gx[864] * gx[1760];
                        gout[6] += gx[384] * gx[800] * gx[1856];
                        gout[7] += gx[256] * gx[1248] * gx[1536];
                        gout[8] += gx[224] * gx[1152] * gx[1664];
                        gout[9] += gx[96] * gx[1376] * gx[1568];
                        gout[10] += gx[64] * gx[1344] * gx[1632];
                        gout[11] += gx[128] * gx[1152] * gx[1760];
                        gout[12] += gx[0] * gx[1280] * gx[1760];
                        gout[13] += gx[352] * gx[768] * gx[1920];
                        gout[14] += gx[224] * gx[864] * gx[1952];
                        gout[15] += gx[192] * gx[800] * gx[2048];
                        gout[16] += gx[64] * gx[1056] * gx[1920];
                        gout[17] += gx[32] * gx[960] * gx[2048];
                        gout[18] += gx[96] * gx[800] * gx[2144];
                        gout[19] += gx[64] * gx[768] * gx[2208];
                        break;
                        case 5:
                        gout[0] += gx[672] * gx[768] * gx[1600];
                        gout[1] += gx[608] * gx[800] * gx[1632];
                        gout[2] += gx[480] * gx[1024] * gx[1536];
                        gout[3] += gx[384] * gx[1056] * gx[1600];
                        gout[4] += gx[512] * gx[800] * gx[1728];
                        gout[5] += gx[384] * gx[928] * gx[1728];
                        gout[6] += gx[384] * gx[768] * gx[1888];
                        gout[7] += gx[224] * gx[1280] * gx[1536];
                        gout[8] += gx[192] * gx[1216] * gx[1632];
                        gout[9] += gx[96] * gx[1344] * gx[1600];
                        gout[10] += gx[32] * gx[1376] * gx[1632];
                        gout[11] += gx[96] * gx[1216] * gx[1728];
                        gout[12] += gx[0] * gx[1248] * gx[1792];
                        gout[13] += gx[320] * gx[800] * gx[1920];
                        gout[14] += gx[192] * gx[928] * gx[1920];
                        gout[15] += gx[192] * gx[768] * gx[2080];
                        gout[16] += gx[32] * gx[1088] * gx[1920];
                        gout[17] += gx[0] * gx[1024] * gx[2016];
                        gout[18] += gx[96] * gx[768] * gx[2176];
                        gout[19] += gx[32] * gx[800] * gx[2208];
                        break;
                        case 6:
                        gout[0] += gx[640] * gx[864] * gx[1536];
                        gout[1] += gx[608] * gx[768] * gx[1664];
                        gout[2] += gx[480] * gx[992] * gx[1568];
                        gout[3] += gx[448] * gx[960] * gx[1632];
                        gout[4] += gx[512] * gx[768] * gx[1760];
                        gout[5] += gx[384] * gx[896] * gx[1760];
                        gout[6] += gx[352] * gx[1152] * gx[1536];
                        gout[7] += gx[224] * gx[1248] * gx[1568];
                        gout[8] += gx[192] * gx[1184] * gx[1664];
                        gout[9] += gx[64] * gx[1440] * gx[1536];
                        gout[10] += gx[32] * gx[1344] * gx[1664];
                        gout[11] += gx[96] * gx[1184] * gx[1760];
                        gout[12] += gx[64] * gx[1152] * gx[1824];
                        gout[13] += gx[320] * gx[768] * gx[1952];
                        gout[14] += gx[192] * gx[896] * gx[1952];
                        gout[15] += gx[160] * gx[960] * gx[1920];
                        gout[16] += gx[32] * gx[1056] * gx[1952];
                        gout[17] += gx[0] * gx[992] * gx[2048];
                        gout[18] += gx[64] * gx[864] * gx[2112];
                        gout[19] += gx[32] * gx[768] * gx[2240];
                        break;
                        case 7:
                        gout[0] += gx[608] * gx[896] * gx[1536];
                        gout[1] += gx[576] * gx[832] * gx[1632];
                        gout[2] += gx[480] * gx[960] * gx[1600];
                        gout[3] += gx[416] * gx[992] * gx[1632];
                        gout[4] += gx[480] * gx[832] * gx[1728];
                        gout[5] += gx[384] * gx[864] * gx[1792];
                        gout[6] += gx[320] * gx[1184] * gx[1536];
                        gout[7] += gx[192] * gx[1312] * gx[1536];
                        gout[8] += gx[192] * gx[1152] * gx[1696];
                        gout[9] += gx[32] * gx[1472] * gx[1536];
                        gout[10] += gx[0] * gx[1408] * gx[1632];
                        gout[11] += gx[96] * gx[1152] * gx[1792];
                        gout[12] += gx[32] * gx[1184] * gx[1824];
                        gout[13] += gx[288] * gx[832] * gx[1920];
                        gout[14] += gx[192] * gx[864] * gx[1984];
                        gout[15] += gx[128] * gx[992] * gx[1920];
                        gout[16] += gx[0] * gx[1120] * gx[1920];
                        gout[17] += gx[0] * gx[960] * gx[2080];
                        gout[18] += gx[32] * gx[896] * gx[2112];
                        gout[19] += gx[0] * gx[832] * gx[2208];
                        break;
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao * nao;
                    double *vk = kmat.vk + i_dm * nao * nao;
                    switch (gout_id) {
                    case 0: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[3]*dm_jk_11 + gout[6]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_01 = gout[9]*dm_jk_01 + gout[12]*dm_jk_12;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_02 = gout[15]*dm_jk_20 + gout[18]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_20 = gout[1]*dm_jk_10 + gout[4]*dm_jk_21;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[7]*dm_jk_00 + gout[10]*dm_jk_11 + gout[13]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[16]*dm_jk_01 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[2]*dm_jk_20 + gout[5]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[8]*dm_jk_10 + gout[11]*dm_jk_21;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[14]*dm_jk_00 + gout[17]*dm_jk_11 + gout[20]*dm_jk_22;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    break; }
                    case 1: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_10 = gout[0]*dm_jk_00 + gout[3]*dm_jk_11 + gout[6]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_11 = gout[9]*dm_jk_01 + gout[12]*dm_jk_12;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_12 = gout[15]*dm_jk_20 + gout[18]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_30 = gout[1]*dm_jk_10 + gout[4]*dm_jk_21;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[7]*dm_jk_00 + gout[10]*dm_jk_11 + gout[13]*dm_jk_22;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[16]*dm_jk_01 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[2]*dm_jk_20 + gout[5]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[8]*dm_jk_10 + gout[11]*dm_jk_21;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[14]*dm_jk_00 + gout[17]*dm_jk_11 + gout[20]*dm_jk_22;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    break; }
                    case 2: {
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_00 = gout[2]*dm_jk_01 + gout[5]*dm_jk_12;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_01 = gout[8]*dm_jk_20 + gout[11]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_02 = gout[14]*dm_jk_10 + gout[17]*dm_jk_21;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_20 = gout[0]*dm_jk_00 + gout[3]*dm_jk_11 + gout[6]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[9]*dm_jk_01 + gout[12]*dm_jk_12;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[15]*dm_jk_20 + gout[18]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[1]*dm_jk_10 + gout[4]*dm_jk_21;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[7]*dm_jk_00 + gout[10]*dm_jk_11 + gout[13]*dm_jk_22;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[16]*dm_jk_01 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    break; }
                    case 3: {
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_10 = gout[2]*dm_jk_01 + gout[5]*dm_jk_12;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_11 = gout[8]*dm_jk_20 + gout[11]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_12 = gout[14]*dm_jk_10 + gout[17]*dm_jk_21;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_30 = gout[0]*dm_jk_00 + gout[3]*dm_jk_11 + gout[6]*dm_jk_22;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[9]*dm_jk_01 + gout[12]*dm_jk_12;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[15]*dm_jk_20 + gout[18]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[1]*dm_jk_10 + gout[4]*dm_jk_21;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[7]*dm_jk_00 + gout[10]*dm_jk_11 + gout[13]*dm_jk_22;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[16]*dm_jk_01 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    break; }
                    case 4: {
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[1]*dm_jk_20 + gout[4]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_01 = gout[7]*dm_jk_10 + gout[10]*dm_jk_21;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_02 = gout[13]*dm_jk_00 + gout[16]*dm_jk_11 + gout[19]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_20 = gout[2]*dm_jk_01 + gout[5]*dm_jk_12;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[8]*dm_jk_20 + gout[11]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[14]*dm_jk_10 + gout[17]*dm_jk_21;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[0]*dm_jk_00 + gout[3]*dm_jk_11 + gout[6]*dm_jk_22;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[9]*dm_jk_01 + gout[12]*dm_jk_12;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[15]*dm_jk_20 + gout[18]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    break; }
                    case 5: {
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_10 = gout[1]*dm_jk_20 + gout[4]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_11 = gout[7]*dm_jk_10 + gout[10]*dm_jk_21;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_12 = gout[13]*dm_jk_00 + gout[16]*dm_jk_11 + gout[19]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_30 = gout[2]*dm_jk_01 + gout[5]*dm_jk_12;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[8]*dm_jk_20 + gout[11]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[14]*dm_jk_10 + gout[17]*dm_jk_21;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[0]*dm_jk_00 + gout[3]*dm_jk_11 + gout[6]*dm_jk_22;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[9]*dm_jk_01 + gout[12]*dm_jk_12;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[15]*dm_jk_20 + gout[18]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    break; }
                    case 6: {
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_00 = gout[0]*dm_jk_10 + gout[3]*dm_jk_21;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_01 = gout[6]*dm_jk_00 + gout[9]*dm_jk_11 + gout[12]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_02 = gout[15]*dm_jk_01 + gout[18]*dm_jk_12;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_20 = gout[1]*dm_jk_20 + gout[4]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[7]*dm_jk_10 + gout[10]*dm_jk_21;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[13]*dm_jk_00 + gout[16]*dm_jk_11 + gout[19]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[2]*dm_jk_01 + gout[5]*dm_jk_12;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[8]*dm_jk_20 + gout[11]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[14]*dm_jk_10 + gout[17]*dm_jk_21;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    break; }
                    case 7: {
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double vil_10 = gout[0]*dm_jk_10 + gout[3]*dm_jk_21;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_11 = gout[6]*dm_jk_00 + gout[9]*dm_jk_11 + gout[12]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_12 = gout[15]*dm_jk_01 + gout[18]*dm_jk_12;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_30 = gout[1]*dm_jk_20 + gout[4]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[7]*dm_jk_10 + gout[10]*dm_jk_21;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[13]*dm_jk_00 + gout[16]*dm_jk_11 + gout[19]*dm_jk_22;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[2]*dm_jk_01 + gout[5]*dm_jk_12;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[8]*dm_jk_20 + gout[11]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[14]*dm_jk_10 + gout[17]*dm_jk_21;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    break; }
                    }
                    switch (gout_id) {
                    case 0: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[15]*dm_jl_22;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_01 = gout[3]*dm_jl_10 + gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_02 = gout[6]*dm_jl_20 + gout[12]*dm_jl_11 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[1]*dm_jl_10 + gout[7]*dm_jl_01;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[4]*dm_jl_20 + gout[10]*dm_jl_11 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_22 = gout[13]*dm_jl_21 + gout[19]*dm_jl_12;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[2]*dm_jl_20 + gout[8]*dm_jl_11 + gout[14]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[11]*dm_jl_21 + gout[17]*dm_jl_12;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[5]*dm_jl_00 + gout[20]*dm_jl_22;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    break; }
                    case 1: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_10 = gout[0]*dm_jl_00 + gout[15]*dm_jl_22;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_11 = gout[3]*dm_jl_10 + gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_12 = gout[6]*dm_jl_20 + gout[12]*dm_jl_11 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[1]*dm_jl_10 + gout[7]*dm_jl_01;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[4]*dm_jl_20 + gout[10]*dm_jl_11 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_32 = gout[13]*dm_jl_21 + gout[19]*dm_jl_12;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[2]*dm_jl_20 + gout[8]*dm_jl_11 + gout[14]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_21 + gout[17]*dm_jl_12;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[5]*dm_jl_00 + gout[20]*dm_jl_22;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    break; }
                    case 2: {
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_00 = gout[8]*dm_jl_21 + gout[14]*dm_jl_12;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_01 = gout[2]*dm_jl_00 + gout[17]*dm_jl_22;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_02 = gout[5]*dm_jl_10 + gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[0]*dm_jl_00 + gout[15]*dm_jl_22;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[3]*dm_jl_10 + gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_22 = gout[6]*dm_jl_20 + gout[12]*dm_jl_11 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[1]*dm_jl_10 + gout[7]*dm_jl_01;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[4]*dm_jl_20 + gout[10]*dm_jl_11 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[13]*dm_jl_21 + gout[19]*dm_jl_12;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    break; }
                    case 3: {
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_10 = gout[8]*dm_jl_21 + gout[14]*dm_jl_12;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_11 = gout[2]*dm_jl_00 + gout[17]*dm_jl_22;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_12 = gout[5]*dm_jl_10 + gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[0]*dm_jl_00 + gout[15]*dm_jl_22;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[3]*dm_jl_10 + gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_32 = gout[6]*dm_jl_20 + gout[12]*dm_jl_11 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[1]*dm_jl_10 + gout[7]*dm_jl_01;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[4]*dm_jl_20 + gout[10]*dm_jl_11 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[13]*dm_jl_21 + gout[19]*dm_jl_12;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    break; }
                    case 4: {
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_00 = gout[1]*dm_jl_20 + gout[7]*dm_jl_11 + gout[13]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_01 = gout[10]*dm_jl_21 + gout[16]*dm_jl_12;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_02 = gout[4]*dm_jl_00 + gout[19]*dm_jl_22;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[8]*dm_jl_21 + gout[14]*dm_jl_12;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[2]*dm_jl_00 + gout[17]*dm_jl_22;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_22 = gout[5]*dm_jl_10 + gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[0]*dm_jl_00 + gout[15]*dm_jl_22;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[3]*dm_jl_10 + gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[6]*dm_jl_20 + gout[12]*dm_jl_11 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    break; }
                    case 5: {
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_10 = gout[1]*dm_jl_20 + gout[7]*dm_jl_11 + gout[13]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_11 = gout[10]*dm_jl_21 + gout[16]*dm_jl_12;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_12 = gout[4]*dm_jl_00 + gout[19]*dm_jl_22;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[8]*dm_jl_21 + gout[14]*dm_jl_12;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[2]*dm_jl_00 + gout[17]*dm_jl_22;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_32 = gout[5]*dm_jl_10 + gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[0]*dm_jl_00 + gout[15]*dm_jl_22;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[3]*dm_jl_10 + gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[6]*dm_jl_20 + gout[12]*dm_jl_11 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    break; }
                    case 6: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_00 = gout[0]*dm_jl_10 + gout[6]*dm_jl_01;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_01 = gout[3]*dm_jl_20 + gout[9]*dm_jl_11 + gout[15]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_02 = gout[12]*dm_jl_21 + gout[18]*dm_jl_12;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[1]*dm_jl_20 + gout[7]*dm_jl_11 + gout[13]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[10]*dm_jl_21 + gout[16]*dm_jl_12;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_22 = gout[4]*dm_jl_00 + gout[19]*dm_jl_22;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[8]*dm_jl_21 + gout[14]*dm_jl_12;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[2]*dm_jl_00 + gout[17]*dm_jl_22;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[5]*dm_jl_10 + gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    break; }
                    case 7: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_10 = gout[0]*dm_jl_10 + gout[6]*dm_jl_01;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_11 = dm[(j0+1)*nao+(l0+1)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_11 = gout[3]*dm_jl_20 + gout[9]*dm_jl_11 + gout[15]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double dm_jl_21 = dm[(j0+2)*nao+(l0+1)];
                    double dm_jl_12 = dm[(j0+1)*nao+(l0+2)];
                    double vik_12 = gout[12]*dm_jl_21 + gout[18]*dm_jl_12;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[1]*dm_jl_20 + gout[7]*dm_jl_11 + gout[13]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[10]*dm_jl_21 + gout[16]*dm_jl_12;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_22 = dm[(j0+2)*nao+(l0+2)];
                    double vik_32 = gout[4]*dm_jl_00 + gout[19]*dm_jl_22;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[8]*dm_jl_21 + gout[14]*dm_jl_12;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[2]*dm_jl_00 + gout[17]*dm_jl_22;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[5]*dm_jl_10 + gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    break; }
                    }
                    if (ish != jsh) {
                    double vjl_00 = 0;
                    double vjl_01 = 0;
                    double vjl_02 = 0;
                    double vjl_10 = 0;
                    double vjl_11 = 0;
                    double vjl_12 = 0;
                    double vjl_20 = 0;
                    double vjl_21 = 0;
                    double vjl_22 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_00;
                    vjl_22 += gout[15] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_01 += gout[9] * dm_ik_01;
                    vjl_10 += gout[3] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_02 += gout[18] * dm_ik_02;
                    vjl_11 += gout[12] * dm_ik_02;
                    vjl_20 += gout[6] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_01 += gout[7] * dm_ik_20;
                    vjl_10 += gout[1] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_02 += gout[16] * dm_ik_21;
                    vjl_11 += gout[10] * dm_ik_21;
                    vjl_20 += gout[4] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_12 += gout[19] * dm_ik_22;
                    vjl_21 += gout[13] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_02 += gout[14] * dm_ik_40;
                    vjl_11 += gout[8] * dm_ik_40;
                    vjl_20 += gout[2] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_12 += gout[17] * dm_ik_41;
                    vjl_21 += gout[11] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_00 += gout[5] * dm_ik_42;
                    vjl_22 += gout[20] * dm_ik_42;
                    break; }
                    case 1: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_10;
                    vjl_22 += gout[15] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_01 += gout[9] * dm_ik_11;
                    vjl_10 += gout[3] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_02 += gout[18] * dm_ik_12;
                    vjl_11 += gout[12] * dm_ik_12;
                    vjl_20 += gout[6] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_01 += gout[7] * dm_ik_30;
                    vjl_10 += gout[1] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_02 += gout[16] * dm_ik_31;
                    vjl_11 += gout[10] * dm_ik_31;
                    vjl_20 += gout[4] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_12 += gout[19] * dm_ik_32;
                    vjl_21 += gout[13] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_02 += gout[14] * dm_ik_50;
                    vjl_11 += gout[8] * dm_ik_50;
                    vjl_20 += gout[2] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_12 += gout[17] * dm_ik_51;
                    vjl_21 += gout[11] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_00 += gout[5] * dm_ik_52;
                    vjl_22 += gout[20] * dm_ik_52;
                    break; }
                    case 2: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_12 += gout[14] * dm_ik_00;
                    vjl_21 += gout[8] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_01;
                    vjl_22 += gout[17] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_01 += gout[11] * dm_ik_02;
                    vjl_10 += gout[5] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_20;
                    vjl_22 += gout[15] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_01 += gout[9] * dm_ik_21;
                    vjl_10 += gout[3] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_02 += gout[18] * dm_ik_22;
                    vjl_11 += gout[12] * dm_ik_22;
                    vjl_20 += gout[6] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_01 += gout[7] * dm_ik_40;
                    vjl_10 += gout[1] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_02 += gout[16] * dm_ik_41;
                    vjl_11 += gout[10] * dm_ik_41;
                    vjl_20 += gout[4] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_12 += gout[19] * dm_ik_42;
                    vjl_21 += gout[13] * dm_ik_42;
                    break; }
                    case 3: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_12 += gout[14] * dm_ik_10;
                    vjl_21 += gout[8] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_11;
                    vjl_22 += gout[17] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_01 += gout[11] * dm_ik_12;
                    vjl_10 += gout[5] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_30;
                    vjl_22 += gout[15] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_01 += gout[9] * dm_ik_31;
                    vjl_10 += gout[3] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_02 += gout[18] * dm_ik_32;
                    vjl_11 += gout[12] * dm_ik_32;
                    vjl_20 += gout[6] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_01 += gout[7] * dm_ik_50;
                    vjl_10 += gout[1] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_02 += gout[16] * dm_ik_51;
                    vjl_11 += gout[10] * dm_ik_51;
                    vjl_20 += gout[4] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_12 += gout[19] * dm_ik_52;
                    vjl_21 += gout[13] * dm_ik_52;
                    break; }
                    case 4: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_02 += gout[13] * dm_ik_00;
                    vjl_11 += gout[7] * dm_ik_00;
                    vjl_20 += gout[1] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_12 += gout[16] * dm_ik_01;
                    vjl_21 += gout[10] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_00 += gout[4] * dm_ik_02;
                    vjl_22 += gout[19] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_12 += gout[14] * dm_ik_20;
                    vjl_21 += gout[8] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_21;
                    vjl_22 += gout[17] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_01 += gout[11] * dm_ik_22;
                    vjl_10 += gout[5] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_40;
                    vjl_22 += gout[15] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_01 += gout[9] * dm_ik_41;
                    vjl_10 += gout[3] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_02 += gout[18] * dm_ik_42;
                    vjl_11 += gout[12] * dm_ik_42;
                    vjl_20 += gout[6] * dm_ik_42;
                    break; }
                    case 5: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_02 += gout[13] * dm_ik_10;
                    vjl_11 += gout[7] * dm_ik_10;
                    vjl_20 += gout[1] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_12 += gout[16] * dm_ik_11;
                    vjl_21 += gout[10] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_00 += gout[4] * dm_ik_12;
                    vjl_22 += gout[19] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_12 += gout[14] * dm_ik_30;
                    vjl_21 += gout[8] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_31;
                    vjl_22 += gout[17] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_01 += gout[11] * dm_ik_32;
                    vjl_10 += gout[5] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_50;
                    vjl_22 += gout[15] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_01 += gout[9] * dm_ik_51;
                    vjl_10 += gout[3] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_02 += gout[18] * dm_ik_52;
                    vjl_11 += gout[12] * dm_ik_52;
                    vjl_20 += gout[6] * dm_ik_52;
                    break; }
                    case 6: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_01 += gout[6] * dm_ik_00;
                    vjl_10 += gout[0] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_02 += gout[15] * dm_ik_01;
                    vjl_11 += gout[9] * dm_ik_01;
                    vjl_20 += gout[3] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_12 += gout[18] * dm_ik_02;
                    vjl_21 += gout[12] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_02 += gout[13] * dm_ik_20;
                    vjl_11 += gout[7] * dm_ik_20;
                    vjl_20 += gout[1] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_12 += gout[16] * dm_ik_21;
                    vjl_21 += gout[10] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_00 += gout[4] * dm_ik_22;
                    vjl_22 += gout[19] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_12 += gout[14] * dm_ik_40;
                    vjl_21 += gout[8] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_41;
                    vjl_22 += gout[17] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_01 += gout[11] * dm_ik_42;
                    vjl_10 += gout[5] * dm_ik_42;
                    break; }
                    case 7: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_01 += gout[6] * dm_ik_10;
                    vjl_10 += gout[0] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_02 += gout[15] * dm_ik_11;
                    vjl_11 += gout[9] * dm_ik_11;
                    vjl_20 += gout[3] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_12 += gout[18] * dm_ik_12;
                    vjl_21 += gout[12] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_02 += gout[13] * dm_ik_30;
                    vjl_11 += gout[7] * dm_ik_30;
                    vjl_20 += gout[1] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_12 += gout[16] * dm_ik_31;
                    vjl_21 += gout[10] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_00 += gout[4] * dm_ik_32;
                    vjl_22 += gout[19] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_12 += gout[14] * dm_ik_50;
                    vjl_21 += gout[8] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_51;
                    vjl_22 += gout[17] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_01 += gout[11] * dm_ik_52;
                    vjl_10 += gout[5] * dm_ik_52;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                    atomicAdd(vk+(j0+0)*nao+(l0+1), vjl_01);
                    atomicAdd(vk+(j0+0)*nao+(l0+2), vjl_02);
                    atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                    atomicAdd(vk+(j0+1)*nao+(l0+1), vjl_11);
                    atomicAdd(vk+(j0+1)*nao+(l0+2), vjl_12);
                    atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                    atomicAdd(vk+(j0+2)*nao+(l0+1), vjl_21);
                    atomicAdd(vk+(j0+2)*nao+(l0+2), vjl_22);
                    double vjk_00 = 0;
                    double vjk_01 = 0;
                    double vjk_02 = 0;
                    double vjk_10 = 0;
                    double vjk_11 = 0;
                    double vjk_12 = 0;
                    double vjk_20 = 0;
                    double vjk_21 = 0;
                    double vjk_22 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_00;
                    vjk_11 += gout[3] * dm_il_00;
                    vjk_22 += gout[6] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_01 += gout[9] * dm_il_01;
                    vjk_12 += gout[12] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_02 += gout[18] * dm_il_02;
                    vjk_20 += gout[15] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_10 += gout[1] * dm_il_20;
                    vjk_21 += gout[4] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_00 += gout[7] * dm_il_21;
                    vjk_11 += gout[10] * dm_il_21;
                    vjk_22 += gout[13] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_01 += gout[16] * dm_il_22;
                    vjk_12 += gout[19] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_02 += gout[5] * dm_il_40;
                    vjk_20 += gout[2] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_10 += gout[8] * dm_il_41;
                    vjk_21 += gout[11] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_00 += gout[14] * dm_il_42;
                    vjk_11 += gout[17] * dm_il_42;
                    vjk_22 += gout[20] * dm_il_42;
                    break; }
                    case 1: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_10;
                    vjk_11 += gout[3] * dm_il_10;
                    vjk_22 += gout[6] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_01 += gout[9] * dm_il_11;
                    vjk_12 += gout[12] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_02 += gout[18] * dm_il_12;
                    vjk_20 += gout[15] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_10 += gout[1] * dm_il_30;
                    vjk_21 += gout[4] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_00 += gout[7] * dm_il_31;
                    vjk_11 += gout[10] * dm_il_31;
                    vjk_22 += gout[13] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_01 += gout[16] * dm_il_32;
                    vjk_12 += gout[19] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_02 += gout[5] * dm_il_50;
                    vjk_20 += gout[2] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_10 += gout[8] * dm_il_51;
                    vjk_21 += gout[11] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_00 += gout[14] * dm_il_52;
                    vjk_11 += gout[17] * dm_il_52;
                    vjk_22 += gout[20] * dm_il_52;
                    break; }
                    case 2: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_00;
                    vjk_12 += gout[5] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_02 += gout[11] * dm_il_01;
                    vjk_20 += gout[8] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_10 += gout[14] * dm_il_02;
                    vjk_21 += gout[17] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_20;
                    vjk_11 += gout[3] * dm_il_20;
                    vjk_22 += gout[6] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_01 += gout[9] * dm_il_21;
                    vjk_12 += gout[12] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_02 += gout[18] * dm_il_22;
                    vjk_20 += gout[15] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_10 += gout[1] * dm_il_40;
                    vjk_21 += gout[4] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_00 += gout[7] * dm_il_41;
                    vjk_11 += gout[10] * dm_il_41;
                    vjk_22 += gout[13] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_01 += gout[16] * dm_il_42;
                    vjk_12 += gout[19] * dm_il_42;
                    break; }
                    case 3: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_10;
                    vjk_12 += gout[5] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_02 += gout[11] * dm_il_11;
                    vjk_20 += gout[8] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_10 += gout[14] * dm_il_12;
                    vjk_21 += gout[17] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_30;
                    vjk_11 += gout[3] * dm_il_30;
                    vjk_22 += gout[6] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_01 += gout[9] * dm_il_31;
                    vjk_12 += gout[12] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_02 += gout[18] * dm_il_32;
                    vjk_20 += gout[15] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_10 += gout[1] * dm_il_50;
                    vjk_21 += gout[4] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_00 += gout[7] * dm_il_51;
                    vjk_11 += gout[10] * dm_il_51;
                    vjk_22 += gout[13] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_01 += gout[16] * dm_il_52;
                    vjk_12 += gout[19] * dm_il_52;
                    break; }
                    case 4: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_02 += gout[4] * dm_il_00;
                    vjk_20 += gout[1] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_10 += gout[7] * dm_il_01;
                    vjk_21 += gout[10] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_00 += gout[13] * dm_il_02;
                    vjk_11 += gout[16] * dm_il_02;
                    vjk_22 += gout[19] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_20;
                    vjk_12 += gout[5] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_02 += gout[11] * dm_il_21;
                    vjk_20 += gout[8] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_10 += gout[14] * dm_il_22;
                    vjk_21 += gout[17] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_40;
                    vjk_11 += gout[3] * dm_il_40;
                    vjk_22 += gout[6] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_01 += gout[9] * dm_il_41;
                    vjk_12 += gout[12] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_02 += gout[18] * dm_il_42;
                    vjk_20 += gout[15] * dm_il_42;
                    break; }
                    case 5: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_02 += gout[4] * dm_il_10;
                    vjk_20 += gout[1] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_10 += gout[7] * dm_il_11;
                    vjk_21 += gout[10] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_00 += gout[13] * dm_il_12;
                    vjk_11 += gout[16] * dm_il_12;
                    vjk_22 += gout[19] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_30;
                    vjk_12 += gout[5] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_02 += gout[11] * dm_il_31;
                    vjk_20 += gout[8] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_10 += gout[14] * dm_il_32;
                    vjk_21 += gout[17] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_50;
                    vjk_11 += gout[3] * dm_il_50;
                    vjk_22 += gout[6] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_01 += gout[9] * dm_il_51;
                    vjk_12 += gout[12] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_02 += gout[18] * dm_il_52;
                    vjk_20 += gout[15] * dm_il_52;
                    break; }
                    case 6: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_10 += gout[0] * dm_il_00;
                    vjk_21 += gout[3] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_00 += gout[6] * dm_il_01;
                    vjk_11 += gout[9] * dm_il_01;
                    vjk_22 += gout[12] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_01 += gout[15] * dm_il_02;
                    vjk_12 += gout[18] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_02 += gout[4] * dm_il_20;
                    vjk_20 += gout[1] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_10 += gout[7] * dm_il_21;
                    vjk_21 += gout[10] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_00 += gout[13] * dm_il_22;
                    vjk_11 += gout[16] * dm_il_22;
                    vjk_22 += gout[19] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_40;
                    vjk_12 += gout[5] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_02 += gout[11] * dm_il_41;
                    vjk_20 += gout[8] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_10 += gout[14] * dm_il_42;
                    vjk_21 += gout[17] * dm_il_42;
                    break; }
                    case 7: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_10 += gout[0] * dm_il_10;
                    vjk_21 += gout[3] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_00 += gout[6] * dm_il_11;
                    vjk_11 += gout[9] * dm_il_11;
                    vjk_22 += gout[12] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_01 += gout[15] * dm_il_12;
                    vjk_12 += gout[18] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_02 += gout[4] * dm_il_30;
                    vjk_20 += gout[1] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_10 += gout[7] * dm_il_31;
                    vjk_21 += gout[10] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_00 += gout[13] * dm_il_32;
                    vjk_11 += gout[16] * dm_il_32;
                    vjk_22 += gout[19] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_50;
                    vjk_12 += gout[5] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_02 += gout[11] * dm_il_51;
                    vjk_20 += gout[8] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_10 += gout[14] * dm_il_52;
                    vjk_21 += gout[17] * dm_il_52;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2120(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int gout_id = item.get_local_id(0);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif
    int t_id = 64 * gout_id + sq_id;
    constexpr int threads = 256;
    constexpr int nsq_per_block = 64;
    constexpr int g_size = 18;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+8);

while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
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
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0] = xlxk;
                rlrk[64] = ylyk;
                rlrk[128] = zlzk;
            }
            double gout[27];

            #pragma unroll
            for (int n = 0; n < 27; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                    int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
                    int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                    int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0];
                    double ylyk = rlrk[64];
                    double zlzk = rlrk[128];
                    double theta_kl = ak * al_akl;
                    double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = PI_FAC;
                    if (task_id < ntasks) {
                        if (ksh == lsh) fac_sym *= .5;
                        if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                    } else {
                        fac_sym = 0;
                    }
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + rjri[0] * aj_aij;
                    double yij = ri[1] + rjri[1] * aj_aij;
                    double zij = ri[2] + rjri[2] * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double s0, s1, s2;
                        double rt = rw[irys*128];
                        double aij = aij_cache[0];
                        double rt_aa = rt / (aij + akl);
                        double akl = akl_cache[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        for (int n = gout_id; n < 3; n += 4) {
                            if (n == 2) {
                                gx[2304] = rw[irys*128+64];
                            }
                            double *_gx = gx + n * 1152;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aij_cache[1];
                            double c0x = Rpa - rt_aij * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[64] = s1;
                            s2 = c0x * s1 + 1 * b10 * s0;
                            _gx[128] = s2;
                            s0 = s1;
                            s1 = s2;
                            s2 = c0x * s1 + 2 * b10 * s0;
                            _gx[192] = s2;
                            double xlxk = rlrk[n*64];
                            double Rqc = xlxk * akl_cache[64];
                            double cpx = Rqc + rt_akl * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[384] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            _gx[768] = s2;
                            s0 = _gx[64];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[448] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 1 * b00 * _gx[384];
                            _gx[832] = s2;
                            s0 = _gx[128];
                            s1 = cpx * s0;
                            s1 += 2 * b00 * _gx[64];
                            _gx[512] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 2 * b00 * _gx[448];
                            _gx[896] = s2;
                            s0 = _gx[192];
                            s1 = cpx * s0;
                            s1 += 3 * b00 * _gx[128];
                            _gx[576] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 3 * b00 * _gx[512];
                            _gx[960] = s2;
                            s1 = _gx[192];
                            s0 = _gx[128];
                            _gx[320] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[64];
                            _gx[256] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[0];
                            _gx[192] = s1 - xjxi * s0;
                            s1 = _gx[576];
                            s0 = _gx[512];
                            _gx[704] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[448];
                            _gx[640] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[384];
                            _gx[576] = s1 - xjxi * s0;
                            s1 = _gx[960];
                            s0 = _gx[896];
                            _gx[1088] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[832];
                            _gx[1024] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[768];
                            _gx[960] = s1 - xjxi * s0;
                        }
                        __syncthreads();
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[1088] * gx[1152] * gx[2304];
                        gout[1] += gx[960] * gx[1216] * gx[2368];
                        gout[2] += gx[832] * gx[1344] * gx[2368];
                        gout[3] += gx[896] * gx[1152] * gx[2496];
                        gout[4] += gx[768] * gx[1216] * gx[2560];
                        gout[5] += gx[640] * gx[1536] * gx[2368];
                        gout[6] += gx[512] * gx[1728] * gx[2304];
                        gout[7] += gx[384] * gx[1792] * gx[2368];
                        gout[8] += gx[448] * gx[1536] * gx[2560];
                        gout[9] += gx[704] * gx[1152] * gx[2688];
                        gout[10] += gx[576] * gx[1216] * gx[2752];
                        gout[11] += gx[448] * gx[1344] * gx[2752];
                        gout[12] += gx[512] * gx[1152] * gx[2880];
                        gout[13] += gx[384] * gx[1216] * gx[2944];
                        gout[14] += gx[256] * gx[1920] * gx[2368];
                        gout[15] += gx[128] * gx[2112] * gx[2304];
                        gout[16] += gx[0] * gx[2176] * gx[2368];
                        gout[17] += gx[64] * gx[1920] * gx[2560];
                        gout[18] += gx[320] * gx[1536] * gx[2688];
                        gout[19] += gx[192] * gx[1600] * gx[2752];
                        gout[20] += gx[64] * gx[1728] * gx[2752];
                        gout[21] += gx[128] * gx[1536] * gx[2880];
                        gout[22] += gx[0] * gx[1600] * gx[2944];
                        gout[23] += gx[256] * gx[1152] * gx[3136];
                        gout[24] += gx[128] * gx[1344] * gx[3072];
                        gout[25] += gx[0] * gx[1408] * gx[3136];
                        gout[26] += gx[64] * gx[1152] * gx[3328];
                        break;
                        case 1:
                        gout[0] += gx[1024] * gx[1216] * gx[2304];
                        gout[1] += gx[960] * gx[1152] * gx[2432];
                        gout[2] += gx[768] * gx[1472] * gx[2304];
                        gout[3] += gx[832] * gx[1216] * gx[2496];
                        gout[4] += gx[768] * gx[1152] * gx[2624];
                        gout[5] += gx[576] * gx[1664] * gx[2304];
                        gout[6] += gx[448] * gx[1792] * gx[2304];
                        gout[7] += gx[384] * gx[1728] * gx[2432];
                        gout[8] += gx[384] * gx[1664] * gx[2496];
                        gout[9] += gx[640] * gx[1216] * gx[2688];
                        gout[10] += gx[576] * gx[1152] * gx[2816];
                        gout[11] += gx[384] * gx[1472] * gx[2688];
                        gout[12] += gx[448] * gx[1216] * gx[2880];
                        gout[13] += gx[384] * gx[1152] * gx[3008];
                        gout[14] += gx[192] * gx[2048] * gx[2304];
                        gout[15] += gx[64] * gx[2176] * gx[2304];
                        gout[16] += gx[0] * gx[2112] * gx[2432];
                        gout[17] += gx[0] * gx[2048] * gx[2496];
                        gout[18] += gx[256] * gx[1600] * gx[2688];
                        gout[19] += gx[192] * gx[1536] * gx[2816];
                        gout[20] += gx[0] * gx[1856] * gx[2688];
                        gout[21] += gx[64] * gx[1600] * gx[2880];
                        gout[22] += gx[0] * gx[1536] * gx[3008];
                        gout[23] += gx[192] * gx[1280] * gx[3072];
                        gout[24] += gx[64] * gx[1408] * gx[3072];
                        gout[25] += gx[0] * gx[1344] * gx[3200];
                        gout[26] += gx[0] * gx[1280] * gx[3264];
                        break;
                        case 2:
                        gout[0] += gx[1024] * gx[1152] * gx[2368];
                        gout[1] += gx[896] * gx[1344] * gx[2304];
                        gout[2] += gx[768] * gx[1408] * gx[2368];
                        gout[3] += gx[832] * gx[1152] * gx[2560];
                        gout[4] += gx[704] * gx[1536] * gx[2304];
                        gout[5] += gx[576] * gx[1600] * gx[2368];
                        gout[6] += gx[448] * gx[1728] * gx[2368];
                        gout[7] += gx[512] * gx[1536] * gx[2496];
                        gout[8] += gx[384] * gx[1600] * gx[2560];
                        gout[9] += gx[640] * gx[1152] * gx[2752];
                        gout[10] += gx[512] * gx[1344] * gx[2688];
                        gout[11] += gx[384] * gx[1408] * gx[2752];
                        gout[12] += gx[448] * gx[1152] * gx[2944];
                        gout[13] += gx[320] * gx[1920] * gx[2304];
                        gout[14] += gx[192] * gx[1984] * gx[2368];
                        gout[15] += gx[64] * gx[2112] * gx[2368];
                        gout[16] += gx[128] * gx[1920] * gx[2496];
                        gout[17] += gx[0] * gx[1984] * gx[2560];
                        gout[18] += gx[256] * gx[1536] * gx[2752];
                        gout[19] += gx[128] * gx[1728] * gx[2688];
                        gout[20] += gx[0] * gx[1792] * gx[2752];
                        gout[21] += gx[64] * gx[1536] * gx[2944];
                        gout[22] += gx[320] * gx[1152] * gx[3072];
                        gout[23] += gx[192] * gx[1216] * gx[3136];
                        gout[24] += gx[64] * gx[1344] * gx[3136];
                        gout[25] += gx[128] * gx[1152] * gx[3264];
                        gout[26] += gx[0] * gx[1216] * gx[3328];
                        break;
                        case 3:
                        gout[0] += gx[960] * gx[1280] * gx[2304];
                        gout[1] += gx[832] * gx[1408] * gx[2304];
                        gout[2] += gx[768] * gx[1344] * gx[2432];
                        gout[3] += gx[768] * gx[1280] * gx[2496];
                        gout[4] += gx[640] * gx[1600] * gx[2304];
                        gout[5] += gx[576] * gx[1536] * gx[2432];
                        gout[6] += gx[384] * gx[1856] * gx[2304];
                        gout[7] += gx[448] * gx[1600] * gx[2496];
                        gout[8] += gx[384] * gx[1536] * gx[2624];
                        gout[9] += gx[576] * gx[1280] * gx[2688];
                        gout[10] += gx[448] * gx[1408] * gx[2688];
                        gout[11] += gx[384] * gx[1344] * gx[2816];
                        gout[12] += gx[384] * gx[1280] * gx[2880];
                        gout[13] += gx[256] * gx[1984] * gx[2304];
                        gout[14] += gx[192] * gx[1920] * gx[2432];
                        gout[15] += gx[0] * gx[2240] * gx[2304];
                        gout[16] += gx[64] * gx[1984] * gx[2496];
                        gout[17] += gx[0] * gx[1920] * gx[2624];
                        gout[18] += gx[192] * gx[1664] * gx[2688];
                        gout[19] += gx[64] * gx[1792] * gx[2688];
                        gout[20] += gx[0] * gx[1728] * gx[2816];
                        gout[21] += gx[0] * gx[1664] * gx[2880];
                        gout[22] += gx[256] * gx[1216] * gx[3072];
                        gout[23] += gx[192] * gx[1152] * gx[3200];
                        gout[24] += gx[0] * gx[1472] * gx[3072];
                        gout[25] += gx[64] * gx[1216] * gx[3264];
                        gout[26] += gx[0] * gx[1152] * gx[3392];
                        break;
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao * nao;
                    double *vk = kmat.vk + i_dm * nao * nao;
                    double vil_00 = 0;
                    double vil_10 = 0;
                    double vil_20 = 0;
                    double vil_30 = 0;
                    double vil_40 = 0;
                    double vil_50 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_00 += gout[0] * dm_jk_00;
                    vil_40 += gout[1] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_20 += gout[5] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_00 += gout[9] * dm_jk_02;
                    vil_40 += gout[10] * dm_jk_02;
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    vil_20 += gout[14] * dm_jk_03;
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    vil_00 += gout[18] * dm_jk_04;
                    vil_40 += gout[19] * dm_jk_04;
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    vil_20 += gout[23] * dm_jk_05;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_20 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_00 += gout[6] * dm_jk_11;
                    vil_40 += gout[7] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_20 += gout[11] * dm_jk_12;
                    double dm_jk_13 = dm[(j0+1)*nao+(k0+3)];
                    vil_00 += gout[15] * dm_jk_13;
                    vil_40 += gout[16] * dm_jk_13;
                    double dm_jk_14 = dm[(j0+1)*nao+(k0+4)];
                    vil_20 += gout[20] * dm_jk_14;
                    double dm_jk_15 = dm[(j0+1)*nao+(k0+5)];
                    vil_00 += gout[24] * dm_jk_15;
                    vil_40 += gout[25] * dm_jk_15;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_00 += gout[3] * dm_jk_20;
                    vil_40 += gout[4] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_20 += gout[8] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_00 += gout[12] * dm_jk_22;
                    vil_40 += gout[13] * dm_jk_22;
                    double dm_jk_23 = dm[(j0+2)*nao+(k0+3)];
                    vil_20 += gout[17] * dm_jk_23;
                    double dm_jk_24 = dm[(j0+2)*nao+(k0+4)];
                    vil_00 += gout[21] * dm_jk_24;
                    vil_40 += gout[22] * dm_jk_24;
                    double dm_jk_25 = dm[(j0+2)*nao+(k0+5)];
                    vil_20 += gout[26] * dm_jk_25;
                    break; }
                    case 1: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_10 += gout[0] * dm_jk_00;
                    vil_50 += gout[1] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_30 += gout[5] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_10 += gout[9] * dm_jk_02;
                    vil_50 += gout[10] * dm_jk_02;
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    vil_30 += gout[14] * dm_jk_03;
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    vil_10 += gout[18] * dm_jk_04;
                    vil_50 += gout[19] * dm_jk_04;
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    vil_30 += gout[23] * dm_jk_05;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_30 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_10 += gout[6] * dm_jk_11;
                    vil_50 += gout[7] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_30 += gout[11] * dm_jk_12;
                    double dm_jk_13 = dm[(j0+1)*nao+(k0+3)];
                    vil_10 += gout[15] * dm_jk_13;
                    vil_50 += gout[16] * dm_jk_13;
                    double dm_jk_14 = dm[(j0+1)*nao+(k0+4)];
                    vil_30 += gout[20] * dm_jk_14;
                    double dm_jk_15 = dm[(j0+1)*nao+(k0+5)];
                    vil_10 += gout[24] * dm_jk_15;
                    vil_50 += gout[25] * dm_jk_15;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_10 += gout[3] * dm_jk_20;
                    vil_50 += gout[4] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_30 += gout[8] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_10 += gout[12] * dm_jk_22;
                    vil_50 += gout[13] * dm_jk_22;
                    double dm_jk_23 = dm[(j0+2)*nao+(k0+3)];
                    vil_30 += gout[17] * dm_jk_23;
                    double dm_jk_24 = dm[(j0+2)*nao+(k0+4)];
                    vil_10 += gout[21] * dm_jk_24;
                    vil_50 += gout[22] * dm_jk_24;
                    double dm_jk_25 = dm[(j0+2)*nao+(k0+5)];
                    vil_30 += gout[26] * dm_jk_25;
                    break; }
                    case 2: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_20 += gout[0] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_00 += gout[4] * dm_jk_01;
                    vil_40 += gout[5] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_20 += gout[9] * dm_jk_02;
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    vil_00 += gout[13] * dm_jk_03;
                    vil_40 += gout[14] * dm_jk_03;
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    vil_20 += gout[18] * dm_jk_04;
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    vil_00 += gout[22] * dm_jk_05;
                    vil_40 += gout[23] * dm_jk_05;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_00 += gout[1] * dm_jk_10;
                    vil_40 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_20 += gout[6] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_00 += gout[10] * dm_jk_12;
                    vil_40 += gout[11] * dm_jk_12;
                    double dm_jk_13 = dm[(j0+1)*nao+(k0+3)];
                    vil_20 += gout[15] * dm_jk_13;
                    double dm_jk_14 = dm[(j0+1)*nao+(k0+4)];
                    vil_00 += gout[19] * dm_jk_14;
                    vil_40 += gout[20] * dm_jk_14;
                    double dm_jk_15 = dm[(j0+1)*nao+(k0+5)];
                    vil_20 += gout[24] * dm_jk_15;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_20 += gout[3] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_00 += gout[7] * dm_jk_21;
                    vil_40 += gout[8] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_20 += gout[12] * dm_jk_22;
                    double dm_jk_23 = dm[(j0+2)*nao+(k0+3)];
                    vil_00 += gout[16] * dm_jk_23;
                    vil_40 += gout[17] * dm_jk_23;
                    double dm_jk_24 = dm[(j0+2)*nao+(k0+4)];
                    vil_20 += gout[21] * dm_jk_24;
                    double dm_jk_25 = dm[(j0+2)*nao+(k0+5)];
                    vil_00 += gout[25] * dm_jk_25;
                    vil_40 += gout[26] * dm_jk_25;
                    break; }
                    case 3: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_30 += gout[0] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_10 += gout[4] * dm_jk_01;
                    vil_50 += gout[5] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_30 += gout[9] * dm_jk_02;
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    vil_10 += gout[13] * dm_jk_03;
                    vil_50 += gout[14] * dm_jk_03;
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    vil_30 += gout[18] * dm_jk_04;
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    vil_10 += gout[22] * dm_jk_05;
                    vil_50 += gout[23] * dm_jk_05;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_10 += gout[1] * dm_jk_10;
                    vil_50 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_30 += gout[6] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_10 += gout[10] * dm_jk_12;
                    vil_50 += gout[11] * dm_jk_12;
                    double dm_jk_13 = dm[(j0+1)*nao+(k0+3)];
                    vil_30 += gout[15] * dm_jk_13;
                    double dm_jk_14 = dm[(j0+1)*nao+(k0+4)];
                    vil_10 += gout[19] * dm_jk_14;
                    vil_50 += gout[20] * dm_jk_14;
                    double dm_jk_15 = dm[(j0+1)*nao+(k0+5)];
                    vil_30 += gout[24] * dm_jk_15;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_30 += gout[3] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_10 += gout[7] * dm_jk_21;
                    vil_50 += gout[8] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_30 += gout[12] * dm_jk_22;
                    double dm_jk_23 = dm[(j0+2)*nao+(k0+3)];
                    vil_10 += gout[16] * dm_jk_23;
                    vil_50 += gout[17] * dm_jk_23;
                    double dm_jk_24 = dm[(j0+2)*nao+(k0+4)];
                    vil_30 += gout[21] * dm_jk_24;
                    double dm_jk_25 = dm[(j0+2)*nao+(k0+5)];
                    vil_10 += gout[25] * dm_jk_25;
                    vil_50 += gout[26] * dm_jk_25;
                    break; }
                    }
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    switch (gout_id) {
                    case 0: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_01 = gout[6]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_03 = gout[15]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+3), vik_03);
                    double vik_04 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+4), vik_04);
                    double vik_05 = gout[24]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+5), vik_05);
                    double vik_20 = gout[2]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[5]*dm_jl_00 + gout[8]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_23 = gout[14]*dm_jl_00 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+3), vik_23);
                    double vik_24 = gout[20]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+4), vik_24);
                    double vik_25 = gout[23]*dm_jl_00 + gout[26]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+5), vik_25);
                    double vik_40 = gout[1]*dm_jl_00 + gout[4]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[7]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[10]*dm_jl_00 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_43 = gout[16]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+3), vik_43);
                    double vik_44 = gout[19]*dm_jl_00 + gout[22]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+4), vik_44);
                    double vik_45 = gout[25]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+5), vik_45);
                    break; }
                    case 1: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_10 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_11 = gout[6]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_13 = gout[15]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+3), vik_13);
                    double vik_14 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+4), vik_14);
                    double vik_15 = gout[24]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+5), vik_15);
                    double vik_30 = gout[2]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[5]*dm_jl_00 + gout[8]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_33 = gout[14]*dm_jl_00 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+3), vik_33);
                    double vik_34 = gout[20]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+4), vik_34);
                    double vik_35 = gout[23]*dm_jl_00 + gout[26]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+5), vik_35);
                    double vik_50 = gout[1]*dm_jl_00 + gout[4]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[7]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[10]*dm_jl_00 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_53 = gout[16]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+3), vik_53);
                    double vik_54 = gout[19]*dm_jl_00 + gout[22]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+4), vik_54);
                    double vik_55 = gout[25]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+5), vik_55);
                    break; }
                    case 2: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_00 = gout[1]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_01 = gout[4]*dm_jl_00 + gout[7]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[10]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_03 = gout[13]*dm_jl_00 + gout[16]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+3), vik_03);
                    double vik_04 = gout[19]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+4), vik_04);
                    double vik_05 = gout[22]*dm_jl_00 + gout[25]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+5), vik_05);
                    double vik_20 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[6]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_23 = gout[15]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+3), vik_23);
                    double vik_24 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+4), vik_24);
                    double vik_25 = gout[24]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+5), vik_25);
                    double vik_40 = gout[2]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[5]*dm_jl_00 + gout[8]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_43 = gout[14]*dm_jl_00 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+3), vik_43);
                    double vik_44 = gout[20]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+4), vik_44);
                    double vik_45 = gout[23]*dm_jl_00 + gout[26]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+5), vik_45);
                    break; }
                    case 3: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_10 = gout[1]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_11 = gout[4]*dm_jl_00 + gout[7]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[10]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_13 = gout[13]*dm_jl_00 + gout[16]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+3), vik_13);
                    double vik_14 = gout[19]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+4), vik_14);
                    double vik_15 = gout[22]*dm_jl_00 + gout[25]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+5), vik_15);
                    double vik_30 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[6]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_33 = gout[15]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+3), vik_33);
                    double vik_34 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+4), vik_34);
                    double vik_35 = gout[24]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+5), vik_35);
                    double vik_50 = gout[2]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[5]*dm_jl_00 + gout[8]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_53 = gout[14]*dm_jl_00 + gout[17]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+3), vik_53);
                    double vik_54 = gout[20]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+4), vik_54);
                    double vik_55 = gout[23]*dm_jl_00 + gout[26]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+5), vik_55);
                    break; }
                    }
                    if (ish != jsh) {
                    double vjl_00 = 0;
                    double vjl_10 = 0;
                    double vjl_20 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_00;
                    vjl_20 += gout[3] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_10 += gout[6] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_00 += gout[9] * dm_ik_02;
                    vjl_20 += gout[12] * dm_ik_02;
                    double dm_ik_03 = dm[(i0+0)*nao+(k0+3)];
                    vjl_10 += gout[15] * dm_ik_03;
                    double dm_ik_04 = dm[(i0+0)*nao+(k0+4)];
                    vjl_00 += gout[18] * dm_ik_04;
                    vjl_20 += gout[21] * dm_ik_04;
                    double dm_ik_05 = dm[(i0+0)*nao+(k0+5)];
                    vjl_10 += gout[24] * dm_ik_05;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_00 += gout[5] * dm_ik_21;
                    vjl_20 += gout[8] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_10 += gout[11] * dm_ik_22;
                    double dm_ik_23 = dm[(i0+2)*nao+(k0+3)];
                    vjl_00 += gout[14] * dm_ik_23;
                    vjl_20 += gout[17] * dm_ik_23;
                    double dm_ik_24 = dm[(i0+2)*nao+(k0+4)];
                    vjl_10 += gout[20] * dm_ik_24;
                    double dm_ik_25 = dm[(i0+2)*nao+(k0+5)];
                    vjl_00 += gout[23] * dm_ik_25;
                    vjl_20 += gout[26] * dm_ik_25;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_40;
                    vjl_20 += gout[4] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_10 += gout[7] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_00 += gout[10] * dm_ik_42;
                    vjl_20 += gout[13] * dm_ik_42;
                    double dm_ik_43 = dm[(i0+4)*nao+(k0+3)];
                    vjl_10 += gout[16] * dm_ik_43;
                    double dm_ik_44 = dm[(i0+4)*nao+(k0+4)];
                    vjl_00 += gout[19] * dm_ik_44;
                    vjl_20 += gout[22] * dm_ik_44;
                    double dm_ik_45 = dm[(i0+4)*nao+(k0+5)];
                    vjl_10 += gout[25] * dm_ik_45;
                    break; }
                    case 1: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_10;
                    vjl_20 += gout[3] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_10 += gout[6] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_00 += gout[9] * dm_ik_12;
                    vjl_20 += gout[12] * dm_ik_12;
                    double dm_ik_13 = dm[(i0+1)*nao+(k0+3)];
                    vjl_10 += gout[15] * dm_ik_13;
                    double dm_ik_14 = dm[(i0+1)*nao+(k0+4)];
                    vjl_00 += gout[18] * dm_ik_14;
                    vjl_20 += gout[21] * dm_ik_14;
                    double dm_ik_15 = dm[(i0+1)*nao+(k0+5)];
                    vjl_10 += gout[24] * dm_ik_15;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_00 += gout[5] * dm_ik_31;
                    vjl_20 += gout[8] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_10 += gout[11] * dm_ik_32;
                    double dm_ik_33 = dm[(i0+3)*nao+(k0+3)];
                    vjl_00 += gout[14] * dm_ik_33;
                    vjl_20 += gout[17] * dm_ik_33;
                    double dm_ik_34 = dm[(i0+3)*nao+(k0+4)];
                    vjl_10 += gout[20] * dm_ik_34;
                    double dm_ik_35 = dm[(i0+3)*nao+(k0+5)];
                    vjl_00 += gout[23] * dm_ik_35;
                    vjl_20 += gout[26] * dm_ik_35;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_50;
                    vjl_20 += gout[4] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_10 += gout[7] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_00 += gout[10] * dm_ik_52;
                    vjl_20 += gout[13] * dm_ik_52;
                    double dm_ik_53 = dm[(i0+5)*nao+(k0+3)];
                    vjl_10 += gout[16] * dm_ik_53;
                    double dm_ik_54 = dm[(i0+5)*nao+(k0+4)];
                    vjl_00 += gout[19] * dm_ik_54;
                    vjl_20 += gout[22] * dm_ik_54;
                    double dm_ik_55 = dm[(i0+5)*nao+(k0+5)];
                    vjl_10 += gout[25] * dm_ik_55;
                    break; }
                    case 2: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_10 += gout[1] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_00 += gout[4] * dm_ik_01;
                    vjl_20 += gout[7] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_10 += gout[10] * dm_ik_02;
                    double dm_ik_03 = dm[(i0+0)*nao+(k0+3)];
                    vjl_00 += gout[13] * dm_ik_03;
                    vjl_20 += gout[16] * dm_ik_03;
                    double dm_ik_04 = dm[(i0+0)*nao+(k0+4)];
                    vjl_10 += gout[19] * dm_ik_04;
                    double dm_ik_05 = dm[(i0+0)*nao+(k0+5)];
                    vjl_00 += gout[22] * dm_ik_05;
                    vjl_20 += gout[25] * dm_ik_05;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_20;
                    vjl_20 += gout[3] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_10 += gout[6] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_00 += gout[9] * dm_ik_22;
                    vjl_20 += gout[12] * dm_ik_22;
                    double dm_ik_23 = dm[(i0+2)*nao+(k0+3)];
                    vjl_10 += gout[15] * dm_ik_23;
                    double dm_ik_24 = dm[(i0+2)*nao+(k0+4)];
                    vjl_00 += gout[18] * dm_ik_24;
                    vjl_20 += gout[21] * dm_ik_24;
                    double dm_ik_25 = dm[(i0+2)*nao+(k0+5)];
                    vjl_10 += gout[24] * dm_ik_25;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_00 += gout[5] * dm_ik_41;
                    vjl_20 += gout[8] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_10 += gout[11] * dm_ik_42;
                    double dm_ik_43 = dm[(i0+4)*nao+(k0+3)];
                    vjl_00 += gout[14] * dm_ik_43;
                    vjl_20 += gout[17] * dm_ik_43;
                    double dm_ik_44 = dm[(i0+4)*nao+(k0+4)];
                    vjl_10 += gout[20] * dm_ik_44;
                    double dm_ik_45 = dm[(i0+4)*nao+(k0+5)];
                    vjl_00 += gout[23] * dm_ik_45;
                    vjl_20 += gout[26] * dm_ik_45;
                    break; }
                    case 3: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_10 += gout[1] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_00 += gout[4] * dm_ik_11;
                    vjl_20 += gout[7] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_10 += gout[10] * dm_ik_12;
                    double dm_ik_13 = dm[(i0+1)*nao+(k0+3)];
                    vjl_00 += gout[13] * dm_ik_13;
                    vjl_20 += gout[16] * dm_ik_13;
                    double dm_ik_14 = dm[(i0+1)*nao+(k0+4)];
                    vjl_10 += gout[19] * dm_ik_14;
                    double dm_ik_15 = dm[(i0+1)*nao+(k0+5)];
                    vjl_00 += gout[22] * dm_ik_15;
                    vjl_20 += gout[25] * dm_ik_15;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_30;
                    vjl_20 += gout[3] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_10 += gout[6] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_00 += gout[9] * dm_ik_32;
                    vjl_20 += gout[12] * dm_ik_32;
                    double dm_ik_33 = dm[(i0+3)*nao+(k0+3)];
                    vjl_10 += gout[15] * dm_ik_33;
                    double dm_ik_34 = dm[(i0+3)*nao+(k0+4)];
                    vjl_00 += gout[18] * dm_ik_34;
                    vjl_20 += gout[21] * dm_ik_34;
                    double dm_ik_35 = dm[(i0+3)*nao+(k0+5)];
                    vjl_10 += gout[24] * dm_ik_35;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_00 += gout[5] * dm_ik_51;
                    vjl_20 += gout[8] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_10 += gout[11] * dm_ik_52;
                    double dm_ik_53 = dm[(i0+5)*nao+(k0+3)];
                    vjl_00 += gout[14] * dm_ik_53;
                    vjl_20 += gout[17] * dm_ik_53;
                    double dm_ik_54 = dm[(i0+5)*nao+(k0+4)];
                    vjl_10 += gout[20] * dm_ik_54;
                    double dm_ik_55 = dm[(i0+5)*nao+(k0+5)];
                    vjl_00 += gout[23] * dm_ik_55;
                    vjl_20 += gout[26] * dm_ik_55;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                    atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                    atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                    switch (gout_id) {
                    case 0: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    double vjk_01 = gout[5]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[9]*dm_il_00 + gout[10]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double vjk_03 = gout[14]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                    double vjk_04 = gout[18]*dm_il_00 + gout[19]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                    double vjk_05 = gout[23]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    double vjk_10 = gout[2]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[6]*dm_il_00 + gout[7]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[11]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_13 = gout[15]*dm_il_00 + gout[16]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+3), vjk_13);
                    double vjk_14 = gout[20]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+4), vjk_14);
                    double vjk_15 = gout[24]*dm_il_00 + gout[25]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+5), vjk_15);
                    double vjk_20 = gout[3]*dm_il_00 + gout[4]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[8]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[12]*dm_il_00 + gout[13]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_23 = gout[17]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+3), vjk_23);
                    double vjk_24 = gout[21]*dm_il_00 + gout[22]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+4), vjk_24);
                    double vjk_25 = gout[26]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+5), vjk_25);
                    break; }
                    case 1: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_10 + gout[1]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    double vjk_01 = gout[5]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[9]*dm_il_10 + gout[10]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double vjk_03 = gout[14]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                    double vjk_04 = gout[18]*dm_il_10 + gout[19]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                    double vjk_05 = gout[23]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    double vjk_10 = gout[2]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[6]*dm_il_10 + gout[7]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[11]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_13 = gout[15]*dm_il_10 + gout[16]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+3), vjk_13);
                    double vjk_14 = gout[20]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+4), vjk_14);
                    double vjk_15 = gout[24]*dm_il_10 + gout[25]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+5), vjk_15);
                    double vjk_20 = gout[3]*dm_il_10 + gout[4]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[8]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[12]*dm_il_10 + gout[13]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_23 = gout[17]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+3), vjk_23);
                    double vjk_24 = gout[21]*dm_il_10 + gout[22]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+4), vjk_24);
                    double vjk_25 = gout[26]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+5), vjk_25);
                    break; }
                    case 2: {
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    double vjk_01 = gout[4]*dm_il_00 + gout[5]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[9]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double vjk_03 = gout[13]*dm_il_00 + gout[14]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                    double vjk_04 = gout[18]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                    double vjk_05 = gout[22]*dm_il_00 + gout[23]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    double vjk_10 = gout[1]*dm_il_00 + gout[2]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[6]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[10]*dm_il_00 + gout[11]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_13 = gout[15]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+3), vjk_13);
                    double vjk_14 = gout[19]*dm_il_00 + gout[20]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+4), vjk_14);
                    double vjk_15 = gout[24]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+5), vjk_15);
                    double vjk_20 = gout[3]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[7]*dm_il_00 + gout[8]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[12]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_23 = gout[16]*dm_il_00 + gout[17]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+3), vjk_23);
                    double vjk_24 = gout[21]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+4), vjk_24);
                    double vjk_25 = gout[25]*dm_il_00 + gout[26]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+5), vjk_25);
                    break; }
                    case 3: {
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    double vjk_01 = gout[4]*dm_il_10 + gout[5]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[9]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double vjk_03 = gout[13]*dm_il_10 + gout[14]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                    double vjk_04 = gout[18]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                    double vjk_05 = gout[22]*dm_il_10 + gout[23]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    double vjk_10 = gout[1]*dm_il_10 + gout[2]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[6]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[10]*dm_il_10 + gout[11]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_13 = gout[15]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+3), vjk_13);
                    double vjk_14 = gout[19]*dm_il_10 + gout[20]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+4), vjk_14);
                    double vjk_15 = gout[24]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+5), vjk_15);
                    double vjk_20 = gout[3]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[7]*dm_il_10 + gout[8]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[12]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_23 = gout[16]*dm_il_10 + gout[17]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+3), vjk_23);
                    double vjk_24 = gout[21]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+4), vjk_24);
                    double vjk_25 = gout[25]*dm_il_10 + gout[26]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+5), vjk_25);
                    break; }
                    }
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2200(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[36];
#pragma unroll
            for (int n = 0; n < 36; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                        gout[0] += hrr_2200x * fac * wt;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_1200x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_1200x * fac * trr_10z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += hrr_0200x * trr_20y * wt;
                        gout[4] += hrr_0200x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += hrr_0200x * fac * trr_20z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[6] += hrr_2100x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[7] += hrr_1100x * hrr_1100y * wt;
                        gout[8] += hrr_1100x * hrr_0100y * trr_10z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout[9] += hrr_0100x * hrr_2100y * wt;
                        gout[10] += hrr_0100x * hrr_1100y * trr_10z;
                        gout[11] += hrr_0100x * hrr_0100y * trr_20z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[12] += hrr_2100x * fac * hrr_0100z;
                        gout[13] += hrr_1100x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[14] += hrr_1100x * fac * hrr_1100z;
                        gout[15] += hrr_0100x * trr_20y * hrr_0100z;
                        gout[16] += hrr_0100x * trr_10y * hrr_1100z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout[17] += hrr_0100x * fac * hrr_2100z;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gout[18] += trr_20x * hrr_0200y * wt;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gout[19] += trr_10x * hrr_1200y * wt;
                        gout[20] += trr_10x * hrr_0200y * trr_10z;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                        gout[21] += 1 * hrr_2200y * wt;
                        gout[22] += 1 * hrr_1200y * trr_10z;
                        gout[23] += 1 * hrr_0200y * trr_20z;
                        gout[24] += trr_20x * hrr_0100y * hrr_0100z;
                        gout[25] += trr_10x * hrr_1100y * hrr_0100z;
                        gout[26] += trr_10x * hrr_0100y * hrr_1100z;
                        gout[27] += 1 * hrr_2100y * hrr_0100z;
                        gout[28] += 1 * hrr_1100y * hrr_1100z;
                        gout[29] += 1 * hrr_0100y * hrr_2100z;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gout[30] += trr_20x * fac * hrr_0200z;
                        gout[31] += trr_10x * trr_10y * hrr_0200z;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gout[32] += trr_10x * fac * hrr_1200z;
                        gout[33] += 1 * trr_20y * hrr_0200z;
                        gout[34] += 1 * trr_10y * hrr_1200z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                        gout[35] += 1 * fac * hrr_2200z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_30 = dm[(j0+3)*nao+(k0+0)];
                    double dm_jk_40 = dm[(j0+4)*nao+(k0+0)];
                    double dm_jk_50 = dm[(j0+5)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[6]*dm_jk_10 + gout[12]*dm_jk_20 + gout[18]*dm_jk_30 + gout[24]*dm_jk_40 + gout[30]*dm_jk_50;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[7]*dm_jk_10 + gout[13]*dm_jk_20 + gout[19]*dm_jk_30 + gout[25]*dm_jk_40 + gout[31]*dm_jk_50;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[8]*dm_jk_10 + gout[14]*dm_jk_20 + gout[20]*dm_jk_30 + gout[26]*dm_jk_40 + gout[32]*dm_jk_50;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[9]*dm_jk_10 + gout[15]*dm_jk_20 + gout[21]*dm_jk_30 + gout[27]*dm_jk_40 + gout[33]*dm_jk_50;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[10]*dm_jk_10 + gout[16]*dm_jk_20 + gout[22]*dm_jk_30 + gout[28]*dm_jk_40 + gout[34]*dm_jk_50;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[11]*dm_jk_10 + gout[17]*dm_jk_20 + gout[23]*dm_jk_30 + gout[29]*dm_jk_40 + gout[35]*dm_jk_50;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_30 = dm[(j0+3)*nao+(l0+0)];
                    double dm_jl_40 = dm[(j0+4)*nao+(l0+0)];
                    double dm_jl_50 = dm[(j0+5)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[6]*dm_jl_10 + gout[12]*dm_jl_20 + gout[18]*dm_jl_30 + gout[24]*dm_jl_40 + gout[30]*dm_jl_50;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[7]*dm_jl_10 + gout[13]*dm_jl_20 + gout[19]*dm_jl_30 + gout[25]*dm_jl_40 + gout[31]*dm_jl_50;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[8]*dm_jl_10 + gout[14]*dm_jl_20 + gout[20]*dm_jl_30 + gout[26]*dm_jl_40 + gout[32]*dm_jl_50;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00 + gout[9]*dm_jl_10 + gout[15]*dm_jl_20 + gout[21]*dm_jl_30 + gout[27]*dm_jl_40 + gout[33]*dm_jl_50;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00 + gout[10]*dm_jl_10 + gout[16]*dm_jl_20 + gout[22]*dm_jl_30 + gout[28]*dm_jl_40 + gout[34]*dm_jl_50;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00 + gout[11]*dm_jl_10 + gout[17]*dm_jl_20 + gout[23]*dm_jl_30 + gout[29]*dm_jl_40 + gout[35]*dm_jl_50;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    if (ish != jsh) {
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        double vjl_00 = gout[0]*dm_ik_00 + gout[1]*dm_ik_10 + gout[2]*dm_ik_20 + gout[3]*dm_ik_30 + gout[4]*dm_ik_40 + gout[5]*dm_ik_50;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjl_10 = gout[6]*dm_ik_00 + gout[7]*dm_ik_10 + gout[8]*dm_ik_20 + gout[9]*dm_ik_30 + gout[10]*dm_ik_40 + gout[11]*dm_ik_50;
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        double vjl_20 = gout[12]*dm_ik_00 + gout[13]*dm_ik_10 + gout[14]*dm_ik_20 + gout[15]*dm_ik_30 + gout[16]*dm_ik_40 + gout[17]*dm_ik_50;
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double vjl_30 = gout[18]*dm_ik_00 + gout[19]*dm_ik_10 + gout[20]*dm_ik_20 + gout[21]*dm_ik_30 + gout[22]*dm_ik_40 + gout[23]*dm_ik_50;
                        atomicAdd(vk+(j0+3)*nao+(l0+0), vjl_30);
                        double vjl_40 = gout[24]*dm_ik_00 + gout[25]*dm_ik_10 + gout[26]*dm_ik_20 + gout[27]*dm_ik_30 + gout[28]*dm_ik_40 + gout[29]*dm_ik_50;
                        atomicAdd(vk+(j0+4)*nao+(l0+0), vjl_40);
                        double vjl_50 = gout[30]*dm_ik_00 + gout[31]*dm_ik_10 + gout[32]*dm_ik_20 + gout[33]*dm_ik_30 + gout[34]*dm_ik_40 + gout[35]*dm_ik_50;
                        atomicAdd(vk+(j0+5)*nao+(l0+0), vjl_50);
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_10 + gout[2]*dm_il_20 + gout[3]*dm_il_30 + gout[4]*dm_il_40 + gout[5]*dm_il_50;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        double vjk_10 = gout[6]*dm_il_00 + gout[7]*dm_il_10 + gout[8]*dm_il_20 + gout[9]*dm_il_30 + gout[10]*dm_il_40 + gout[11]*dm_il_50;
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        double vjk_20 = gout[12]*dm_il_00 + gout[13]*dm_il_10 + gout[14]*dm_il_20 + gout[15]*dm_il_30 + gout[16]*dm_il_40 + gout[17]*dm_il_50;
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                        double vjk_30 = gout[18]*dm_il_00 + gout[19]*dm_il_10 + gout[20]*dm_il_20 + gout[21]*dm_il_30 + gout[22]*dm_il_40 + gout[23]*dm_il_50;
                        atomicAdd(vk+(j0+3)*nao+(k0+0), vjk_30);
                        double vjk_40 = gout[24]*dm_il_00 + gout[25]*dm_il_10 + gout[26]*dm_il_20 + gout[27]*dm_il_30 + gout[28]*dm_il_40 + gout[29]*dm_il_50;
                        atomicAdd(vk+(j0+4)*nao+(k0+0), vjk_40);
                        double vjk_50 = gout[30]*dm_il_00 + gout[31]*dm_il_10 + gout[32]*dm_il_20 + gout[33]*dm_il_30 + gout[34]*dm_il_40 + gout[35]*dm_il_50;
                        atomicAdd(vk+(j0+5)*nao+(k0+0), vjk_50);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_2210(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int gout_id = item.get_local_id(0);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif
    int t_id = 64 * gout_id + sq_id;
    constexpr int threads = 256;
    constexpr int nsq_per_block = 64;
    constexpr int g_size = 18;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+8);

while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
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
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0] = xlxk;
                rlrk[64] = ylyk;
                rlrk[128] = zlzk;
            }
            double gout[27];

            #pragma unroll
            for (int n = 0; n < 27; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                    int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
                    int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                    int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0];
                    double ylyk = rlrk[64];
                    double zlzk = rlrk[128];
                    double theta_kl = ak * al_akl;
                    double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = PI_FAC;
                    if (task_id < ntasks) {
                        if (ksh == lsh) fac_sym *= .5;
                        if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                    } else {
                        fac_sym = 0;
                    }
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + rjri[0] * aj_aij;
                    double yij = ri[1] + rjri[1] * aj_aij;
                    double zij = ri[2] + rjri[2] * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double s0, s1, s2;
                        double rt = rw[irys*128];
                        double aij = aij_cache[0];
                        double rt_aa = rt / (aij + akl);
                        double akl = akl_cache[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        for (int n = gout_id; n < 3; n += 4) {
                            if (n == 2) {
                                gx[2304] = rw[irys*128+64];
                            }
                            double *_gx = gx + n * 1152;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aij_cache[1];
                            double c0x = Rpa - rt_aij * Rpq[n*64];
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
                            double xlxk = rlrk[n*64];
                            double Rqc = xlxk * akl_cache[64];
                            double cpx = Rqc + rt_akl * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[576] = s1;
                            s0 = _gx[64];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[640] = s1;
                            s0 = _gx[128];
                            s1 = cpx * s0;
                            s1 += 2 * b00 * _gx[64];
                            _gx[704] = s1;
                            s0 = _gx[192];
                            s1 = cpx * s0;
                            s1 += 3 * b00 * _gx[128];
                            _gx[768] = s1;
                            s0 = _gx[256];
                            s1 = cpx * s0;
                            s1 += 4 * b00 * _gx[192];
                            _gx[832] = s1;
                            s1 = _gx[256];
                            s0 = _gx[192];
                            _gx[384] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[128];
                            _gx[320] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[64];
                            _gx[256] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[0];
                            _gx[192] = s1 - xjxi * s0;
                            s1 = _gx[384];
                            s0 = _gx[320];
                            _gx[512] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[256];
                            _gx[448] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[192];
                            _gx[384] = s1 - xjxi * s0;
                            s1 = _gx[832];
                            s0 = _gx[768];
                            _gx[960] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[704];
                            _gx[896] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[640];
                            _gx[832] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[576];
                            _gx[768] = s1 - xjxi * s0;
                            s1 = _gx[960];
                            s0 = _gx[896];
                            _gx[1088] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[832];
                            _gx[1024] = s1 - xjxi * s0;
                            s1 = s0;
                            s0 = _gx[768];
                            _gx[960] = s1 - xjxi * s0;
                        }
                        __syncthreads();
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[1088] * gx[1152] * gx[2304];
                        gout[1] += gx[960] * gx[1216] * gx[2368];
                        gout[2] += gx[832] * gx[1344] * gx[2368];
                        gout[3] += gx[896] * gx[1152] * gx[2496];
                        gout[4] += gx[768] * gx[1216] * gx[2560];
                        gout[5] += gx[640] * gx[1536] * gx[2368];
                        gout[6] += gx[704] * gx[1344] * gx[2496];
                        gout[7] += gx[576] * gx[1408] * gx[2560];
                        gout[8] += gx[640] * gx[1152] * gx[2752];
                        gout[9] += gx[512] * gx[1728] * gx[2304];
                        gout[10] += gx[384] * gx[1792] * gx[2368];
                        gout[11] += gx[256] * gx[1920] * gx[2368];
                        gout[12] += gx[320] * gx[1728] * gx[2496];
                        gout[13] += gx[192] * gx[1792] * gx[2560];
                        gout[14] += gx[64] * gx[2112] * gx[2368];
                        gout[15] += gx[128] * gx[1920] * gx[2496];
                        gout[16] += gx[0] * gx[1984] * gx[2560];
                        gout[17] += gx[64] * gx[1728] * gx[2752];
                        gout[18] += gx[512] * gx[1152] * gx[2880];
                        gout[19] += gx[384] * gx[1216] * gx[2944];
                        gout[20] += gx[256] * gx[1344] * gx[2944];
                        gout[21] += gx[320] * gx[1152] * gx[3072];
                        gout[22] += gx[192] * gx[1216] * gx[3136];
                        gout[23] += gx[64] * gx[1536] * gx[2944];
                        gout[24] += gx[128] * gx[1344] * gx[3072];
                        gout[25] += gx[0] * gx[1408] * gx[3136];
                        gout[26] += gx[64] * gx[1152] * gx[3328];
                        break;
                        case 1:
                        gout[0] += gx[1024] * gx[1216] * gx[2304];
                        gout[1] += gx[960] * gx[1152] * gx[2432];
                        gout[2] += gx[768] * gx[1472] * gx[2304];
                        gout[3] += gx[832] * gx[1216] * gx[2496];
                        gout[4] += gx[768] * gx[1152] * gx[2624];
                        gout[5] += gx[576] * gx[1664] * gx[2304];
                        gout[6] += gx[640] * gx[1408] * gx[2496];
                        gout[7] += gx[576] * gx[1344] * gx[2624];
                        gout[8] += gx[576] * gx[1280] * gx[2688];
                        gout[9] += gx[448] * gx[1792] * gx[2304];
                        gout[10] += gx[384] * gx[1728] * gx[2432];
                        gout[11] += gx[192] * gx[2048] * gx[2304];
                        gout[12] += gx[256] * gx[1792] * gx[2496];
                        gout[13] += gx[192] * gx[1728] * gx[2624];
                        gout[14] += gx[0] * gx[2240] * gx[2304];
                        gout[15] += gx[64] * gx[1984] * gx[2496];
                        gout[16] += gx[0] * gx[1920] * gx[2624];
                        gout[17] += gx[0] * gx[1856] * gx[2688];
                        gout[18] += gx[448] * gx[1216] * gx[2880];
                        gout[19] += gx[384] * gx[1152] * gx[3008];
                        gout[20] += gx[192] * gx[1472] * gx[2880];
                        gout[21] += gx[256] * gx[1216] * gx[3072];
                        gout[22] += gx[192] * gx[1152] * gx[3200];
                        gout[23] += gx[0] * gx[1664] * gx[2880];
                        gout[24] += gx[64] * gx[1408] * gx[3072];
                        gout[25] += gx[0] * gx[1344] * gx[3200];
                        gout[26] += gx[0] * gx[1280] * gx[3264];
                        break;
                        case 2:
                        gout[0] += gx[1024] * gx[1152] * gx[2368];
                        gout[1] += gx[896] * gx[1344] * gx[2304];
                        gout[2] += gx[768] * gx[1408] * gx[2368];
                        gout[3] += gx[832] * gx[1152] * gx[2560];
                        gout[4] += gx[704] * gx[1536] * gx[2304];
                        gout[5] += gx[576] * gx[1600] * gx[2368];
                        gout[6] += gx[640] * gx[1344] * gx[2560];
                        gout[7] += gx[704] * gx[1152] * gx[2688];
                        gout[8] += gx[576] * gx[1216] * gx[2752];
                        gout[9] += gx[448] * gx[1728] * gx[2368];
                        gout[10] += gx[320] * gx[1920] * gx[2304];
                        gout[11] += gx[192] * gx[1984] * gx[2368];
                        gout[12] += gx[256] * gx[1728] * gx[2560];
                        gout[13] += gx[128] * gx[2112] * gx[2304];
                        gout[14] += gx[0] * gx[2176] * gx[2368];
                        gout[15] += gx[64] * gx[1920] * gx[2560];
                        gout[16] += gx[128] * gx[1728] * gx[2688];
                        gout[17] += gx[0] * gx[1792] * gx[2752];
                        gout[18] += gx[448] * gx[1152] * gx[2944];
                        gout[19] += gx[320] * gx[1344] * gx[2880];
                        gout[20] += gx[192] * gx[1408] * gx[2944];
                        gout[21] += gx[256] * gx[1152] * gx[3136];
                        gout[22] += gx[128] * gx[1536] * gx[2880];
                        gout[23] += gx[0] * gx[1600] * gx[2944];
                        gout[24] += gx[64] * gx[1344] * gx[3136];
                        gout[25] += gx[128] * gx[1152] * gx[3264];
                        gout[26] += gx[0] * gx[1216] * gx[3328];
                        break;
                        case 3:
                        gout[0] += gx[960] * gx[1280] * gx[2304];
                        gout[1] += gx[832] * gx[1408] * gx[2304];
                        gout[2] += gx[768] * gx[1344] * gx[2432];
                        gout[3] += gx[768] * gx[1280] * gx[2496];
                        gout[4] += gx[640] * gx[1600] * gx[2304];
                        gout[5] += gx[576] * gx[1536] * gx[2432];
                        gout[6] += gx[576] * gx[1472] * gx[2496];
                        gout[7] += gx[640] * gx[1216] * gx[2688];
                        gout[8] += gx[576] * gx[1152] * gx[2816];
                        gout[9] += gx[384] * gx[1856] * gx[2304];
                        gout[10] += gx[256] * gx[1984] * gx[2304];
                        gout[11] += gx[192] * gx[1920] * gx[2432];
                        gout[12] += gx[192] * gx[1856] * gx[2496];
                        gout[13] += gx[64] * gx[2176] * gx[2304];
                        gout[14] += gx[0] * gx[2112] * gx[2432];
                        gout[15] += gx[0] * gx[2048] * gx[2496];
                        gout[16] += gx[64] * gx[1792] * gx[2688];
                        gout[17] += gx[0] * gx[1728] * gx[2816];
                        gout[18] += gx[384] * gx[1280] * gx[2880];
                        gout[19] += gx[256] * gx[1408] * gx[2880];
                        gout[20] += gx[192] * gx[1344] * gx[3008];
                        gout[21] += gx[192] * gx[1280] * gx[3072];
                        gout[22] += gx[64] * gx[1600] * gx[2880];
                        gout[23] += gx[0] * gx[1536] * gx[3008];
                        gout[24] += gx[0] * gx[1472] * gx[3072];
                        gout[25] += gx[64] * gx[1216] * gx[3264];
                        gout[26] += gx[0] * gx[1152] * gx[3392];
                        break;
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao * nao;
                    double *vk = kmat.vk + i_dm * nao * nao;
                    double vil_00 = 0;
                    double vil_10 = 0;
                    double vil_20 = 0;
                    double vil_30 = 0;
                    double vil_40 = 0;
                    double vil_50 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_00 += gout[0] * dm_jk_00;
                    vil_40 += gout[1] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_00 += gout[9] * dm_jk_01;
                    vil_40 += gout[10] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_00 += gout[18] * dm_jk_02;
                    vil_40 += gout[19] * dm_jk_02;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_20 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_20 += gout[11] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_20 += gout[20] * dm_jk_12;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_00 += gout[3] * dm_jk_20;
                    vil_40 += gout[4] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_00 += gout[12] * dm_jk_21;
                    vil_40 += gout[13] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_00 += gout[21] * dm_jk_22;
                    vil_40 += gout[22] * dm_jk_22;
                    double dm_jk_30 = dm[(j0+3)*nao+(k0+0)];
                    vil_20 += gout[5] * dm_jk_30;
                    double dm_jk_31 = dm[(j0+3)*nao+(k0+1)];
                    vil_20 += gout[14] * dm_jk_31;
                    double dm_jk_32 = dm[(j0+3)*nao+(k0+2)];
                    vil_20 += gout[23] * dm_jk_32;
                    double dm_jk_40 = dm[(j0+4)*nao+(k0+0)];
                    vil_00 += gout[6] * dm_jk_40;
                    vil_40 += gout[7] * dm_jk_40;
                    double dm_jk_41 = dm[(j0+4)*nao+(k0+1)];
                    vil_00 += gout[15] * dm_jk_41;
                    vil_40 += gout[16] * dm_jk_41;
                    double dm_jk_42 = dm[(j0+4)*nao+(k0+2)];
                    vil_00 += gout[24] * dm_jk_42;
                    vil_40 += gout[25] * dm_jk_42;
                    double dm_jk_50 = dm[(j0+5)*nao+(k0+0)];
                    vil_20 += gout[8] * dm_jk_50;
                    double dm_jk_51 = dm[(j0+5)*nao+(k0+1)];
                    vil_20 += gout[17] * dm_jk_51;
                    double dm_jk_52 = dm[(j0+5)*nao+(k0+2)];
                    vil_20 += gout[26] * dm_jk_52;
                    break; }
                    case 1: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_10 += gout[0] * dm_jk_00;
                    vil_50 += gout[1] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_10 += gout[9] * dm_jk_01;
                    vil_50 += gout[10] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_10 += gout[18] * dm_jk_02;
                    vil_50 += gout[19] * dm_jk_02;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_30 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_30 += gout[11] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_30 += gout[20] * dm_jk_12;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_10 += gout[3] * dm_jk_20;
                    vil_50 += gout[4] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_10 += gout[12] * dm_jk_21;
                    vil_50 += gout[13] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_10 += gout[21] * dm_jk_22;
                    vil_50 += gout[22] * dm_jk_22;
                    double dm_jk_30 = dm[(j0+3)*nao+(k0+0)];
                    vil_30 += gout[5] * dm_jk_30;
                    double dm_jk_31 = dm[(j0+3)*nao+(k0+1)];
                    vil_30 += gout[14] * dm_jk_31;
                    double dm_jk_32 = dm[(j0+3)*nao+(k0+2)];
                    vil_30 += gout[23] * dm_jk_32;
                    double dm_jk_40 = dm[(j0+4)*nao+(k0+0)];
                    vil_10 += gout[6] * dm_jk_40;
                    vil_50 += gout[7] * dm_jk_40;
                    double dm_jk_41 = dm[(j0+4)*nao+(k0+1)];
                    vil_10 += gout[15] * dm_jk_41;
                    vil_50 += gout[16] * dm_jk_41;
                    double dm_jk_42 = dm[(j0+4)*nao+(k0+2)];
                    vil_10 += gout[24] * dm_jk_42;
                    vil_50 += gout[25] * dm_jk_42;
                    double dm_jk_50 = dm[(j0+5)*nao+(k0+0)];
                    vil_30 += gout[8] * dm_jk_50;
                    double dm_jk_51 = dm[(j0+5)*nao+(k0+1)];
                    vil_30 += gout[17] * dm_jk_51;
                    double dm_jk_52 = dm[(j0+5)*nao+(k0+2)];
                    vil_30 += gout[26] * dm_jk_52;
                    break; }
                    case 2: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_20 += gout[0] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_20 += gout[9] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_20 += gout[18] * dm_jk_02;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_00 += gout[1] * dm_jk_10;
                    vil_40 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_00 += gout[10] * dm_jk_11;
                    vil_40 += gout[11] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_00 += gout[19] * dm_jk_12;
                    vil_40 += gout[20] * dm_jk_12;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_20 += gout[3] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_20 += gout[12] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_20 += gout[21] * dm_jk_22;
                    double dm_jk_30 = dm[(j0+3)*nao+(k0+0)];
                    vil_00 += gout[4] * dm_jk_30;
                    vil_40 += gout[5] * dm_jk_30;
                    double dm_jk_31 = dm[(j0+3)*nao+(k0+1)];
                    vil_00 += gout[13] * dm_jk_31;
                    vil_40 += gout[14] * dm_jk_31;
                    double dm_jk_32 = dm[(j0+3)*nao+(k0+2)];
                    vil_00 += gout[22] * dm_jk_32;
                    vil_40 += gout[23] * dm_jk_32;
                    double dm_jk_40 = dm[(j0+4)*nao+(k0+0)];
                    vil_20 += gout[6] * dm_jk_40;
                    double dm_jk_41 = dm[(j0+4)*nao+(k0+1)];
                    vil_20 += gout[15] * dm_jk_41;
                    double dm_jk_42 = dm[(j0+4)*nao+(k0+2)];
                    vil_20 += gout[24] * dm_jk_42;
                    double dm_jk_50 = dm[(j0+5)*nao+(k0+0)];
                    vil_00 += gout[7] * dm_jk_50;
                    vil_40 += gout[8] * dm_jk_50;
                    double dm_jk_51 = dm[(j0+5)*nao+(k0+1)];
                    vil_00 += gout[16] * dm_jk_51;
                    vil_40 += gout[17] * dm_jk_51;
                    double dm_jk_52 = dm[(j0+5)*nao+(k0+2)];
                    vil_00 += gout[25] * dm_jk_52;
                    vil_40 += gout[26] * dm_jk_52;
                    break; }
                    case 3: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    vil_30 += gout[0] * dm_jk_00;
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    vil_30 += gout[9] * dm_jk_01;
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    vil_30 += gout[18] * dm_jk_02;
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    vil_10 += gout[1] * dm_jk_10;
                    vil_50 += gout[2] * dm_jk_10;
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    vil_10 += gout[10] * dm_jk_11;
                    vil_50 += gout[11] * dm_jk_11;
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    vil_10 += gout[19] * dm_jk_12;
                    vil_50 += gout[20] * dm_jk_12;
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    vil_30 += gout[3] * dm_jk_20;
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    vil_30 += gout[12] * dm_jk_21;
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    vil_30 += gout[21] * dm_jk_22;
                    double dm_jk_30 = dm[(j0+3)*nao+(k0+0)];
                    vil_10 += gout[4] * dm_jk_30;
                    vil_50 += gout[5] * dm_jk_30;
                    double dm_jk_31 = dm[(j0+3)*nao+(k0+1)];
                    vil_10 += gout[13] * dm_jk_31;
                    vil_50 += gout[14] * dm_jk_31;
                    double dm_jk_32 = dm[(j0+3)*nao+(k0+2)];
                    vil_10 += gout[22] * dm_jk_32;
                    vil_50 += gout[23] * dm_jk_32;
                    double dm_jk_40 = dm[(j0+4)*nao+(k0+0)];
                    vil_30 += gout[6] * dm_jk_40;
                    double dm_jk_41 = dm[(j0+4)*nao+(k0+1)];
                    vil_30 += gout[15] * dm_jk_41;
                    double dm_jk_42 = dm[(j0+4)*nao+(k0+2)];
                    vil_30 += gout[24] * dm_jk_42;
                    double dm_jk_50 = dm[(j0+5)*nao+(k0+0)];
                    vil_10 += gout[7] * dm_jk_50;
                    vil_50 += gout[8] * dm_jk_50;
                    double dm_jk_51 = dm[(j0+5)*nao+(k0+1)];
                    vil_10 += gout[16] * dm_jk_51;
                    vil_50 += gout[17] * dm_jk_51;
                    double dm_jk_52 = dm[(j0+5)*nao+(k0+2)];
                    vil_10 += gout[25] * dm_jk_52;
                    vil_50 += gout[26] * dm_jk_52;
                    break; }
                    }
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    switch (gout_id) {
                    case 0: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_40 = dm[(j0+4)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20 + gout[6]*dm_jl_40;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20 + gout[15]*dm_jl_40;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20 + gout[24]*dm_jl_40;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_30 = dm[(j0+3)*nao+(l0+0)];
                    double dm_jl_50 = dm[(j0+5)*nao+(l0+0)];
                    double vik_20 = gout[2]*dm_jl_10 + gout[5]*dm_jl_30 + gout[8]*dm_jl_50;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[11]*dm_jl_10 + gout[14]*dm_jl_30 + gout[17]*dm_jl_50;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[20]*dm_jl_10 + gout[23]*dm_jl_30 + gout[26]*dm_jl_50;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[1]*dm_jl_00 + gout[4]*dm_jl_20 + gout[7]*dm_jl_40;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[10]*dm_jl_00 + gout[13]*dm_jl_20 + gout[16]*dm_jl_40;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[19]*dm_jl_00 + gout[22]*dm_jl_20 + gout[25]*dm_jl_40;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    break; }
                    case 1: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_40 = dm[(j0+4)*nao+(l0+0)];
                    double vik_10 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20 + gout[6]*dm_jl_40;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20 + gout[15]*dm_jl_40;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20 + gout[24]*dm_jl_40;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_30 = dm[(j0+3)*nao+(l0+0)];
                    double dm_jl_50 = dm[(j0+5)*nao+(l0+0)];
                    double vik_30 = gout[2]*dm_jl_10 + gout[5]*dm_jl_30 + gout[8]*dm_jl_50;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[11]*dm_jl_10 + gout[14]*dm_jl_30 + gout[17]*dm_jl_50;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[20]*dm_jl_10 + gout[23]*dm_jl_30 + gout[26]*dm_jl_50;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[1]*dm_jl_00 + gout[4]*dm_jl_20 + gout[7]*dm_jl_40;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[10]*dm_jl_00 + gout[13]*dm_jl_20 + gout[16]*dm_jl_40;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[19]*dm_jl_00 + gout[22]*dm_jl_20 + gout[25]*dm_jl_40;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    break; }
                    case 2: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_30 = dm[(j0+3)*nao+(l0+0)];
                    double dm_jl_50 = dm[(j0+5)*nao+(l0+0)];
                    double vik_00 = gout[1]*dm_jl_10 + gout[4]*dm_jl_30 + gout[7]*dm_jl_50;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[10]*dm_jl_10 + gout[13]*dm_jl_30 + gout[16]*dm_jl_50;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[19]*dm_jl_10 + gout[22]*dm_jl_30 + gout[25]*dm_jl_50;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_40 = dm[(j0+4)*nao+(l0+0)];
                    double vik_20 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20 + gout[6]*dm_jl_40;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20 + gout[15]*dm_jl_40;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20 + gout[24]*dm_jl_40;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[2]*dm_jl_10 + gout[5]*dm_jl_30 + gout[8]*dm_jl_50;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[11]*dm_jl_10 + gout[14]*dm_jl_30 + gout[17]*dm_jl_50;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[20]*dm_jl_10 + gout[23]*dm_jl_30 + gout[26]*dm_jl_50;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    break; }
                    case 3: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_30 = dm[(j0+3)*nao+(l0+0)];
                    double dm_jl_50 = dm[(j0+5)*nao+(l0+0)];
                    double vik_10 = gout[1]*dm_jl_10 + gout[4]*dm_jl_30 + gout[7]*dm_jl_50;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[10]*dm_jl_10 + gout[13]*dm_jl_30 + gout[16]*dm_jl_50;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[19]*dm_jl_10 + gout[22]*dm_jl_30 + gout[25]*dm_jl_50;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_40 = dm[(j0+4)*nao+(l0+0)];
                    double vik_30 = gout[0]*dm_jl_00 + gout[3]*dm_jl_20 + gout[6]*dm_jl_40;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[9]*dm_jl_00 + gout[12]*dm_jl_20 + gout[15]*dm_jl_40;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[18]*dm_jl_00 + gout[21]*dm_jl_20 + gout[24]*dm_jl_40;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[2]*dm_jl_10 + gout[5]*dm_jl_30 + gout[8]*dm_jl_50;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_10 + gout[14]*dm_jl_30 + gout[17]*dm_jl_50;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[20]*dm_jl_10 + gout[23]*dm_jl_30 + gout[26]*dm_jl_50;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    break; }
                    }
                    if (ish != jsh) {
                    double vjl_00 = 0;
                    double vjl_10 = 0;
                    double vjl_20 = 0;
                    double vjl_30 = 0;
                    double vjl_40 = 0;
                    double vjl_50 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_00;
                    vjl_20 += gout[3] * dm_ik_00;
                    vjl_40 += gout[6] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_01;
                    vjl_20 += gout[12] * dm_ik_01;
                    vjl_40 += gout[15] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_00 += gout[18] * dm_ik_02;
                    vjl_20 += gout[21] * dm_ik_02;
                    vjl_40 += gout[24] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_20;
                    vjl_30 += gout[5] * dm_ik_20;
                    vjl_50 += gout[8] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_21;
                    vjl_30 += gout[14] * dm_ik_21;
                    vjl_50 += gout[17] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_10 += gout[20] * dm_ik_22;
                    vjl_30 += gout[23] * dm_ik_22;
                    vjl_50 += gout[26] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_40;
                    vjl_20 += gout[4] * dm_ik_40;
                    vjl_40 += gout[7] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_00 += gout[10] * dm_ik_41;
                    vjl_20 += gout[13] * dm_ik_41;
                    vjl_40 += gout[16] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_00 += gout[19] * dm_ik_42;
                    vjl_20 += gout[22] * dm_ik_42;
                    vjl_40 += gout[25] * dm_ik_42;
                    break; }
                    case 1: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_10;
                    vjl_20 += gout[3] * dm_ik_10;
                    vjl_40 += gout[6] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_11;
                    vjl_20 += gout[12] * dm_ik_11;
                    vjl_40 += gout[15] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_00 += gout[18] * dm_ik_12;
                    vjl_20 += gout[21] * dm_ik_12;
                    vjl_40 += gout[24] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_30;
                    vjl_30 += gout[5] * dm_ik_30;
                    vjl_50 += gout[8] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_31;
                    vjl_30 += gout[14] * dm_ik_31;
                    vjl_50 += gout[17] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_10 += gout[20] * dm_ik_32;
                    vjl_30 += gout[23] * dm_ik_32;
                    vjl_50 += gout[26] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_50;
                    vjl_20 += gout[4] * dm_ik_50;
                    vjl_40 += gout[7] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_00 += gout[10] * dm_ik_51;
                    vjl_20 += gout[13] * dm_ik_51;
                    vjl_40 += gout[16] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_00 += gout[19] * dm_ik_52;
                    vjl_20 += gout[22] * dm_ik_52;
                    vjl_40 += gout[25] * dm_ik_52;
                    break; }
                    case 2: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_10 += gout[1] * dm_ik_00;
                    vjl_30 += gout[4] * dm_ik_00;
                    vjl_50 += gout[7] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_10 += gout[10] * dm_ik_01;
                    vjl_30 += gout[13] * dm_ik_01;
                    vjl_50 += gout[16] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_10 += gout[19] * dm_ik_02;
                    vjl_30 += gout[22] * dm_ik_02;
                    vjl_50 += gout[25] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_20;
                    vjl_20 += gout[3] * dm_ik_20;
                    vjl_40 += gout[6] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_21;
                    vjl_20 += gout[12] * dm_ik_21;
                    vjl_40 += gout[15] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_00 += gout[18] * dm_ik_22;
                    vjl_20 += gout[21] * dm_ik_22;
                    vjl_40 += gout[24] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_40;
                    vjl_30 += gout[5] * dm_ik_40;
                    vjl_50 += gout[8] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_41;
                    vjl_30 += gout[14] * dm_ik_41;
                    vjl_50 += gout[17] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_10 += gout[20] * dm_ik_42;
                    vjl_30 += gout[23] * dm_ik_42;
                    vjl_50 += gout[26] * dm_ik_42;
                    break; }
                    case 3: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_10 += gout[1] * dm_ik_10;
                    vjl_30 += gout[4] * dm_ik_10;
                    vjl_50 += gout[7] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_10 += gout[10] * dm_ik_11;
                    vjl_30 += gout[13] * dm_ik_11;
                    vjl_50 += gout[16] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_10 += gout[19] * dm_ik_12;
                    vjl_30 += gout[22] * dm_ik_12;
                    vjl_50 += gout[25] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_30;
                    vjl_20 += gout[3] * dm_ik_30;
                    vjl_40 += gout[6] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_31;
                    vjl_20 += gout[12] * dm_ik_31;
                    vjl_40 += gout[15] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_00 += gout[18] * dm_ik_32;
                    vjl_20 += gout[21] * dm_ik_32;
                    vjl_40 += gout[24] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_50;
                    vjl_30 += gout[5] * dm_ik_50;
                    vjl_50 += gout[8] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_51;
                    vjl_30 += gout[14] * dm_ik_51;
                    vjl_50 += gout[17] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_10 += gout[20] * dm_ik_52;
                    vjl_30 += gout[23] * dm_ik_52;
                    vjl_50 += gout[26] * dm_ik_52;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                    atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                    atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                    atomicAdd(vk+(j0+3)*nao+(l0+0), vjl_30);
                    atomicAdd(vk+(j0+4)*nao+(l0+0), vjl_40);
                    atomicAdd(vk+(j0+5)*nao+(l0+0), vjl_50);
                    switch (gout_id) {
                    case 0: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_00 + gout[1]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double vjk_01 = gout[9]*dm_il_00 + gout[10]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[18]*dm_il_00 + gout[19]*dm_il_40;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    double vjk_10 = gout[2]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[11]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[20]*dm_il_20;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_20 = gout[3]*dm_il_00 + gout[4]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[12]*dm_il_00 + gout[13]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[21]*dm_il_00 + gout[22]*dm_il_40;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_30 = gout[5]*dm_il_20;
                    atomicAdd(vk+(j0+3)*nao+(k0+0), vjk_30);
                    double vjk_31 = gout[14]*dm_il_20;
                    atomicAdd(vk+(j0+3)*nao+(k0+1), vjk_31);
                    double vjk_32 = gout[23]*dm_il_20;
                    atomicAdd(vk+(j0+3)*nao+(k0+2), vjk_32);
                    double vjk_40 = gout[6]*dm_il_00 + gout[7]*dm_il_40;
                    atomicAdd(vk+(j0+4)*nao+(k0+0), vjk_40);
                    double vjk_41 = gout[15]*dm_il_00 + gout[16]*dm_il_40;
                    atomicAdd(vk+(j0+4)*nao+(k0+1), vjk_41);
                    double vjk_42 = gout[24]*dm_il_00 + gout[25]*dm_il_40;
                    atomicAdd(vk+(j0+4)*nao+(k0+2), vjk_42);
                    double vjk_50 = gout[8]*dm_il_20;
                    atomicAdd(vk+(j0+5)*nao+(k0+0), vjk_50);
                    double vjk_51 = gout[17]*dm_il_20;
                    atomicAdd(vk+(j0+5)*nao+(k0+1), vjk_51);
                    double vjk_52 = gout[26]*dm_il_20;
                    atomicAdd(vk+(j0+5)*nao+(k0+2), vjk_52);
                    break; }
                    case 1: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_10 + gout[1]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double vjk_01 = gout[9]*dm_il_10 + gout[10]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[18]*dm_il_10 + gout[19]*dm_il_50;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    double vjk_10 = gout[2]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[11]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[20]*dm_il_30;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_20 = gout[3]*dm_il_10 + gout[4]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[12]*dm_il_10 + gout[13]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[21]*dm_il_10 + gout[22]*dm_il_50;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_30 = gout[5]*dm_il_30;
                    atomicAdd(vk+(j0+3)*nao+(k0+0), vjk_30);
                    double vjk_31 = gout[14]*dm_il_30;
                    atomicAdd(vk+(j0+3)*nao+(k0+1), vjk_31);
                    double vjk_32 = gout[23]*dm_il_30;
                    atomicAdd(vk+(j0+3)*nao+(k0+2), vjk_32);
                    double vjk_40 = gout[6]*dm_il_10 + gout[7]*dm_il_50;
                    atomicAdd(vk+(j0+4)*nao+(k0+0), vjk_40);
                    double vjk_41 = gout[15]*dm_il_10 + gout[16]*dm_il_50;
                    atomicAdd(vk+(j0+4)*nao+(k0+1), vjk_41);
                    double vjk_42 = gout[24]*dm_il_10 + gout[25]*dm_il_50;
                    atomicAdd(vk+(j0+4)*nao+(k0+2), vjk_42);
                    double vjk_50 = gout[8]*dm_il_30;
                    atomicAdd(vk+(j0+5)*nao+(k0+0), vjk_50);
                    double vjk_51 = gout[17]*dm_il_30;
                    atomicAdd(vk+(j0+5)*nao+(k0+1), vjk_51);
                    double vjk_52 = gout[26]*dm_il_30;
                    atomicAdd(vk+(j0+5)*nao+(k0+2), vjk_52);
                    break; }
                    case 2: {
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double vjk_01 = gout[9]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[18]*dm_il_20;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    double vjk_10 = gout[1]*dm_il_00 + gout[2]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[10]*dm_il_00 + gout[11]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[19]*dm_il_00 + gout[20]*dm_il_40;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_20 = gout[3]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[12]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[21]*dm_il_20;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_30 = gout[4]*dm_il_00 + gout[5]*dm_il_40;
                    atomicAdd(vk+(j0+3)*nao+(k0+0), vjk_30);
                    double vjk_31 = gout[13]*dm_il_00 + gout[14]*dm_il_40;
                    atomicAdd(vk+(j0+3)*nao+(k0+1), vjk_31);
                    double vjk_32 = gout[22]*dm_il_00 + gout[23]*dm_il_40;
                    atomicAdd(vk+(j0+3)*nao+(k0+2), vjk_32);
                    double vjk_40 = gout[6]*dm_il_20;
                    atomicAdd(vk+(j0+4)*nao+(k0+0), vjk_40);
                    double vjk_41 = gout[15]*dm_il_20;
                    atomicAdd(vk+(j0+4)*nao+(k0+1), vjk_41);
                    double vjk_42 = gout[24]*dm_il_20;
                    atomicAdd(vk+(j0+4)*nao+(k0+2), vjk_42);
                    double vjk_50 = gout[7]*dm_il_00 + gout[8]*dm_il_40;
                    atomicAdd(vk+(j0+5)*nao+(k0+0), vjk_50);
                    double vjk_51 = gout[16]*dm_il_00 + gout[17]*dm_il_40;
                    atomicAdd(vk+(j0+5)*nao+(k0+1), vjk_51);
                    double vjk_52 = gout[25]*dm_il_00 + gout[26]*dm_il_40;
                    atomicAdd(vk+(j0+5)*nao+(k0+2), vjk_52);
                    break; }
                    case 3: {
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    double vjk_00 = gout[0]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    double vjk_01 = gout[9]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    double vjk_02 = gout[18]*dm_il_30;
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    double vjk_10 = gout[1]*dm_il_10 + gout[2]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    double vjk_11 = gout[10]*dm_il_10 + gout[11]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    double vjk_12 = gout[19]*dm_il_10 + gout[20]*dm_il_50;
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    double vjk_20 = gout[3]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    double vjk_21 = gout[12]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    double vjk_22 = gout[21]*dm_il_30;
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    double vjk_30 = gout[4]*dm_il_10 + gout[5]*dm_il_50;
                    atomicAdd(vk+(j0+3)*nao+(k0+0), vjk_30);
                    double vjk_31 = gout[13]*dm_il_10 + gout[14]*dm_il_50;
                    atomicAdd(vk+(j0+3)*nao+(k0+1), vjk_31);
                    double vjk_32 = gout[22]*dm_il_10 + gout[23]*dm_il_50;
                    atomicAdd(vk+(j0+3)*nao+(k0+2), vjk_32);
                    double vjk_40 = gout[6]*dm_il_30;
                    atomicAdd(vk+(j0+4)*nao+(k0+0), vjk_40);
                    double vjk_41 = gout[15]*dm_il_30;
                    atomicAdd(vk+(j0+4)*nao+(k0+1), vjk_41);
                    double vjk_42 = gout[24]*dm_il_30;
                    atomicAdd(vk+(j0+4)*nao+(k0+2), vjk_42);
                    double vjk_50 = gout[7]*dm_il_10 + gout[8]*dm_il_50;
                    atomicAdd(vk+(j0+5)*nao+(k0+0), vjk_50);
                    double vjk_51 = gout[16]*dm_il_10 + gout[17]*dm_il_50;
                    atomicAdd(vk+(j0+5)*nao+(k0+1), vjk_51);
                    double vjk_52 = gout[25]*dm_il_10 + gout[26]*dm_il_50;
                    atomicAdd(vk+(j0+5)*nao+(k0+2), vjk_52);
                    break; }
                    }
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_3000(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[10];
#pragma unroll
            for (int n = 0; n < 10; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        gout[0] += trr_30x * fac * wt;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_20x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_20x * fac * trr_10z;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += trr_10x * trr_20y * wt;
                        gout[4] += trr_10x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += trr_10x * fac * trr_20z;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout[6] += 1 * trr_30y * wt;
                        gout[7] += 1 * trr_20y * trr_10z;
                        gout[8] += 1 * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout[9] += 1 * fac * trr_30z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_60 = gout[6]*dm_jk_00;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_70 = gout[7]*dm_jk_00;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_80 = gout[8]*dm_jk_00;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_90 = gout[9]*dm_jk_00;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_60 = gout[6]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_70 = gout[7]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_80 = gout[8]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_90 = gout[9]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                        vjl_00 += gout[6] * dm_ik_60;
                        double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                        vjl_00 += gout[7] * dm_ik_70;
                        double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                        vjl_00 += gout[8] * dm_ik_80;
                        double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                        vjl_00 += gout[9] * dm_ik_90;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                        vjk_00 += gout[6] * dm_il_60;
                        double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                        vjk_00 += gout[7] * dm_il_70;
                        double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                        vjk_00 += gout[8] * dm_il_80;
                        double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                        vjk_00 += gout[9] * dm_il_90;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_3010(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[30];
#pragma unroll
            for (int n = 0; n < 30; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
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
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        gout[0] += trr_31x * fac * wt;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_21x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_21x * fac * trr_10z;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += trr_11x * trr_20y * wt;
                        gout[4] += trr_11x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += trr_11x * fac * trr_20z;
                        double trr_01x = cpx * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout[6] += trr_01x * trr_30y * wt;
                        gout[7] += trr_01x * trr_20y * trr_10z;
                        gout[8] += trr_01x * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout[9] += trr_01x * fac * trr_30z;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[10] += trr_30x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[11] += trr_20x * trr_11y * wt;
                        gout[12] += trr_20x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout[13] += trr_10x * trr_21y * wt;
                        gout[14] += trr_10x * trr_11y * trr_10z;
                        gout[15] += trr_10x * trr_01y * trr_20z;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        gout[16] += 1 * trr_31y * wt;
                        gout[17] += 1 * trr_21y * trr_10z;
                        gout[18] += 1 * trr_11y * trr_20z;
                        gout[19] += 1 * trr_01y * trr_30z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[20] += trr_30x * fac * trr_01z;
                        gout[21] += trr_20x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[22] += trr_20x * fac * trr_11z;
                        gout[23] += trr_10x * trr_20y * trr_01z;
                        gout[24] += trr_10x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout[25] += trr_10x * fac * trr_21z;
                        gout[26] += 1 * trr_30y * trr_01z;
                        gout[27] += 1 * trr_20y * trr_11z;
                        gout[28] += 1 * trr_10y * trr_21z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        gout[29] += 1 * fac * trr_31z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[10]*dm_jk_01 + gout[20]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[11]*dm_jk_01 + gout[21]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[12]*dm_jk_01 + gout[22]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[13]*dm_jk_01 + gout[23]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[14]*dm_jk_01 + gout[24]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[15]*dm_jk_01 + gout[25]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_60 = gout[6]*dm_jk_00 + gout[16]*dm_jk_01 + gout[26]*dm_jk_02;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_70 = gout[7]*dm_jk_00 + gout[17]*dm_jk_01 + gout[27]*dm_jk_02;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_80 = gout[8]*dm_jk_00 + gout[18]*dm_jk_01 + gout[28]*dm_jk_02;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_90 = gout[9]*dm_jk_00 + gout[19]*dm_jk_01 + gout[29]*dm_jk_02;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[10]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[20]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[11]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[21]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[12]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[22]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[13]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[23]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[14]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[24]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[15]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[25]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_60 = gout[6]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_61 = gout[16]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+1), vik_61);
                    double vik_62 = gout[26]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+2), vik_62);
                    double vik_70 = gout[7]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_71 = gout[17]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+1), vik_71);
                    double vik_72 = gout[27]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+2), vik_72);
                    double vik_80 = gout[8]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_81 = gout[18]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+1), vik_81);
                    double vik_82 = gout[28]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+2), vik_82);
                    double vik_90 = gout[9]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    double vik_91 = gout[19]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+1), vik_91);
                    double vik_92 = gout[29]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+2), vik_92);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[20] * dm_ik_02;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[21] * dm_ik_12;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[12] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[22] * dm_ik_22;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[13] * dm_ik_31;
                        double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[23] * dm_ik_32;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[14] * dm_ik_41;
                        double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[24] * dm_ik_42;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[15] * dm_ik_51;
                        double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[25] * dm_ik_52;
                        double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                        vjl_00 += gout[6] * dm_ik_60;
                        double dm_ik_61 = dm[(i0+6)*nao+(k0+1)];
                        vjl_00 += gout[16] * dm_ik_61;
                        double dm_ik_62 = dm[(i0+6)*nao+(k0+2)];
                        vjl_00 += gout[26] * dm_ik_62;
                        double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                        vjl_00 += gout[7] * dm_ik_70;
                        double dm_ik_71 = dm[(i0+7)*nao+(k0+1)];
                        vjl_00 += gout[17] * dm_ik_71;
                        double dm_ik_72 = dm[(i0+7)*nao+(k0+2)];
                        vjl_00 += gout[27] * dm_ik_72;
                        double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                        vjl_00 += gout[8] * dm_ik_80;
                        double dm_ik_81 = dm[(i0+8)*nao+(k0+1)];
                        vjl_00 += gout[18] * dm_ik_81;
                        double dm_ik_82 = dm[(i0+8)*nao+(k0+2)];
                        vjl_00 += gout[28] * dm_ik_82;
                        double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                        vjl_00 += gout[9] * dm_ik_90;
                        double dm_ik_91 = dm[(i0+9)*nao+(k0+1)];
                        vjl_00 += gout[19] * dm_ik_91;
                        double dm_ik_92 = dm[(i0+9)*nao+(k0+2)];
                        vjl_00 += gout[29] * dm_ik_92;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[10] * dm_il_00;
                        vjk_02 += gout[20] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[11] * dm_il_10;
                        vjk_02 += gout[21] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[12] * dm_il_20;
                        vjk_02 += gout[22] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_01 += gout[13] * dm_il_30;
                        vjk_02 += gout[23] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_01 += gout[14] * dm_il_40;
                        vjk_02 += gout[24] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_01 += gout[15] * dm_il_50;
                        vjk_02 += gout[25] * dm_il_50;
                        double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                        vjk_00 += gout[6] * dm_il_60;
                        vjk_01 += gout[16] * dm_il_60;
                        vjk_02 += gout[26] * dm_il_60;
                        double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                        vjk_00 += gout[7] * dm_il_70;
                        vjk_01 += gout[17] * dm_il_70;
                        vjk_02 += gout[27] * dm_il_70;
                        double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                        vjk_00 += gout[8] * dm_il_80;
                        vjk_01 += gout[18] * dm_il_80;
                        vjk_02 += gout[28] * dm_il_80;
                        double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                        vjk_00 += gout[9] * dm_il_90;
                        vjk_01 += gout[19] * dm_il_90;
                        vjk_02 += gout[29] * dm_il_90;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_3011(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int gout_id = item.get_local_id(0);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif
    int t_id = 64 * gout_id + sq_id;
    constexpr int threads = 256;
    constexpr int nsq_per_block = 64;
    constexpr int g_size = 16;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+8);

while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
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
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0] = xlxk;
                rlrk[64] = ylyk;
                rlrk[128] = zlzk;
            }
            double gout[23];

            #pragma unroll
            for (int n = 0; n < 23; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                    int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
                    int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                    int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0];
                    double ylyk = rlrk[64];
                    double zlzk = rlrk[128];
                    double theta_kl = ak * al_akl;
                    double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = PI_FAC;
                    if (task_id < ntasks) {
                        if (ksh == lsh) fac_sym *= .5;
                        if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                    } else {
                        fac_sym = 0;
                    }
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + rjri[0] * aj_aij;
                    double yij = ri[1] + rjri[1] * aj_aij;
                    double zij = ri[2] + rjri[2] * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double s0, s1, s2;
                        double rt = rw[irys*128];
                        double aij = aij_cache[0];
                        double rt_aa = rt / (aij + akl);
                        double akl = akl_cache[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        double b01 = .5/akl * (1 - rt_akl);
                        for (int n = gout_id; n < 3; n += 4) {
                            if (n == 2) {
                                gx[2048] = rw[irys*128+64];
                            }
                            double *_gx = gx + n * 1024;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aij_cache[1];
                            double c0x = Rpa - rt_aij * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = c0x * s0;
                            _gx[64] = s1;
                            s2 = c0x * s1 + 1 * b10 * s0;
                            _gx[128] = s2;
                            s0 = s1;
                            s1 = s2;
                            s2 = c0x * s1 + 2 * b10 * s0;
                            _gx[192] = s2;
                            double xlxk = rlrk[n*64];
                            double Rqc = xlxk * akl_cache[64];
                            double cpx = Rqc + rt_akl * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[256] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            _gx[512] = s2;
                            s0 = _gx[64];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[320] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 1 * b00 * _gx[256];
                            _gx[576] = s2;
                            s0 = _gx[128];
                            s1 = cpx * s0;
                            s1 += 2 * b00 * _gx[64];
                            _gx[384] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 2 * b00 * _gx[320];
                            _gx[640] = s2;
                            s0 = _gx[192];
                            s1 = cpx * s0;
                            s1 += 3 * b00 * _gx[128];
                            _gx[448] = s1;
                            s2 = cpx*s1 + 1 * b01 *s0;
                            s2 += 3 * b00 * _gx[384];
                            _gx[704] = s2;
                            s1 = _gx[512];
                            s0 = _gx[256];
                            _gx[768] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[0];
                            _gx[512] = s1 - xlxk * s0;
                            s1 = _gx[576];
                            s0 = _gx[320];
                            _gx[832] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[64];
                            _gx[576] = s1 - xlxk * s0;
                            s1 = _gx[640];
                            s0 = _gx[384];
                            _gx[896] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[128];
                            _gx[640] = s1 - xlxk * s0;
                            s1 = _gx[704];
                            s0 = _gx[448];
                            _gx[960] = s1 - xlxk * s0;
                            s1 = s0;
                            s0 = _gx[192];
                            _gx[704] = s1 - xlxk * s0;
                        }
                        __syncthreads();
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[960] * gx[1024] * gx[2048];
                        gout[1] += gx[832] * gx[1088] * gx[2112];
                        gout[2] += gx[768] * gx[1088] * gx[2176];
                        gout[3] += gx[640] * gx[1280] * gx[2112];
                        gout[4] += gx[512] * gx[1472] * gx[2048];
                        gout[5] += gx[704] * gx[1024] * gx[2304];
                        gout[6] += gx[576] * gx[1088] * gx[2368];
                        gout[7] += gx[512] * gx[1088] * gx[2432];
                        gout[8] += gx[384] * gx[1536] * gx[2112];
                        gout[9] += gx[256] * gx[1728] * gx[2048];
                        gout[10] += gx[192] * gx[1792] * gx[2048];
                        gout[11] += gx[64] * gx[1856] * gx[2112];
                        gout[12] += gx[0] * gx[1856] * gx[2176];
                        gout[13] += gx[128] * gx[1536] * gx[2368];
                        gout[14] += gx[0] * gx[1728] * gx[2304];
                        gout[15] += gx[448] * gx[1024] * gx[2560];
                        gout[16] += gx[320] * gx[1088] * gx[2624];
                        gout[17] += gx[256] * gx[1088] * gx[2688];
                        gout[18] += gx[128] * gx[1280] * gx[2624];
                        gout[19] += gx[0] * gx[1472] * gx[2560];
                        gout[20] += gx[192] * gx[1024] * gx[2816];
                        gout[21] += gx[64] * gx[1088] * gx[2880];
                        gout[22] += gx[0] * gx[1088] * gx[2944];
                        break;
                        case 1:
                        gout[0] += gx[896] * gx[1088] * gx[2048];
                        gout[1] += gx[832] * gx[1024] * gx[2176];
                        gout[2] += gx[768] * gx[1024] * gx[2240];
                        gout[3] += gx[576] * gx[1408] * gx[2048];
                        gout[4] += gx[512] * gx[1408] * gx[2112];
                        gout[5] += gx[640] * gx[1088] * gx[2304];
                        gout[6] += gx[576] * gx[1024] * gx[2432];
                        gout[7] += gx[512] * gx[1024] * gx[2496];
                        gout[8] += gx[320] * gx[1664] * gx[2048];
                        gout[9] += gx[256] * gx[1664] * gx[2112];
                        gout[10] += gx[128] * gx[1856] * gx[2048];
                        gout[11] += gx[64] * gx[1792] * gx[2176];
                        gout[12] += gx[0] * gx[1792] * gx[2240];
                        gout[13] += gx[64] * gx[1664] * gx[2304];
                        gout[14] += gx[0] * gx[1664] * gx[2368];
                        gout[15] += gx[384] * gx[1088] * gx[2560];
                        gout[16] += gx[320] * gx[1024] * gx[2688];
                        gout[17] += gx[256] * gx[1024] * gx[2752];
                        gout[18] += gx[64] * gx[1408] * gx[2560];
                        gout[19] += gx[0] * gx[1408] * gx[2624];
                        gout[20] += gx[128] * gx[1088] * gx[2816];
                        gout[21] += gx[64] * gx[1024] * gx[2944];
                        gout[22] += gx[0] * gx[1024] * gx[3008];
                        break;
                        case 2:
                        gout[0] += gx[896] * gx[1024] * gx[2112];
                        gout[1] += gx[768] * gx[1216] * gx[2048];
                        gout[2] += gx[704] * gx[1280] * gx[2048];
                        gout[3] += gx[576] * gx[1344] * gx[2112];
                        gout[4] += gx[512] * gx[1344] * gx[2176];
                        gout[5] += gx[640] * gx[1024] * gx[2368];
                        gout[6] += gx[512] * gx[1216] * gx[2304];
                        gout[7] += gx[448] * gx[1536] * gx[2048];
                        gout[8] += gx[320] * gx[1600] * gx[2112];
                        gout[9] += gx[256] * gx[1600] * gx[2176];
                        gout[10] += gx[128] * gx[1792] * gx[2112];
                        gout[11] += gx[0] * gx[1984] * gx[2048];
                        gout[12] += gx[192] * gx[1536] * gx[2304];
                        gout[13] += gx[64] * gx[1600] * gx[2368];
                        gout[14] += gx[0] * gx[1600] * gx[2432];
                        gout[15] += gx[384] * gx[1024] * gx[2624];
                        gout[16] += gx[256] * gx[1216] * gx[2560];
                        gout[17] += gx[192] * gx[1280] * gx[2560];
                        gout[18] += gx[64] * gx[1344] * gx[2624];
                        gout[19] += gx[0] * gx[1344] * gx[2688];
                        gout[20] += gx[128] * gx[1024] * gx[2880];
                        gout[21] += gx[0] * gx[1216] * gx[2816];
                        break;
                        case 3:
                        gout[0] += gx[832] * gx[1152] * gx[2048];
                        gout[1] += gx[768] * gx[1152] * gx[2112];
                        gout[2] += gx[640] * gx[1344] * gx[2048];
                        gout[3] += gx[576] * gx[1280] * gx[2176];
                        gout[4] += gx[512] * gx[1280] * gx[2240];
                        gout[5] += gx[576] * gx[1152] * gx[2304];
                        gout[6] += gx[512] * gx[1152] * gx[2368];
                        gout[7] += gx[384] * gx[1600] * gx[2048];
                        gout[8] += gx[320] * gx[1536] * gx[2176];
                        gout[9] += gx[256] * gx[1536] * gx[2240];
                        gout[10] += gx[64] * gx[1920] * gx[2048];
                        gout[11] += gx[0] * gx[1920] * gx[2112];
                        gout[12] += gx[128] * gx[1600] * gx[2304];
                        gout[13] += gx[64] * gx[1536] * gx[2432];
                        gout[14] += gx[0] * gx[1536] * gx[2496];
                        gout[15] += gx[320] * gx[1152] * gx[2560];
                        gout[16] += gx[256] * gx[1152] * gx[2624];
                        gout[17] += gx[128] * gx[1344] * gx[2560];
                        gout[18] += gx[64] * gx[1280] * gx[2688];
                        gout[19] += gx[0] * gx[1280] * gx[2752];
                        gout[20] += gx[64] * gx[1152] * gx[2816];
                        gout[21] += gx[0] * gx[1152] * gx[2880];
                        break;
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao * nao;
                    double *vk = kmat.vk + i_dm * nao * nao;
                    switch (gout_id) {
                    case 0: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[5]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double vil_01 = gout[10]*dm_jk_01;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[15]*dm_jk_00 + gout[20]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double vil_20 = gout[3]*dm_jk_01;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[8]*dm_jk_00 + gout[13]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[18]*dm_jk_01;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[1]*dm_jk_00 + gout[6]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[11]*dm_jk_01;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[16]*dm_jk_00 + gout[21]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    double vil_60 = gout[4]*dm_jk_01;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_61 = gout[9]*dm_jk_00 + gout[14]*dm_jk_02;
                    atomicAdd(vk+(i0+6)*nao+(l0+1), vil_61);
                    double vil_62 = gout[19]*dm_jk_01;
                    atomicAdd(vk+(i0+6)*nao+(l0+2), vil_62);
                    double vil_80 = gout[2]*dm_jk_00 + gout[7]*dm_jk_02;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_81 = gout[12]*dm_jk_01;
                    atomicAdd(vk+(i0+8)*nao+(l0+1), vil_81);
                    double vil_82 = gout[17]*dm_jk_00 + gout[22]*dm_jk_02;
                    atomicAdd(vk+(i0+8)*nao+(l0+2), vil_82);
                    break; }
                    case 1: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_10 = gout[0]*dm_jk_00 + gout[5]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double vil_11 = gout[10]*dm_jk_01;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[15]*dm_jk_00 + gout[20]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double vil_30 = gout[3]*dm_jk_01;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[8]*dm_jk_00 + gout[13]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[18]*dm_jk_01;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[1]*dm_jk_00 + gout[6]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[11]*dm_jk_01;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[16]*dm_jk_00 + gout[21]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    double vil_70 = gout[4]*dm_jk_01;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_71 = gout[9]*dm_jk_00 + gout[14]*dm_jk_02;
                    atomicAdd(vk+(i0+7)*nao+(l0+1), vil_71);
                    double vil_72 = gout[19]*dm_jk_01;
                    atomicAdd(vk+(i0+7)*nao+(l0+2), vil_72);
                    double vil_90 = gout[2]*dm_jk_00 + gout[7]*dm_jk_02;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double vil_91 = gout[12]*dm_jk_01;
                    atomicAdd(vk+(i0+9)*nao+(l0+1), vil_91);
                    double vil_92 = gout[17]*dm_jk_00 + gout[22]*dm_jk_02;
                    atomicAdd(vk+(i0+9)*nao+(l0+2), vil_92);
                    break; }
                    case 2: {
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double vil_00 = gout[2]*dm_jk_01;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_01 = gout[7]*dm_jk_00 + gout[12]*dm_jk_02;
                    atomicAdd(vk+(i0+0)*nao+(l0+1), vil_01);
                    double vil_02 = gout[17]*dm_jk_01;
                    atomicAdd(vk+(i0+0)*nao+(l0+2), vil_02);
                    double vil_20 = gout[0]*dm_jk_00 + gout[5]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_21 = gout[10]*dm_jk_01;
                    atomicAdd(vk+(i0+2)*nao+(l0+1), vil_21);
                    double vil_22 = gout[15]*dm_jk_00 + gout[20]*dm_jk_02;
                    atomicAdd(vk+(i0+2)*nao+(l0+2), vil_22);
                    double vil_40 = gout[3]*dm_jk_01;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_41 = gout[8]*dm_jk_00 + gout[13]*dm_jk_02;
                    atomicAdd(vk+(i0+4)*nao+(l0+1), vil_41);
                    double vil_42 = gout[18]*dm_jk_01;
                    atomicAdd(vk+(i0+4)*nao+(l0+2), vil_42);
                    double vil_60 = gout[1]*dm_jk_00 + gout[6]*dm_jk_02;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_61 = gout[11]*dm_jk_01;
                    atomicAdd(vk+(i0+6)*nao+(l0+1), vil_61);
                    double vil_62 = gout[16]*dm_jk_00 + gout[21]*dm_jk_02;
                    atomicAdd(vk+(i0+6)*nao+(l0+2), vil_62);
                    double vil_80 = gout[4]*dm_jk_01;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_81 = gout[9]*dm_jk_00 + gout[14]*dm_jk_02;
                    atomicAdd(vk+(i0+8)*nao+(l0+1), vil_81);
                    double vil_82 = gout[19]*dm_jk_01;
                    atomicAdd(vk+(i0+8)*nao+(l0+2), vil_82);
                    break; }
                    case 3: {
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double vil_10 = gout[2]*dm_jk_01;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double vil_11 = gout[7]*dm_jk_00 + gout[12]*dm_jk_02;
                    atomicAdd(vk+(i0+1)*nao+(l0+1), vil_11);
                    double vil_12 = gout[17]*dm_jk_01;
                    atomicAdd(vk+(i0+1)*nao+(l0+2), vil_12);
                    double vil_30 = gout[0]*dm_jk_00 + gout[5]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_31 = gout[10]*dm_jk_01;
                    atomicAdd(vk+(i0+3)*nao+(l0+1), vil_31);
                    double vil_32 = gout[15]*dm_jk_00 + gout[20]*dm_jk_02;
                    atomicAdd(vk+(i0+3)*nao+(l0+2), vil_32);
                    double vil_50 = gout[3]*dm_jk_01;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_51 = gout[8]*dm_jk_00 + gout[13]*dm_jk_02;
                    atomicAdd(vk+(i0+5)*nao+(l0+1), vil_51);
                    double vil_52 = gout[18]*dm_jk_01;
                    atomicAdd(vk+(i0+5)*nao+(l0+2), vil_52);
                    double vil_70 = gout[1]*dm_jk_00 + gout[6]*dm_jk_02;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_71 = gout[11]*dm_jk_01;
                    atomicAdd(vk+(i0+7)*nao+(l0+1), vil_71);
                    double vil_72 = gout[16]*dm_jk_00 + gout[21]*dm_jk_02;
                    atomicAdd(vk+(i0+7)*nao+(l0+2), vil_72);
                    double vil_90 = gout[4]*dm_jk_01;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double vil_91 = gout[9]*dm_jk_00 + gout[14]*dm_jk_02;
                    atomicAdd(vk+(i0+9)*nao+(l0+1), vil_91);
                    double vil_92 = gout[19]*dm_jk_01;
                    atomicAdd(vk+(i0+9)*nao+(l0+2), vil_92);
                    break; }
                    }
                    switch (gout_id) {
                    case 0: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[15]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_01 = gout[10]*dm_jl_01;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[5]*dm_jl_00 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[8]*dm_jl_01;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[3]*dm_jl_00 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[13]*dm_jl_01;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[1]*dm_jl_00 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[6]*dm_jl_00 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_60 = gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_61 = gout[4]*dm_jl_00 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+6)*nao+(k0+1), vik_61);
                    double vik_62 = gout[14]*dm_jl_01;
                    atomicAdd(vk+(i0+6)*nao+(k0+2), vik_62);
                    double vik_80 = gout[2]*dm_jl_00 + gout[17]*dm_jl_02;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_81 = gout[12]*dm_jl_01;
                    atomicAdd(vk+(i0+8)*nao+(k0+1), vik_81);
                    double vik_82 = gout[7]*dm_jl_00 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+8)*nao+(k0+2), vik_82);
                    break; }
                    case 1: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_10 = gout[0]*dm_jl_00 + gout[15]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_11 = gout[10]*dm_jl_01;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[5]*dm_jl_00 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[8]*dm_jl_01;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[3]*dm_jl_00 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[13]*dm_jl_01;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[1]*dm_jl_00 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[6]*dm_jl_00 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_70 = gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_71 = gout[4]*dm_jl_00 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+7)*nao+(k0+1), vik_71);
                    double vik_72 = gout[14]*dm_jl_01;
                    atomicAdd(vk+(i0+7)*nao+(k0+2), vik_72);
                    double vik_90 = gout[2]*dm_jl_00 + gout[17]*dm_jl_02;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    double vik_91 = gout[12]*dm_jl_01;
                    atomicAdd(vk+(i0+9)*nao+(k0+1), vik_91);
                    double vik_92 = gout[7]*dm_jl_00 + gout[22]*dm_jl_02;
                    atomicAdd(vk+(i0+9)*nao+(k0+2), vik_92);
                    break; }
                    case 2: {
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_00 = gout[7]*dm_jl_01;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_01 = gout[2]*dm_jl_00 + gout[17]*dm_jl_02;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[12]*dm_jl_01;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[0]*dm_jl_00 + gout[15]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[10]*dm_jl_01;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[5]*dm_jl_00 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[8]*dm_jl_01;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[3]*dm_jl_00 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[13]*dm_jl_01;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_60 = gout[1]*dm_jl_00 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_61 = gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+6)*nao+(k0+1), vik_61);
                    double vik_62 = gout[6]*dm_jl_00 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+6)*nao+(k0+2), vik_62);
                    double vik_80 = gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_81 = gout[4]*dm_jl_00 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+8)*nao+(k0+1), vik_81);
                    double vik_82 = gout[14]*dm_jl_01;
                    atomicAdd(vk+(i0+8)*nao+(k0+2), vik_82);
                    break; }
                    case 3: {
                    double dm_jl_01 = dm[(j0+0)*nao+(l0+1)];
                    double vik_10 = gout[7]*dm_jl_01;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_02 = dm[(j0+0)*nao+(l0+2)];
                    double vik_11 = gout[2]*dm_jl_00 + gout[17]*dm_jl_02;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[12]*dm_jl_01;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[0]*dm_jl_00 + gout[15]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[10]*dm_jl_01;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[5]*dm_jl_00 + gout[20]*dm_jl_02;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[8]*dm_jl_01;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[3]*dm_jl_00 + gout[18]*dm_jl_02;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[13]*dm_jl_01;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_70 = gout[1]*dm_jl_00 + gout[16]*dm_jl_02;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_71 = gout[11]*dm_jl_01;
                    atomicAdd(vk+(i0+7)*nao+(k0+1), vik_71);
                    double vik_72 = gout[6]*dm_jl_00 + gout[21]*dm_jl_02;
                    atomicAdd(vk+(i0+7)*nao+(k0+2), vik_72);
                    double vik_90 = gout[9]*dm_jl_01;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    double vik_91 = gout[4]*dm_jl_00 + gout[19]*dm_jl_02;
                    atomicAdd(vk+(i0+9)*nao+(k0+1), vik_91);
                    double vik_92 = gout[14]*dm_jl_01;
                    atomicAdd(vk+(i0+9)*nao+(k0+2), vik_92);
                    break; }
                    }
                    if (ish != jsh) {
                    double vjl_00 = 0;
                    double vjl_01 = 0;
                    double vjl_02 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_00;
                    vjl_02 += gout[15] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_01 += gout[10] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_00 += gout[5] * dm_ik_02;
                    vjl_02 += gout[20] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_01 += gout[8] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_00 += gout[3] * dm_ik_21;
                    vjl_02 += gout[18] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_01 += gout[13] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_40;
                    vjl_02 += gout[16] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_01 += gout[11] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_00 += gout[6] * dm_ik_42;
                    vjl_02 += gout[21] * dm_ik_42;
                    double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                    vjl_01 += gout[9] * dm_ik_60;
                    double dm_ik_61 = dm[(i0+6)*nao+(k0+1)];
                    vjl_00 += gout[4] * dm_ik_61;
                    vjl_02 += gout[19] * dm_ik_61;
                    double dm_ik_62 = dm[(i0+6)*nao+(k0+2)];
                    vjl_01 += gout[14] * dm_ik_62;
                    double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                    vjl_00 += gout[2] * dm_ik_80;
                    vjl_02 += gout[17] * dm_ik_80;
                    double dm_ik_81 = dm[(i0+8)*nao+(k0+1)];
                    vjl_01 += gout[12] * dm_ik_81;
                    double dm_ik_82 = dm[(i0+8)*nao+(k0+2)];
                    vjl_00 += gout[7] * dm_ik_82;
                    vjl_02 += gout[22] * dm_ik_82;
                    break; }
                    case 1: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_10;
                    vjl_02 += gout[15] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_01 += gout[10] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_00 += gout[5] * dm_ik_12;
                    vjl_02 += gout[20] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_01 += gout[8] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_00 += gout[3] * dm_ik_31;
                    vjl_02 += gout[18] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_01 += gout[13] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_50;
                    vjl_02 += gout[16] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_01 += gout[11] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_00 += gout[6] * dm_ik_52;
                    vjl_02 += gout[21] * dm_ik_52;
                    double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                    vjl_01 += gout[9] * dm_ik_70;
                    double dm_ik_71 = dm[(i0+7)*nao+(k0+1)];
                    vjl_00 += gout[4] * dm_ik_71;
                    vjl_02 += gout[19] * dm_ik_71;
                    double dm_ik_72 = dm[(i0+7)*nao+(k0+2)];
                    vjl_01 += gout[14] * dm_ik_72;
                    double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                    vjl_00 += gout[2] * dm_ik_90;
                    vjl_02 += gout[17] * dm_ik_90;
                    double dm_ik_91 = dm[(i0+9)*nao+(k0+1)];
                    vjl_01 += gout[12] * dm_ik_91;
                    double dm_ik_92 = dm[(i0+9)*nao+(k0+2)];
                    vjl_00 += gout[7] * dm_ik_92;
                    vjl_02 += gout[22] * dm_ik_92;
                    break; }
                    case 2: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_01 += gout[7] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_01;
                    vjl_02 += gout[17] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_01 += gout[12] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_20;
                    vjl_02 += gout[15] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_01 += gout[10] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_00 += gout[5] * dm_ik_22;
                    vjl_02 += gout[20] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_01 += gout[8] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_00 += gout[3] * dm_ik_41;
                    vjl_02 += gout[18] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_01 += gout[13] * dm_ik_42;
                    double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_60;
                    vjl_02 += gout[16] * dm_ik_60;
                    double dm_ik_61 = dm[(i0+6)*nao+(k0+1)];
                    vjl_01 += gout[11] * dm_ik_61;
                    double dm_ik_62 = dm[(i0+6)*nao+(k0+2)];
                    vjl_00 += gout[6] * dm_ik_62;
                    vjl_02 += gout[21] * dm_ik_62;
                    double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                    vjl_01 += gout[9] * dm_ik_80;
                    double dm_ik_81 = dm[(i0+8)*nao+(k0+1)];
                    vjl_00 += gout[4] * dm_ik_81;
                    vjl_02 += gout[19] * dm_ik_81;
                    double dm_ik_82 = dm[(i0+8)*nao+(k0+2)];
                    vjl_01 += gout[14] * dm_ik_82;
                    break; }
                    case 3: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_01 += gout[7] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_00 += gout[2] * dm_ik_11;
                    vjl_02 += gout[17] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_01 += gout[12] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_30;
                    vjl_02 += gout[15] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_01 += gout[10] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_00 += gout[5] * dm_ik_32;
                    vjl_02 += gout[20] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_01 += gout[8] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_00 += gout[3] * dm_ik_51;
                    vjl_02 += gout[18] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_01 += gout[13] * dm_ik_52;
                    double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_70;
                    vjl_02 += gout[16] * dm_ik_70;
                    double dm_ik_71 = dm[(i0+7)*nao+(k0+1)];
                    vjl_01 += gout[11] * dm_ik_71;
                    double dm_ik_72 = dm[(i0+7)*nao+(k0+2)];
                    vjl_00 += gout[6] * dm_ik_72;
                    vjl_02 += gout[21] * dm_ik_72;
                    double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                    vjl_01 += gout[9] * dm_ik_90;
                    double dm_ik_91 = dm[(i0+9)*nao+(k0+1)];
                    vjl_00 += gout[4] * dm_ik_91;
                    vjl_02 += gout[19] * dm_ik_91;
                    double dm_ik_92 = dm[(i0+9)*nao+(k0+2)];
                    vjl_01 += gout[14] * dm_ik_92;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                    atomicAdd(vk+(j0+0)*nao+(l0+1), vjl_01);
                    atomicAdd(vk+(j0+0)*nao+(l0+2), vjl_02);
                    double vjk_00 = 0;
                    double vjk_01 = 0;
                    double vjk_02 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_00;
                    vjk_02 += gout[5] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_01 += gout[10] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_00 += gout[15] * dm_il_02;
                    vjk_02 += gout[20] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_01 += gout[3] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_00 += gout[8] * dm_il_21;
                    vjk_02 += gout[13] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_01 += gout[18] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_40;
                    vjk_02 += gout[6] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_00 += gout[16] * dm_il_42;
                    vjk_02 += gout[21] * dm_il_42;
                    double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                    vjk_01 += gout[4] * dm_il_60;
                    double dm_il_61 = dm[(i0+6)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_61;
                    vjk_02 += gout[14] * dm_il_61;
                    double dm_il_62 = dm[(i0+6)*nao+(l0+2)];
                    vjk_01 += gout[19] * dm_il_62;
                    double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                    vjk_00 += gout[2] * dm_il_80;
                    vjk_02 += gout[7] * dm_il_80;
                    double dm_il_81 = dm[(i0+8)*nao+(l0+1)];
                    vjk_01 += gout[12] * dm_il_81;
                    double dm_il_82 = dm[(i0+8)*nao+(l0+2)];
                    vjk_00 += gout[17] * dm_il_82;
                    vjk_02 += gout[22] * dm_il_82;
                    break; }
                    case 1: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_10;
                    vjk_02 += gout[5] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_01 += gout[10] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_00 += gout[15] * dm_il_12;
                    vjk_02 += gout[20] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_01 += gout[3] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_00 += gout[8] * dm_il_31;
                    vjk_02 += gout[13] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_01 += gout[18] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_50;
                    vjk_02 += gout[6] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_00 += gout[16] * dm_il_52;
                    vjk_02 += gout[21] * dm_il_52;
                    double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                    vjk_01 += gout[4] * dm_il_70;
                    double dm_il_71 = dm[(i0+7)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_71;
                    vjk_02 += gout[14] * dm_il_71;
                    double dm_il_72 = dm[(i0+7)*nao+(l0+2)];
                    vjk_01 += gout[19] * dm_il_72;
                    double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                    vjk_00 += gout[2] * dm_il_90;
                    vjk_02 += gout[7] * dm_il_90;
                    double dm_il_91 = dm[(i0+9)*nao+(l0+1)];
                    vjk_01 += gout[12] * dm_il_91;
                    double dm_il_92 = dm[(i0+9)*nao+(l0+2)];
                    vjk_00 += gout[17] * dm_il_92;
                    vjk_02 += gout[22] * dm_il_92;
                    break; }
                    case 2: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_00;
                    double dm_il_01 = dm[(i0+0)*nao+(l0+1)];
                    vjk_00 += gout[7] * dm_il_01;
                    vjk_02 += gout[12] * dm_il_01;
                    double dm_il_02 = dm[(i0+0)*nao+(l0+2)];
                    vjk_01 += gout[17] * dm_il_02;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_20;
                    vjk_02 += gout[5] * dm_il_20;
                    double dm_il_21 = dm[(i0+2)*nao+(l0+1)];
                    vjk_01 += gout[10] * dm_il_21;
                    double dm_il_22 = dm[(i0+2)*nao+(l0+2)];
                    vjk_00 += gout[15] * dm_il_22;
                    vjk_02 += gout[20] * dm_il_22;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_01 += gout[3] * dm_il_40;
                    double dm_il_41 = dm[(i0+4)*nao+(l0+1)];
                    vjk_00 += gout[8] * dm_il_41;
                    vjk_02 += gout[13] * dm_il_41;
                    double dm_il_42 = dm[(i0+4)*nao+(l0+2)];
                    vjk_01 += gout[18] * dm_il_42;
                    double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_60;
                    vjk_02 += gout[6] * dm_il_60;
                    double dm_il_61 = dm[(i0+6)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_61;
                    double dm_il_62 = dm[(i0+6)*nao+(l0+2)];
                    vjk_00 += gout[16] * dm_il_62;
                    vjk_02 += gout[21] * dm_il_62;
                    double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                    vjk_01 += gout[4] * dm_il_80;
                    double dm_il_81 = dm[(i0+8)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_81;
                    vjk_02 += gout[14] * dm_il_81;
                    double dm_il_82 = dm[(i0+8)*nao+(l0+2)];
                    vjk_01 += gout[19] * dm_il_82;
                    break; }
                    case 3: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_01 += gout[2] * dm_il_10;
                    double dm_il_11 = dm[(i0+1)*nao+(l0+1)];
                    vjk_00 += gout[7] * dm_il_11;
                    vjk_02 += gout[12] * dm_il_11;
                    double dm_il_12 = dm[(i0+1)*nao+(l0+2)];
                    vjk_01 += gout[17] * dm_il_12;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_30;
                    vjk_02 += gout[5] * dm_il_30;
                    double dm_il_31 = dm[(i0+3)*nao+(l0+1)];
                    vjk_01 += gout[10] * dm_il_31;
                    double dm_il_32 = dm[(i0+3)*nao+(l0+2)];
                    vjk_00 += gout[15] * dm_il_32;
                    vjk_02 += gout[20] * dm_il_32;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_01 += gout[3] * dm_il_50;
                    double dm_il_51 = dm[(i0+5)*nao+(l0+1)];
                    vjk_00 += gout[8] * dm_il_51;
                    vjk_02 += gout[13] * dm_il_51;
                    double dm_il_52 = dm[(i0+5)*nao+(l0+2)];
                    vjk_01 += gout[18] * dm_il_52;
                    double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_70;
                    vjk_02 += gout[6] * dm_il_70;
                    double dm_il_71 = dm[(i0+7)*nao+(l0+1)];
                    vjk_01 += gout[11] * dm_il_71;
                    double dm_il_72 = dm[(i0+7)*nao+(l0+2)];
                    vjk_00 += gout[16] * dm_il_72;
                    vjk_02 += gout[21] * dm_il_72;
                    double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                    vjk_01 += gout[4] * dm_il_90;
                    double dm_il_91 = dm[(i0+9)*nao+(l0+1)];
                    vjk_00 += gout[9] * dm_il_91;
                    vjk_02 += gout[14] * dm_il_91;
                    double dm_il_92 = dm[(i0+9)*nao+(l0+2)];
                    vjk_01 += gout[19] * dm_il_92;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    }
                }
            }
        }
    }
}
}

__global__ static
 void rys_k_3020(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                     float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                     float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                     uint32_t *pool, int *head
                     #ifdef USE_SYCL
                     , sycl::nd_item<2> &item, double *shared_memory
                     #endif
                     )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[60];
#pragma unroll
            for (int n = 0; n < 60; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double b00 = .5 * rt_aa;
                        double rt_akl = rt_aa * aij;
                        double b01 = .5/akl * (1 - rt_akl);
                        double cpx = xlxk*al_akl + xpq*rt_akl;
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                        double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                        double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                        gout[0] += trr_32x * fac * wt;
                        double trr_11x = cpx * trr_10x + 1*b00 * 1;
                        double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += trr_22x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += trr_22x * fac * trr_10z;
                        double trr_01x = cpx * 1;
                        double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += trr_12x * trr_20y * wt;
                        gout[4] += trr_12x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += trr_12x * fac * trr_20z;
                        double trr_02x = cpx * trr_01x + 1*b01 * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout[6] += trr_02x * trr_30y * wt;
                        gout[7] += trr_02x * trr_20y * trr_10z;
                        gout[8] += trr_02x * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout[9] += trr_02x * fac * trr_30z;
                        double cpy = ylyk*al_akl + ypq*rt_akl;
                        double trr_01y = cpy * fac;
                        gout[10] += trr_31x * trr_01y * wt;
                        double trr_11y = cpy * trr_10y + 1*b00 * fac;
                        gout[11] += trr_21x * trr_11y * wt;
                        gout[12] += trr_21x * trr_01y * trr_10z;
                        double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                        gout[13] += trr_11x * trr_21y * wt;
                        gout[14] += trr_11x * trr_11y * trr_10z;
                        gout[15] += trr_11x * trr_01y * trr_20z;
                        double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                        gout[16] += trr_01x * trr_31y * wt;
                        gout[17] += trr_01x * trr_21y * trr_10z;
                        gout[18] += trr_01x * trr_11y * trr_20z;
                        gout[19] += trr_01x * trr_01y * trr_30z;
                        double cpz = zlzk*al_akl + zpq*rt_akl;
                        double trr_01z = cpz * wt;
                        gout[20] += trr_31x * fac * trr_01z;
                        gout[21] += trr_21x * trr_10y * trr_01z;
                        double trr_11z = cpz * trr_10z + 1*b00 * wt;
                        gout[22] += trr_21x * fac * trr_11z;
                        gout[23] += trr_11x * trr_20y * trr_01z;
                        gout[24] += trr_11x * trr_10y * trr_11z;
                        double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                        gout[25] += trr_11x * fac * trr_21z;
                        gout[26] += trr_01x * trr_30y * trr_01z;
                        gout[27] += trr_01x * trr_20y * trr_11z;
                        gout[28] += trr_01x * trr_10y * trr_21z;
                        double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                        gout[29] += trr_01x * fac * trr_31z;
                        double trr_02y = cpy * trr_01y + 1*b01 * fac;
                        gout[30] += trr_30x * trr_02y * wt;
                        double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                        gout[31] += trr_20x * trr_12y * wt;
                        gout[32] += trr_20x * trr_02y * trr_10z;
                        double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                        gout[33] += trr_10x * trr_22y * wt;
                        gout[34] += trr_10x * trr_12y * trr_10z;
                        gout[35] += trr_10x * trr_02y * trr_20z;
                        double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                        gout[36] += 1 * trr_32y * wt;
                        gout[37] += 1 * trr_22y * trr_10z;
                        gout[38] += 1 * trr_12y * trr_20z;
                        gout[39] += 1 * trr_02y * trr_30z;
                        gout[40] += trr_30x * trr_01y * trr_01z;
                        gout[41] += trr_20x * trr_11y * trr_01z;
                        gout[42] += trr_20x * trr_01y * trr_11z;
                        gout[43] += trr_10x * trr_21y * trr_01z;
                        gout[44] += trr_10x * trr_11y * trr_11z;
                        gout[45] += trr_10x * trr_01y * trr_21z;
                        gout[46] += 1 * trr_31y * trr_01z;
                        gout[47] += 1 * trr_21y * trr_11z;
                        gout[48] += 1 * trr_11y * trr_21z;
                        gout[49] += 1 * trr_01y * trr_31z;
                        double trr_02z = cpz * trr_01z + 1*b01 * wt;
                        gout[50] += trr_30x * fac * trr_02z;
                        gout[51] += trr_20x * trr_10y * trr_02z;
                        double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                        gout[52] += trr_20x * fac * trr_12z;
                        gout[53] += trr_10x * trr_20y * trr_02z;
                        gout[54] += trr_10x * trr_10y * trr_12z;
                        double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                        gout[55] += trr_10x * fac * trr_22z;
                        gout[56] += 1 * trr_30y * trr_02z;
                        gout[57] += 1 * trr_20y * trr_12z;
                        gout[58] += 1 * trr_10y * trr_22z;
                        double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                        gout[59] += 1 * fac * trr_32z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_03 = dm[(j0+0)*nao+(k0+3)];
                    double dm_jk_04 = dm[(j0+0)*nao+(k0+4)];
                    double dm_jk_05 = dm[(j0+0)*nao+(k0+5)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[10]*dm_jk_01 + gout[20]*dm_jk_02 + gout[30]*dm_jk_03 + gout[40]*dm_jk_04 + gout[50]*dm_jk_05;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[11]*dm_jk_01 + gout[21]*dm_jk_02 + gout[31]*dm_jk_03 + gout[41]*dm_jk_04 + gout[51]*dm_jk_05;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[12]*dm_jk_01 + gout[22]*dm_jk_02 + gout[32]*dm_jk_03 + gout[42]*dm_jk_04 + gout[52]*dm_jk_05;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[13]*dm_jk_01 + gout[23]*dm_jk_02 + gout[33]*dm_jk_03 + gout[43]*dm_jk_04 + gout[53]*dm_jk_05;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[14]*dm_jk_01 + gout[24]*dm_jk_02 + gout[34]*dm_jk_03 + gout[44]*dm_jk_04 + gout[54]*dm_jk_05;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[15]*dm_jk_01 + gout[25]*dm_jk_02 + gout[35]*dm_jk_03 + gout[45]*dm_jk_04 + gout[55]*dm_jk_05;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_60 = gout[6]*dm_jk_00 + gout[16]*dm_jk_01 + gout[26]*dm_jk_02 + gout[36]*dm_jk_03 + gout[46]*dm_jk_04 + gout[56]*dm_jk_05;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_70 = gout[7]*dm_jk_00 + gout[17]*dm_jk_01 + gout[27]*dm_jk_02 + gout[37]*dm_jk_03 + gout[47]*dm_jk_04 + gout[57]*dm_jk_05;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_80 = gout[8]*dm_jk_00 + gout[18]*dm_jk_01 + gout[28]*dm_jk_02 + gout[38]*dm_jk_03 + gout[48]*dm_jk_04 + gout[58]*dm_jk_05;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_90 = gout[9]*dm_jk_00 + gout[19]*dm_jk_01 + gout[29]*dm_jk_02 + gout[39]*dm_jk_03 + gout[49]*dm_jk_04 + gout[59]*dm_jk_05;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_01 = gout[10]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[20]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_03 = gout[30]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+3), vik_03);
                    double vik_04 = gout[40]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+4), vik_04);
                    double vik_05 = gout[50]*dm_jl_00;
                    atomicAdd(vk+(i0+0)*nao+(k0+5), vik_05);
                    double vik_10 = gout[1]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_11 = gout[11]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[21]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_13 = gout[31]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+3), vik_13);
                    double vik_14 = gout[41]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+4), vik_14);
                    double vik_15 = gout[51]*dm_jl_00;
                    atomicAdd(vk+(i0+1)*nao+(k0+5), vik_15);
                    double vik_20 = gout[2]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[12]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[22]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_23 = gout[32]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+3), vik_23);
                    double vik_24 = gout[42]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+4), vik_24);
                    double vik_25 = gout[52]*dm_jl_00;
                    atomicAdd(vk+(i0+2)*nao+(k0+5), vik_25);
                    double vik_30 = gout[3]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[13]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[23]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_33 = gout[33]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+3), vik_33);
                    double vik_34 = gout[43]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+4), vik_34);
                    double vik_35 = gout[53]*dm_jl_00;
                    atomicAdd(vk+(i0+3)*nao+(k0+5), vik_35);
                    double vik_40 = gout[4]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[14]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[24]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_43 = gout[34]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+3), vik_43);
                    double vik_44 = gout[44]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+4), vik_44);
                    double vik_45 = gout[54]*dm_jl_00;
                    atomicAdd(vk+(i0+4)*nao+(k0+5), vik_45);
                    double vik_50 = gout[5]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[15]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[25]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_53 = gout[35]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+3), vik_53);
                    double vik_54 = gout[45]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+4), vik_54);
                    double vik_55 = gout[55]*dm_jl_00;
                    atomicAdd(vk+(i0+5)*nao+(k0+5), vik_55);
                    double vik_60 = gout[6]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_61 = gout[16]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+1), vik_61);
                    double vik_62 = gout[26]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+2), vik_62);
                    double vik_63 = gout[36]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+3), vik_63);
                    double vik_64 = gout[46]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+4), vik_64);
                    double vik_65 = gout[56]*dm_jl_00;
                    atomicAdd(vk+(i0+6)*nao+(k0+5), vik_65);
                    double vik_70 = gout[7]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_71 = gout[17]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+1), vik_71);
                    double vik_72 = gout[27]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+2), vik_72);
                    double vik_73 = gout[37]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+3), vik_73);
                    double vik_74 = gout[47]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+4), vik_74);
                    double vik_75 = gout[57]*dm_jl_00;
                    atomicAdd(vk+(i0+7)*nao+(k0+5), vik_75);
                    double vik_80 = gout[8]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_81 = gout[18]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+1), vik_81);
                    double vik_82 = gout[28]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+2), vik_82);
                    double vik_83 = gout[38]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+3), vik_83);
                    double vik_84 = gout[48]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+4), vik_84);
                    double vik_85 = gout[58]*dm_jl_00;
                    atomicAdd(vk+(i0+8)*nao+(k0+5), vik_85);
                    double vik_90 = gout[9]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    double vik_91 = gout[19]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+1), vik_91);
                    double vik_92 = gout[29]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+2), vik_92);
                    double vik_93 = gout[39]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+3), vik_93);
                    double vik_94 = gout[49]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+4), vik_94);
                    double vik_95 = gout[59]*dm_jl_00;
                    atomicAdd(vk+(i0+9)*nao+(k0+5), vik_95);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                        vjl_00 += gout[10] * dm_ik_01;
                        double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                        vjl_00 += gout[20] * dm_ik_02;
                        double dm_ik_03 = dm[(i0+0)*nao+(k0+3)];
                        vjl_00 += gout[30] * dm_ik_03;
                        double dm_ik_04 = dm[(i0+0)*nao+(k0+4)];
                        vjl_00 += gout[40] * dm_ik_04;
                        double dm_ik_05 = dm[(i0+0)*nao+(k0+5)];
                        vjl_00 += gout[50] * dm_ik_05;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                        vjl_00 += gout[11] * dm_ik_11;
                        double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                        vjl_00 += gout[21] * dm_ik_12;
                        double dm_ik_13 = dm[(i0+1)*nao+(k0+3)];
                        vjl_00 += gout[31] * dm_ik_13;
                        double dm_ik_14 = dm[(i0+1)*nao+(k0+4)];
                        vjl_00 += gout[41] * dm_ik_14;
                        double dm_ik_15 = dm[(i0+1)*nao+(k0+5)];
                        vjl_00 += gout[51] * dm_ik_15;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                        vjl_00 += gout[12] * dm_ik_21;
                        double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                        vjl_00 += gout[22] * dm_ik_22;
                        double dm_ik_23 = dm[(i0+2)*nao+(k0+3)];
                        vjl_00 += gout[32] * dm_ik_23;
                        double dm_ik_24 = dm[(i0+2)*nao+(k0+4)];
                        vjl_00 += gout[42] * dm_ik_24;
                        double dm_ik_25 = dm[(i0+2)*nao+(k0+5)];
                        vjl_00 += gout[52] * dm_ik_25;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                        vjl_00 += gout[13] * dm_ik_31;
                        double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                        vjl_00 += gout[23] * dm_ik_32;
                        double dm_ik_33 = dm[(i0+3)*nao+(k0+3)];
                        vjl_00 += gout[33] * dm_ik_33;
                        double dm_ik_34 = dm[(i0+3)*nao+(k0+4)];
                        vjl_00 += gout[43] * dm_ik_34;
                        double dm_ik_35 = dm[(i0+3)*nao+(k0+5)];
                        vjl_00 += gout[53] * dm_ik_35;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                        vjl_00 += gout[14] * dm_ik_41;
                        double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                        vjl_00 += gout[24] * dm_ik_42;
                        double dm_ik_43 = dm[(i0+4)*nao+(k0+3)];
                        vjl_00 += gout[34] * dm_ik_43;
                        double dm_ik_44 = dm[(i0+4)*nao+(k0+4)];
                        vjl_00 += gout[44] * dm_ik_44;
                        double dm_ik_45 = dm[(i0+4)*nao+(k0+5)];
                        vjl_00 += gout[54] * dm_ik_45;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                        vjl_00 += gout[15] * dm_ik_51;
                        double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                        vjl_00 += gout[25] * dm_ik_52;
                        double dm_ik_53 = dm[(i0+5)*nao+(k0+3)];
                        vjl_00 += gout[35] * dm_ik_53;
                        double dm_ik_54 = dm[(i0+5)*nao+(k0+4)];
                        vjl_00 += gout[45] * dm_ik_54;
                        double dm_ik_55 = dm[(i0+5)*nao+(k0+5)];
                        vjl_00 += gout[55] * dm_ik_55;
                        double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                        vjl_00 += gout[6] * dm_ik_60;
                        double dm_ik_61 = dm[(i0+6)*nao+(k0+1)];
                        vjl_00 += gout[16] * dm_ik_61;
                        double dm_ik_62 = dm[(i0+6)*nao+(k0+2)];
                        vjl_00 += gout[26] * dm_ik_62;
                        double dm_ik_63 = dm[(i0+6)*nao+(k0+3)];
                        vjl_00 += gout[36] * dm_ik_63;
                        double dm_ik_64 = dm[(i0+6)*nao+(k0+4)];
                        vjl_00 += gout[46] * dm_ik_64;
                        double dm_ik_65 = dm[(i0+6)*nao+(k0+5)];
                        vjl_00 += gout[56] * dm_ik_65;
                        double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                        vjl_00 += gout[7] * dm_ik_70;
                        double dm_ik_71 = dm[(i0+7)*nao+(k0+1)];
                        vjl_00 += gout[17] * dm_ik_71;
                        double dm_ik_72 = dm[(i0+7)*nao+(k0+2)];
                        vjl_00 += gout[27] * dm_ik_72;
                        double dm_ik_73 = dm[(i0+7)*nao+(k0+3)];
                        vjl_00 += gout[37] * dm_ik_73;
                        double dm_ik_74 = dm[(i0+7)*nao+(k0+4)];
                        vjl_00 += gout[47] * dm_ik_74;
                        double dm_ik_75 = dm[(i0+7)*nao+(k0+5)];
                        vjl_00 += gout[57] * dm_ik_75;
                        double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                        vjl_00 += gout[8] * dm_ik_80;
                        double dm_ik_81 = dm[(i0+8)*nao+(k0+1)];
                        vjl_00 += gout[18] * dm_ik_81;
                        double dm_ik_82 = dm[(i0+8)*nao+(k0+2)];
                        vjl_00 += gout[28] * dm_ik_82;
                        double dm_ik_83 = dm[(i0+8)*nao+(k0+3)];
                        vjl_00 += gout[38] * dm_ik_83;
                        double dm_ik_84 = dm[(i0+8)*nao+(k0+4)];
                        vjl_00 += gout[48] * dm_ik_84;
                        double dm_ik_85 = dm[(i0+8)*nao+(k0+5)];
                        vjl_00 += gout[58] * dm_ik_85;
                        double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                        vjl_00 += gout[9] * dm_ik_90;
                        double dm_ik_91 = dm[(i0+9)*nao+(k0+1)];
                        vjl_00 += gout[19] * dm_ik_91;
                        double dm_ik_92 = dm[(i0+9)*nao+(k0+2)];
                        vjl_00 += gout[29] * dm_ik_92;
                        double dm_ik_93 = dm[(i0+9)*nao+(k0+3)];
                        vjl_00 += gout[39] * dm_ik_93;
                        double dm_ik_94 = dm[(i0+9)*nao+(k0+4)];
                        vjl_00 += gout[49] * dm_ik_94;
                        double dm_ik_95 = dm[(i0+9)*nao+(k0+5)];
                        vjl_00 += gout[59] * dm_ik_95;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        double vjk_00 = 0;
                        double vjk_01 = 0;
                        double vjk_02 = 0;
                        double vjk_03 = 0;
                        double vjk_04 = 0;
                        double vjk_05 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_01 += gout[10] * dm_il_00;
                        vjk_02 += gout[20] * dm_il_00;
                        vjk_03 += gout[30] * dm_il_00;
                        vjk_04 += gout[40] * dm_il_00;
                        vjk_05 += gout[50] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_01 += gout[11] * dm_il_10;
                        vjk_02 += gout[21] * dm_il_10;
                        vjk_03 += gout[31] * dm_il_10;
                        vjk_04 += gout[41] * dm_il_10;
                        vjk_05 += gout[51] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_01 += gout[12] * dm_il_20;
                        vjk_02 += gout[22] * dm_il_20;
                        vjk_03 += gout[32] * dm_il_20;
                        vjk_04 += gout[42] * dm_il_20;
                        vjk_05 += gout[52] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_01 += gout[13] * dm_il_30;
                        vjk_02 += gout[23] * dm_il_30;
                        vjk_03 += gout[33] * dm_il_30;
                        vjk_04 += gout[43] * dm_il_30;
                        vjk_05 += gout[53] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_01 += gout[14] * dm_il_40;
                        vjk_02 += gout[24] * dm_il_40;
                        vjk_03 += gout[34] * dm_il_40;
                        vjk_04 += gout[44] * dm_il_40;
                        vjk_05 += gout[54] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_01 += gout[15] * dm_il_50;
                        vjk_02 += gout[25] * dm_il_50;
                        vjk_03 += gout[35] * dm_il_50;
                        vjk_04 += gout[45] * dm_il_50;
                        vjk_05 += gout[55] * dm_il_50;
                        double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                        vjk_00 += gout[6] * dm_il_60;
                        vjk_01 += gout[16] * dm_il_60;
                        vjk_02 += gout[26] * dm_il_60;
                        vjk_03 += gout[36] * dm_il_60;
                        vjk_04 += gout[46] * dm_il_60;
                        vjk_05 += gout[56] * dm_il_60;
                        double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                        vjk_00 += gout[7] * dm_il_70;
                        vjk_01 += gout[17] * dm_il_70;
                        vjk_02 += gout[27] * dm_il_70;
                        vjk_03 += gout[37] * dm_il_70;
                        vjk_04 += gout[47] * dm_il_70;
                        vjk_05 += gout[57] * dm_il_70;
                        double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                        vjk_00 += gout[8] * dm_il_80;
                        vjk_01 += gout[18] * dm_il_80;
                        vjk_02 += gout[28] * dm_il_80;
                        vjk_03 += gout[38] * dm_il_80;
                        vjk_04 += gout[48] * dm_il_80;
                        vjk_05 += gout[58] * dm_il_80;
                        double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                        vjk_00 += gout[9] * dm_il_90;
                        vjk_01 += gout[19] * dm_il_90;
                        vjk_02 += gout[29] * dm_il_90;
                        vjk_03 += gout[39] * dm_il_90;
                        vjk_04 += gout[49] * dm_il_90;
                        vjk_05 += gout[59] * dm_il_90;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                        atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                        atomicAdd(vk+(j0+0)*nao+(k0+3), vjk_03);
                        atomicAdd(vk+(j0+0)*nao+(k0+4), vjk_04);
                        atomicAdd(vk+(j0+0)*nao+(k0+5), vjk_05);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_3100(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double *&expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    double *&expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<double*>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;


while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[30];
#pragma unroll
            for (int n = 0; n < 30; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        gout[0] += hrr_3100x * fac * wt;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_2100x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_2100x * fac * trr_10z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += hrr_1100x * trr_20y * wt;
                        gout[4] += hrr_1100x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += hrr_1100x * fac * trr_20z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout[6] += hrr_0100x * trr_30y * wt;
                        gout[7] += hrr_0100x * trr_20y * trr_10z;
                        gout[8] += hrr_0100x * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout[9] += hrr_0100x * fac * trr_30z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[10] += trr_30x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[11] += trr_20x * hrr_1100y * wt;
                        gout[12] += trr_20x * hrr_0100y * trr_10z;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout[13] += trr_10x * hrr_2100y * wt;
                        gout[14] += trr_10x * hrr_1100y * trr_10z;
                        gout[15] += trr_10x * hrr_0100y * trr_20z;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        gout[16] += 1 * hrr_3100y * wt;
                        gout[17] += 1 * hrr_2100y * trr_10z;
                        gout[18] += 1 * hrr_1100y * trr_20z;
                        gout[19] += 1 * hrr_0100y * trr_30z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[20] += trr_30x * fac * hrr_0100z;
                        gout[21] += trr_20x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[22] += trr_20x * fac * hrr_1100z;
                        gout[23] += trr_10x * trr_20y * hrr_0100z;
                        gout[24] += trr_10x * trr_10y * hrr_1100z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout[25] += trr_10x * fac * hrr_2100z;
                        gout[26] += 1 * trr_30y * hrr_0100z;
                        gout[27] += 1 * trr_20y * hrr_1100z;
                        gout[28] += 1 * trr_10y * hrr_2100z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        gout[29] += 1 * fac * hrr_3100z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[10]*dm_jk_10 + gout[20]*dm_jk_20;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[11]*dm_jk_10 + gout[21]*dm_jk_20;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[12]*dm_jk_10 + gout[22]*dm_jk_20;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[13]*dm_jk_10 + gout[23]*dm_jk_20;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[14]*dm_jk_10 + gout[24]*dm_jk_20;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[15]*dm_jk_10 + gout[25]*dm_jk_20;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_60 = gout[6]*dm_jk_00 + gout[16]*dm_jk_10 + gout[26]*dm_jk_20;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_70 = gout[7]*dm_jk_00 + gout[17]*dm_jk_10 + gout[27]*dm_jk_20;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_80 = gout[8]*dm_jk_00 + gout[18]*dm_jk_10 + gout[28]*dm_jk_20;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_90 = gout[9]*dm_jk_00 + gout[19]*dm_jk_10 + gout[29]*dm_jk_20;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[10]*dm_jl_10 + gout[20]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[11]*dm_jl_10 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[12]*dm_jl_10 + gout[22]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00 + gout[13]*dm_jl_10 + gout[23]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00 + gout[14]*dm_jl_10 + gout[24]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00 + gout[15]*dm_jl_10 + gout[25]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_60 = gout[6]*dm_jl_00 + gout[16]*dm_jl_10 + gout[26]*dm_jl_20;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_70 = gout[7]*dm_jl_00 + gout[17]*dm_jl_10 + gout[27]*dm_jl_20;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_80 = gout[8]*dm_jl_00 + gout[18]*dm_jl_10 + gout[28]*dm_jl_20;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_90 = gout[9]*dm_jl_00 + gout[19]*dm_jl_10 + gout[29]*dm_jl_20;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[10] * dm_ik_00;
                        vjl_20 += gout[20] * dm_ik_00;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[11] * dm_ik_10;
                        vjl_20 += gout[21] * dm_ik_10;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[12] * dm_ik_20;
                        vjl_20 += gout[22] * dm_ik_20;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        vjl_10 += gout[13] * dm_ik_30;
                        vjl_20 += gout[23] * dm_ik_30;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        vjl_10 += gout[14] * dm_ik_40;
                        vjl_20 += gout[24] * dm_ik_40;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        vjl_10 += gout[15] * dm_ik_50;
                        vjl_20 += gout[25] * dm_ik_50;
                        double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                        vjl_00 += gout[6] * dm_ik_60;
                        vjl_10 += gout[16] * dm_ik_60;
                        vjl_20 += gout[26] * dm_ik_60;
                        double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                        vjl_00 += gout[7] * dm_ik_70;
                        vjl_10 += gout[17] * dm_ik_70;
                        vjl_20 += gout[27] * dm_ik_70;
                        double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                        vjl_00 += gout[8] * dm_ik_80;
                        vjl_10 += gout[18] * dm_ik_80;
                        vjl_20 += gout[28] * dm_ik_80;
                        double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                        vjl_00 += gout[9] * dm_ik_90;
                        vjl_10 += gout[19] * dm_ik_90;
                        vjl_20 += gout[29] * dm_ik_90;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        double vjk_00 = 0;
                        double vjk_10 = 0;
                        double vjk_20 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_10 += gout[10] * dm_il_00;
                        vjk_20 += gout[20] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_10 += gout[11] * dm_il_10;
                        vjk_20 += gout[21] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_10 += gout[12] * dm_il_20;
                        vjk_20 += gout[22] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_10 += gout[13] * dm_il_30;
                        vjk_20 += gout[23] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_10 += gout[14] * dm_il_40;
                        vjk_20 += gout[24] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_10 += gout[15] * dm_il_50;
                        vjk_20 += gout[25] * dm_il_50;
                        double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                        vjk_00 += gout[6] * dm_il_60;
                        vjk_10 += gout[16] * dm_il_60;
                        vjk_20 += gout[26] * dm_il_60;
                        double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                        vjk_00 += gout[7] * dm_il_70;
                        vjk_10 += gout[17] * dm_il_70;
                        vjk_20 += gout[27] * dm_il_70;
                        double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                        vjk_00 += gout[8] * dm_il_80;
                        vjk_10 += gout[18] * dm_il_80;
                        vjk_20 += gout[28] * dm_il_80;
                        double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                        vjk_00 += gout[9] * dm_il_90;
                        vjk_10 += gout[19] * dm_il_90;
                        vjk_20 += gout[29] * dm_il_90;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    }
                }
            }
        }
    }
}
}

__global__ static
__global__ static
void rys_k_3110(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int gout_id = item.get_local_id(0);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&aij_cache)[2] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[2]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ double aij_cache[2];
    __shared__ int expi;
    __shared__ int expj;
    #endif
    int t_id = 64 * gout_id + sq_id;
    constexpr int threads = 256;
    constexpr int nsq_per_block = 64;
    constexpr int g_size = 16;
    uint32_t nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;

    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + nsq_per_block * 3 + sq_id;
    double *akl_cache = shared_memory + nsq_per_block * 6 + sq_id;
    double *gx = shared_memory + nsq_per_block * 8 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+8) + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * (g_size*3+bounds.nroots*2+8);

while (1) {
    if (t_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (t_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
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
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            __syncthreads();
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            if (gout_id == 0) {
                double xlxk = env[rl+0] - env[rk+0];
                double ylyk = env[rl+1] - env[rk+1];
                double zlzk = env[rl+2] - env[rk+2];
                rlrk[0] = xlxk;
                rlrk[64] = ylyk;
                rlrk[128] = zlzk;
            }
            double gout[23];

            #pragma unroll
            for (int n = 0; n < 23; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                __syncthreads();
                if (gout_id == 0) {
                    int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
                    int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
                    int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
                    int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
                    int kp = klp / lprim;
                    int lp = klp % lprim;
                    double ak = env[expk+kp];
                    double al = env[expl+lp];
                    double akl = ak + al;
                    double al_akl = al / akl;
                    double xlxk = rlrk[0];
                    double ylyk = rlrk[64];
                    double zlzk = rlrk[128];
                    double theta_kl = ak * al_akl;
                    double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                    double ckcl = env[ck+kp] * env[cl+lp] * Kcd;
                    double fac_sym = PI_FAC;
                    if (task_id < ntasks) {
                        if (ksh == lsh) fac_sym *= .5;
                        if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                    } else {
                        fac_sym = 0;
                    }
                    gx[0] = fac_sym * ckcl;
                    akl_cache[0] = akl;
                    akl_cache[nsq_per_block] = al_akl;
                }
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
                    double akl = akl_cache[0];
                    double al_akl = akl_cache[nsq_per_block];
                    double xij = ri[0] + rjri[0] * aj_aij;
                    double yij = ri[1] + rjri[1] * aj_aij;
                    double zij = ri[2] + rjri[2] * aj_aij;
                    double xkl = env[rk+0] + rlrk[0*nsq_per_block] * al_akl;
                    double ykl = env[rk+1] + rlrk[1*nsq_per_block] * al_akl;
                    double zkl = env[rk+2] + rlrk[2*nsq_per_block] * al_akl;
                    double xpq = xij - xkl;
                    double ypq = yij - ykl;
                    double zpq = zij - zkl;
                    double theta = aij * akl / (aij + akl);
                    double rr = xpq * xpq + ypq * ypq + zpq * zpq;
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (gout_id == 0) {
                        Rpq[0*nsq_per_block] = xpq;
                        Rpq[1*nsq_per_block] = ypq;
                        Rpq[2*nsq_per_block] = zpq;
                        double cicj = cicj_cache[ijp];
                        gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                        if (sq_id == 0) {
                            aij_cache[0] = aij;
                            aij_cache[1] = aj_aij;
                        }
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        __syncthreads();
                        double s0, s1, s2;
                        double rt = rw[irys*128];
                        double aij = aij_cache[0];
                        double rt_aa = rt / (aij + akl);
                        double akl = akl_cache[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double rt_akl = rt_aa * aij;
                        double b00 = .5 * rt_aa;
                        for (int n = gout_id; n < 3; n += 4) {
                            if (n == 2) {
                                gx[2048] = rw[irys*128+64];
                            }
                            double *_gx = gx + n * 1024;
                            double xjxi = rjri[n];
                            double Rpa = xjxi * aij_cache[1];
                            double c0x = Rpa - rt_aij * Rpq[n*64];
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
                            double xlxk = rlrk[n*64];
                            double Rqc = xlxk * akl_cache[64];
                            double cpx = Rqc + rt_akl * Rpq[n*64];
                            s0 = _gx[0];
                            s1 = cpx * s0;
                            _gx[512] = s1;
                            s0 = _gx[64];
                            s1 = cpx * s0;
                            s1 += 1 * b00 * _gx[0];
                            _gx[576] = s1;
                            s0 = _gx[128];
                            s1 = cpx * s0;
                            s1 += 2 * b00 * _gx[64];
                            _gx[640] = s1;
                            s0 = _gx[192];
                            s1 = cpx * s0;
                            s1 += 3 * b00 * _gx[128];
                            _gx[704] = s1;
                            s0 = _gx[256];
                            s1 = cpx * s0;
                            s1 += 4 * b00 * _gx[192];
                            _gx[768] = s1;
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
                        }
                        __syncthreads();
                        switch (gout_id) {
                        case 0:
                        gout[0] += gx[960] * gx[1024] * gx[2048];
                        gout[1] += gx[832] * gx[1088] * gx[2112];
                        gout[2] += gx[768] * gx[1088] * gx[2176];
                        gout[3] += gx[640] * gx[1280] * gx[2112];
                        gout[4] += gx[512] * gx[1472] * gx[2048];
                        gout[5] += gx[704] * gx[1024] * gx[2304];
                        gout[6] += gx[576] * gx[1088] * gx[2368];
                        gout[7] += gx[512] * gx[1088] * gx[2432];
                        gout[8] += gx[384] * gx[1536] * gx[2112];
                        gout[9] += gx[256] * gx[1728] * gx[2048];
                        gout[10] += gx[192] * gx[1792] * gx[2048];
                        gout[11] += gx[64] * gx[1856] * gx[2112];
                        gout[12] += gx[0] * gx[1856] * gx[2176];
                        gout[13] += gx[128] * gx[1536] * gx[2368];
                        gout[14] += gx[0] * gx[1728] * gx[2304];
                        gout[15] += gx[448] * gx[1024] * gx[2560];
                        gout[16] += gx[320] * gx[1088] * gx[2624];
                        gout[17] += gx[256] * gx[1088] * gx[2688];
                        gout[18] += gx[128] * gx[1280] * gx[2624];
                        gout[19] += gx[0] * gx[1472] * gx[2560];
                        gout[20] += gx[192] * gx[1024] * gx[2816];
                        gout[21] += gx[64] * gx[1088] * gx[2880];
                        gout[22] += gx[0] * gx[1088] * gx[2944];
                        break;
                        case 1:
                        gout[0] += gx[896] * gx[1088] * gx[2048];
                        gout[1] += gx[832] * gx[1024] * gx[2176];
                        gout[2] += gx[768] * gx[1024] * gx[2240];
                        gout[3] += gx[576] * gx[1408] * gx[2048];
                        gout[4] += gx[512] * gx[1408] * gx[2112];
                        gout[5] += gx[640] * gx[1088] * gx[2304];
                        gout[6] += gx[576] * gx[1024] * gx[2432];
                        gout[7] += gx[512] * gx[1024] * gx[2496];
                        gout[8] += gx[320] * gx[1664] * gx[2048];
                        gout[9] += gx[256] * gx[1664] * gx[2112];
                        gout[10] += gx[128] * gx[1856] * gx[2048];
                        gout[11] += gx[64] * gx[1792] * gx[2176];
                        gout[12] += gx[0] * gx[1792] * gx[2240];
                        gout[13] += gx[64] * gx[1664] * gx[2304];
                        gout[14] += gx[0] * gx[1664] * gx[2368];
                        gout[15] += gx[384] * gx[1088] * gx[2560];
                        gout[16] += gx[320] * gx[1024] * gx[2688];
                        gout[17] += gx[256] * gx[1024] * gx[2752];
                        gout[18] += gx[64] * gx[1408] * gx[2560];
                        gout[19] += gx[0] * gx[1408] * gx[2624];
                        gout[20] += gx[128] * gx[1088] * gx[2816];
                        gout[21] += gx[64] * gx[1024] * gx[2944];
                        gout[22] += gx[0] * gx[1024] * gx[3008];
                        break;
                        case 2:
                        gout[0] += gx[896] * gx[1024] * gx[2112];
                        gout[1] += gx[768] * gx[1216] * gx[2048];
                        gout[2] += gx[704] * gx[1280] * gx[2048];
                        gout[3] += gx[576] * gx[1344] * gx[2112];
                        gout[4] += gx[512] * gx[1344] * gx[2176];
                        gout[5] += gx[640] * gx[1024] * gx[2368];
                        gout[6] += gx[512] * gx[1216] * gx[2304];
                        gout[7] += gx[448] * gx[1536] * gx[2048];
                        gout[8] += gx[320] * gx[1600] * gx[2112];
                        gout[9] += gx[256] * gx[1600] * gx[2176];
                        gout[10] += gx[128] * gx[1792] * gx[2112];
                        gout[11] += gx[0] * gx[1984] * gx[2048];
                        gout[12] += gx[192] * gx[1536] * gx[2304];
                        gout[13] += gx[64] * gx[1600] * gx[2368];
                        gout[14] += gx[0] * gx[1600] * gx[2432];
                        gout[15] += gx[384] * gx[1024] * gx[2624];
                        gout[16] += gx[256] * gx[1216] * gx[2560];
                        gout[17] += gx[192] * gx[1280] * gx[2560];
                        gout[18] += gx[64] * gx[1344] * gx[2624];
                        gout[19] += gx[0] * gx[1344] * gx[2688];
                        gout[20] += gx[128] * gx[1024] * gx[2880];
                        gout[21] += gx[0] * gx[1216] * gx[2816];
                        break;
                        case 3:
                        gout[0] += gx[832] * gx[1152] * gx[2048];
                        gout[1] += gx[768] * gx[1152] * gx[2112];
                        gout[2] += gx[640] * gx[1344] * gx[2048];
                        gout[3] += gx[576] * gx[1280] * gx[2176];
                        gout[4] += gx[512] * gx[1280] * gx[2240];
                        gout[5] += gx[576] * gx[1152] * gx[2304];
                        gout[6] += gx[512] * gx[1152] * gx[2368];
                        gout[7] += gx[384] * gx[1600] * gx[2048];
                        gout[8] += gx[320] * gx[1536] * gx[2176];
                        gout[9] += gx[256] * gx[1536] * gx[2240];
                        gout[10] += gx[64] * gx[1920] * gx[2048];
                        gout[11] += gx[0] * gx[1920] * gx[2112];
                        gout[12] += gx[128] * gx[1600] * gx[2304];
                        gout[13] += gx[64] * gx[1536] * gx[2432];
                        gout[14] += gx[0] * gx[1536] * gx[2496];
                        gout[15] += gx[320] * gx[1152] * gx[2560];
                        gout[16] += gx[256] * gx[1152] * gx[2624];
                        gout[17] += gx[128] * gx[1344] * gx[2560];
                        gout[18] += gx[64] * gx[1280] * gx[2688];
                        gout[19] += gx[0] * gx[1280] * gx[2752];
                        gout[20] += gx[64] * gx[1152] * gx[2816];
                        gout[21] += gx[0] * gx[1152] * gx[2880];
                        break;
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao * nao;
                    double *vk = kmat.vk + i_dm * nao * nao;
                    switch (gout_id) {
                    case 0: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[5]*dm_jk_20 + gout[10]*dm_jk_11 + gout[15]*dm_jk_02 + gout[20]*dm_jk_22;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_20 = gout[3]*dm_jk_10 + gout[8]*dm_jk_01 + gout[13]*dm_jk_21 + gout[18]*dm_jk_12;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_40 = gout[1]*dm_jk_00 + gout[6]*dm_jk_20 + gout[11]*dm_jk_11 + gout[16]*dm_jk_02 + gout[21]*dm_jk_22;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_60 = gout[4]*dm_jk_10 + gout[9]*dm_jk_01 + gout[14]*dm_jk_21 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_80 = gout[2]*dm_jk_00 + gout[7]*dm_jk_20 + gout[12]*dm_jk_11 + gout[17]*dm_jk_02 + gout[22]*dm_jk_22;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    break; }
                    case 1: {
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_10 = gout[0]*dm_jk_00 + gout[5]*dm_jk_20 + gout[10]*dm_jk_11 + gout[15]*dm_jk_02 + gout[20]*dm_jk_22;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_30 = gout[3]*dm_jk_10 + gout[8]*dm_jk_01 + gout[13]*dm_jk_21 + gout[18]*dm_jk_12;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_50 = gout[1]*dm_jk_00 + gout[6]*dm_jk_20 + gout[11]*dm_jk_11 + gout[16]*dm_jk_02 + gout[21]*dm_jk_22;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_70 = gout[4]*dm_jk_10 + gout[9]*dm_jk_01 + gout[14]*dm_jk_21 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_90 = gout[2]*dm_jk_00 + gout[7]*dm_jk_20 + gout[12]*dm_jk_11 + gout[17]*dm_jk_02 + gout[22]*dm_jk_22;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    break; }
                    case 2: {
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_00 = gout[2]*dm_jk_10 + gout[7]*dm_jk_01 + gout[12]*dm_jk_21 + gout[17]*dm_jk_12;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_20 = gout[0]*dm_jk_00 + gout[5]*dm_jk_20 + gout[10]*dm_jk_11 + gout[15]*dm_jk_02 + gout[20]*dm_jk_22;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_40 = gout[3]*dm_jk_10 + gout[8]*dm_jk_01 + gout[13]*dm_jk_21 + gout[18]*dm_jk_12;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_60 = gout[1]*dm_jk_00 + gout[6]*dm_jk_20 + gout[11]*dm_jk_11 + gout[16]*dm_jk_02 + gout[21]*dm_jk_22;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_80 = gout[4]*dm_jk_10 + gout[9]*dm_jk_01 + gout[14]*dm_jk_21 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    break; }
                    case 3: {
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_01 = dm[(j0+0)*nao+(k0+1)];
                    double dm_jk_21 = dm[(j0+2)*nao+(k0+1)];
                    double dm_jk_12 = dm[(j0+1)*nao+(k0+2)];
                    double vil_10 = gout[2]*dm_jk_10 + gout[7]*dm_jk_01 + gout[12]*dm_jk_21 + gout[17]*dm_jk_12;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_11 = dm[(j0+1)*nao+(k0+1)];
                    double dm_jk_02 = dm[(j0+0)*nao+(k0+2)];
                    double dm_jk_22 = dm[(j0+2)*nao+(k0+2)];
                    double vil_30 = gout[0]*dm_jk_00 + gout[5]*dm_jk_20 + gout[10]*dm_jk_11 + gout[15]*dm_jk_02 + gout[20]*dm_jk_22;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_50 = gout[3]*dm_jk_10 + gout[8]*dm_jk_01 + gout[13]*dm_jk_21 + gout[18]*dm_jk_12;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_70 = gout[1]*dm_jk_00 + gout[6]*dm_jk_20 + gout[11]*dm_jk_11 + gout[16]*dm_jk_02 + gout[21]*dm_jk_22;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_90 = gout[4]*dm_jk_10 + gout[9]*dm_jk_01 + gout[14]*dm_jk_21 + gout[19]*dm_jk_12;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    break; }
                    }
                    switch (gout_id) {
                    case 0: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[5]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_01 = gout[10]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[15]*dm_jl_00 + gout[20]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[3]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[8]*dm_jl_00 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[18]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[1]*dm_jl_00 + gout[6]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[16]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_60 = gout[4]*dm_jl_10;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_61 = gout[9]*dm_jl_00 + gout[14]*dm_jl_20;
                    atomicAdd(vk+(i0+6)*nao+(k0+1), vik_61);
                    double vik_62 = gout[19]*dm_jl_10;
                    atomicAdd(vk+(i0+6)*nao+(k0+2), vik_62);
                    double vik_80 = gout[2]*dm_jl_00 + gout[7]*dm_jl_20;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_81 = gout[12]*dm_jl_10;
                    atomicAdd(vk+(i0+8)*nao+(k0+1), vik_81);
                    double vik_82 = gout[17]*dm_jl_00 + gout[22]*dm_jl_20;
                    atomicAdd(vk+(i0+8)*nao+(k0+2), vik_82);
                    break; }
                    case 1: {
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_10 = gout[0]*dm_jl_00 + gout[5]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_11 = gout[10]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[15]*dm_jl_00 + gout[20]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[3]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[8]*dm_jl_00 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[18]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[1]*dm_jl_00 + gout[6]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[16]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_70 = gout[4]*dm_jl_10;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_71 = gout[9]*dm_jl_00 + gout[14]*dm_jl_20;
                    atomicAdd(vk+(i0+7)*nao+(k0+1), vik_71);
                    double vik_72 = gout[19]*dm_jl_10;
                    atomicAdd(vk+(i0+7)*nao+(k0+2), vik_72);
                    double vik_90 = gout[2]*dm_jl_00 + gout[7]*dm_jl_20;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    double vik_91 = gout[12]*dm_jl_10;
                    atomicAdd(vk+(i0+9)*nao+(k0+1), vik_91);
                    double vik_92 = gout[17]*dm_jl_00 + gout[22]*dm_jl_20;
                    atomicAdd(vk+(i0+9)*nao+(k0+2), vik_92);
                    break; }
                    case 2: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_00 = gout[2]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_01 = gout[7]*dm_jl_00 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+0)*nao+(k0+1), vik_01);
                    double vik_02 = gout[17]*dm_jl_10;
                    atomicAdd(vk+(i0+0)*nao+(k0+2), vik_02);
                    double vik_20 = gout[0]*dm_jl_00 + gout[5]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_21 = gout[10]*dm_jl_10;
                    atomicAdd(vk+(i0+2)*nao+(k0+1), vik_21);
                    double vik_22 = gout[15]*dm_jl_00 + gout[20]*dm_jl_20;
                    atomicAdd(vk+(i0+2)*nao+(k0+2), vik_22);
                    double vik_40 = gout[3]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_41 = gout[8]*dm_jl_00 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+4)*nao+(k0+1), vik_41);
                    double vik_42 = gout[18]*dm_jl_10;
                    atomicAdd(vk+(i0+4)*nao+(k0+2), vik_42);
                    double vik_60 = gout[1]*dm_jl_00 + gout[6]*dm_jl_20;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_61 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+6)*nao+(k0+1), vik_61);
                    double vik_62 = gout[16]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+6)*nao+(k0+2), vik_62);
                    double vik_80 = gout[4]*dm_jl_10;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_81 = gout[9]*dm_jl_00 + gout[14]*dm_jl_20;
                    atomicAdd(vk+(i0+8)*nao+(k0+1), vik_81);
                    double vik_82 = gout[19]*dm_jl_10;
                    atomicAdd(vk+(i0+8)*nao+(k0+2), vik_82);
                    break; }
                    case 3: {
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double vik_10 = gout[2]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double vik_11 = gout[7]*dm_jl_00 + gout[12]*dm_jl_20;
                    atomicAdd(vk+(i0+1)*nao+(k0+1), vik_11);
                    double vik_12 = gout[17]*dm_jl_10;
                    atomicAdd(vk+(i0+1)*nao+(k0+2), vik_12);
                    double vik_30 = gout[0]*dm_jl_00 + gout[5]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_31 = gout[10]*dm_jl_10;
                    atomicAdd(vk+(i0+3)*nao+(k0+1), vik_31);
                    double vik_32 = gout[15]*dm_jl_00 + gout[20]*dm_jl_20;
                    atomicAdd(vk+(i0+3)*nao+(k0+2), vik_32);
                    double vik_50 = gout[3]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_51 = gout[8]*dm_jl_00 + gout[13]*dm_jl_20;
                    atomicAdd(vk+(i0+5)*nao+(k0+1), vik_51);
                    double vik_52 = gout[18]*dm_jl_10;
                    atomicAdd(vk+(i0+5)*nao+(k0+2), vik_52);
                    double vik_70 = gout[1]*dm_jl_00 + gout[6]*dm_jl_20;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_71 = gout[11]*dm_jl_10;
                    atomicAdd(vk+(i0+7)*nao+(k0+1), vik_71);
                    double vik_72 = gout[16]*dm_jl_00 + gout[21]*dm_jl_20;
                    atomicAdd(vk+(i0+7)*nao+(k0+2), vik_72);
                    double vik_90 = gout[4]*dm_jl_10;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    double vik_91 = gout[9]*dm_jl_00 + gout[14]*dm_jl_20;
                    atomicAdd(vk+(i0+9)*nao+(k0+1), vik_91);
                    double vik_92 = gout[19]*dm_jl_10;
                    atomicAdd(vk+(i0+9)*nao+(k0+2), vik_92);
                    break; }
                    }
                    if (ish != jsh) {
                    double vjl_00 = 0;
                    double vjl_10 = 0;
                    double vjl_20 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_00;
                    vjl_20 += gout[5] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_10 += gout[10] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_00 += gout[15] * dm_ik_02;
                    vjl_20 += gout[20] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_10 += gout[3] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_00 += gout[8] * dm_ik_21;
                    vjl_20 += gout[13] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_10 += gout[18] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_40;
                    vjl_20 += gout[6] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_00 += gout[16] * dm_ik_42;
                    vjl_20 += gout[21] * dm_ik_42;
                    double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                    vjl_10 += gout[4] * dm_ik_60;
                    double dm_ik_61 = dm[(i0+6)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_61;
                    vjl_20 += gout[14] * dm_ik_61;
                    double dm_ik_62 = dm[(i0+6)*nao+(k0+2)];
                    vjl_10 += gout[19] * dm_ik_62;
                    double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                    vjl_00 += gout[2] * dm_ik_80;
                    vjl_20 += gout[7] * dm_ik_80;
                    double dm_ik_81 = dm[(i0+8)*nao+(k0+1)];
                    vjl_10 += gout[12] * dm_ik_81;
                    double dm_ik_82 = dm[(i0+8)*nao+(k0+2)];
                    vjl_00 += gout[17] * dm_ik_82;
                    vjl_20 += gout[22] * dm_ik_82;
                    break; }
                    case 1: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_10;
                    vjl_20 += gout[5] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_10 += gout[10] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_00 += gout[15] * dm_ik_12;
                    vjl_20 += gout[20] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_10 += gout[3] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_00 += gout[8] * dm_ik_31;
                    vjl_20 += gout[13] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_10 += gout[18] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_50;
                    vjl_20 += gout[6] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_00 += gout[16] * dm_ik_52;
                    vjl_20 += gout[21] * dm_ik_52;
                    double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                    vjl_10 += gout[4] * dm_ik_70;
                    double dm_ik_71 = dm[(i0+7)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_71;
                    vjl_20 += gout[14] * dm_ik_71;
                    double dm_ik_72 = dm[(i0+7)*nao+(k0+2)];
                    vjl_10 += gout[19] * dm_ik_72;
                    double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                    vjl_00 += gout[2] * dm_ik_90;
                    vjl_20 += gout[7] * dm_ik_90;
                    double dm_ik_91 = dm[(i0+9)*nao+(k0+1)];
                    vjl_10 += gout[12] * dm_ik_91;
                    double dm_ik_92 = dm[(i0+9)*nao+(k0+2)];
                    vjl_00 += gout[17] * dm_ik_92;
                    vjl_20 += gout[22] * dm_ik_92;
                    break; }
                    case 2: {
                    double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_00;
                    double dm_ik_01 = dm[(i0+0)*nao+(k0+1)];
                    vjl_00 += gout[7] * dm_ik_01;
                    vjl_20 += gout[12] * dm_ik_01;
                    double dm_ik_02 = dm[(i0+0)*nao+(k0+2)];
                    vjl_10 += gout[17] * dm_ik_02;
                    double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_20;
                    vjl_20 += gout[5] * dm_ik_20;
                    double dm_ik_21 = dm[(i0+2)*nao+(k0+1)];
                    vjl_10 += gout[10] * dm_ik_21;
                    double dm_ik_22 = dm[(i0+2)*nao+(k0+2)];
                    vjl_00 += gout[15] * dm_ik_22;
                    vjl_20 += gout[20] * dm_ik_22;
                    double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                    vjl_10 += gout[3] * dm_ik_40;
                    double dm_ik_41 = dm[(i0+4)*nao+(k0+1)];
                    vjl_00 += gout[8] * dm_ik_41;
                    vjl_20 += gout[13] * dm_ik_41;
                    double dm_ik_42 = dm[(i0+4)*nao+(k0+2)];
                    vjl_10 += gout[18] * dm_ik_42;
                    double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_60;
                    vjl_20 += gout[6] * dm_ik_60;
                    double dm_ik_61 = dm[(i0+6)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_61;
                    double dm_ik_62 = dm[(i0+6)*nao+(k0+2)];
                    vjl_00 += gout[16] * dm_ik_62;
                    vjl_20 += gout[21] * dm_ik_62;
                    double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                    vjl_10 += gout[4] * dm_ik_80;
                    double dm_ik_81 = dm[(i0+8)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_81;
                    vjl_20 += gout[14] * dm_ik_81;
                    double dm_ik_82 = dm[(i0+8)*nao+(k0+2)];
                    vjl_10 += gout[19] * dm_ik_82;
                    break; }
                    case 3: {
                    double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                    vjl_10 += gout[2] * dm_ik_10;
                    double dm_ik_11 = dm[(i0+1)*nao+(k0+1)];
                    vjl_00 += gout[7] * dm_ik_11;
                    vjl_20 += gout[12] * dm_ik_11;
                    double dm_ik_12 = dm[(i0+1)*nao+(k0+2)];
                    vjl_10 += gout[17] * dm_ik_12;
                    double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                    vjl_00 += gout[0] * dm_ik_30;
                    vjl_20 += gout[5] * dm_ik_30;
                    double dm_ik_31 = dm[(i0+3)*nao+(k0+1)];
                    vjl_10 += gout[10] * dm_ik_31;
                    double dm_ik_32 = dm[(i0+3)*nao+(k0+2)];
                    vjl_00 += gout[15] * dm_ik_32;
                    vjl_20 += gout[20] * dm_ik_32;
                    double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                    vjl_10 += gout[3] * dm_ik_50;
                    double dm_ik_51 = dm[(i0+5)*nao+(k0+1)];
                    vjl_00 += gout[8] * dm_ik_51;
                    vjl_20 += gout[13] * dm_ik_51;
                    double dm_ik_52 = dm[(i0+5)*nao+(k0+2)];
                    vjl_10 += gout[18] * dm_ik_52;
                    double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                    vjl_00 += gout[1] * dm_ik_70;
                    vjl_20 += gout[6] * dm_ik_70;
                    double dm_ik_71 = dm[(i0+7)*nao+(k0+1)];
                    vjl_10 += gout[11] * dm_ik_71;
                    double dm_ik_72 = dm[(i0+7)*nao+(k0+2)];
                    vjl_00 += gout[16] * dm_ik_72;
                    vjl_20 += gout[21] * dm_ik_72;
                    double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                    vjl_10 += gout[4] * dm_ik_90;
                    double dm_ik_91 = dm[(i0+9)*nao+(k0+1)];
                    vjl_00 += gout[9] * dm_ik_91;
                    vjl_20 += gout[14] * dm_ik_91;
                    double dm_ik_92 = dm[(i0+9)*nao+(k0+2)];
                    vjl_10 += gout[19] * dm_ik_92;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                    atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                    atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                    double vjk_00 = 0;
                    double vjk_01 = 0;
                    double vjk_02 = 0;
                    double vjk_10 = 0;
                    double vjk_11 = 0;
                    double vjk_12 = 0;
                    double vjk_20 = 0;
                    double vjk_21 = 0;
                    double vjk_22 = 0;
                    switch (gout_id) {
                    case 0: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_00;
                    vjk_02 += gout[15] * dm_il_00;
                    vjk_11 += gout[10] * dm_il_00;
                    vjk_20 += gout[5] * dm_il_00;
                    vjk_22 += gout[20] * dm_il_00;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_01 += gout[8] * dm_il_20;
                    vjk_10 += gout[3] * dm_il_20;
                    vjk_12 += gout[18] * dm_il_20;
                    vjk_21 += gout[13] * dm_il_20;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_40;
                    vjk_02 += gout[16] * dm_il_40;
                    vjk_11 += gout[11] * dm_il_40;
                    vjk_20 += gout[6] * dm_il_40;
                    vjk_22 += gout[21] * dm_il_40;
                    double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                    vjk_01 += gout[9] * dm_il_60;
                    vjk_10 += gout[4] * dm_il_60;
                    vjk_12 += gout[19] * dm_il_60;
                    vjk_21 += gout[14] * dm_il_60;
                    double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                    vjk_00 += gout[2] * dm_il_80;
                    vjk_02 += gout[17] * dm_il_80;
                    vjk_11 += gout[12] * dm_il_80;
                    vjk_20 += gout[7] * dm_il_80;
                    vjk_22 += gout[22] * dm_il_80;
                    break; }
                    case 1: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_10;
                    vjk_02 += gout[15] * dm_il_10;
                    vjk_11 += gout[10] * dm_il_10;
                    vjk_20 += gout[5] * dm_il_10;
                    vjk_22 += gout[20] * dm_il_10;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_01 += gout[8] * dm_il_30;
                    vjk_10 += gout[3] * dm_il_30;
                    vjk_12 += gout[18] * dm_il_30;
                    vjk_21 += gout[13] * dm_il_30;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_50;
                    vjk_02 += gout[16] * dm_il_50;
                    vjk_11 += gout[11] * dm_il_50;
                    vjk_20 += gout[6] * dm_il_50;
                    vjk_22 += gout[21] * dm_il_50;
                    double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                    vjk_01 += gout[9] * dm_il_70;
                    vjk_10 += gout[4] * dm_il_70;
                    vjk_12 += gout[19] * dm_il_70;
                    vjk_21 += gout[14] * dm_il_70;
                    double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                    vjk_00 += gout[2] * dm_il_90;
                    vjk_02 += gout[17] * dm_il_90;
                    vjk_11 += gout[12] * dm_il_90;
                    vjk_20 += gout[7] * dm_il_90;
                    vjk_22 += gout[22] * dm_il_90;
                    break; }
                    case 2: {
                    double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                    vjk_01 += gout[7] * dm_il_00;
                    vjk_10 += gout[2] * dm_il_00;
                    vjk_12 += gout[17] * dm_il_00;
                    vjk_21 += gout[12] * dm_il_00;
                    double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_20;
                    vjk_02 += gout[15] * dm_il_20;
                    vjk_11 += gout[10] * dm_il_20;
                    vjk_20 += gout[5] * dm_il_20;
                    vjk_22 += gout[20] * dm_il_20;
                    double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                    vjk_01 += gout[8] * dm_il_40;
                    vjk_10 += gout[3] * dm_il_40;
                    vjk_12 += gout[18] * dm_il_40;
                    vjk_21 += gout[13] * dm_il_40;
                    double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_60;
                    vjk_02 += gout[16] * dm_il_60;
                    vjk_11 += gout[11] * dm_il_60;
                    vjk_20 += gout[6] * dm_il_60;
                    vjk_22 += gout[21] * dm_il_60;
                    double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                    vjk_01 += gout[9] * dm_il_80;
                    vjk_10 += gout[4] * dm_il_80;
                    vjk_12 += gout[19] * dm_il_80;
                    vjk_21 += gout[14] * dm_il_80;
                    break; }
                    case 3: {
                    double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                    vjk_01 += gout[7] * dm_il_10;
                    vjk_10 += gout[2] * dm_il_10;
                    vjk_12 += gout[17] * dm_il_10;
                    vjk_21 += gout[12] * dm_il_10;
                    double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                    vjk_00 += gout[0] * dm_il_30;
                    vjk_02 += gout[15] * dm_il_30;
                    vjk_11 += gout[10] * dm_il_30;
                    vjk_20 += gout[5] * dm_il_30;
                    vjk_22 += gout[20] * dm_il_30;
                    double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                    vjk_01 += gout[8] * dm_il_50;
                    vjk_10 += gout[3] * dm_il_50;
                    vjk_12 += gout[18] * dm_il_50;
                    vjk_21 += gout[13] * dm_il_50;
                    double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                    vjk_00 += gout[1] * dm_il_70;
                    vjk_02 += gout[16] * dm_il_70;
                    vjk_11 += gout[11] * dm_il_70;
                    vjk_20 += gout[6] * dm_il_70;
                    vjk_22 += gout[21] * dm_il_70;
                    double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                    vjk_01 += gout[9] * dm_il_90;
                    vjk_10 += gout[4] * dm_il_90;
                    vjk_12 += gout[19] * dm_il_90;
                    vjk_21 += gout[14] * dm_il_90;
                    break; }
                    }
                    atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                    atomicAdd(vk+(j0+0)*nao+(k0+1), vjk_01);
                    atomicAdd(vk+(j0+0)*nao+(k0+2), vjk_02);
                    atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                    atomicAdd(vk+(j0+1)*nao+(k0+1), vjk_11);
                    atomicAdd(vk+(j0+1)*nao+(k0+2), vjk_12);
                    atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                    atomicAdd(vk+(j0+2)*nao+(k0+1), vjk_21);
                    atomicAdd(vk+(j0+2)*nao+(k0+2), vjk_22);
                    }
                }
            }
        }
    }
}
}

__global__ static
void rys_k_3200(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                uint32_t *pool, int *head
                #ifdef USE_SYCL
                , sycl::nd_item<2> &item, double *shared_memory
                #endif
                )
{
    #ifdef USE_SYCL
    int sq_id = item.get_local_id(1);
    int nsq_per_block = item.get_local_range(1);
    uint32_t *bas_kl_idx = pool + item.get_group(1) * QUEUE_DEPTH;

    auto thread_block = item.get_group();
    int &ntasks = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_ij = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &pair_kl0 = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &ish = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &jsh = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    double (&ri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    double (&rjri)[3] = *sycl::ext::oneapi::group_local_memory_for_overwrite<double[3]>(thread_block);
    int &expi = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    int &expj = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(thread_block);
    #else
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    uint32_t *bas_kl_idx = pool + blockIdx.x * QUEUE_DEPTH;

    extern __shared__ double shared_memory[];

    __shared__ int ntasks, pair_ij, pair_kl0;
    __shared__ int ish;
    __shared__ int jsh;
    __shared__ double ri[3];
    __shared__ double rjri[3];
    __shared__ int expi;
    __shared__ int expj;
    #endif

    double *rw = shared_memory + sq_id;
    double *cicj_cache = shared_memory + nsq_per_block * bounds.nroots*2;

while (1) {
    if (sq_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    if (pair_ij >= bounds.npairs_ij) {
        break;
    }

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    uint32_t bas_ij = bounds.pair_ij_mapping[pair_ij];
    if (sq_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij % nbas;
        expi = bas[ish*BAS_SLOTS+PTR_EXP];
        expj = bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    __syncthreads();
    if (sq_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[sq_id] = env[ri_ptr+sq_id];
        rjri[sq_id] = env[rj_ptr+sq_id] - ri[sq_id];
    }
    __syncthreads();
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double xjxi = rjri[0];
    double yjyi = rjri[1];
    double zjzi = rjri[2];
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    for (int ij = sq_id; ij < iprim*jprim; ij += nsq_per_block) {
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

    if (sq_id == 0) {
        pair_kl0 = 0;
    }
    __syncthreads();
    while (pair_kl0 < bounds.npairs_kl) {
        if (kmat.lr_factor != 0) {
            _fill_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                           q_cond_ij, q_cond_kl, dm_penalty, envs, bounds, shared_memory);
        } else {
            _fill_sr_vk_tasks(ntasks, pair_kl0, bas_kl_idx, pair_ij, ish, jsh,
                              q_cond_ij, q_cond_kl, dm_penalty,
                              s_cond_ij, s_cond_kl, diffuse_exps, envs, bounds, shared_memory);
        }
        if (ntasks == 0) {
            continue;
        }
        for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += nsq_per_block) {
            int iprim = bounds.iprim;
            int jprim = bounds.jprim;
            int kprim = bounds.kprim;
            int lprim = bounds.lprim;
            uint32_t bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int expl = bas[lsh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int cl = bas[lsh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            int rl = bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = env[rl+0] - env[rk+0];
            double ylyk = env[rl+1] - env[rk+1];
            double zlzk = env[rl+2] - env[rk+2];
            double gout[60];
#pragma unroll
            for (int n = 0; n < 60; ++n) { gout[n] = 0; }
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                int kp = klp / lprim;
                int lp = klp % lprim;
                double ak = env[expk+kp];
                double al = env[expl+lp];
                double akl = ak + al;
                double al_akl = al / akl;
                double theta_kl = ak * al_akl;
                double Kcd = exp(-theta_kl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                double fac_sym = PI_FAC;
                if (task_id < ntasks) {
                    if (ksh == lsh) fac_sym *= .5;
                    if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
                } else {
                    fac_sym = 0;
                }
                double ckcl = fac_sym * env[ck+kp] * env[cl+lp] * Kcd;
                for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                    __syncthreads();
                    int ip = ijp / jprim;
                    int jp = ijp % jprim;
                    double ai = env[expi+ip];
                    double aj = env[expj+jp];
                    double aij = ai + aj;
                    double aj_aij = aj / aij;
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
                    int nroots = bounds.nroots;
                    rys_roots_for_k(nroots, theta, rr, rw, kmat.omega, kmat.lr_factor, kmat.sr_factor);
                    if (task_id >= ntasks) {
                        continue;
                    }
                    for (int irys = 0; irys < nroots; ++irys) {
                        double wt = rw[(2*irys+1)*nsq_per_block];
                        double rt = rw[ 2*irys   *nsq_per_block];
                        double rt_aa = rt / (aij + akl);
                        double xjxi = rjri[0];
                        double rt_aij = rt_aa * akl;
                        double b10 = .5/aij * (1 - rt_aij);
                        double c0x = xjxi*aj_aij - xpq*rt_aij;
                        double trr_10x = c0x * 1;
                        double trr_20x = c0x * trr_10x + 1*b10 * 1;
                        double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                        double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                        double trr_50x = c0x * trr_40x + 4*b10 * trr_30x;
                        double hrr_4100x = trr_50x - xjxi * trr_40x;
                        double hrr_3100x = trr_40x - xjxi * trr_30x;
                        double hrr_3200x = hrr_4100x - xjxi * hrr_3100x;
                        gout[0] += hrr_3200x * fac * wt;
                        double hrr_2100x = trr_30x - xjxi * trr_20x;
                        double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                        double yjyi = rjri[1];
                        double c0y = yjyi*aj_aij - ypq*rt_aij;
                        double trr_10y = c0y * fac;
                        gout[1] += hrr_2200x * trr_10y * wt;
                        double zjzi = rjri[2];
                        double c0z = zjzi*aj_aij - zpq*rt_aij;
                        double trr_10z = c0z * wt;
                        gout[2] += hrr_2200x * fac * trr_10z;
                        double hrr_1100x = trr_20x - xjxi * trr_10x;
                        double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                        double trr_20y = c0y * trr_10y + 1*b10 * fac;
                        gout[3] += hrr_1200x * trr_20y * wt;
                        gout[4] += hrr_1200x * trr_10y * trr_10z;
                        double trr_20z = c0z * trr_10z + 1*b10 * wt;
                        gout[5] += hrr_1200x * fac * trr_20z;
                        double hrr_0100x = trr_10x - xjxi * 1;
                        double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                        double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                        gout[6] += hrr_0200x * trr_30y * wt;
                        gout[7] += hrr_0200x * trr_20y * trr_10z;
                        gout[8] += hrr_0200x * trr_10y * trr_20z;
                        double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                        gout[9] += hrr_0200x * fac * trr_30z;
                        double hrr_0100y = trr_10y - yjyi * fac;
                        gout[10] += hrr_3100x * hrr_0100y * wt;
                        double hrr_1100y = trr_20y - yjyi * trr_10y;
                        gout[11] += hrr_2100x * hrr_1100y * wt;
                        gout[12] += hrr_2100x * hrr_0100y * trr_10z;
                        double hrr_2100y = trr_30y - yjyi * trr_20y;
                        gout[13] += hrr_1100x * hrr_2100y * wt;
                        gout[14] += hrr_1100x * hrr_1100y * trr_10z;
                        gout[15] += hrr_1100x * hrr_0100y * trr_20z;
                        double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                        double hrr_3100y = trr_40y - yjyi * trr_30y;
                        gout[16] += hrr_0100x * hrr_3100y * wt;
                        gout[17] += hrr_0100x * hrr_2100y * trr_10z;
                        gout[18] += hrr_0100x * hrr_1100y * trr_20z;
                        gout[19] += hrr_0100x * hrr_0100y * trr_30z;
                        double hrr_0100z = trr_10z - zjzi * wt;
                        gout[20] += hrr_3100x * fac * hrr_0100z;
                        gout[21] += hrr_2100x * trr_10y * hrr_0100z;
                        double hrr_1100z = trr_20z - zjzi * trr_10z;
                        gout[22] += hrr_2100x * fac * hrr_1100z;
                        gout[23] += hrr_1100x * trr_20y * hrr_0100z;
                        gout[24] += hrr_1100x * trr_10y * hrr_1100z;
                        double hrr_2100z = trr_30z - zjzi * trr_20z;
                        gout[25] += hrr_1100x * fac * hrr_2100z;
                        gout[26] += hrr_0100x * trr_30y * hrr_0100z;
                        gout[27] += hrr_0100x * trr_20y * hrr_1100z;
                        gout[28] += hrr_0100x * trr_10y * hrr_2100z;
                        double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                        double hrr_3100z = trr_40z - zjzi * trr_30z;
                        gout[29] += hrr_0100x * fac * hrr_3100z;
                        double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                        gout[30] += trr_30x * hrr_0200y * wt;
                        double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                        gout[31] += trr_20x * hrr_1200y * wt;
                        gout[32] += trr_20x * hrr_0200y * trr_10z;
                        double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                        gout[33] += trr_10x * hrr_2200y * wt;
                        gout[34] += trr_10x * hrr_1200y * trr_10z;
                        gout[35] += trr_10x * hrr_0200y * trr_20z;
                        double trr_50y = c0y * trr_40y + 4*b10 * trr_30y;
                        double hrr_4100y = trr_50y - yjyi * trr_40y;
                        double hrr_3200y = hrr_4100y - yjyi * hrr_3100y;
                        gout[36] += 1 * hrr_3200y * wt;
                        gout[37] += 1 * hrr_2200y * trr_10z;
                        gout[38] += 1 * hrr_1200y * trr_20z;
                        gout[39] += 1 * hrr_0200y * trr_30z;
                        gout[40] += trr_30x * hrr_0100y * hrr_0100z;
                        gout[41] += trr_20x * hrr_1100y * hrr_0100z;
                        gout[42] += trr_20x * hrr_0100y * hrr_1100z;
                        gout[43] += trr_10x * hrr_2100y * hrr_0100z;
                        gout[44] += trr_10x * hrr_1100y * hrr_1100z;
                        gout[45] += trr_10x * hrr_0100y * hrr_2100z;
                        gout[46] += 1 * hrr_3100y * hrr_0100z;
                        gout[47] += 1 * hrr_2100y * hrr_1100z;
                        gout[48] += 1 * hrr_1100y * hrr_2100z;
                        gout[49] += 1 * hrr_0100y * hrr_3100z;
                        double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                        gout[50] += trr_30x * fac * hrr_0200z;
                        gout[51] += trr_20x * trr_10y * hrr_0200z;
                        double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                        gout[52] += trr_20x * fac * hrr_1200z;
                        gout[53] += trr_10x * trr_20y * hrr_0200z;
                        gout[54] += trr_10x * trr_10y * hrr_1200z;
                        double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                        gout[55] += trr_10x * fac * hrr_2200z;
                        gout[56] += 1 * trr_30y * hrr_0200z;
                        gout[57] += 1 * trr_20y * hrr_1200z;
                        gout[58] += 1 * trr_10y * hrr_2200z;
                        double trr_50z = c0z * trr_40z + 4*b10 * trr_30z;
                        double hrr_4100z = trr_50z - zjzi * trr_40z;
                        double hrr_3200z = hrr_4100z - zjzi * hrr_3100z;
                        gout[59] += 1 * fac * hrr_3200z;
                    }
                }
            }
            if (task_id < ntasks) {
                int *ao_loc = envs.ao_loc;
                int nao = ao_loc[nbas];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int k0 = ao_loc[ksh];
                int l0 = ao_loc[lsh];
                size_t nao2 = (size_t)nao * nao;
                for (int i_dm = 0; i_dm < kmat.n_dm; ++i_dm) {
                    double *dm = kmat.dm + i_dm * nao2;
                    double *vk = kmat.vk + i_dm * nao2;
                    double dm_jk_00 = dm[(j0+0)*nao+(k0+0)];
                    double dm_jk_10 = dm[(j0+1)*nao+(k0+0)];
                    double dm_jk_20 = dm[(j0+2)*nao+(k0+0)];
                    double dm_jk_30 = dm[(j0+3)*nao+(k0+0)];
                    double dm_jk_40 = dm[(j0+4)*nao+(k0+0)];
                    double dm_jk_50 = dm[(j0+5)*nao+(k0+0)];
                    double vil_00 = gout[0]*dm_jk_00 + gout[10]*dm_jk_10 + gout[20]*dm_jk_20 + gout[30]*dm_jk_30 + gout[40]*dm_jk_40 + gout[50]*dm_jk_50;
                    atomicAdd(vk+(i0+0)*nao+(l0+0), vil_00);
                    double vil_10 = gout[1]*dm_jk_00 + gout[11]*dm_jk_10 + gout[21]*dm_jk_20 + gout[31]*dm_jk_30 + gout[41]*dm_jk_40 + gout[51]*dm_jk_50;
                    atomicAdd(vk+(i0+1)*nao+(l0+0), vil_10);
                    double vil_20 = gout[2]*dm_jk_00 + gout[12]*dm_jk_10 + gout[22]*dm_jk_20 + gout[32]*dm_jk_30 + gout[42]*dm_jk_40 + gout[52]*dm_jk_50;
                    atomicAdd(vk+(i0+2)*nao+(l0+0), vil_20);
                    double vil_30 = gout[3]*dm_jk_00 + gout[13]*dm_jk_10 + gout[23]*dm_jk_20 + gout[33]*dm_jk_30 + gout[43]*dm_jk_40 + gout[53]*dm_jk_50;
                    atomicAdd(vk+(i0+3)*nao+(l0+0), vil_30);
                    double vil_40 = gout[4]*dm_jk_00 + gout[14]*dm_jk_10 + gout[24]*dm_jk_20 + gout[34]*dm_jk_30 + gout[44]*dm_jk_40 + gout[54]*dm_jk_50;
                    atomicAdd(vk+(i0+4)*nao+(l0+0), vil_40);
                    double vil_50 = gout[5]*dm_jk_00 + gout[15]*dm_jk_10 + gout[25]*dm_jk_20 + gout[35]*dm_jk_30 + gout[45]*dm_jk_40 + gout[55]*dm_jk_50;
                    atomicAdd(vk+(i0+5)*nao+(l0+0), vil_50);
                    double vil_60 = gout[6]*dm_jk_00 + gout[16]*dm_jk_10 + gout[26]*dm_jk_20 + gout[36]*dm_jk_30 + gout[46]*dm_jk_40 + gout[56]*dm_jk_50;
                    atomicAdd(vk+(i0+6)*nao+(l0+0), vil_60);
                    double vil_70 = gout[7]*dm_jk_00 + gout[17]*dm_jk_10 + gout[27]*dm_jk_20 + gout[37]*dm_jk_30 + gout[47]*dm_jk_40 + gout[57]*dm_jk_50;
                    atomicAdd(vk+(i0+7)*nao+(l0+0), vil_70);
                    double vil_80 = gout[8]*dm_jk_00 + gout[18]*dm_jk_10 + gout[28]*dm_jk_20 + gout[38]*dm_jk_30 + gout[48]*dm_jk_40 + gout[58]*dm_jk_50;
                    atomicAdd(vk+(i0+8)*nao+(l0+0), vil_80);
                    double vil_90 = gout[9]*dm_jk_00 + gout[19]*dm_jk_10 + gout[29]*dm_jk_20 + gout[39]*dm_jk_30 + gout[49]*dm_jk_40 + gout[59]*dm_jk_50;
                    atomicAdd(vk+(i0+9)*nao+(l0+0), vil_90);
                    double dm_jl_00 = dm[(j0+0)*nao+(l0+0)];
                    double dm_jl_10 = dm[(j0+1)*nao+(l0+0)];
                    double dm_jl_20 = dm[(j0+2)*nao+(l0+0)];
                    double dm_jl_30 = dm[(j0+3)*nao+(l0+0)];
                    double dm_jl_40 = dm[(j0+4)*nao+(l0+0)];
                    double dm_jl_50 = dm[(j0+5)*nao+(l0+0)];
                    double vik_00 = gout[0]*dm_jl_00 + gout[10]*dm_jl_10 + gout[20]*dm_jl_20 + gout[30]*dm_jl_30 + gout[40]*dm_jl_40 + gout[50]*dm_jl_50;
                    atomicAdd(vk+(i0+0)*nao+(k0+0), vik_00);
                    double vik_10 = gout[1]*dm_jl_00 + gout[11]*dm_jl_10 + gout[21]*dm_jl_20 + gout[31]*dm_jl_30 + gout[41]*dm_jl_40 + gout[51]*dm_jl_50;
                    atomicAdd(vk+(i0+1)*nao+(k0+0), vik_10);
                    double vik_20 = gout[2]*dm_jl_00 + gout[12]*dm_jl_10 + gout[22]*dm_jl_20 + gout[32]*dm_jl_30 + gout[42]*dm_jl_40 + gout[52]*dm_jl_50;
                    atomicAdd(vk+(i0+2)*nao+(k0+0), vik_20);
                    double vik_30 = gout[3]*dm_jl_00 + gout[13]*dm_jl_10 + gout[23]*dm_jl_20 + gout[33]*dm_jl_30 + gout[43]*dm_jl_40 + gout[53]*dm_jl_50;
                    atomicAdd(vk+(i0+3)*nao+(k0+0), vik_30);
                    double vik_40 = gout[4]*dm_jl_00 + gout[14]*dm_jl_10 + gout[24]*dm_jl_20 + gout[34]*dm_jl_30 + gout[44]*dm_jl_40 + gout[54]*dm_jl_50;
                    atomicAdd(vk+(i0+4)*nao+(k0+0), vik_40);
                    double vik_50 = gout[5]*dm_jl_00 + gout[15]*dm_jl_10 + gout[25]*dm_jl_20 + gout[35]*dm_jl_30 + gout[45]*dm_jl_40 + gout[55]*dm_jl_50;
                    atomicAdd(vk+(i0+5)*nao+(k0+0), vik_50);
                    double vik_60 = gout[6]*dm_jl_00 + gout[16]*dm_jl_10 + gout[26]*dm_jl_20 + gout[36]*dm_jl_30 + gout[46]*dm_jl_40 + gout[56]*dm_jl_50;
                    atomicAdd(vk+(i0+6)*nao+(k0+0), vik_60);
                    double vik_70 = gout[7]*dm_jl_00 + gout[17]*dm_jl_10 + gout[27]*dm_jl_20 + gout[37]*dm_jl_30 + gout[47]*dm_jl_40 + gout[57]*dm_jl_50;
                    atomicAdd(vk+(i0+7)*nao+(k0+0), vik_70);
                    double vik_80 = gout[8]*dm_jl_00 + gout[18]*dm_jl_10 + gout[28]*dm_jl_20 + gout[38]*dm_jl_30 + gout[48]*dm_jl_40 + gout[58]*dm_jl_50;
                    atomicAdd(vk+(i0+8)*nao+(k0+0), vik_80);
                    double vik_90 = gout[9]*dm_jl_00 + gout[19]*dm_jl_10 + gout[29]*dm_jl_20 + gout[39]*dm_jl_30 + gout[49]*dm_jl_40 + gout[59]*dm_jl_50;
                    atomicAdd(vk+(i0+9)*nao+(k0+0), vik_90);
                    if (ish != jsh) {
                        double vjl_00 = 0;
                        double vjl_10 = 0;
                        double vjl_20 = 0;
                        double vjl_30 = 0;
                        double vjl_40 = 0;
                        double vjl_50 = 0;
                        double dm_ik_00 = dm[(i0+0)*nao+(k0+0)];
                        vjl_00 += gout[0] * dm_ik_00;
                        vjl_10 += gout[10] * dm_ik_00;
                        vjl_20 += gout[20] * dm_ik_00;
                        vjl_30 += gout[30] * dm_ik_00;
                        vjl_40 += gout[40] * dm_ik_00;
                        vjl_50 += gout[50] * dm_ik_00;
                        double dm_ik_10 = dm[(i0+1)*nao+(k0+0)];
                        vjl_00 += gout[1] * dm_ik_10;
                        vjl_10 += gout[11] * dm_ik_10;
                        vjl_20 += gout[21] * dm_ik_10;
                        vjl_30 += gout[31] * dm_ik_10;
                        vjl_40 += gout[41] * dm_ik_10;
                        vjl_50 += gout[51] * dm_ik_10;
                        double dm_ik_20 = dm[(i0+2)*nao+(k0+0)];
                        vjl_00 += gout[2] * dm_ik_20;
                        vjl_10 += gout[12] * dm_ik_20;
                        vjl_20 += gout[22] * dm_ik_20;
                        vjl_30 += gout[32] * dm_ik_20;
                        vjl_40 += gout[42] * dm_ik_20;
                        vjl_50 += gout[52] * dm_ik_20;
                        double dm_ik_30 = dm[(i0+3)*nao+(k0+0)];
                        vjl_00 += gout[3] * dm_ik_30;
                        vjl_10 += gout[13] * dm_ik_30;
                        vjl_20 += gout[23] * dm_ik_30;
                        vjl_30 += gout[33] * dm_ik_30;
                        vjl_40 += gout[43] * dm_ik_30;
                        vjl_50 += gout[53] * dm_ik_30;
                        double dm_ik_40 = dm[(i0+4)*nao+(k0+0)];
                        vjl_00 += gout[4] * dm_ik_40;
                        vjl_10 += gout[14] * dm_ik_40;
                        vjl_20 += gout[24] * dm_ik_40;
                        vjl_30 += gout[34] * dm_ik_40;
                        vjl_40 += gout[44] * dm_ik_40;
                        vjl_50 += gout[54] * dm_ik_40;
                        double dm_ik_50 = dm[(i0+5)*nao+(k0+0)];
                        vjl_00 += gout[5] * dm_ik_50;
                        vjl_10 += gout[15] * dm_ik_50;
                        vjl_20 += gout[25] * dm_ik_50;
                        vjl_30 += gout[35] * dm_ik_50;
                        vjl_40 += gout[45] * dm_ik_50;
                        vjl_50 += gout[55] * dm_ik_50;
                        double dm_ik_60 = dm[(i0+6)*nao+(k0+0)];
                        vjl_00 += gout[6] * dm_ik_60;
                        vjl_10 += gout[16] * dm_ik_60;
                        vjl_20 += gout[26] * dm_ik_60;
                        vjl_30 += gout[36] * dm_ik_60;
                        vjl_40 += gout[46] * dm_ik_60;
                        vjl_50 += gout[56] * dm_ik_60;
                        double dm_ik_70 = dm[(i0+7)*nao+(k0+0)];
                        vjl_00 += gout[7] * dm_ik_70;
                        vjl_10 += gout[17] * dm_ik_70;
                        vjl_20 += gout[27] * dm_ik_70;
                        vjl_30 += gout[37] * dm_ik_70;
                        vjl_40 += gout[47] * dm_ik_70;
                        vjl_50 += gout[57] * dm_ik_70;
                        double dm_ik_80 = dm[(i0+8)*nao+(k0+0)];
                        vjl_00 += gout[8] * dm_ik_80;
                        vjl_10 += gout[18] * dm_ik_80;
                        vjl_20 += gout[28] * dm_ik_80;
                        vjl_30 += gout[38] * dm_ik_80;
                        vjl_40 += gout[48] * dm_ik_80;
                        vjl_50 += gout[58] * dm_ik_80;
                        double dm_ik_90 = dm[(i0+9)*nao+(k0+0)];
                        vjl_00 += gout[9] * dm_ik_90;
                        vjl_10 += gout[19] * dm_ik_90;
                        vjl_20 += gout[29] * dm_ik_90;
                        vjl_30 += gout[39] * dm_ik_90;
                        vjl_40 += gout[49] * dm_ik_90;
                        vjl_50 += gout[59] * dm_ik_90;
                        atomicAdd(vk+(j0+0)*nao+(l0+0), vjl_00);
                        atomicAdd(vk+(j0+1)*nao+(l0+0), vjl_10);
                        atomicAdd(vk+(j0+2)*nao+(l0+0), vjl_20);
                        atomicAdd(vk+(j0+3)*nao+(l0+0), vjl_30);
                        atomicAdd(vk+(j0+4)*nao+(l0+0), vjl_40);
                        atomicAdd(vk+(j0+5)*nao+(l0+0), vjl_50);
                        double vjk_00 = 0;
                        double vjk_10 = 0;
                        double vjk_20 = 0;
                        double vjk_30 = 0;
                        double vjk_40 = 0;
                        double vjk_50 = 0;
                        double dm_il_00 = dm[(i0+0)*nao+(l0+0)];
                        vjk_00 += gout[0] * dm_il_00;
                        vjk_10 += gout[10] * dm_il_00;
                        vjk_20 += gout[20] * dm_il_00;
                        vjk_30 += gout[30] * dm_il_00;
                        vjk_40 += gout[40] * dm_il_00;
                        vjk_50 += gout[50] * dm_il_00;
                        double dm_il_10 = dm[(i0+1)*nao+(l0+0)];
                        vjk_00 += gout[1] * dm_il_10;
                        vjk_10 += gout[11] * dm_il_10;
                        vjk_20 += gout[21] * dm_il_10;
                        vjk_30 += gout[31] * dm_il_10;
                        vjk_40 += gout[41] * dm_il_10;
                        vjk_50 += gout[51] * dm_il_10;
                        double dm_il_20 = dm[(i0+2)*nao+(l0+0)];
                        vjk_00 += gout[2] * dm_il_20;
                        vjk_10 += gout[12] * dm_il_20;
                        vjk_20 += gout[22] * dm_il_20;
                        vjk_30 += gout[32] * dm_il_20;
                        vjk_40 += gout[42] * dm_il_20;
                        vjk_50 += gout[52] * dm_il_20;
                        double dm_il_30 = dm[(i0+3)*nao+(l0+0)];
                        vjk_00 += gout[3] * dm_il_30;
                        vjk_10 += gout[13] * dm_il_30;
                        vjk_20 += gout[23] * dm_il_30;
                        vjk_30 += gout[33] * dm_il_30;
                        vjk_40 += gout[43] * dm_il_30;
                        vjk_50 += gout[53] * dm_il_30;
                        double dm_il_40 = dm[(i0+4)*nao+(l0+0)];
                        vjk_00 += gout[4] * dm_il_40;
                        vjk_10 += gout[14] * dm_il_40;
                        vjk_20 += gout[24] * dm_il_40;
                        vjk_30 += gout[34] * dm_il_40;
                        vjk_40 += gout[44] * dm_il_40;
                        vjk_50 += gout[54] * dm_il_40;
                        double dm_il_50 = dm[(i0+5)*nao+(l0+0)];
                        vjk_00 += gout[5] * dm_il_50;
                        vjk_10 += gout[15] * dm_il_50;
                        vjk_20 += gout[25] * dm_il_50;
                        vjk_30 += gout[35] * dm_il_50;
                        vjk_40 += gout[45] * dm_il_50;
                        vjk_50 += gout[55] * dm_il_50;
                        double dm_il_60 = dm[(i0+6)*nao+(l0+0)];
                        vjk_00 += gout[6] * dm_il_60;
                        vjk_10 += gout[16] * dm_il_60;
                        vjk_20 += gout[26] * dm_il_60;
                        vjk_30 += gout[36] * dm_il_60;
                        vjk_40 += gout[46] * dm_il_60;
                        vjk_50 += gout[56] * dm_il_60;
                        double dm_il_70 = dm[(i0+7)*nao+(l0+0)];
                        vjk_00 += gout[7] * dm_il_70;
                        vjk_10 += gout[17] * dm_il_70;
                        vjk_20 += gout[27] * dm_il_70;
                        vjk_30 += gout[37] * dm_il_70;
                        vjk_40 += gout[47] * dm_il_70;
                        vjk_50 += gout[57] * dm_il_70;
                        double dm_il_80 = dm[(i0+8)*nao+(l0+0)];
                        vjk_00 += gout[8] * dm_il_80;
                        vjk_10 += gout[18] * dm_il_80;
                        vjk_20 += gout[28] * dm_il_80;
                        vjk_30 += gout[38] * dm_il_80;
                        vjk_40 += gout[48] * dm_il_80;
                        vjk_50 += gout[58] * dm_il_80;
                        double dm_il_90 = dm[(i0+9)*nao+(l0+0)];
                        vjk_00 += gout[9] * dm_il_90;
                        vjk_10 += gout[19] * dm_il_90;
                        vjk_20 += gout[29] * dm_il_90;
                        vjk_30 += gout[39] * dm_il_90;
                        vjk_40 += gout[49] * dm_il_90;
                        vjk_50 += gout[59] * dm_il_90;
                        atomicAdd(vk+(j0+0)*nao+(k0+0), vjk_00);
                        atomicAdd(vk+(j0+1)*nao+(k0+0), vjk_10);
                        atomicAdd(vk+(j0+2)*nao+(k0+0), vjk_20);
                        atomicAdd(vk+(j0+3)*nao+(k0+0), vjk_30);
                        atomicAdd(vk+(j0+4)*nao+(k0+0), vjk_40);
                        atomicAdd(vk+(j0+5)*nao+(k0+0), vjk_50);
                    }
                }
            }
        }
    }
}
}

int rys_k_unrolled(RysIntEnvVars *envs, JKMatrix *kmat, BoundsInfo *bounds,
                   float *q_cond_ij, float *q_cond_kl, float dm_penalty,
                   float *s_cond_ij, float *s_cond_kl, float *diffuse_exps,
                   uint32_t *pool, int *head, int workers)
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
    case 156: // (1, 1, 1, 1)
        break;
    case 250: // (2, 0, 0, 0)
        adjust_threads(rys_k_2000, nsq_per_block);
        break;
    case 255: // (2, 0, 1, 0)
        adjust_threads(rys_k_2010, nsq_per_block);
        break;
    case 256: // (2, 0, 1, 1)
        break;
    case 260: // (2, 0, 2, 0)
        break;
    case 261: // (2, 0, 2, 1)
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 275: // (2, 1, 0, 0)
        adjust_threads(rys_k_2100, nsq_per_block);
        break;
    case 280: // (2, 1, 1, 0)
        break;
    case 281: // (2, 1, 1, 1)
        nsq_per_block = 32;
        gout_stride = 8;
        break;
    case 285: // (2, 1, 2, 0)
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 300: // (2, 2, 0, 0)
        break;
    case 305: // (2, 2, 1, 0)
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 375: // (3, 0, 0, 0)
        adjust_threads(rys_k_3000, nsq_per_block);
        break;
    case 380: // (3, 0, 1, 0)
        break;
    case 381: // (3, 0, 1, 1)
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 385: // (3, 0, 2, 0)
        break;
    case 400: // (3, 1, 0, 0)
        break;
    case 405: // (3, 1, 1, 0)
        nsq_per_block = 64;
        gout_stride = 4;
        break;
    case 425: // (3, 2, 0, 0)
        break;
    }

    int iprim = bounds->iprim;
    int jprim = bounds->jprim;
    int buflen = nroots*2 * nsq_per_block + iprim*jprim;

    #ifdef USE_SYCL
    auto dev_envs = *envs;
    auto dev_kmat = *kmat;
    auto dev_bounds = *bounds;

    sycl::queue& stream = *sycl_get_queue();
    sycl::range<2> blocks(1, workers);
    sycl::range<2> threads(gout_stride, nsq_per_block);
    switch (ijkl) {
    case 0:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_0000_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_0000(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 125:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_1000_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_1000(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 130:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_1010_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_1010(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 131:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_1011_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_1011(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 150:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_1100_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_1100(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 155:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_1110_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_1110(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 156:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_1111_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_1111(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 250:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2000_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2000(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 255:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2010_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2010(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 256:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2011_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2011(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 260:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2020_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2020(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 261:
      buflen += 4032;
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2021_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2021(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 275:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2100_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2100(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 280:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2110_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2110(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 281:
      buflen += 2592;
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2111_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2111(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 285:
      buflen += 4032;
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2120_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2120(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 300:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2200_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2200(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 305:
      buflen += 4032;
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_2210_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_2210(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 375:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3000_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3000(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 380:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3010_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3010(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 381:
      buflen += 3648;
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3011_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3011(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 385:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3020_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3020(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 400:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3100_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3100(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 405:
      buflen += 3648;
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3110_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3110(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    case 425:
      stream.submit([&](sycl::handler &cgh) { sycl::local_accessor<double, 1> local_acc(sycl::range<1>(buflen), cgh); cgh.parallel_for<class rys_k_3200_sycl>(sycl::nd_range<2>(blocks * threads, threads), [=](auto item) { rys_k_3200(dev_envs, dev_kmat, dev_bounds, pool, head, item, GPU4PYSCF_IMPL_SYCL_GET_MULTI_PTR(local_acc)); }); }); break;
    default: return 0;
    }
    #else
    dim3 threads(nsq_per_block, gout_stride);
    switch (ijkl) {
    case 0: // (0, 0, 0, 0)
        rys_k_0000<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 125: // (1, 0, 0, 0)
        rys_k_1000<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 130: // (1, 0, 1, 0)
        rys_k_1010<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 131: // (1, 0, 1, 1)
        rys_k_1011<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 150: // (1, 1, 0, 0)
        rys_k_1100<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 155: // (1, 1, 1, 0)
        rys_k_1110<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 156: // (1, 1, 1, 1)
        rys_k_1111<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 250: // (2, 0, 0, 0)
        rys_k_2000<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 255: // (2, 0, 1, 0)
        rys_k_2010<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 256: // (2, 0, 1, 1)
        rys_k_2011<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 260: // (2, 0, 2, 0)
        rys_k_2020<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 261: // (2, 0, 2, 1)
        buflen = 4736 + iprim * jprim;
        rys_k_2021<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 275: // (2, 1, 0, 0)
        rys_k_2100<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 280: // (2, 1, 1, 0)
        rys_k_2110<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 281: // (2, 1, 1, 1)
        buflen = 2944 + iprim * jprim;
        rys_k_2111<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 285: // (2, 1, 2, 0)
        buflen = 4736 + iprim * jprim;
        rys_k_2120<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 300: // (2, 2, 0, 0)
        rys_k_2200<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 305: // (2, 2, 1, 0)
        buflen = 4736 + iprim * jprim;
        rys_k_2210<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 375: // (3, 0, 0, 0)
        rys_k_3000<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 380: // (3, 0, 1, 0)
        rys_k_3010<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 381: // (3, 0, 1, 1)
        buflen = 4352 + iprim * jprim;
        rys_k_3011<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 385: // (3, 0, 2, 0)
        rys_k_3020<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 400: // (3, 1, 0, 0)
        rys_k_3100<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 405: // (3, 1, 1, 0)
        buflen = 4352 + iprim * jprim;
        rys_k_3110<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    case 425: // (3, 2, 0, 0)
        rys_k_3200<<<workers, threads, buflen*sizeof(double)>>>(
            *envs, *kmat, *bounds, q_cond_ij, q_cond_kl, dm_penalty, s_cond_ij, s_cond_kl, diffuse_exps, pool, head); break;
    default: return 0;
    }
    #endif
    return 1;
}
