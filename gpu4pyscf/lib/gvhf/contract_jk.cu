/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include "gint/gint.h"
#include "gint/cint2e.cuh"
#include "gint/reduction.cu"
#include "gvhf.h"

template <int NROOTS, int GSIZE> __device__
static void GINTkernel_direct_getjk(GINTEnvVars envs, JKMatrix jk, double* __restrict__ g,
                      int ish, int jsh, int ksh, int lsh)
{
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ];
    int i1 = ao_loc[ish+1];
    int j0 = ao_loc[jsh  ];
    int j1 = ao_loc[jsh+1];
    int k0 = ao_loc[ksh  ];
    int k1 = ao_loc[ksh+1];
    int l0 = ao_loc[lsh  ];
    int l1 = ao_loc[lsh+1];
    int nfi = i1 - i0;
    int nfj = j1 - j0;
    int nfij = nfi * nfj;

    int nao = jk.nao;
    int i, j, k, l, n, i_dm;

    int n_dm = jk.n_dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double* __restrict__ dm = jk.dm;

    int nf = envs.nf;
    int16_t *idx = c_idx4c;
    if (nf > NFffff){
        idx = envs.idx;
    }
    int16_t *idy = idx + nf;
    int16_t *idz = idx + nf * 2;
    
    if (vk == NULL) {
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            int ngout = 0;
            for (l = l0; l < l1; ++l) {
                for (k = k0; k < k1; ++k) {
                    double v_kl = 0;
                    double d_kl = dm[k+nao*l];
                    for (n = 0, j = j0; j < j1; ++j) {
                        for (i = i0; i < i1; ++i, ++n) {
                            int ng = n + ngout;
                            int ix = idx[ng];
                            int iy = idy[ng];
                            int iz = idz[ng];
                            double s = 0.0;
#pragma unroll
                            for (int r = 0; r < NROOTS; r++){
                                s += g[ix+r] * g[iy+r] * g[iz+r];
                            }
                            double v_ij  = s * d_kl;   
                            atomicAdd(vj+i+nao*j, v_ij);
                            v_kl += s * dm[i+j*nao];
                        }
                    }
                    atomicAdd(vj+k+nao*l, v_kl);
                    ngout += nfij;
                }
            }
            dm += nao * nao;
            vj += nao * nao;
        }
        return;
    }
    
    if (vj == NULL){
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            int ngout = 0;
            for (l = l0; l < l1; ++l) {
                for (k = k0; k < k1; ++k) {
                    double gout[GPU_AO_NF * GPU_AO_NF];
                    for (n = 0, j = j0; j < j1; ++j) {
                        int jp = j - j0;
                        for (i = i0; i < i1; ++i, ++n) {
                            int ip = i - i0;
                            int ng = n + ngout;
                            int ix = idx[ng];
                            int iy = idy[ng];
                            int iz = idz[ng];
                            double s = 0.0;
                            for (int r = 0; r < NROOTS; r++){
                                s += g[ix+r] * g[iy+r] * g[iz+r];
                            }
                            gout[ip + GPU_AO_NF * jp] = s;
                        }
                    }

                    double v_ik[GPU_AO_NF];
                    double d_ik[GPU_AO_NF];
                    double v_il[GPU_AO_NF];
                    double d_il[GPU_AO_NF];
                    for (i = 0; i < i1-i0; ++i){ 
                        v_il[i] = 0.0; 
                        d_il[i] = dm[i+i0+l*nao];
                        v_ik[i] = 0.0; 
                        d_ik[i] = dm[i+i0+k*nao];
                    }

                    for (j = j0; j < j1; ++j){
                        int jp = j - j0;
                        double v_jk = 0.0;
                        double v_jl = 0.0;
                        double d_jk = dm[j+nao*k];
                        double d_jl = dm[j+nao*l];
                        for (i = i0; i < i1; ++i){
                            int ip = i - i0;
                            double s = gout[ip + GPU_AO_NF * jp];
                            v_il[ip] += s * d_jk;
                            v_ik[ip] += s * d_jl;

                            v_jl += s * d_ik[ip];
                            v_jk += s * d_il[ip];
                        }
                        atomicAdd(vk+j+nao*k, v_jk);
                        atomicAdd(vk+j+nao*l, v_jl);
                    }
                    for (i = 0; i < i1-i0; i++){ 
                        atomicAdd(vk+i+i0+nao*k, v_ik[i]); 
                        atomicAdd(vk+i+i0+nao*l, v_il[i]);  
                    }
                    ngout += nfij;
                }
            }
            dm += nao * nao;
            vk += nao * nao;
        }
        return;

    }

    double v_il[GPU_AO_NF];
    double v_ik[GPU_AO_NF];

    double d_ik[GPU_AO_NF];
    double d_il[GPU_AO_NF];

    // vj != NULL and vk != NULL
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        int ngout = 0;
        for (l = l0; l < l1; ++l) {
            for (i = 0; i < i1-i0; ++i){ 
                v_il[i] = 0.0; 
                d_il[i] = dm[i+i0+l*nao];
            }
            for (k = k0; k < k1; ++k) {
                for (i = 0; i < i1-i0; ++i){ 
                    v_ik[i] = 0.0; 
                    d_ik[i] = dm[i+i0+k*nao];
                }
                double v_kl = 0;
                double d_kl = dm[k+nao*l];
                for (n = 0, j = j0; j < j1; ++j) {
                    double v_jk = 0.0;
                    double v_jl = 0.0;
                    double d_jk = dm[j+nao*k];
                    double d_jl = dm[j+nao*l];
                    
                    for (i = i0; i < i1; ++i, ++n) {
                        int ip = i - i0;
                        int ng = n + ngout;
                        int ix = idx[ng];
                        int iy = idy[ng];
                        int iz = idz[ng];
                        double s = 0.0;
                        for (int r = 0; r < NROOTS; r++){
                            s += g[ix+r] * g[iy+r] * g[iz+r];
                        }
                        double v_ij  = s * d_kl;   
                        atomicAdd(vj+i+nao*j, v_ij);

                        v_il[ip] += s * d_jk;
                        v_ik[ip] += s * d_jl;

                        v_jl += s * d_ik[ip];
                        v_jk += s * d_il[ip];

                        v_kl += s * dm[i+j*nao];
                    }
                    atomicAdd(vk+j+nao*k, v_jk);
                    atomicAdd(vk+j+nao*l, v_jl);
                }
                for (i = 0; i < i1-i0; i++){ 
                    atomicAdd(vk+i+i0+nao*k, v_ik[i]); 
                }
                atomicAdd(vj+k+nao*l, v_kl);
                ngout += nfij;
            }
            for (i = 0; i < i1-i0; i++){ 
                atomicAdd(vk+i+i0+nao*l, v_il[i]);  
            }
        }

        dm += nao * nao;
        vj += nao * nao;
        vk += nao * nao;
    }
}

/*
__device__
static void GINTkernel_getjk(JKMatrix jk, double* __restrict__ gout,
                      int ish, int jsh, int ksh, int lsh)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int task_id = ty * THREADSX + tx;
    int *ao_loc = c_bpcache.ao_loc;
    int i0 = ao_loc[ish  ];
    int i1 = ao_loc[ish+1];
    int j0 = ao_loc[jsh  ];
    int j1 = ao_loc[jsh+1];
    int k0 = ao_loc[ksh  ];
    int k1 = ao_loc[ksh+1];
    int l0 = ao_loc[lsh  ];
    int l1 = ao_loc[lsh+1];
    int nfi = i1 - i0;
    int nfj = j1 - j0;
    int nfk = k1 - k0;
    //int nfl = l1 - l0;
    int nfij = nfi * nfj;

    int nao = jk.nao;
    int i, j, k, l, n, i_dm;
    int ip, jp, kp, lp;
    double s;
    double d_kl, d_jk, d_jl;
    double v_ij, v_kl, v_ik, v_il, v_jk, v_jl;
    // enough to hold (g,s) shells
    __shared__ double _buf[THREADS*(GPU_CART_MAX*2+1)];
    int n_dm = jk.n_dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    double* __restrict__ dm = jk.dm;

    if (vk == NULL) {
        if (nfij > (GPU_CART_MAX*2+1)) {
            double* __restrict__  buf_ij = gout + envs.nf;
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
                for (ip = 0; ip < nfij; ++ip) {
                    buf_ij[ip] = 0;
                }
                double* __restrict__ pgout = gout;
                for (l = l0; l < l1; ++l) {
                    for (k = k0; k < k1; ++k) {
                        v_kl = 0;
                        d_kl = dm[k+nao*l];
                        for (n = 0, j = j0; j < j1; ++j) {
                            for (i = i0; i < i1; ++i, ++n) {
                                s = pgout[n];
                                v_ij  = s * d_kl;
                                v_kl += s * dm[i+nao*j];
                                buf_ij[n] += v_ij;
                            }
                        }
                        atomicAdd(vj+k+nao*l, v_kl);
                        pgout += nfij;
                    }
                }
                for (n = 0, j = j0; j < j1; ++j) {
                    for (i = i0; i < i1; ++i, ++n) {
                        atomicAdd(vj+i+nao*j, buf_ij[n]);
                    }
                }
                dm += nao * nao;
                vj += nao * nao;
            }

        } else {
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
                for (ip = 0; ip < nfij; ++ip) {
                    _buf[ip*THREADS+task_id] = 0;
                }
                double* __restrict__ pgout = gout;
                for (l = l0; l < l1; ++l) {
                    for (k = k0; k < k1; ++k) {
                        v_kl = 0;
                        d_kl = dm[k+nao*l];
                        for (n = 0, j = j0; j < j1; ++j) {
                            for (i = i0; i < i1; ++i, ++n) {
                                s = pgout[n];
                                v_ij  = s * d_kl;
                                v_kl += s * dm[i+nao*j];
                                _buf[n*THREADS+task_id] += v_ij;
                            }
                        }
                        atomicAdd(vj+k+nao*l, v_kl);
                        pgout += nfij;
                    }
                }
                for (n = 0, j = j0; j < j1; ++j) {
                    for (i = i0; i < i1; ++i, ++n) {
                        atomicAdd(vj+i+nao*j, _buf[n*THREADS+task_id]);
                    }
                }
                dm += nao * nao;
                vj += nao * nao;
            }
        }
        return;
    }

    // vk != NULL
    double* __restrict__ buf_i = _buf;
    double* __restrict__ buf_j = _buf + nfi * THREADS;

    if (vj != NULL) {
        if (nfij > SHARED_MEM_NFIJ_MAX) {
            double* __restrict__  buf_ij = gout + envs.nf;
                for (i_dm = 0; i_dm < n_dm; ++i_dm) {
                for (ip = 0; ip < nfij; ++ip) {
                    buf_ij[ip] = 0;
                }
                double* __restrict__ pgout = gout;
                for (l = l0; l < l1; ++l) {
                    for (ip = 0; ip < nfi; ++ip) {
                        buf_i[ip*THREADS+task_id] = 0;
                    }
                    for (jp = 0; jp < nfj; ++jp) {
                        buf_j[jp*THREADS+task_id] = 0;
                    }

                    for (k = k0; k < k1; ++k) {
                        v_kl = 0;
                        d_kl = dm[k+nao*l];
                        for (n = 0, j = j0; j < j1; ++j) { jp = j - j0;
                            v_il = 0;
                            v_jl = 0;
                            d_jk = dm[j+nao*k];
                            for (i = i0; i < i1; ++i, ++n) { ip = i - i0;
                                s = pgout[n];
                                v_ij  = s * d_kl;
                                v_il  = s * d_jk;
                                v_jl += s * dm[i+nao*k];
                                v_kl += s * dm[i+nao*j];
                                buf_ij[n] += v_ij;
                                buf_i[ip*THREADS+task_id] += v_il;
                            }
                            buf_j[jp*THREADS+task_id] += v_jl;
                        }
                        atomicAdd(vj+k+nao*l, v_kl);
                        pgout += nfij;
                    }
                    for (ip = 0; ip < nfi; ++ip) {
                        atomicAdd(vk+i0+ip+nao*l, buf_i[ip*THREADS+task_id]);
                    }
                    for (jp = 0; jp < nfj; ++jp) {
                        atomicAdd(vk+j0+jp+nao*l, buf_j[jp*THREADS+task_id]);
                    }
                }
                for (n = 0, j = j0; j < j1; ++j) {
                    for (i = i0; i < i1; ++i, ++n) {
                        atomicAdd(vj+i+nao*j, buf_ij[n]);
                    }
                }
                dm += nao * nao;
                vj += nao * nao;
                vk += nao * nao;
            }

        } else {  // nfij <= SHARED_MEM_NFIJ_MAX
            double* __restrict__ buf_ij = buf_j + nfj * THREADS;
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
                for (ip = 0; ip < nfij; ++ip) {
                    buf_ij[ip*THREADS+task_id] = 0;
                }
                double* __restrict__ pgout = gout;
                for (l = l0; l < l1; ++l) {
                    for (ip = 0; ip < nfi; ++ip) {
                        buf_i[ip*THREADS+task_id] = 0;
                    }
                    for (jp = 0; jp < nfj; ++jp) {
                        buf_j[jp*THREADS+task_id] = 0;
                    }

                    for (k = k0; k < k1; ++k) {
                        v_kl = 0;
                        d_kl = dm[k+nao*l];
                        for (n = 0, j = j0; j < j1; ++j) { jp = j - j0;
                            v_il = 0;
                            v_jl = 0;
                            d_jk = dm[j+nao*k];
                            for (i = i0; i < i1; ++i, ++n) { ip = i - i0;
                                s = pgout[n];
                                v_ij  = s * d_kl;
                                v_il  = s * d_jk;
                                v_jl += s * dm[i+nao*k];
                                v_kl += s * dm[i+nao*j];
                                buf_ij[n*THREADS+task_id] += v_ij;
                                buf_i[ip*THREADS+task_id] += v_il;
                            }
                            buf_j[jp*THREADS+task_id] += v_jl;
                        }
                        atomicAdd(vj+k+nao*l, v_kl);
                        pgout += nfij;
                    }
                    for (ip = 0; ip < nfi; ++ip) {
                        atomicAdd(vk+i0+ip+nao*l, buf_i[ip*THREADS+task_id]);
                    }
                    for (jp = 0; jp < nfj; ++jp) {
                        atomicAdd(vk+j0+jp+nao*l, buf_j[jp*THREADS+task_id]);
                    }
                }
                for (n = 0, j = j0; j < j1; ++j) {
                    for (i = i0; i < i1; ++i, ++n) {
                        atomicAdd(vj+i+nao*j, buf_ij[n*THREADS+task_id]);
                    }
                }
                dm += nao * nao;
                vj += nao * nao;
                vk += nao * nao;
            }
        }

    } else {  // vj == NULL, vk != NULL
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            for (n = 0, l = l0; l < l1; ++l) {
                for (ip = 0; ip < nfi; ++ip) {
                    buf_i[ip*THREADS+task_id] = 0;
                }
                for (jp = 0; jp < nfj; ++jp) {
                    buf_j[jp*THREADS+task_id] = 0;
                }

                for (k = k0; k < k1; ++k) {
                    for (j = j0; j < j1; ++j) { jp = j - j0;
                        v_il = 0;
                        v_jl = 0;
                        for (i = i0; i < i1; ++i, ++n) { ip = i - i0;
                            s = gout[n];
                            v_il  = s * d_jk;
                            v_jl += s * dm[i+nao*k];
                            buf_i[ip*THREADS+task_id] += v_il;
                        }
                        buf_j[jp*THREADS+task_id] += v_jl;
                    }
                }
                for (ip = 0; ip < nfi; ++ip) {
                    atomicAdd(vk+i0+ip+nao*l, buf_i[ip*THREADS+task_id]);
                }
                for (jp = 0; jp < nfj; ++jp) {
                    atomicAdd(vk+j0+jp+nao*l, buf_j[jp*THREADS+task_id]);
                }
            }
            dm += nao * nao;
            vk += nao * nao;
        }
    }

    // vj == NULL, vk != NULL
    vk = jk.vk;
    dm = jk.dm;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (k = k0; k < k1; ++k) { kp = k - k0;
            for (ip = 0; ip < nfi; ++ip) {
                buf_i[ip*THREADS+task_id] = 0;
            }
            for (jp = 0; jp < nfj; ++jp) {
                buf_j[jp*THREADS+task_id] = 0;
            }

            for (l = l0; l < l1; ++l) { lp = l - l0;
                n = nfij * (lp * nfk + kp);
                for (j = j0; j < j1; ++j) { jp = j - j0;
                    v_ik = 0;
                    v_jk = 0;
                    d_jl = dm[j+nao*l];
                    for (i = i0; i < i1; ++i, ++n) { ip = i - i0;
                        s = gout[n];
                        v_ik  = s * d_jl;
                        v_jk += s * dm[i+nao*l];
                        buf_i[ip*THREADS+task_id] += v_ik;
                    }
                    buf_j[jp*THREADS+task_id] += v_jk;
                }
            }
            for (ip = 0; ip < nfi; ++ip) {
                atomicAdd(vk+i0+ip+nao*k, buf_i[ip*THREADS+task_id]);
            }
            for (jp = 0; jp < nfj; ++jp) {
                atomicAdd(vk+j0+jp+nao*k, buf_j[jp*THREADS+task_id]);
            }
        }
        dm += nao * nao;
        vk += nao * nao;
    }
}
*/

__device__
static int is_skip(JKMatrix jk, double log_q_ij, double log_q_kl, int ish, int jsh, int ksh, int lsh, double log_cutoff)
{
    double max_dm = -9999;
    int nshls = jk.nshls;
    max_dm = MAX(max_dm, jk.dm_sh[ish * nshls + jsh]);
    max_dm = MAX(max_dm, jk.dm_sh[ksh * nshls + lsh]);
    max_dm = MAX(max_dm, jk.dm_sh[ish * nshls + ksh]);
    max_dm = MAX(max_dm, jk.dm_sh[jsh * nshls + ksh]);
    max_dm = MAX(max_dm, jk.dm_sh[ish * nshls + lsh]);
    max_dm = MAX(max_dm, jk.dm_sh[jsh * nshls + lsh]);
    
    if(log_q_ij + log_q_kl + max_dm < log_cutoff){
        return 1;
    }
    else{
        return 0;
    }
}
