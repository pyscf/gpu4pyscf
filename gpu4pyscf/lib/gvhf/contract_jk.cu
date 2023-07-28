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

__device__
void GINTkernel_getjk(JKMatrix jk, double* __restrict__ gout,
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
            double* __restrict__  buf_ij = gout + c_envs.nf;
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
            double* __restrict__  buf_ij = gout + c_envs.nf;
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
