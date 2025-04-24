# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Gaussian density fitting with real integrals for Gamma point
'''

__all__ = [
    'get_jk'
]

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import contract, unpack_tril
from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0
from gpu4pyscf.df import df_jk as mol_df_jk

# TOOD: refactor and reuse the molecule df functions

def get_jk(mydf, dm, hermi=1, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    log = logger.new_logger(mydf)
    t0 = t1 = log.init_timer()
    if mydf._cderi is None:
        assert mydf.is_gamma_point
        mydf.build()
        t0 = log.timer_debug1('Init get_jk', *t0)
    assert hermi == 1
    assert dm.dtype == np.float64
    assert with_j or with_k

    out_shape = dm.shape
    out_cupy = isinstance(dm, cp.ndarray)
    nao = out_shape[-1]
    assert nao == mydf.nao
    dms = cp.asarray(dm).reshape(-1,nao,nao)
    nset = len(dms)

    vj = vk = None
    if with_j:
        rows, cols, diag = mydf._cderi_idx
        dm_sparse = dms[:,rows,cols]
        dm_sparse *= 2
        dm_sparse[:,diag] *= .5

    if getattr(dm, 'mo_coeff', None) is not None:
        nmo = dm.mo_occ.shape[-1]
        mo_occ = dm.mo_occ.reshape(nset,nmo)
        mo_coeff = dm.mo_coeff.reshape(nset,nao,nmo)
        occ_coeff = []
        for c, occ in zip(mo_coeff, mo_occ):
            mask = occ > 0
            occ_coeff.append(c[:,mask] * occ[mask]**0.5)

        def proc():
            vj_packed = vk = None
            if with_j:
                _dm_sparse = cp.asarray(dm_sparse)
                vj_packed = cp.zeros_like(dm_sparse)
            nocc = 0
            if with_k:
                _occ_coeff = [cp.asarray(x) for x in occ_coeff]
                vk = cp.zeros_like(dms)
                nocc = max(x.shape[1] for x in occ_coeff)
            blksize = mydf.get_blksize(extra=nao*nocc)
            for cderi, cderi_sparse in mydf.loop(blksize=blksize, unpack=with_k):
                if with_j:
                    rhoj = _dm_sparse.dot(cderi_sparse)
                    vj_packed += rhoj.dot(cderi_sparse.T)
                cderi_sparse = rhoj = None
                if with_k:
                    for i in range(nset):
                        rhok = contract('Lji,jk->Lki', cderi, _occ_coeff[i])
                        rhok = rhok.reshape([-1,nao])
                        vk[i] += cp.dot(rhok.T, rhok)
                        rhok = None
                cderi = None
            return vj_packed, vk
    else:
        def proc():
            vj_packed = vk = None
            if with_j:
                _dm_sparse = cp.asarray(dm_sparse)
                vj_packed = cp.zeros_like(dm_sparse)
            if with_k:
                _dms = cp.asarray(dms)
                vk = cp.zeros_like(dms)
            blksize = mydf.get_blksize(extra=nao*nao)
            for cderi, cderi_sparse in mydf.loop(blksize=blksize, unpack=with_k):
                if with_j:
                    rhoj = _dm_sparse.dot(cderi_sparse)
                    vj_packed += rhoj.dot(cderi_sparse.T)
                cderi_sparse = rhoj = None
                if with_k:
                    for k in range(nset):
                        rhok = contract('Lij,jk->Lki', cderi, _dms[k])
                        rhok = rhok.reshape([-1,nao])
                        vk[k] += cp.dot(rhok.T, cderi.reshape([-1,nao]))
                        rhok = None
                cderi = None
            return vj_packed, vk

    results = multi_gpu.run(proc, non_blocking=True)

    vj = vk = None
    if with_j:
        vj_packed = [j for j, k in results]
        vj_packed = multi_gpu.array_reduce(vj_packed, inplace=True)
        vj = cp.zeros_like(dms)
        vj[:,cols,rows] = vj[:,rows,cols] = vj_packed
        vj = vj.reshape(out_shape)
        if not out_cupy: vj = vj.get()

    if with_k:
        vk = [k for j, k in results]
        vk = multi_gpu.array_reduce(vk, inplace=True)
        if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(mydf.cell, np.zeros(3), dms, vk)
        vk = vk.reshape(out_shape)
        if not out_cupy: vk = vk.get()

    t1 = log.timer_debug1('vj and vk', *t1)
    return vj, vk
