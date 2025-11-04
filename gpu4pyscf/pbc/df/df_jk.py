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

__all__ = [
    'density_fit', 'get_j_kpts', 'get_k_kpts',
    'get_j_kpts_kshift', 'get_k_kpts_kshift',
    'get_jk'
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib.cupy_helper import contract, unpack_tril, get_avail_mem
from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0, _format_dms, _format_jks
from gpu4pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.lib import kpts_helper

def density_fit(mf, auxbasis=None, with_df=None):
    '''Generate density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        with_df : DF object
    '''
    from gpu4pyscf.pbc.df.df import GDF
    if with_df is None:
        if getattr(mf, 'kpts', None) is not None:
            kpts = mf.kpts
        else:
            kpts = np.reshape(mf.kpt, (1,3))
        with_df = GDF(mf.cell, kpts)
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf = mf.copy().reset()
    mf.with_df = with_df
    return mf


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None):
    log = logger.new_logger(mydf)
    t0 = log.init_timer()
    assert kpts_band is None or kpts_band is kpts
    assert mydf.has_kpts(kpts)
    if mydf._cderi is None:
        mydf.build(j_only=True, kpts_band=kpts_band)
        t0 = log.timer_debug1('Init get_j_kpts', *t0)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    # Alter the contraction order for
    # rho = einsum('piLj,LK,Kji->p', cderi, expLk, dm)
    # dm_sparse = einsum('LK,Kji->iLj', expLk, dm)[cderi_idx]
    expLk = kpts_helper.fft_matrix(mydf.kmesh)
    dm_sparse = contract('LK,nKji->niLj', expLk, dms)
    contract('LK,nKji->njLi', expLk.conj(), dms, beta=1, out=dm_sparse)
    dm_sparse = dm_sparse.reshape(nset, -1)
    ao_pair_mapping, diag = mydf._cderi_idx
    dm_sparse = dm_sparse[:,ao_pair_mapping]
    dm_sparse[:,diag] *= .5

    avail_mem = get_avail_mem() * .8
    npairs = len(ao_pair_mapping)
    blksize = avail_mem/16 / ((nkpts+1)*npairs)
    if blksize < 16:
        raise RuntimeError('Insufficient GPU memory')
    blksize = min(int(blksize), mydf.blockdim)
    logger.debug2(mydf, 'max_memory %d MB, blksize %d', avail_mem*1e-6, blksize)
    naux = mydf.get_naoaux()
    aux_iter = iter((0, p0, p1) for p0, p1 in lib.prange(0, naux, blksize))

    def proc():
        _dm_sparse = cp.asarray(dm_sparse)
        vj_packed = cp.zeros_like(dm_sparse)
        buf = cp.empty(nkpts*blksize*npairs, dtype=np.complex128)
        for k_aux, Lpq, sign in mydf.loop(blksize, unpack=False, kpts=kpts,
                                          aux_iter=aux_iter, out=buf):
            rho = sign * _dm_sparse.dot(Lpq.T)
            vj_packed += rho.dot(Lpq)
        return vj_packed

    results = multi_gpu.run(proc, non_blocking=True)
    vj_packed = multi_gpu.array_reduce(results, inplace=True)
    kk_conserv = k2gamma.double_translation_indices(mydf.kmesh)
    # The ao-pair in vj_packed has the same storage order like the ao-pair in
    # cderi tensor. It can be unpacked using rsdf_builder.unpack_cderi. This
    # function returns a tensor sorted as [nkpt,naux,nao,nao]. vj for multiple
    # dms should be stored as [ndm,nkpt,nao,nao].
    vj = rsdf_builder.unpack_cderi(
        vj_packed, mydf._cderi_idx, 0, kk_conserv, expLk, nao)
    vj = vj.transpose(1,0,2,3)
    if is_zero(kpts_band) and not np.iscomplexobj(dms):
        vj = vj.real
    vj *= 1./nkpts
    return _format_jks(vj, dm_kpts, input_band, kpts)


def get_j_kpts_kshift(mydf, dm_kpts, kshift, hermi=0, kpts=None, kpts_band=None):
    raise NotImplementedError

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               exxdiv=None):
    cell = mydf.cell
    log = logger.new_logger(mydf)

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('GDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)

    t0 = (logger.process_clock(), logger.perf_counter())
    assert kpts_band is None or kpts_band is kpts
    assert mydf.has_kpts(kpts)
    if mydf._cderi is None:
        mydf.build(kpts_band=kpts_band)
        t0 = log.timer_debug1('Init get_k_kpts', *t0)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    vk = cp.zeros((nset,nband,nao,nao), dtype=np.complex128)

    ''' math
    K(p,q; k2 from k1)
        = V(r k1, q k2, p k2, s k1) * D(s,r; k1)
        = V(L, r k1, q k2) * V(L, s k1, p k2).conj() * D(s,r; k1)         eqn (1)
    --> in case of Hermitian & PSD DM
        = ( V(L, s k1, p k2) * C(s,i; k1).conj() ).conj()
          * V(L, r k1, q k2) * C(r,i; k1).conj()                          eqn (2)
        = W(L, i k1, p k2).conj() * W(L, i k1, q k2)                      eqn (3)
    --> in case of non-Hermitian or non-PSD DM
        = ( V(L, s k1, p k2) * A(s,i; k1).conj() ).conj()
          * V(L, r k1, q k2) * B(r,i; k1).conj()                          eqn (4)
        = X(L, i k1, p k2).conj() * Y(L, i k1, q k2)                      eqn (5)

    if swap_2e:
    K(p,q; k1 from k2)
        = V(p k1, s k2, r k2, q k1) * D(s,r; k2)
        = V(L, p k1, s k2) * V(L, q k1, r k2).conj() * D(s,r; k2)         eqn (1')
    --> in case of Hermitian & PSD DM
        = V(L, p k1, s k2) * C(s,i; k2)
          * ( V(L, q k1, r k2) * C(r,i; k2) ).conj()                      eqn (2')
        = W(L, p k1, i k2) * W(L, q k1, i k2).conj()                      eqn (3')
    --> in case of non-Hermitian or non-PSD DM
        = V(L, p k1, s k2) * A(s,i; k2)
          * ( V(L, q k1, r k2) * B(r,i; k2) ).conj()                      eqn (4')
        = X(L, p k1, i k2) * Y(L, q k1, i k2).conj()                      eqn (5')

    Mode 1: DM-based K-build uses eqn (1) and eqn (1')
    (NA) Mode 2: Symm MO-based K-build uses eqns (2,3) and eqns (2',3')
    (NA) Mode 3: Asymm MO-based K-build uses eqns (4,5) and eqns (4',5')
    '''
    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    # input dm is not Hermitian/PSD --> build K from dm
    log.debug2('get_k_kpts: build K from dm')

    avail_mem = get_avail_mem() * .8
    blksize = avail_mem/16 / (nkpts*nao**2*3)
    if blksize < 16:
        raise RuntimeError('Insufficient GPU memory')
    blksize = min(int(blksize), mydf.blockdim)
    logger.debug1(mydf, 'max_memory %d MB, blksize %d', avail_mem*1e-6, blksize)
    naux = mydf.get_naoaux()
    aux_iter = iter((kp, p0, p1)
                    for p0, p1 in lib.prange(0, naux, blksize)
                    for kp in mydf._cderi)
    k_adapt_dic = {}
    for kp, kp_conj, ki_idx, kj_idx in kpts_helper.kk_adapted_iter(mydf.kmesh):
        # ki_idx is already sorted
        k_adapt_dic[kp] = kp_conj, kj_idx

    if (is_zero(kpts) and is_zero(kpts_band) and
        not np.iscomplexobj(dm_kpts)):
        dtype = np.float64
    else:
        dtype = np.complex128

    def proc():
        _dms = cp.asarray(dms)
        vk = cp.zeros(_dms.shape, dtype=dtype)
        buf = cp.empty((3, nkpts*blksize*nao**2), dtype=dtype)
        for kp, Lpq, sign in mydf.loop(blksize, kpts=kpts, aux_iter=aux_iter,
                                       buf=buf[1], out=buf[0]):
            kp_conj, kj = k_adapt_dic[kp]
            Lpq_conj = cp.ndarray(Lpq.shape, dtype=dtype, memptr=buf[1].data)
            Lpq_conj = cp.conjugate(Lpq, out=Lpq_conj)
            tmp = cp.ndarray(Lpq.shape, dtype=dtype, memptr=buf[2].data)
            for i in range(nset):
                tmp = contract('nLij,njk->nLik', Lpq, _dms[i,kj], alpha=sign, out=tmp)
                contract('nLlk,nLik->nil', Lpq_conj, tmp, beta=1, out=vk[i])
                if kp != kp_conj:
                    tmp = contract('nLij,nli->nLlj', Lpq, _dms[i], alpha=sign, out=tmp)
                    vk[i,kj] += contract('nLlk,nLlj->nkj', Lpq_conj, tmp)
        return vk

    results = multi_gpu.run(proc, non_blocking=True)
    vk = multi_gpu.array_reduce(results, inplace=True)
    vk *= 1./nkpts
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band)
    log.timer('get_k_kpts', *t0)
    return _format_jks(vk, dm_kpts, input_band, kpts)

def get_k_kpts_kshift(mydf, dm_kpts, kshift, hermi=0, kpts=None, kpts_band=None,
                      exxdiv=None):
    raise NotImplementedError


##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    from gpu4pyscf.pbc.df import df_jk_real
    assert kpts_band is None
    if not is_zero(kpt):
        raise NotImplementedError(f'get_jk for single k-point {kpt}')

    if mydf._cderi is None:
        mydf.build()

    if dm.dtype == np.float64:
        return df_jk_real.get_jk(mydf, dm, hermi, with_j, with_k, exxdiv)
    else:
        kpts = kpt.reshape(1, 3)
        vj = vk = None
        if with_k:
            vk = get_k_kpts(mydf, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, dm, hermi, kpts, kpts_band)
        return vj, vk
