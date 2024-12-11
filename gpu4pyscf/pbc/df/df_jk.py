# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
    'get_j_kpts_kshift', 'get_k_kpts_kshift'
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, unpack_tril
from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0, _format_dms, _format_jks

def density_fit(mf, auxbasis=None, mesh=None, with_df=None):
    '''Generate density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        mesh : tuple
            number of grids in each direction
        with_df : DF object
    '''
    from gpu4pyscf.pbc.df.df import GDF
    if with_df is None:
        if getattr(mf, 'kpts', None) is not None:
            kpts = mf.kpts
        else:
            kpts = np.reshape(mf.kpt, (1,3))
        with_df = GDF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if mesh is not None:
            with_df.mesh = mesh

    mf = mf.copy()
    mf.with_df = with_df
    mf._eri = None
    return mf


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    log = logger.new_logger(mydf)
    t0 = log.init_timer()
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(j_only=True, kpts_band=kpts_band)
        t0 = log.timer_debug1('Init get_j_kpts', *t0)

    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if mydf.auxcell is None:
        # If mydf._cderi is the file that generated from another calculation,
        # guess naux based on the contents of the integral file.
        naux = mydf.get_naoaux()
    else:
        naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    rho = cp.zeros((nset,naux), dtype=np.complex128)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))
    for k, kpt in enumerate(kpts):
        kptii = np.asarray((kpt,kpt))
        p1 = 0
        for Lpq, sign in mydf.sr_loop(kptii, max_memory, False):
            Lpq = Lpq.reshape(-1,nao,nao)
            p0, p1 = p1, p1+Lpq.shape[0]
            rho[:,p0:p1] += sign * contract('Lpq,xqp->xL', Lpq, dms[:,k])
    t1 = log.timer_debug1('get_j pass 1', *t0)

    rho *= 1./nkpts
    if hermi == 0:
        aos2symm = False
        vj = cp.zeros((nset,nband,nao**2), dtype=np.complex128)
    else:
        aos2symm = True
        vj = cp.zeros((nset,nband,nao_pair), dtype=np.complex128)

    for k, kpt in enumerate(kpts_band):
        kptii = np.asarray((kpt,kpt))
        p1 = 0
        for Lpq, sign in mydf.sr_loop(kptii, max_memory, aos2symm):
            nrow = Lpq.shape[0]
            p0, p1 = p1, p1+nrow
            Lpq = Lpq.reshape(nrow, -1)
            vj[:,k] += cp.dot(rho[:,p0:p1], Lpq)
    t1 = log.timer_debug1('get_j pass 2', *t1)

    if aos2symm:
        vj = unpack_tril(vj.reshape(-1,nao_pair))
    j_real = is_zero(kpts_band) and not np.iscomplexobj(dms)
    if j_real:
        vj = vj.real
    vj = vj.reshape(nset,nband,nao,nao)

    log.timer('get_j', *t0)

    return _format_jks(vj, dm_kpts, input_band, kpts)


def get_j_kpts_kshift(mydf, dm_kpts, kshift, hermi=0, kpts=np.zeros((1,3)), kpts_band=None):
    raise NotImplementedError

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    cell = mydf.cell
    log = logger.new_logger(mydf)

    if exxdiv is not None and exxdiv != 'ewald':
        log.warn('GDF does not support exxdiv %s. '
                 'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)

    t0 = (logger.process_clock(), logger.perf_counter())
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(kpts_band=kpts_band)
        t0 = log.timer_debug1('Init get_k_kpts', *t0)

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
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
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    def make_kpt(ki, kj, swap_2e, inverse_idx=None):
        kpti = kpts[ki]
        kptj = kpts_band[kj]
        #TODO: utilize kk_adapted_iter with time_reversal_symmetry, as that in aft_jk
        for Lpq, sign in mydf.sr_loop((kpti,kptj), max_memory, compact=False):
            Lpq = Lpq.reshape(-1, nao, nao)
            tmp = contract('njk,Lkl->nLjl', dms[:,ki], Lpq)
            if sign > 0:
                vk[:,kj] += contract('Lji,nLjl->nil', Lpq.conj(), tmp)
            else:
                vk[:,kj] -= contract('Lji,nLjl->nil', Lpq.conj(), tmp)

            if swap_2e:
                tmp = contract('Lkl,nli->nLki', Lpq, dms[:,kj])
                if sign > 0:
                    vk[:,ki] += contract('nLki,Lji->nkj', tmp, Lpq.conj())
                else:
                    vk[:,ki] -= contract('nLki,Lji->nkj', tmp, Lpq.conj())

    t1 = log.init_timer()
    if kpts_band is kpts:  # normal k-points HF/DFT
        for ki in range(nkpts):
            for kj in range(ki):
                make_kpt(ki, kj, True)
            make_kpt(ki, ki, False)
            t1 = log.timer_debug1('get_k_kpts: make_kpt ki>=kj (%d,*)'%ki, *t1)
    else:
        raise NotImplementedError

    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band)

    if (is_zero(kpts) and is_zero(kpts_band) and
        not np.iscomplexobj(dm_kpts)):
        vk = vk.real
    vk *= 1./nkpts

    log.timer('get_k_kpts', *t0)
    return _format_jks(vk, dm_kpts, input_band, kpts)

def get_k_kpts_kshift(mydf, dm_kpts, kshift, hermi=0, kpts=np.zeros((1,3)), kpts_band=None,
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
    log = logger.new_logger(mydf)
    t0 = log.init_timer()
    if mydf._cderi is None or not mydf.has_kpts(kpts_band):
        if mydf._cderi is not None:
            log.warn('DF integrals for band k-points were not found %s. '
                     'DF integrals will be rebuilt to include band k-points.',
                     mydf._cderi)
        mydf.build(j_only=not with_k, kpts_band=kpts_band)
        t0 = log.timer_debug1('Init get_jk', *t0)

    vj = vk = None
    if kpts_band is not None and abs(kpt-kpts_band).sum() > 1e-9:
        kpt = np.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(mydf, dm, hermi, kpt, kpts_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, dm, hermi, kpt, kpts_band)
        return vj, vk

    cell = mydf.cell
    dm = np.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    kptii = np.asarray((kpt,kpt))
    if with_j:
        vj = cp.zeros((nset,nao,nao), dtype=np.complex128)
    if with_k:
        ''' math
        Mode 1: DM-based K-build:
            K(p,q)
                = V(r,q,p,s) * D(s,r)
                = V(L,r,q) * V(L,s,p).conj() * D(s,r)    eqn (1)

        (NA) Mode 2: Symm MO-based K-build:
        In case of Hermitian & PSD DM, eqn (1) can be rewritten as
            K(p,q)
                = W(L,i,p).conj() * W(L,i,q)
        where
            W(L,i,p) = V(L,s,p) * C(s,i).conj()
            D(s,r) = C(s,i) * C(r,i).conj()

        (NA) Mode 3: Asymm MO-based K-build:
        In case of non-Hermitian or Hermitian but non-PSD DM, eqn (1) can be rewritten as
            K(p,q)
                = X(L,i,p).conj() * Y(L,i,q)
            where
                X(L,i,p) = V(L,s,p) * A(s,i).conj()
                Y(L,i,q) = V(L,r,q) * B(r,i).conj()
                D(s,r) = A(s,i) * B(r,i).conj()
        '''
        vk = cp.zeros((nset,nao,nao), dtype=np.complex128)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))
    for Lpq, sign in mydf.sr_loop(kptii, max_memory, False):
        if with_j:
            #:rho_coeff = np.einsum('Lpq,xqp->xL', Lpq, dms)
            #:vj += np.dot(rho_coeff, Lpq.reshape(-1,nao**2))
            rho = contract('Lpq,xqp->xL', Lpq, dms)
            vj += sign * contract('xL,Lpq->xpq', rho, Lpq)
        if with_k:
            tmp = contract('njk,Lkl->nLjl', dms, Lpq)
            if sign > 0:
                vk += contract('Lji,nLjl->nil', Lpq.conj(), tmp)
            else:
                vk -= contract('Lji,nLjl->nil', Lpq.conj(), tmp)

    if with_j:
        j_real = is_zero(kpt) and hermi == 1
        if j_real:
            vj = vj.real
        vj = vj.reshape(dm.shape)
    if with_k:
        k_real = is_zero(kpt) and not np.iscomplexobj(dms)
        if k_real:
            vk = vk.real
        if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
        vk = vk.reshape(dm.shape)

    log.timer('sr jk', *t0)
    return vj, vk
