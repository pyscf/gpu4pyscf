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
JK with analytic Fourier transformation
'''

__all__ = [
    'get_j_kpts', 'get_k_kpts', 'get_jk'
]

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.lib.kpts_helper import (is_zero, group_by_conj_pairs,
                                       kk_adapted_iter)
from pyscf.pbc.tools import k2gamma
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.df.ft_ao import FTOpt
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks, _ewald_exxdiv_for_G0
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.lib import logger

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    if kpts_band is not None:
        return get_j_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band)

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    vj_kpts = cp.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)
    kpt_allow = np.zeros(3)
    coulG = mydf.weighted_coulG(kpt_allow, False)
    for Gpq, p0, p1 in mydf.ft_loop(q=kpt_allow, kpts=kpts):
        _update_vj_(vj_kpts, Gpq, dms, coulG[p0:p1])
    vj_kpts *= 1./len(kpts)

    if is_zero(kpts):
        vj_kpts = vj_kpts.real
    return _format_jks(vj_kpts, dm_kpts, kpts_band, kpts)

def _update_vj_(vj_kpts, Gpq, dms, coulG, weight=None):
    r'''Compute the Coulomb matrix
    J_{kl} = \sum_{ij} \sum_G 4\pi/G^2 * FT(\rho_{ij}) IFT(\rho_{kl}) dm_{ji}
    for analytical FT tensor FT(\rho_{ij})
    '''
    rho = contract('nkij,kgij->ng', dms, Gpq.conj())
    if weight is not None:
        coulG = coulG * weight
    vG = coulG * rho

    if vj_kpts.dtype == np.double:
        vj_kpts += contract('ng,kgij->nkij', vG.real, Gpq.real)
        vj_kpts -= contract('ng,kgij->nkij', vG.imag, Gpq.imag)
    else:
        vj_kpts += contract('ng,kgij->nkij', vG, Gpq)
    return vj_kpts

def get_j_for_bands(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    kpt_allow = np.zeros(3)
    coulG = mydf.weighted_coulG(kpt_allow, False)
    ngrids = len(coulG)
    rhoG = cp.zeros((nset,ngrids), dtype=np.complex128)

    for Gpq, p0, p1 in mydf.ft_loop(q=kpt_allow, kpts=kpts):
        rhoG[:,p0:p1] += contract('nkij,kLij->nL', dms, Gpq.conj())
    vG = rhoG * coulG
    vG *= 1./len(kpts)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vj_kpts = cp.zeros((nset,nband,nao,nao), dtype=np.complex128)
    for Gpq, p0, p1 in mydf.ft_loop(q=kpt_allow, kpts=kpts_band):
        vj_kpts += contract('nL,kLij->nij', vG[:,p0:p1], Gpq)

    if is_zero(kpts_band):
        vj_kpts = vj_kpts.real
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    if kpts_band is not None:
        return get_k_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)

    log = logger.new_logger(mydf)
    cpu0 = cpu1 = log.init_timer()
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
    mo_occ = getattr(dm_kpts, 'mo_occ', None)
    dm_kpts = cp.asarray(dm_kpts)

    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    vk_kpts = cp.zeros((n_dm,nkpts,nao,nao), dtype=np.complex128)
    weight = 1. / nkpts
    # Add ewald_exxdiv contribution because G=0 was not included in the
    # non-uniform grids
    if (exxdiv == 'ewald' and
        (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
         (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts)

    t_rev_pairs = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    try:
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    except TypeError:
        t_rev_pairs = [[k, k] if k_conj is None else [k, k_conj]
                       for k, k_conj in t_rev_pairs]
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    log.debug1('Num time-reversal pairs %d', len(t_rev_pairs))

    time_reversal_symmetry = mydf.time_reversal_symmetry
    if time_reversal_symmetry:
        for k, k_conj in t_rev_pairs:
            if k != k_conj and abs(dms[:,k_conj] - dms[:,k].conj()).max() > 1e-6:
                time_reversal_symmetry = False
                log.debug2('Disable time_reversal_symmetry')
                break

    if time_reversal_symmetry:
        k_to_compute = np.zeros(nkpts, dtype=np.int8)
        k_to_compute[t_rev_pairs[:,0]] = 1
    else:
        k_to_compute = np.ones(nkpts, dtype=np.int8)

    bvk_kmesh = kpts_to_kmesh(cell, kpts)
    log.debug('bvk_kmesh = %s', bvk_kmesh)
    bvk_ncells = np.prod(bvk_kmesh)

    if mo_coeff is None:
        update_vk = _update_vk_
    else:
        # dm ~= dm_factor * dm_factor.T
        n_dm, nkpts, nao = dms.shape[:3]
        # mo_coeff, mo_occ may not be a list of aligned array if
        # remove_lin_dep was applied to scf object.
        # We assume they are of the same length in this version.
        mo_occ = cp.asarray(mo_occ)
        nocc = cp.count_nonzero(mo_occ > 0, axis=-1).max()
        if dm_kpts.ndim == 4:  # KUHF
            mo_coeff = cp.asarray(mo_coeff)[:,:,:,:nocc]
            occs = mo_occ[:,:,:nocc]
            dm_factor = cp.array(mo_coeff, dtype=np.complex128, order='C')
            dm_factor *= cp.sqrt(cp.array(occs, dtype=np.double))[:,:,None,:]
        else:  # KRHF
            mo_coeff = cp.asarray(mo_coeff)[:,:,:nocc]
            occs = mo_occ[:,:nocc]
            dm_factor = cp.array(mo_coeff, dtype=np.complex128, order='C')
            dm_factor *= cp.sqrt(cp.array(occs, dtype=np.double))[:,None,:]
            dm_factor = dm_factor[None]
        dms, dm_factor = dm_factor, None

        log.debug2('time_reversal_symmetry = %s bvk_ncells = %d '
                   'cell0_nao = %d nocc = %d n_dm = %d',
                   time_reversal_symmetry, bvk_ncells, nao, nocc, n_dm)
        update_vk = _update_vk_dmf
    log.debug2('set update_vk to %s', update_vk)

    # TODO: apply ft_opt.coeff to the dms; skip the AO ordering transformation
    # in ft_kern.
    ft_opt = FTOpt(cell, bvk_kmesh=bvk_kmesh)
    ft_kern = ft_opt.gen_ft_kernel(verbose=log)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    avail_mem = get_avail_mem() * .8
    Gblksize = max(16, int(avail_mem/(2*16*nao**2*bvk_ncells))//8*8)
    Gblksize = min(Gblksize, ngrids, 16384)
    log.debug1('Gblksize = %d', Gblksize)

    for group_id, (kpt, ki_idx, kj_idx, self_conj) \
            in enumerate(kk_adapted_iter(cell, kpts)):
        vkcoulG = mydf.weighted_coulG(kpt, exxdiv, mesh)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            log.debug3('update_vk [%s:%s]', p0, p1)
            Gpq = ft_kern(Gv[p0:p1], kpt, kpts)
            update_vk(vk_kpts, Gpq, dms, vkcoulG[p0:p1] * weight, ki_idx, kj_idx,
                      not self_conj, k_to_compute, t_rev_pairs)
            Gpq = None
        cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)

    if is_zero(kpts) and not np.iscomplexobj(dm_kpts):
        vk_kpts = vk_kpts.real

    if time_reversal_symmetry:
        for k, k_conj in t_rev_pairs:
            if k != k_conj:
                vk_kpts[:,k_conj] = vk_kpts[:,k].conj()
    log.timer_debug1('get_k_kpts', *cpu0)
    return vk_kpts.reshape(dm_kpts.shape)

def get_k_for_bands(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
                    exxdiv=None):
    raise NotImplementedError

def _update_vk_(vk, Gpq, dms, wcoulG, kpti_idx, kptj_idx, swap_2e,
                k_to_compute, t_rev_pairs):
    '''
    contraction for exchange matrices:
    '''
    if Gpq.dtype == np.float64:
        Gpq_conj = Gpq * wcoulG[:,None,None]
    else:
        Gpq_conj = Gpq.conj()
        Gpq_conj *= wcoulG[:,None,None]
    k_mask = k_to_compute[kpti_idx] == 1
    ki = kpti_idx[k_mask]
    kj = kptj_idx[k_mask]
    tmp = contract('ngij,snjk->sngik', Gpq[kj], dms[:,kj])
    vk[:,ki] += contract('sngik,nglk->snil', tmp, Gpq_conj[kj])

    if swap_2e:
        k_mask = k_to_compute[kptj_idx] == 1
        ki = kpti_idx[k_mask]
        kj = kptj_idx[k_mask]
        tmp = contract('ngij,snli->snglj', Gpq[kj], dms[:,ki])
        vk[:,kj] += contract('nglk,snglj->snkj', Gpq_conj[kj], tmp)
    return vk

def _update_vk_dmf(vk, Gpq, dmf, wcoulG, kpti_idx, kptj_idx, swap_2e,
                   k_to_compute, t_rev_pairs):
    '''
    dmf is the factorized dm, dm = dmf * dmf.conj().T
    Computing exchange matrices with dmf:
    '''
    k_mask = k_to_compute[kpti_idx] == 1
    ki = kpti_idx[k_mask]
    kj = kptj_idx[k_mask]
    Gpi = contract('ngij,snjp->sngpi', Gpq[kj], dmf[:,kj])
    if Gpi.dtype == np.float64:
        Gpi_conj = Gpi * wcoulG[:,None,None]
    else:
        Gpi_conj = Gpi.conj()
        Gpi_conj *= wcoulG[:,None,None]
    vk[:,ki] += contract('sngpi,sngpj->snij', Gpi, Gpi_conj)

    if swap_2e:
        k_mask = k_to_compute[kptj_idx] == 1
        ki = kpti_idx[k_mask]
        kj = kptj_idx[k_mask]
        Gpi = contract('ngij,snip->sngpj', Gpq[kj], dmf[:,ki].conj())
        Gpi_conj = Gpi.conj()
        Gpi_conj *= wcoulG[:,None,None]
        vk[:,kj] += contract('sngpi,sngpj->snij', Gpi_conj, Gpi)
    return vk

##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    vj = vk = None
    if kpts_band is not None and abs(kpt-kpts_band).max() > 1e-9:
        kpt = np.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(mydf, dm, hermi, kpt, kpts_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, dm, hermi, kpt, kpts_band)
        return vj, vk

    cell = mydf.cell
    log = logger.new_logger(mydf)
    dm = cp.asarray(dm, order='C')
    dms = _format_dms(dm, kpt.reshape(1, 3))
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = is_zero(kpt)
    k_real = is_zero(kpt) and not np.iscomplexobj(dms)

    mesh = mydf.mesh
    kptii = np.asarray((kpt,kpt))
    kpt_allow = np.zeros(3)

    if with_j:
        vjcoulG = mydf.weighted_coulG(kpt_allow, False, mesh)
        vj = cp.zeros((nset,nao,nao), dtype=np.complex128)
    if with_k:
        vkcoulG = mydf.weighted_coulG(kpt_allow, exxdiv, mesh)
        vk = cp.zeros((nset,nao,nao), dtype=np.complex128)

    # TODO: apply ft_opt.coeff to the dms; skip the AO ordering transformation
    # in ft_kern.
    bvk_kmesh = kpts_to_kmesh(cell, kptii)
    ft_opt = FTOpt(cell, bvk_kmesh=bvk_kmesh)
    ft_kern = ft_opt.gen_ft_kernel(verbose=log)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    avail_mem = get_avail_mem() * .8
    Gblksize = max(16, int(avail_mem/(16*nao**2*2)//8*8))
    Gblksize = min(Gblksize, ngrids, 16384)
    log.debug1('Gblksize = %d', Gblksize)

    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        Gpq = ft_kern(Gv[p0:p1], kpt_allow, kpt.reshape(1, 3))[0]
        if with_j:
            rho = contract('npq,Gpq->nG', dms.conj(), Gpq).conj()
            rho *= vjcoulG[p0:p1]
            vj += contract('nG,Gpq->npq', rho, Gpq)
        if with_k:
            Gpq_conj = Gpq.conj() * vkcoulG[p0:p1,None,None]
            tmp = contract('Gij,njk->nGik', Gpq, dms)
            vk += contract('nGik,Glk->nil', tmp, Gpq_conj)

    if with_j:
        if j_real:
            vj = vj.real
        vj = vj.reshape(dm.shape)
    if with_k:
        # Add ewald_exxdiv contribution because G=0 was not included in the
        # non-uniform grids
        if (exxdiv == 'ewald' and
            (cell.dimension < 2 or  # 0D and 1D are computed with inf_vacuum
             (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum'))):
            _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
        if k_real:
            vk = vk.real
        vk = vk.reshape(dm.shape)
    return vj, vk
