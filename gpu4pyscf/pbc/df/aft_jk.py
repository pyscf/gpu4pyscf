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
    'get_j_kpts', 'get_k_kpts', 'get_jk',
    'get_ej_ip1', 'get_ek_ip1'
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
from gpu4pyscf.pbc.df.ft_ao import FTOpt, libpbc
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks, _ewald_exxdiv_for_G0
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem, asarray
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.jk import apply_coeff_C_mat_CT, SHM_SIZE

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None):
    if kpts_band is not None:
        return get_j_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band)

    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)

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

def get_j_for_bands(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None):
    raise NotImplementedError
    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)
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

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               exxdiv=None):
    if kpts_band is not None:
        return get_k_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)

    is_single_kpt = kpts is not None and kpts.ndim == 1
    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)

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
        mo_coeff = cp.asarray(mo_coeff)
        mo_occ = cp.asarray(mo_occ)
        if is_single_kpt:
            if mo_coeff.ndim == 3:
                mo_coeff = mo_coeff[:,None]
                mo_occ = mo_occ[:,None]
            else:
                mo_coeff = mo_coeff[None]
                mo_occ = mo_occ[None]
        nocc = cp.count_nonzero(mo_occ > 0, axis=-1).max()
        if mo_coeff.ndim == 4:  # KUHF
            mo_coeff = mo_coeff[:,:,:,:nocc]
            occs = cp.array(mo_occ[:,:,:nocc], dtype=np.double)
            dm_factor = cp.array(mo_coeff, dtype=np.complex128, order='C')
            dm_factor *= cp.sqrt(occs)[:,:,None,:]
        else:  # KRHF
            mo_coeff = mo_coeff[None,:,:,:nocc]
            occs = cp.asarray(mo_occ[None,:,:nocc], dtype=np.double)
            dm_factor = cp.array(mo_coeff, dtype=np.complex128, order='C')
            dm_factor *= cp.sqrt(occs)[:,:,None,:]
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
        vkcoulG = mydf.weighted_coulG(kpt, exxdiv, mesh, kpts=kpts) * weight
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            log.debug3('update_vk [%s:%s]', p0, p1)
            Gpq = ft_kern(Gv[p0:p1], kpt, kpts)
            update_vk(vk_kpts, Gpq, dms, vkcoulG[p0:p1], ki_idx, kj_idx,
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

def get_k_for_bands(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None,
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
    if len(kj) == len(Gpq):
        idx = np.empty_like(ki)
        idx[kj] = ki
        tmp = contract('ngij,snjk->sngik', Gpq, dms)
        vk[:,idx] += contract('sngik,nglk->snil', tmp, Gpq_conj)
    else:
        # TODO: grouped gemm
        tmp = contract('ngij,snjk->sngik', Gpq[kj], dms[:,kj])
        vk[:,ki] += contract('sngik,nglk->snil', tmp, Gpq_conj[kj])

    if swap_2e:
        k_mask = k_to_compute[kptj_idx] == 1
        ki = kpti_idx[k_mask]
        kj = kptj_idx[k_mask]
        if len(ki) == len(Gpq):
            idx = np.empty_like(ki)
            idx[kj] = ki
            tmp = contract('ngij,snli->snglj', Gpq, dms[:,idx])
            vk += contract('nglk,snglj->snkj', Gpq_conj, tmp)
        else:
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
    if len(ki) == len(Gpq):
        idx = np.empty_like(ki)
        idx[kj] = ki
        Gpi = contract('ngij,snjp->sngpi', Gpq, dmf)
        if Gpi.dtype == np.float64:
            Gpi_conj = Gpi * wcoulG[:,None,None]
        else:
            Gpi_conj = Gpi.conj()
            Gpi_conj *= wcoulG[:,None,None]
        vk[:,idx] += contract('sngpi,sngpj->snij', Gpi, Gpi_conj)
    else:
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
        if len(ki) == len(Gpq):
            idx = np.empty_like(ki)
            idx[kj] = ki
            Gpi = contract('ngij,snip->sngpj', Gpq, dmf[:,idx].conj())
            Gpi_conj = Gpi.conj()
            Gpi_conj *= wcoulG[:,None,None]
            vk += contract('sngpi,sngpj->snij', Gpi_conj, Gpi)
        else:
            Gpi = contract('ngij,snip->sngpj', Gpq[kj], dmf[:,ki].conj())
            Gpi_conj = Gpi.conj()
            Gpi_conj *= wcoulG[:,None,None]
            vk[:,kj] += contract('sngpi,sngpj->snij', Gpi_conj, Gpi)
    return vk

def get_ej_ip1(mydf, dm, kpts=None):
    '''The first order energy derivatives from Coulomb matrix'''
    log = logger.new_logger(mydf)
    cell = mydf.cell
    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)
    is_gamma_point = is_zero(kpts)
    dms = _format_dms(dm, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    assert nkpts == len(kpts)
    if n_dm == 2:
        dms = dms[0] + dms[1]
    elif n_dm > 1:
        raise NotImplementedError

    ft_opt = FTOpt(cell, kpts).build()
    ft_kern = ft_opt.gen_ft_kernel()
    sorted_cell = ft_opt.sorted_cell
    dms = cp.asarray(dms.reshape(-1,nao,nao))
    dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, ft_opt.uniq_l_ctr,
                               ft_opt.l_ctr_offsets, ft_opt.ao_idx)
    if is_gamma_point:
        dms_bvkcell = cp.asarray(dms.real, order='C')
    else:
        expLk = cp.exp(1j*cp.asarray(ft_opt.bvkmesh_Ls).dot(cp.asarray(kpts).T))
        dms_bvkcell = contract('Lk,kpq->Lpq', expLk, dms)
        assert abs(dms_bvkcell.imag).max() < 1e-6
        dms_bvkcell = cp.asarray(dms_bvkcell.real, order='C')
        expLk = None

    bvk_ncells = np.prod(ft_opt.bvk_kmesh)
    nao = ft_opt.sorted_cell.nao
    Gv = cell.get_Gv(mydf.mesh)
    ngrids = len(Gv)
    # memory buffer required by ft_kern
    avail_mem = get_avail_mem() * .8
    blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
    blksize = min(blksize, ngrids, 16384)

    kpt_allow = np.zeros(3)
    wcoulG = mydf.weighted_coulG()

    bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = _generate_shl_pairs(ft_opt)
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    aft_envs = ft_opt.aft_envs

    lmax = ft_opt.uniq_l_ctr[:,0].max()
    ls = np.arange(lmax+1)
    gx_len = (ls[:,None]+2)*(ls+1) * 6*32
    nsp_per_block = np.ones_like(gx_len)
    for m in [2, 4, 8]:
        nsp_per_block[(gx_len + 3)*m*8 < SHM_SIZE] = m
    shm_size = (nsp_per_block * (gx_len + 3)).max() * 8

    log.debug('bas_ij_idx=%d nbatches=%d shm_size=%d blksize=%d',
              len(bas_ij_idx), nbatches_shl_pair, shm_size, blksize)

    kern = libpbc.PBC_ft_aopair_ej_ip1
    vG = cp.zeros(blksize+256, dtype=np.complex128)
    GvT = cp.zeros(3*blksize+256)
    ej = cp.zeros((cell.natm, 3))
    for p0, p1 in lib.prange(0, ngrids, blksize):
        nGv = p1 - p0
        # TODO: Gpq are transformed to the k-points adapted representation in
        # gen_ft_kernel. This transfomration can be skipped.
        Gpq = ft_kern(Gv[p0:p1], kpt_allow, kpts, transform_ao=False)
        Gpq = Gpq.transpose(0,2,3,1)
        vG[:nGv] = contract('kji,kijg->g', dms, Gpq).conj()
        vG[:nGv] *= wcoulG[p0:p1]
        GvT[:3*nGv].set(Gv[p0:p1].T.ravel())
        Gpq = None
        err = kern(
            ctypes.cast(ej.data.ptr, ctypes.c_void_p),
            ctypes.cast(dms_bvkcell.data.ptr, ctypes.c_void_p),
            ctypes.cast(vG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aft_envs),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.c_int(nGv),
            ctypes.c_int(shm_size),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
            sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
            sorted_cell._env.ctypes)
        if err != 0:
            raise RuntimeError('PBC_ft_aopair_ej_ip1 failed')
    ej = ej.get()
    ej /= nkpts**2
    return ej

def get_ek_ip1(mydf, dm, kpts=None, exxdiv=None):
    '''The first order energy derivatives from exact exchange'''
    log = logger.new_logger(mydf)
    cpu0 = cpu1 = log.init_timer()
    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)
    is_gamma_point = is_zero(kpts)
    cell = mydf.cell
    dms = _format_dms(dm, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    assert nkpts == len(kpts)
    if n_dm > 2:
        raise NotImplementedError

    ft_opt = FTOpt(cell, kpts).build()
    ft_kern = ft_opt.gen_ft_kernel()
    sorted_cell = ft_opt.sorted_cell
    dms = cp.asarray(dms.reshape(-1,nao,nao))
    dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, ft_opt.uniq_l_ctr,
                               ft_opt.l_ctr_offsets, ft_opt.ao_idx)
    nao = dms.shape[-1]
    dms = dms.reshape(n_dm,nkpts,nao,nao)

    if not is_gamma_point:
        expLk = cp.exp(1j*cp.asarray(ft_opt.bvkmesh_Ls).dot(cp.asarray(kpts).T))

    bvk_ncells = np.prod(ft_opt.bvk_kmesh)
    Gv = cell.get_Gv(mydf.mesh)
    ngrids = len(Gv)
    # memory buffer required by ft_kern
    avail_mem = get_avail_mem() * .8
    blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
    blksize = min(blksize, ngrids, 16384)

    bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = _generate_shl_pairs(ft_opt)
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    aft_envs = ft_opt.aft_envs

    lmax = ft_opt.uniq_l_ctr[:,0].max()
    ls = np.arange(lmax+1)
    gx_len = (ls[:,None]+2)*(ls+1) * 6*32
    nsp_per_block = np.ones_like(gx_len)
    for m in [2, 4, 8]:
        nsp_per_block[(gx_len + 3)*m*8 < SHM_SIZE] = m
    shm_size = (nsp_per_block * (gx_len + 3)).max() * 8

    log.debug('bas_ij_idx=%d nbatches=%d shm_size=%d blksize=%d',
              len(bas_ij_idx), nbatches_shl_pair, shm_size, blksize)

    kern = libpbc.PBC_ft_aopair_ek_ip1
    GvT = cp.zeros(3*blksize+256)
    ek = cp.zeros((cell.natm, 3))
    for group_id, (kpt, ki_idx, kj_idx, self_conj) \
            in enumerate(kk_adapted_iter(cell, kpts)):
        wcoulG = mydf.weighted_coulG(kpt, exxdiv, mydf.mesh, kpts=kpts)
        swap_2e = not self_conj
        for p0, p1 in lib.prange(0, ngrids, blksize):
            nGv = p1 - p0
            #:Gpq = ft_kern(Gv[p0:p1], kpt, kpts, transform_ao=False)
            #:Gpq = Gpq.transpose(0,2,3,1)
            #:Gpq_conj = Gpq.conj()
            # Gpq.conj() can be computed equivalently as
            Gpq_conj = ft_kern(-Gv[p0:p1], -kpt, -kpts, transform_ao=False)
            Gpq_conj = Gpq_conj.transpose(0,2,3,1)

            if is_gamma_point:
                tmp = contract('sjk,lkg->sjlg', dms[:,0], Gpq_conj[0])
                dm_vG = contract('sjlg,sli->jig', tmp, dms[:,0])
            else:
                # einsum(nijG[kj_idx],jk[kj_idx],nlkG*[kj_idx],li[ki_idx])
                # apply derivatives to nlkG*
                #:tmp = contract('nijg,snjk->snikg', Gpq[kj_idx], dms[:,kj_idx])
                #:tmp = contract('snikg,snli->nklg', tmp, dms[:,ki_idx])
                #:dm_vG = contract('Lk,kpqg->Lpqg', expLk[:,kj_idx].conj(), tmp).conj()
                #:if swap_2e:
                # apply derivatives to nijG. This term is equivalent to the
                # derivatives of nlkG*.
                #:    tmp = contract('snjk,nlkg->snjlg', dms[:,kj_idx], Gpq.conj()[kj_idx])
                #:    tmp = contract('snjlg,snli->njig', tmp, dms[:,ki_idx])
                #:    dm_vG += contract('Lk,kpqg->Lpqg', expLk[:,kj_idx], tmp)
                idx = np.empty_like(ki_idx)
                idx[kj_idx] = ki_idx
                tmp = contract('snjk,nlkg->snjlg', dms, Gpq_conj)
                tmp = contract('snjlg,snli->njig', tmp, dms[:,idx])
                dm_vG = contract('Lk,kpqg->Lpqg', expLk, tmp)
            if swap_2e:
                dm_vG *= wcoulG[p0:p1] * 2
            else:
                dm_vG *= wcoulG[p0:p1]
            dm_vG = cp.asarray(dm_vG, order='C')

            GvT[:3*nGv].set((Gv[p0:p1]+kpt).T.ravel())
            err = kern(
                ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(nbatches_shl_pair),
                ctypes.c_int(nGv),
                ctypes.c_int(shm_size),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
                sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
                sorted_cell._env.ctypes)
            Gpq_conj = tmp = dm_vG = None
            if err != 0:
                raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
        cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)
    ek = ek.get()
    if not is_gamma_point:
        ek /= nkpts**2
    log.timer_debug1('get_ek_ip1', *cpu0)
    return ek

def _generate_shl_pairs(ft_opt):
    img_idx_cache = ft_opt.make_img_idx_cache(permutation_symmetry=True)
    bas_ij_idx = []
    bas_ij_img_idx = []
    shl_pair_offsets = []
    sp0 = sp1 = 0
    for i, j in img_idx_cache:
        bas_ij, img_offsets, img_idx = img_idx_cache[i, j]
        # bas_ij is the pair indices within the bvk_cell.
        img_counts = img_offsets[1:] - img_offsets[:-1]
        bas_ij = cp.asarray(np.repeat(bas_ij.get(), img_counts.get()))
        bas_ij_idx.append(bas_ij)
        bas_ij_img_idx.append(img_idx)
        sp0, sp1 = sp1, sp1 + len(bas_ij)
        shl_pair_offsets.append(cp.arange(sp0, sp1, 32, dtype=np.int32))
    shl_pair_offsets.append(np.int32(sp1))
    bas_ij_idx = cp.hstack(bas_ij_idx, dtype=np.int32)
    bas_ij_img_idx = cp.hstack(bas_ij_img_idx, dtype=np.int32)
    shl_pair_offsets = cp.hstack(shl_pair_offsets, dtype=np.int32)
    return bas_ij_idx, bas_ij_img_idx, shl_pair_offsets

def get_ej_strain_deriv(mydf, dm, kpts=None, omega=None):
    '''Strain derivatives from Coulomb matrix'''
    from gpu4pyscf.pbc.grad import rks_stress
    log = logger.new_logger(mydf)
    cell = mydf.cell
    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)
    is_gamma_point = is_zero(kpts)
    dms = _format_dms(dm, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    assert nkpts == len(kpts)
    if n_dm == 2:
        dms = dms[0] + dms[1]
    elif n_dm > 1:
        raise NotImplementedError

    ft_opt = FTOpt(cell, kpts).build()
    ft_kern = ft_opt.gen_ft_kernel()
    sorted_cell = ft_opt.sorted_cell
    dms = cp.asarray(dms.reshape(-1,nao,nao))
    dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, ft_opt.uniq_l_ctr,
                               ft_opt.l_ctr_offsets, ft_opt.ao_idx)
    if is_gamma_point:
        dms_bvkcell = cp.asarray(dms.real, order='C')
    else:
        expLk = cp.exp(1j*cp.asarray(ft_opt.bvkmesh_Ls).dot(cp.asarray(kpts).T))
        dms_bvkcell = contract('Lk,kpq->Lpq', expLk, dms)
        assert abs(dms_bvkcell.imag).max() < 1e-6
        dms_bvkcell = cp.asarray(dms_bvkcell.real, order='C')
        expLk = None

    bvk_ncells = np.prod(ft_opt.bvk_kmesh)
    nao = ft_opt.sorted_cell.nao
    Gv = cell.get_Gv(mydf.mesh)
    ngrids = len(Gv)
    # memory buffer required by ft_kern
    avail_mem = get_avail_mem() * .8
    blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
    blksize = min(blksize, ngrids, 16384)

    kpt_allow = np.zeros(3)
    coulG_0, coulG_1 = rks_stress._get_coulG_strain_derivatives(cell, Gv, omega=omega)
    coulG_0 = asarray(coulG_0)
    coulG_1 = asarray(coulG_1)
    weight_0 = 1/cell.vol
    weight_1 = -1/cell.vol * cp.eye(3)
    wcoulG_0 = weight_0 * coulG_0
    # wcoulG_1 includes two terms, weight_0*coulG_1 + weight_1*coulG_0
    wcoulG_1 = weight_0 * coulG_1
    wcoulG_1 += weight_1[:,:,None] * coulG_0

    bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = _generate_shl_pairs(ft_opt)
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    aft_envs = ft_opt.aft_envs

    lmax = ft_opt.uniq_l_ctr[:,0].max()
    ls = np.arange(lmax+1)
    gx_len = (ls[:,None]+2)*(ls+1) * 6*32
    nsp_per_block = np.ones_like(gx_len)
    for m in [2, 4, 8]:
        nsp_per_block[(gx_len + 6)*m*8 < SHM_SIZE] = m
    shm_size = (nsp_per_block * (gx_len + 6)).max() * 8

    log.debug('bas_ij_idx=%d nbatches=%d shm_size=%d blksize=%d',
              len(bas_ij_idx), nbatches_shl_pair, shm_size, blksize)

    kern = libpbc.PBC_ft_aopair_ej_strain_deriv
    vG = cp.zeros(blksize+256, dtype=np.complex128)
    GvT = cp.zeros(3*blksize+256)
    ej = cp.zeros((cell.natm, 3))
    sigma = cp.zeros((3, 3))
    for p0, p1 in lib.prange(0, ngrids, blksize):
        nGv = p1 - p0
        # TODO: Gpq are transformed to the k-points adapted representation in
        # gen_ft_kernel. This transfomration can be skipped.
        Gpq = ft_kern(Gv[p0:p1], kpt_allow, kpts, transform_ao=False)
        Gpq = Gpq.transpose(0,2,3,1)
        rhoG = contract('kji,kijg->g', dms, Gpq)
        sigma += .25*cp.einsum('xyg,g,g->xy', wcoulG_1[:,:,p0:p1], rhoG.conj(), rhoG).real

        vG[:nGv] = rhoG.conj()
        vG[:nGv] *= wcoulG_0[p0:p1]
        GvT[:3*nGv].set(Gv[p0:p1].T.ravel())
        Gpq = None
        err = kern(
            ctypes.cast(ej.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma.data.ptr, ctypes.c_void_p),
            ctypes.cast(dms_bvkcell.data.ptr, ctypes.c_void_p),
            ctypes.cast(vG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aft_envs),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.c_int(nGv),
            ctypes.c_int(shm_size),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
            sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
            sorted_cell._env.ctypes)
        if err != 0:
            raise RuntimeError('PBC_ft_aopair_ej_strain_deriv failed')
    ej = ej.get()
    if not is_gamma_point:
        ej /= nkpts**2
    sigma = sigma.get()
    sigma *= 2 / nkpts**2
    return sigma

def get_ek_strain_deriv(mydf, dm, kpts=None, exxdiv=None, omega=None):
    '''Strain derivatives from exact exchange'''
    from gpu4pyscf.pbc.grad import rks_stress
    log = logger.new_logger(mydf)
    cpu0 = cpu1 = log.init_timer()
    if kpts is None:
        kpts = np.zeros((1,3))
    else:
        kpts = kpts.reshape(-1, 3)
    is_gamma_point = is_zero(kpts)
    cell = mydf.cell
    dm0 = _format_dms(dm, kpts)
    n_dm, nkpts, nao = dm0.shape[:3]
    assert nkpts == len(kpts)
    if n_dm > 2:
        raise NotImplementedError

    ft_opt = FTOpt(cell, kpts).build()
    ft_kern = ft_opt.gen_ft_kernel()
    sorted_cell = ft_opt.sorted_cell
    dms = cp.asarray(dm0.reshape(-1,nao,nao))
    dms = apply_coeff_C_mat_CT(dms, cell, sorted_cell, ft_opt.uniq_l_ctr,
                               ft_opt.l_ctr_offsets, ft_opt.ao_idx)
    nao = dms.shape[-1]
    dms = dms.reshape(n_dm,nkpts,nao,nao)

    if not is_gamma_point:
        expLk = cp.exp(1j*cp.asarray(ft_opt.bvkmesh_Ls).dot(cp.asarray(kpts).T))

    bvk_ncells = np.prod(ft_opt.bvk_kmesh)
    Gv = cell.get_Gv(mydf.mesh)
    ngrids = len(Gv)
    # memory buffer required by ft_kern
    avail_mem = get_avail_mem() * .8
    blksize = max(16, int(avail_mem/(nao**2*bvk_ncells*16*2))//16*16)
    blksize = min(blksize, ngrids, 16384)

    bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = _generate_shl_pairs(ft_opt)
    nbatches_shl_pair = len(shl_pair_offsets) - 1
    aft_envs = ft_opt.aft_envs

    lmax = ft_opt.uniq_l_ctr[:,0].max()
    ls = np.arange(lmax+1)
    gx_len = (ls[:,None]+2)*(ls+1) * 6*32
    nsp_per_block = np.ones_like(gx_len)
    for m in [2, 4, 8]:
        nsp_per_block[(gx_len + 3)*m*8 < SHM_SIZE] = m
    shm_size = (nsp_per_block * (gx_len + 3)).max() * 8

    log.debug('bas_ij_idx=%d nbatches=%d shm_size=%d blksize=%d',
              len(bas_ij_idx), nbatches_shl_pair, shm_size, blksize)

    kern = libpbc.PBC_ft_aopair_ek_strain_deriv
    GvT = cp.zeros(3*blksize+256)
    ek = cp.zeros((cell.natm, 3))
    sigma = cp.zeros((3, 3))
    sigma1 = cp.zeros((3, 3))
    for group_id, (kpt, ki_idx, kj_idx, self_conj) \
            in enumerate(kk_adapted_iter(cell, kpts)):
        Gvk = Gv + kpt
        coulG_0, coulG_1 = rks_stress._get_coulG_strain_derivatives(
            cell, Gvk, omega=omega, remove_G0=is_zero(kpt))
        coulG_0 = asarray(coulG_0)
        coulG_1 = asarray(coulG_1)
        weight_0 = 1/cell.vol
        weight_1 = -1/cell.vol * cp.eye(3)
        wcoulG_0 = weight_0 * coulG_0
        wcoulG_1 = weight_0 * coulG_1
        wcoulG_1 += weight_1[:,:,None] * coulG_0

        swap_2e = not self_conj
        for p0, p1 in lib.prange(0, ngrids, blksize):
            nGv = p1 - p0
            Gpq = ft_kern(Gv[p0:p1], kpt, kpts, transform_ao=False)
            Gpq = Gpq.transpose(0,2,3,1)
            Gpq_conj = Gpq.conj()
            # Gpq.conj() can be computed equivalently as
            #Gpq_conj = ft_kern(-Gv[p0:p1], -kpt, -kpts, transform_ao=False)
            #Gpq_conj = Gpq_conj.transpose(0,2,3,1)

            if is_gamma_point:
                tmp = contract('sjk,lkg->sjlg', dms[:,0], Gpq_conj[0])
                dm_vG = contract('sjlg,sli->jig', tmp, dms[:,0])
                vkG = cp.einsum('pqg,qpg->g', dm_vG, Gpq[0]).real
                sigma += cp.einsum('xyg,g->xy', wcoulG_1[:,:,p0:p1], vkG)
            else:
                # einsum(nijG[kj_idx],jk[kj_idx],nlkG*[kj_idx],li[ki_idx])
                # apply derivatives to nlkG*
                #:tmp = contract('nijg,snjk->snikg', Gpq[kj_idx], dms[:,kj_idx])
                #:tmp = contract('snikg,snli->nklg', tmp, dms[:,ki_idx])
                #:dm_vG = contract('Lk,kpqg->Lpqg', expLk[:,kj_idx].conj(), tmp).conj()
                #:if swap_2e:
                # apply derivatives to nijG. This term is equivalent to the
                # derivatives of nlkG*.
                #:    tmp = contract('snjk,nlkg->snjlg', dms[:,kj_idx], Gpq.conj()[kj_idx])
                #:    tmp = contract('snjlg,snli->njig', tmp, dms[:,ki_idx])
                #:    dm_vG += contract('Lk,kpqg->Lpqg', expLk[:,kj_idx], tmp)
                idx = np.empty_like(ki_idx)
                idx[kj_idx] = ki_idx
                dm_k = contract('snjk,nlkg->snjlg', dms, Gpq_conj)
                dm_k = contract('snjlg,snli->njig', dm_k, dms[:,idx])
                dm_vG = contract('Lk,kpqg->Lpqg', expLk, dm_k)

                vkG = cp.einsum('njig,nijg->g', dm_k, Gpq).real
                tmp = cp.einsum('xyg,g->xy', wcoulG_1[:,:,p0:p1], vkG)
                if swap_2e:
                    sigma += tmp * 2
                else:
                    sigma += tmp

            if swap_2e:
                dm_vG *= wcoulG_0[p0:p1] * 2
            else:
                dm_vG *= wcoulG_0[p0:p1]
            dm_vG = cp.asarray(dm_vG, order='C')

            GvT[:3*nGv].set(Gvk[p0:p1].T.ravel())
            err = kern(
                ctypes.cast(ek.data.ptr, ctypes.c_void_p),
                ctypes.cast(sigma1.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(nbatches_shl_pair),
                ctypes.c_int(nGv),
                ctypes.c_int(shm_size),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                sorted_cell._atm.ctypes, ctypes.c_int(sorted_cell.natm),
                sorted_cell._bas.ctypes, ctypes.c_int(sorted_cell.nbas),
                sorted_cell._env.ctypes)
            Gpq_conj = tmp = dm_vG = None
            if err != 0:
                raise RuntimeError('PBC_ft_aopair_ek_strain_deriv failed')
        cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)
    ek = ek.get()
    if not is_gamma_point:
        ek /= nkpts**2
    sigma *= .5 / nkpts**2
    # First *2 due to i>=j symmetry in kernel;
    # second *2 due to (d/dX ij|kl) + (ij|d/dX kl)
    sigma1 *= .5 * 2 * 2 / nkpts**2
    sigma += sigma1
    sigma = sigma.get()

    if (exxdiv == 'ewald' and
        (cell.dimension == 3 or
         (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
        from pyscf.pbc.tools.pbc import madelung
        from gpu4pyscf.pbc.gto import int1e
        int1e_opt = int1e._Int1eOpt(cell, kpts)
        s0 = int1e_opt.intor('PBCint1e_ovlp', 1, 1, (0, 0))
        k_dm = contract('nkpq,kqr->nkpr', dm0, s0)
        k_dm = contract('nkpr,nkrs->kps', k_dm, dm0)
        ek_G0 = .5 * cp.einsum('kij,kji->', s0, k_dm).real.get() / nkpts**2

        scaled_kpts = kpts.dot(cell.lattice_vectors().T)
        ewald_G0 = np.empty((3,3))
        disp = max(1e-5, (cell.precision*.1)**.5)
        for i in range(3):
            for j in range(i+1):
                cell1, cell2 = rks_stress._finite_diff_cells(cell, i, j, disp)
                kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
                kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
                e1 = nkpts * madelung(cell1, kpts1, omega=omega)
                e2 = nkpts * madelung(cell2, kpts2, omega=omega)
                ewald_G0[j,i] = ewald_G0[i,j] = (e1-e2)/(2*disp)
        ewald_G0 *= ek_G0
        int1e_opt = int1e._Int1eOptV2(cell)
        ewald_G0 += int1e_opt.get_ovlp_strain_deriv(k_dm, kpts) * madelung(cell, kpts, omega=omega)
        sigma += ewald_G0

    log.timer_debug1('get_ek_ip1', *cpu0)
    return sigma

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
