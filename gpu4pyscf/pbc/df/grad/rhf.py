# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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
Functions for computing nuclear gradients and strain derivatives
'''

import math
import ctypes
import copy
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, transpose_sum, get_avail_mem, empty_aligned)
from gpu4pyscf.gto.mole import RysIntEnvVars, _scale_sp_ctr_coeff
from gpu4pyscf.df.int3c2e_bdiv import _split_l_ctr_pattern, get_ao_pair_loc
from gpu4pyscf.df.grad.rhf import factorize_dm
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, POOL_SIZE, MAX_IMGS_PER_TASK, int3c2e_scheme,
    _get_shl_pair_per_block, _counts_to_offsets)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.rsdf_builder import decompose_j2c, LINEAR_DEP_THR
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.gto.cell import get_Gv_weights
from gpu4pyscf.pbc.grad.krks_stress import (
    _get_weighted_coulG_strain_derivatives as get_wcoulG)
from gpu4pyscf.pbc.tools.pbc import madelung
from gpu4pyscf.__config__ import props as gpu_specs


def _gen_metric_solver(j2c, linear_dep_threshold=LINEAR_DEP_THR,
                       dimension=3):
    '''Generate a pseudo-inverse solver consistent with the CDERI metric.'''
    metric, metric_negative, _ = decompose_j2c(
        j2c, prefer_ed=True, linear_dep_threshold=linear_dep_threshold)
    if metric_negative is not None and dimension != 2:
        raise RuntimeError(
            'Negative auxiliary metric is only supported for 2D systems')
    naux = j2c.shape[0]

    def solve(rhs):
        shape = rhs.shape
        rhs = rhs.reshape(naux, -1)
        out = metric.dot(metric.conj().T.dot(rhs))
        if metric_negative is not None:
            out -= metric_negative.dot(metric_negative.conj().T.dot(rhs))
        return out.reshape(shape)
    return solve

def _get_ejk_derivatives(int3c2e_opt, dm, hermi=0, j_factor=1., k_factor=1.,
                         exxdiv=None, omega=None, verbose=None,
                         linear_dep_threshold=LINEAR_DEP_THR):
    '''
    Computes the first-order derivatives (nuclear gradients and strain
    derivatives) of the energy contributions from J and K terms per atom.
    '''
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        ej_sigma = _get_ej_derivatives(
            int3c2e_opt, dm, hermi, omega, verbose, linear_dep_threshold)
        return ej_sigma * j_factor

    assert hermi == 1 or hermi == 2
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    assert bvk_ncells == 1, 'Gamma derivatives require a one-cell BvK mesh'
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = factorize_dm(dm, hermi)
    # transform to the AO order in sorted_cell
    dm_factor_l = cell.apply_C_dot(dm_factor_l, axis=0)
    assert dm_factor_l.dtype == np.float64
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l
    else:
        dm_factor_r = cell.apply_C_dot(dm_factor_r, axis=0)
    log.debug1('dm_factor shape %s', dm_factor_l.shape)
    nao, nocc = dm_factor_l.shape
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)
    compact_idx = pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    nao_pair = n_compact_pairs = len(pair_addresses)
    dd_ft_opt = int3c2e_opt.dd_ft_opt
    separated_dd = dd_ft_opt is not None
    if separated_dd:
        dd_ao_idx, dd_diag = dd_ft_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)
        pair_addresses = cp.hstack([compact_idx, dd_ao_idx], dtype=np.int32)
        diag_idx = cp.hstack([diag_idx, n_compact_pairs + dd_diag])
        nao_pair = len(pair_addresses)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free
    mem_avail -= naux*nocc**2 * 8  # j3c_oo
    batch_size = max(1, min(naux, int(mem_avail*.5/(max(1, n_compact_pairs)*8*bvk_ncells))))
    blksize = max(1, min(naux, int(mem_avail*.4/(nao**2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    def sr_int3c2e():
        eval_j3c, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
            aux_batch_size=None if batch_size >= naux else batch_size, cart=True)
        aux_batches = len(aux_offsets) - 1

        i_addr, j_addr = divmod(compact_idx, nao)
        aux0 = aux1 = 0
        j3c_full = cp.zeros((nao, nao, blksize))
        max_aux_batch = int(np.diff(aux_offsets).max())
        buf = cp.empty((max_aux_batch, n_compact_pairs))
        buf1 = cp.empty((blksize, nocc, nao))
        j3c_oo = cp.empty((naux, nocc, nocc))
        for kbatch in range(aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)[:,0,:]
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:,:,:dk]
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
                tmp = ndarray((nocc, nao, dk), buffer=buf1)
                contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
        j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
        return j3c_oo

    if n_compact_pairs > 0:
        j3c_oo = sr_int3c2e()
        t0 = log.timer_debug1('contract sr_int3c2e dm', *t0)
    else:
        j3c_oo = cp.zeros((naux, nocc, nocc))

    # Adjust the rcut because the default cell.rcut is estimated based on
    # overlap integrals.
    rcut = _estimate_sr_2c2e_rcut(
        auxcell, int3c2e_opt.omega, auxcell.precision*1e-6)
    with lib.temporary_env(auxcell, rcut=rcut):
        int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False, omega=-int3c2e_opt.omega)

    ################################
    # LR part 0th order
    mesh = int3c2e_opt.mesh
    log.debug('mesh for LR coulG %s', mesh)

    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
    mesh = int3c2e_opt.mesh
    Gv, _, kws = get_Gv_weights(cell, mesh)
    ngrids = len(Gv)
    wcoulG_LR0, wcoulG_LR1 = get_wcoulG(cell, Gv, int3c2e_opt.omega)
    if omega != 0:
        wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gv, omega)
        wcoulG_LR0 -= wcoulG_0
        wcoulG_LR1 -= wcoulG_1
    wcoulG_SR_at_G0 = np.pi / int3c2e_opt.omega**2 * kws
    wcoulG_LR0[0] -= wcoulG_SR_at_G0
    wcoulG_LR1[:,:,0] += wcoulG_SR_at_G0 * cp.eye(3)

    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    eval_compact = None
    if n_compact_pairs > 0:
        eval_compact = ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]
    if separated_dd:
        wcoulG_FR0, wcoulG_FR1 = get_wcoulG(cell, Gv, -omega)

        eval_dd = dd_ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]

    def eval_ft(Gv, out=None):
        # Preserve compact/DD column ordering using each partition's own
        # shell, image and AO offsets through the existing FT interface.
        result = ndarray((nao_pair, len(Gv)), dtype=np.complex128, buffer=out)
        if n_compact_pairs > 0:
            eval_compact(Gv, out=result[:n_compact_pairs])
        if separated_dd:
            eval_dd(Gv, out=result[n_compact_pairs:])
        return result

    def lr_3c2e(j3c_oo):
        i_addr, j_addr = divmod(pair_addresses, nao)
        unit = max(nao**2, naux)  # buf: pqG
        unit += max(nao*nocc, naux)  # buf1: tmp / auxGw
        unit += naux  # buf2: auxG
        unit += nao_pair  # buf3: pqG_compressed
        Gblksize = int(mem_avail*.8//(unit*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)
        buf  = cp.empty(max(nao**2,naux)*Gblksize, dtype=np.complex128)
        buf1 = cp.empty(max(nao*nocc, naux)*Gblksize, dtype=np.complex128)
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        buf3 = cp.empty(nao_pair*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf2).T
            auxGw = ndarray((naux, nGv), dtype=np.complex128, buffer=buf1)
            cp.multiply(auxG, wcoulG_LR0[p0:p1], out=auxGw)
            contract('iG,jG->ij', auxG.view(np.float64), auxGw.view(np.float64),
                     beta=1, out=j2c)
            auxG = auxG.view(np.float64)

            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            pqG_compressed = eval_ft(Gv[p0:p1], out=buf3)
            pqG_compressed[:n_compact_pairs] *= wcoulG_LR0[p0:p1]
            if separated_dd:
                pqG_compressed[n_compact_pairs:] *= wcoulG_FR0[p0:p1]
            pqG = ndarray((nao,nao,nGv), dtype=np.complex128, buffer=buf)
            symmetric_scatter(pqG, i_addr, j_addr, pqG_compressed)
            pqG = pqG.view(np.float64).reshape(nao,nao,nGv*2)
            tmp = ndarray((nocc,nao,nGv*2), buffer=buf1)
            ijG = ndarray((nocc,nocc,nGv*2), buffer=buf)
            contract('pqG,pi->iqG', pqG, dm_factor_r, out=tmp)
            contract('iqG,qj->ijG', tmp, dm_factor_l, out=ijG)
            contract('rG,ijG->rij', auxG, ijG, beta=1, out=j3c_oo)
        return j3c_oo
    j3c_oo = lr_3c2e(j3c_oo)
    t0 = log.timer_debug1('contract lr_int3c2e dm', *t0)

    ################################
    # (d/dX P|Q) contributions
    j2c = auxcell.apply_CT_mat_C(j2c)
    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        aux_coeff = cp.asarray(auxcell.ctr_coeff)
        solve_j2c = _gen_metric_solver(
            j2c, linear_dep_threshold, auxcell.dimension)
        metric = aux_coeff.dot(solve_j2c(aux_coeff.T))
    j2c = aux_coeff = solve_j2c = None
    dm_oo = j3c_oo
    occ_blksize = min(nocc, int(mem_avail*.4//(naux*nocc*8)))
    assert occ_blksize > 0
    for p0, p1 in lib.prange(0, nocc, occ_blksize):
        tmp = contract('uv,vij->uij', metric, dm_oo[:,p0:p1])
        dm_oo[:,p0:p1] = tmp
        tmp = None
    metric = j3c_oo = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)
        dm_sorted = dm_factor_l.dot(dm_factor_r.T)

    if j_factor == 0:
        dm_aux = None
    else:
        dm_aux = auxvec[:,None] * auxvec
    # dm_aux should be symmetric
    dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                      alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
    # ejk = .5 * contract_h1e_dm(auxcell, auxcell.pbc_intor('int2c2e_ip1'), dm_aux)
    ejk_sigma = int2c2e_opt.energy_derivatives(dm_aux, omega=-int3c2e_opt.omega)
    ejk_sigma = cp.asarray(-ejk_sigma)
    t0 = log.timer_debug1('contract int2c2e_deriv', *t0)

    ################################
    # LR part response
    def lr_3c2e_response():
        aft_envs = ft_opt.aft_envs
        aux_ft_envs = RysIntEnvVars.new(
            auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
            _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)

        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = \
                aft_jk._shl_pairs_for_derivative_kernel(ft_opt)
        if separated_dd:
            dd_bas_ij_idx, dd_bas_ij_img_idx, dd_shl_pair_offsets = \
                    aft_jk._shl_pairs_for_derivative_kernel(dd_ft_opt)
            bas_ij_idx = cp.hstack([bas_ij_idx, dd_bas_ij_idx])
            bas_ij_img_idx = cp.hstack([bas_ij_img_idx, dd_bas_ij_img_idx])
            shl_pair_offsets = cp.hstack([
                shl_pair_offsets[:-1], shl_pair_offsets[-1] + dd_shl_pair_offsets])

        i_addr, j_addr = divmod(pair_addresses, nao)
        # The derivative kernel reads (j,i), whereas compressed FT tensors
        # store (i,j). Weight only the entries actually read by the kernel.
        response_idx = j_addr * nao + i_addr

        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        unit = max(nao**2, naux)  # buf: pqG
        unit += max(nao*nocc, naux)  # buf1: tmp / auxGw
        unit += naux  # buf2: auxG_conj / dm_auxG
        unit += nao_pair  # buf3: pqG_compressed
        unit += naux  # buf_auxG
        Gblksize = int(mem_avail*.8//(unit*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        ejk_sigma_lr = cp.zeros([cell.natm+3, 3])
        ejk_sigma_aux = cp.zeros([cell.natm+3, 3])
        sigma_G = cp.zeros((3, 3))

        kern = libpbc.PBC_ft_aopair_ek_deriv
        kern_auxG = libpbc.PBC_ft_ao_deriv
        null_ptr = lib.c_null_ptr()
        buf  = cp.empty(max(nao**2,naux)*Gblksize, dtype=np.complex128)
        buf1 = cp.empty(max(nao*nocc, naux)*Gblksize, dtype=np.complex128)
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        buf3 = cp.empty(nao_pair*Gblksize, dtype=np.complex128)
        buf_auxG = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            GvT = cp.asarray(Gv[p0:p1].T, order='C')
            auxG = ft_ao.ft_ao(auxcell, GvT.T, out=buf_auxG).T

            # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
            auxG_conj = ndarray((naux, nGv), dtype=np.complex128, buffer=buf2)
            auxG_conj = cp.conj(auxG, out=auxG_conj)
            auxG_conj = auxG_conj.view(np.float64)

            # Note: PBC_ft_aopair_ek_deriv kernel only processes the tril part.
            # dm_oo must be symmetric
            dm_vG = ndarray((nao,nao,nGv*2), buffer=buf)
            dm_ooG = ndarray((nocc**2, nGv*2), buffer=buf)
            tmp = ndarray((nocc,nao,nGv*2), buffer=buf1)
            dm_oo.reshape(naux, nocc*nocc).T.dot(auxG_conj, out=dm_ooG)
            dm_ooG = dm_ooG.reshape(nocc,nocc,nGv*2)
            contract('jiG,qi->jqG', dm_ooG, dm_factor_r, out=tmp)
            beta = 0
            if j_factor != 0:
                vG = auxvec.dot(auxG_conj)
                cp.multiply(dm_sorted[:,:,None], vG, out=dm_vG)
                beta = j_factor
            contract('jqG,pj->pqG', tmp, dm_factor_l, -.5*k_factor, beta, out=dm_vG)
            dm_vG = dm_vG.view(np.complex128).reshape(nao*nao, nGv)
            dm_vG_compressed = cp.take(
                dm_vG, pair_addresses, axis=0,
                out=ndarray((nao_pair, nGv), dtype=np.complex128, buffer=buf3))

            if n_compact_pairs > 0:
                indexed_scale(dm_vG, response_idx[:n_compact_pairs], wcoulG_LR0[p0:p1])
            if separated_dd:
                indexed_scale(dm_vG, response_idx[n_compact_pairs:], wcoulG_FR0[p0:p1])
            err = kern(
                ctypes.cast(ejk_sigma_lr[:-3].data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_sigma_lr[-3:].data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(len(shl_pair_offsets) - 1),
                ctypes.c_int(nGv),
                ctypes.c_int(shm_size),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ft_opt.permutation_symmetry))
            if err != 0:
                raise RuntimeError('PBC_ft_aopair_ek_deriv failed')

            pqG_compressed = eval_ft(Gv[p0:p1], out=buf)
            dm_vG_compressed[diag_idx] *= .5
            vG = contract('rg,rg->g', pqG_compressed[:n_compact_pairs],
                          dm_vG_compressed[:n_compact_pairs]).real
            sigma_G += 2 * cp.einsum('g,xyg->xy', vG, wcoulG_LR1[:,:,p0:p1])
            if separated_dd:
                vG = contract('rg,rg->g', pqG_compressed[n_compact_pairs:],
                              dm_vG_compressed[n_compact_pairs:]).real
                sigma_G += 2 * cp.einsum('g,xyg->xy', vG, wcoulG_FR1[:,:,p0:p1])

            # (ij|r)^{[0]} * metric * (r|G)^{[1]} (ji|G)^{[0]}
            pqG_swap = dm_vG_compressed # swap pqG_compressed, to release buf
            cp.multiply(pqG_compressed[:n_compact_pairs], wcoulG_LR0[p0:p1],
                        out=pqG_swap[:n_compact_pairs])
            if separated_dd:
                cp.multiply(pqG_compressed[n_compact_pairs:], wcoulG_FR0[p0:p1],
                            out=pqG_swap[n_compact_pairs:])
            pqGw = ndarray((nao,nao,nGv), dtype=np.complex128, buffer=buf)
            symmetric_scatter(pqGw, i_addr, j_addr, pqG_swap)
            pqGw = pqGw.view(np.float64).reshape(nao,nao,nGv*2)

            beta = 0
            dm_auxG = ndarray((naux,nGv*2), buffer=buf2)
            if j_factor != 0:
                rhoGz = contract('pqG,qp->G', pqGw, dm_sorted)
                cp.multiply(auxvec[:,None], rhoGz, out=dm_auxG)
                beta = j_factor
            # einsum('pqG,pi,qj,rij,Gx,rG->rx', pqGw, c, c, dm_oo, 1j*Gv, conj(auxG))
            tmp = ndarray((nocc,nao,nGv*2), buffer=buf1)
            ijG = ndarray((nocc,nocc,nGv*2), buffer=buf)
            contract('pqG,pi->iqG', pqGw, dm_factor_r, out=tmp)
            contract('iqG,qj->ijG', tmp, dm_factor_l, out=ijG)
            # (ji|r)^{[0]} * metric * (r|G)^{[1]} (G|ij)^{[0]}
            # contracting all [0] order terms -> dm_auxG
            contract('rji,ijG->rG', dm_oo, ijG, -.5*k_factor, beta, out=dm_auxG)
            dm_auxG = dm_auxG.view(np.complex128)

            # (ji|r)^{[0]} * metric * -J2c^{[1]} * metric * (ij|s)^{[0]}
            # = -(ji|r)^{[0]} * metric * (r|G)^{[1]} (G|s)^{[0]} * metric * (ij|s)^{[0]}
            dm_auxG1 = contract('sr,sG->rG', dm_aux, auxG.view(np.float64),
                                out=ndarray((naux,nGv*2), buffer=buf)).view(np.complex128)
            vG = contract('rg,rg->g', dm_auxG1, auxG.conj()).real
            sigma_G -= .5 * cp.einsum('g,xyg->xy', vG, wcoulG_LR1[:,:,p0:p1])

            dm_auxG1 *= wcoulG_LR0[p0:p1]
            dm_auxG -= dm_auxG1
            dm_auxG = dm_auxG.view(np.float64)

            # contract to (r|G)^{[1]}.
            # (r|G)^{[1]} = IFT(nabla_A aux) = IFT(-nabla aux) = (iG IFT(aux))
            # Contributions to derivatives are
            # 1/2 * einsum('ag,ag->a', (iG IFT(aux)), dm_auxG).real
            # = 1/2 * einsum('ag,ag->a', conj(-iG FT(aux)), dm_auxG).real
            # = 1/2 *(einsum('ag,ag->a', Re(-iG FT(aux)), Re(dm_auxG))
            #        +einsum('ag,ag->a', Im(-iG FT(aux)), Im(dm_auxG)))
            # The derivatives also include a term that is contracted to (G|r)^{[1]},
            # which is complex conjugated to this term. The overall
            # contributions are
            # 1/2 * einsum('ag,ag->a', (iG IFT(aux)), dm_auxG) + c.c
            # = (einsum('ag,ag->a', Re(-iG FT(aux)), Re(dm_auxG))
            #   +einsum('ag,ag->a', Im(-iG FT(aux)), Im(dm_auxG)))
            #:ip_auxG = ndarray((naux, nGv), dtype=np.complex128, buffer=buf)
            #:for i in range(3):
            #:    cp.multiply(auxG, -1j*Gv[p0:p1,i], out=ip_auxG)
            #:    partial_daux[i] += cp.einsum('ag,ag->a', ip_auxG.view(np.float64), dm_auxG)
            err = kern_auxG(
                ctypes.cast(ejk_sigma_aux[:-3].data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_sigma_aux[-3:].data.ptr, ctypes.c_void_p),
                null_ptr,
                ctypes.cast(dm_auxG.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aux_ft_envs),
                ctypes.c_int(p1-p0))
            if err != 0:
                raise RuntimeError('ft_ao_deriv failed')

        ejk_sigma_lr *= 2 # due to i>=j symmetry in CUDA kernel
        ejk_sigma_lr += ejk_sigma_aux
        ejk_sigma_lr[-3:] += sigma_G
        return ejk_sigma_lr

    ejk_sigma += lr_3c2e_response()
    t0 = log.timer_debug1('lr_int3c2e_deriv', *t0)
    ft_opt = eval_compact = eval_dd = None
    dm_aux = None

    ################################
    # SR int3c2e response
    # contract the derivatives and the pseudo DM/rho
    if n_compact_pairs > 0:
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            gout_width=54, deriv=(1,0,0))
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

        l_ctr_aux_offsets = _counts_to_offsets(auxcell.l_ctr_counts)
        # Split auxbasis in the unit cell. A large aux_batch can overflow the POOL_SIZE
        aux_batch_size = POOL_SIZE // bvk_ncells // 8
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, auxcell.uniq_l_ctr, aux_batch_size)

        ksh_offsets_cpu = l_ctr_aux_offsets
        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

        pair_per_block = _get_shl_pair_per_block(np.diff(ksh_offsets_cpu), bvk_ncells)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            int3c2e_opt.bas_ij_cache, nsp_per_block=pair_per_block)
        ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache, cart=True)

        diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
        diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
        log_cutoff = math.log(int3c2e_opt.cutoff)

        assert cell.natm == auxcell.natm
        ejk_sigma_sr = cp.zeros([cell.natm+3, 3])
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        int3c2e_envs = int3c2e_opt.int3c2e_envs
        kern = libpbc.PBCsr_ejk_int3c2e_deriv
        aux0 = aux1 = 0
        max_aux_batch = int(np.diff(aux_loc[ksh_offsets_cpu]).max())
        buf = cp.empty(n_compact_pairs*max_aux_batch)
        buf1 = cp.empty((blksize, nao, nao))
        buf2 = cp.empty((blksize, nao, nao))
        for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
            aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
            naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
            compressed = ndarray((n_compact_pairs, naux_in_batch), buffer=buf)
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                dm_tensor = ndarray((nao,nao,dk), buffer=buf1)
                tmp = ndarray((nocc,nao,dk), buffer=buf2)
                beta = 0
                if j_factor != 0:
                    cp.multiply(dm_sorted[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                    beta = j_factor
                contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
                contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
                if hermi == 1:
                    cp.take(dm_tensor.reshape(-1,dk), compact_idx, axis=0,
                            out=compressed[:,k0:k1])
                else:
                    dm_tensor1 = ndarray((nao,nao,dk), buffer=buf2)
                    dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                    dm_tensor1[:] += dm_tensor
                    cp.take(dm_tensor1.reshape(-1,dk), compact_idx, axis=0,
                            out=compressed[:,k0:k1])
            err = kern(
                ctypes.cast(ejk_sigma_sr[:-3].data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_sigma_sr[-3:].data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
                ctypes.c_double(-int3c2e_opt.omega),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(head.data.ptr, ctypes.c_void_p),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(shl_pair_offsets) - 1),
                ctypes.c_int(1),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
                ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(aux_ao_offset),
                ctypes.c_int(auxcell.nbas),
                ctypes.c_int(naux_in_batch),
                ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff))
            if err != 0:
                raise RuntimeError('PBCsr_ejk_int3c2e_deriv failed')
        if hermi == 1:
            ejk_sigma_sr *= 2.
        ejk_sigma += ejk_sigma_sr
        t0 = log.timer_debug1('contract sr_int3c2e_ejk_deriv', *t0)

    ejk_sigma = ejk_sigma.get()

    if (exxdiv == 'ewald' and
        (cell.dimension == 3 or
         (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
        s0 = int1e.int1e_ovlp(cell)
        k_dm = contract('pq,qr->pr', dm, s0)
        k_dm = contract('pr,rs->ps', k_dm, dm)
        kpts = np.zeros((1, 3))
        de_ewald = int1e.ovlp_derivatives(cell, k_dm)
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell.cell, kpts, -omega)
        de_ewald *= -.5 * k_factor * exx_0
        ejk_sigma += de_ewald

        ek_G0 = float(cp.einsum('ij,ji->', s0, k_dm).real.get())
        # *.5 for the factor 1/2 in Coulomb operator; second *.5 for J-K/2 in RHF
        ejk_sigma[-3:] -= k_factor * .5 * .5 * ek_G0 * exx_1
    return ejk_sigma

def _get_ej_derivatives(int3c2e_opt, dm, hermi=0, omega=None, verbose=None,
                        linear_dep_threshold=LINEAR_DEP_THR):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    assert bvk_ncells == 1, 'Gamma derivatives require a one-cell BvK mesh'
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    assert cell.dimension == 3
    assert dm.dtype == np.float64

    dm = cell.apply_C_mat_CT(dm)
    if hermi != 1:
        dm = transpose_sum(dm, inplace=True)
        dm[:] *= .5
    has_compact = len(int3c2e_opt.img_idx) > 0
    if has_compact:
        auxvec = int3c2e_opt.contract_dm(dm, hermi=1)
    else:
        auxvec = cp.zeros(auxcell.cell.nao)

    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    # Adjust the rcut because the default cell.rcut is estimated based on
    # overlap integrals.
    rcut = _estimate_sr_2c2e_rcut(
        auxcell, int3c2e_opt.omega, auxcell.precision*1e-6)
    with lib.temporary_env(auxcell, rcut=rcut):
        int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False, omega=-int3c2e_opt.omega)

    ################################
    # LR part 0th order
    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
    dd_ft_opt = int3c2e_opt.dd_ft_opt
    separated_dd = dd_ft_opt is not None

    def lr_3c2e(ft_opt, wcoulG, update_metric):
        auxvec_LR = cp.zeros(naux)
        rhoG = cp.zeros(ngrids, dtype=np.complex128)

        eval_ft = None
        if len(ft_opt.img_idx) > 0:
            eval_ft = ft_opt.ft_evaluator(
                compressing=True, cart=True, original_ao_order=False)[0]

        if len(ft_opt.img_idx) == 0 and not update_metric:
            return auxvec_LR, rhoG

        pair_addresses, diag_idx = ft_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)
        dm_tril = dm.ravel()[pair_addresses]
        dm_tril[diag_idx] *= .5
        dm_tril *= 2

        mem_avail = get_avail_mem(exclude_memory_pool=True)
        nao_pair = len(dm_tril)
        # Conservative sum: buf reuses storage for pqG and auxG.
        unit = nao_pair  # pqG capacity in buf
        unit += naux  # auxG capacity in buf
        unit += naux  # buf1: auxGw
        Gblksize = int(mem_avail//2//(unit*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)

        buf  = cp.empty(max(nao_pair,naux)*Gblksize, dtype=np.complex128)
        buf1 = cp.empty((naux,Gblksize), dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            if eval_ft is None:
                rhoGz = cp.zeros(nGv*2)
            else:
                pqG = eval_ft(Gv[p0:p1], out=buf)
                rhoGz = cp.einsum('pG,p->G', pqG.view(np.float64), dm_tril)
            rhoG[p0:p1] = rhoGz.view(np.complex128)

            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            auxGw = ndarray((naux, nGv), dtype=np.complex128, buffer=buf1)
            cp.multiply(auxG, wcoulG[p0:p1], out=auxGw)
            auxGw = auxGw.view(np.float64)
            if update_metric:
                contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)
            auxvec_LR += auxGw.dot(rhoGz)
        return auxvec_LR, rhoG

    mesh = int3c2e_opt.mesh
    log.debug('mesh for LR coulG %s', mesh)
    Gv, _, kws = get_Gv_weights(cell, mesh)
    ngrids = len(Gv)
    wcoulG_LR0, wcoulG_LR1 = get_wcoulG(cell, Gv, int3c2e_opt.omega)
    if omega != 0:
        wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gv, omega)
        wcoulG_LR0 -= wcoulG_0
        wcoulG_LR1 -= wcoulG_1
    wcoulG_SR_at_G0 = np.pi / int3c2e_opt.omega**2 * kws
    wcoulG_LR0[0] -= wcoulG_SR_at_G0
    wcoulG_LR1[:,:,0] += wcoulG_SR_at_G0 * cp.eye(3)
    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    auxvec_LR, rhoG_LR = lr_3c2e(ft_opt, wcoulG_LR0, True)

    if separated_dd:
        wcoulG_FR0, wcoulG_FR1 = get_wcoulG(cell, Gv, -omega)
        auxvec_FR, rhoG_FR = lr_3c2e(dd_ft_opt, wcoulG_FR0, False)
        auxvec_LR += auxvec_FR

    auxvec += auxcell.apply_CT_dot(auxvec_LR)
    t0 = log.timer_debug1('contract dm', *t0)

    ################################
    # (d/dX P|Q) contributions
    j2c = auxcell.apply_CT_mat_C(j2c)
    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        auxvec = _gen_metric_solver(
            j2c, linear_dep_threshold, auxcell.dimension)(auxvec)
    auxvec = auxcell.C_dot_mat(auxvec)
    assert auxvec.dtype == np.float64
    j2c = None

    dm_aux = auxvec[:,None] * auxvec
    ej_sigma = int2c2e_opt.energy_derivatives(dm_aux, omega=-int3c2e_opt.omega)
    ej_sigma = cp.asarray(-ej_sigma)
    dm_aux = None
    t0 = log.timer_debug1('contract int2c2e_deriv', *t0)

    #########################
    # LR part response
    def lr_3c2e_response():
        aft_envs = ft_opt.aft_envs
        aux_ft_envs = RysIntEnvVars.new(
            auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
            _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)

        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = \
                aft_jk._shl_pairs_for_derivative_kernel(ft_opt)
        if separated_dd:
            dd_bas_ij_idx, dd_bas_ij_img_idx, dd_shl_pair_offsets = \
                    aft_jk._shl_pairs_for_derivative_kernel(dd_ft_opt)

        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        unit = naux  # buf: auxG
        unit += naux  # additional auxiliary FT workspace allowance
        Gblksize = int(mem_avail//(unit*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        rho_auxG = cp.empty(ngrids, dtype=np.complex128)
        buf = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            rho_auxG[p0:p1] = auxvec.dot(auxG.view(np.float64)).view(np.complex128)

            # (ii|r)^{[0]} * metric * (r|G)^{[1]} (jj|G)^{[0]}
            # = auxvec * (r|G)^{[1]} (jj|G)^{[0]}
            # IFT(nabla_A aux) = IFT(-nabla aux) = (iG IFT(aux)) = (iG conj(FT(aux)))
            #ip_vG = rhoG[p0:p1] * wcoulG_LR0[p0:p1] * 1j * Gv[p0:p1].T
            # (ii|r)^{[0]} * metric * -J2c^{[1]} * metric * (jj|r)^{[0]}
            # = auxvec * J2c^{[1]} * auxvec
            #ip_vG -= rho_auxG[p0:p1] * wcoulG_LR0[p0:p1] * 1j * Gv[p0:p1].T
            #partial_daux += cp.einsum('xg,ag->xa', ip_vG.view(np.float64),
            #                          auxG.view(np.float64))
        vG = (rhoG_LR - rho_auxG) * wcoulG_LR0
        if separated_dd:
            vG_FR = rhoG_FR * wcoulG_FR0
            vG += vG_FR
        GvT = cp.asarray(Gv.T.ravel())
        ej_sigma_aux = cp.zeros([cell.natm+3, 3])
        err = libpbc.PBC_ft_ao_deriv(
            ctypes.cast(ej_sigma_aux[:-3].data.ptr, ctypes.c_void_p),
            ctypes.cast(ej_sigma_aux[-3:].data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.cast(vG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aux_ft_envs),
            ctypes.c_int(ngrids))
        if err != 0:
            raise RuntimeError('ft_ao_deriv failed')

        ej_sigma_lr = cp.zeros([cell.natm+3, 3])
        vG_conj = rho_auxG.conj() * wcoulG_LR0
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        if nbatches_shl_pair > 0:
            err = libpbc.PBC_ft_aopair_ej_deriv(
                ctypes.cast(ej_sigma_lr[:-3].data.ptr, ctypes.c_void_p),
                ctypes.cast(ej_sigma_lr[-3:].data.ptr, ctypes.c_void_p),
                ctypes.cast(dm.data.ptr, ctypes.c_void_p),
                ctypes.cast(vG_conj.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(nbatches_shl_pair),
                ctypes.c_int(ngrids),
                ctypes.c_int(shm_size),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ft_opt.permutation_symmetry))
            if err != 0:
                raise RuntimeError('PBC_ft_aopair_ej_deriv failed')

        if separated_dd:
            vG_conj = rho_auxG.conj() * wcoulG_FR0
            err = libpbc.PBC_ft_aopair_ej_deriv(
                ctypes.cast(ej_sigma_lr[:-3].data.ptr, ctypes.c_void_p),
                ctypes.cast(ej_sigma_lr[-3:].data.ptr, ctypes.c_void_p),
                ctypes.cast(dm.data.ptr, ctypes.c_void_p),
                ctypes.cast(vG_conj.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(len(dd_shl_pair_offsets) - 1),
                ctypes.c_int(ngrids),
                ctypes.c_int(shm_size),
                ctypes.cast(dd_bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(dd_bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(dd_shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ft_opt.permutation_symmetry))
            if err != 0:
                raise RuntimeError('PBC_ft_aopair_ej_deriv failed')

        ej_sigma_lr *= 2 # due to i>=j symmetry in CUDA kernel
        ej_sigma_lr += ej_sigma_aux
        ej_sigma_lr[-3:] += cp.einsum(
            'g,xyg->xy', (rho_auxG*rhoG_LR.conj()).real, wcoulG_LR1)
        if separated_dd:
            ej_sigma_lr[-3:] += cp.einsum(
                'g,xyg->xy', (rho_auxG*rhoG_FR.conj()).real, wcoulG_FR1)
        ej_sigma_lr[-3:] -= .5 * cp.einsum(
            'g,xyg->xy', (rho_auxG*rho_auxG.conj()).real, wcoulG_LR1)
        return ej_sigma_lr

    ej_sigma += lr_3c2e_response()
    t0 = log.timer_debug1('lr_int3c2e_deriv', *t0)

    ################################
    # SR int3c2e response
    if has_compact:
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            gout_width=54, deriv=(1,0,0))
        lmax = cell.uniq_l_ctr[:,0].max()
        laux = auxcell.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

        l_ctr_aux_offsets = _counts_to_offsets(auxcell.l_ctr_counts)
        # Split auxbasis in the unit cell. A large aux_batch can overflow the POOL_SIZE
        aux_batch_size = POOL_SIZE // bvk_ncells // 8
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, auxcell.uniq_l_ctr, aux_batch_size)
        ksh_offsets_cpu = l_ctr_aux_offsets
        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

        nksh_per_batch = ksh_offsets_cpu[1:] - ksh_offsets_cpu[:-1]
        pair_per_block = _get_shl_pair_per_block(nksh_per_batch, bvk_ncells)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            int3c2e_opt.bas_ij_cache, nsp_per_block=pair_per_block)

        diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
        diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
        log_cutoff = math.log(int3c2e_opt.cutoff)

        assert cell.natm == auxcell.natm
        ej_sigma_sr = cp.zeros([cell.natm+3, 3])
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        int3c2e_envs = int3c2e_opt.int3c2e_envs
        kern = libpbc.PBCsr_ejk_int3c2e_deriv
        err = kern(
            ctypes.cast(ej_sigma_sr[:-3].data.ptr, ctypes.c_void_p),
            ctypes.cast(ej_sigma_sr[-3:].data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.c_double(-int3c2e_opt.omega),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(task_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(len(ksh_offsets_gpu) - 1),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(0),
            ctypes.c_int(auxcell.nbas),
            ctypes.c_int(naux),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_deriv failed')
        ej_sigma += ej_sigma_sr * 2
        t0 = log.timer_debug1('contract sr_int3c2e_ejk_deriv', *t0)
    return ej_sigma.get()


def _jk_energy_per_atom(int3c2e_opt, dm, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, omega=None, verbose=None,
                        linear_dep_threshold=LINEAR_DEP_THR):
    '''Compatibility wrapper returning only the atomic J/K derivatives.'''
    return _get_ejk_derivatives(
        int3c2e_opt, dm, hermi, j_factor, k_factor, exxdiv, omega, verbose,
        linear_dep_threshold)[:-3]


def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, omega=None, verbose=None,
                       linear_dep_threshold=LINEAR_DEP_THR):
    '''Compatibility wrapper returning only the atomic Coulomb derivatives.'''
    return _get_ej_derivatives(
        int3c2e_opt, dm, hermi, omega, verbose, linear_dep_threshold)[:-3]


_kernel_registery = {}

def indexed_scale(a, idx, b):
    '''a[idx] *= b'''
    fn_name = 'indexed_scale'
    if fn_name not in _kernel_registery:
        _kernel_registery[fn_name] = cp.RawKernel(r'''
#include <cuComplex.h>
extern "C" __global__
void ''' + fn_name + r'''(cuDoubleComplex *a, int *idx, double *b, int ncol) {
    int i = blockIdx.x;
    size_t row = idx[i];
    size_t off = row * ncol;
    for (int j = threadIdx.x; j < ncol; j += blockDim.x) {
        double w = b[j];
        cuDoubleComplex val = a[off + j];
        val.x *= w;
        val.y *= w;
        a[off + j] = val;
    }
}''', fn_name)
    kernel = _kernel_registery[fn_name]
    idx = cp.asarray(idx, dtype=np.int32)
    ncol = a.shape[1]
    assert a.ndim == 2 and a.flags.c_contiguous
    assert a.dtype == cp.complex128
    assert b.dtype == cp.float64 and b.flags.c_contiguous
    assert len(b) == ncol
    if len(idx) == 0:
        return a
    kernel((len(idx),), (1024,), (a, idx, b, cp.int32(ncol)))
    return a

def symmetric_scatter(a, i_addr, j_addr, b):
    '''a[i_addr,j_addr,:] = a[j_addr,i_addr,:] = b[:,:]'''
    fn_name = 'symmetric_scatter'
    if fn_name not in _kernel_registery:
        _kernel_registery[fn_name] = cp.RawKernel(r'''
extern "C" __global__
void ''' + fn_name + r'''(double *a, double *b, int* i_addr, int* j_addr, int N, int ncol) {
    size_t k = blockIdx.x;
    size_t i = i_addr[k];
    size_t j = j_addr[k];
    size_t base_ij = (i * N + j) * ncol;
    size_t base_ji = (j * N + i) * ncol;
    size_t base_b =  k * ncol;
    for (int c = threadIdx.x; c < ncol; c += blockDim.x) {
        double val = b[base_b + c];
        a[base_ij + c] = val;
        a[base_ji + c] = val;
    }
}
''', fn_name)
    kernel = _kernel_registery[fn_name]
    i_addr = cp.asarray(i_addr, dtype=np.int32)
    j_addr = cp.asarray(j_addr, dtype=np.int32)
    n, ncol = a.shape[1:3]
    assert a.dtype == b.dtype
    a.fill(0.)
    if len(i_addr) == 0:
        return a
    if a.dtype == cp.complex128:
        ncol *= 2
    kernel((len(i_addr),), (1024,), (a, b, i_addr, j_addr, cp.int32(n), cp.int32(ncol)))
    return a
