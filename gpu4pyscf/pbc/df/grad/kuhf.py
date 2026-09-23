#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.tools.k2gamma import double_translation_indices
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, get_avail_mem, empty_aligned, scatter_add)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.pbc.df.int3c2e import libpbc, POOL_SIZE, MAX_IMGS_PER_TASK
from gpu4pyscf.pbc.df.rsdf_builder import LINEAR_DEP_THR, _unpack_cderi_v2
from gpu4pyscf.pbc.tools.pbc import madelung, _Gv_wrap_around
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.grad import uhf
from gpu4pyscf.pbc.df.grad.krhf import (
    _get_ej_derivatives, _get_j3c_block_sizes, _get_lr_block_size)
from gpu4pyscf.pbc.df.grad.rhf import (
    factorize_dm, get_ao_pair_loc, _split_l_ctr_pattern, _gen_metric_solver,
    _get_shl_pair_batch_size, indexed_scale)
from gpu4pyscf.pbc.df.int3c2e import int3c2e_scheme
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.grad.krks_stress import (
    _get_weighted_coulG_strain_derivatives as get_wcoulG)
from gpu4pyscf.gto.mole import RysIntEnvVars, _scale_sp_ctr_coeff
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.lib.kpts_helper import (
    kk_adapted_iter, conj_images_in_bvk_cell)


def _get_ejk_derivatives(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                         exxdiv=None, omega=None, verbose=None,
                         linear_dep_threshold=LINEAR_DEP_THR):
    '''
    Computes the first-order derivatives (nuclear gradients and strain
    derivatives) of the energy contributions from J and K terms per atom.
    '''
    if kpts is None or kpts.ndim == 1:
        assert dm.ndim == 3
        assert dm.dtype == np.float64
        if kpts is not None:
            assert is_zero(kpts)
        return uhf._get_ejk_derivatives(
            int3c2e_opt, dm, hermi, j_factor, k_factor, exxdiv, omega, verbose,
            linear_dep_threshold)

    if hermi == 2:
        j_factor = 0

    if k_factor == 0:
        ej_sigma = _get_ej_derivatives(
            int3c2e_opt, dm[0]+dm[1], kpts, hermi, omega, verbose,
            linear_dep_threshold)
        return ej_sigma * j_factor

    # Must be symmetric density matrices, otherwise, dm_tensor needs to be
    # symmetrized since PBCsr_ejk_int3c2e_deriv only handles the tril pairs
    assert hermi == 1 or hermi == 2
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = factorize_dm(dm, hermi)
    # transform to the AO order in sorted_cell
    assert dm.ndim == 4
    dm_factor_l = cell.apply_C_dot(dm_factor_l, axis=2)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l.conj()
    else:
        dm_factor_r = cell.apply_C_dot(dm_factor_r, axis=2)
    nkpts, nao, nocc = dm_factor_l.shape[1:]

    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)
    compact_idx = pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    compact_diag = diag_idx
    nao_pair = n_compact_pairs = len(pair_addresses)
    dd_ft_opt = int3c2e_opt.dd_ft_opt
    separated_dd = dd_ft_opt is not None
    if separated_dd:
        dd_ao_idx, dd_diag = dd_ft_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)
        pair_addresses = cp.hstack([compact_idx, dd_ao_idx], dtype=np.int32)
        diag_idx = cp.hstack([diag_idx, n_compact_pairs + dd_diag])
        nao_pair = len(pair_addresses)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    assert nkpts == len(kpts)
    expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conj = expLk.conj()
    expLk_conjz = expLk_conj.view(np.float64).reshape(bvk_ncells,nkpts,2)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    batch_size = blksize = 0
    if n_compact_pairs > 0:
        batch_size, blksize = _get_j3c_block_sizes(
            mem_free, nao, n_compact_pairs, naux, nocc, nkpts, bvk_ncells,
            int(np.diff(aux_loc).max()), nspin=2)

    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    # k=ijk_conserv[i,j] provides: -i + j - k = 2n\pi
    # therefore, i=ijk_conserv[k,j]
    ijk_conserv = cp.asarray(double_translation_indices(int3c2e_opt.bvk_kmesh))
    #for ki in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ki,kj] += j3c_tmp[ijk_conserv[ki,kj],ki]
    #        => order_KI = argsort([ki,ijk_conserv[ki,kj]])
    order_KI = cp.argsort((ijk_conserv * nkpts + cp.arange(nkpts)[:,None]).ravel())
    #for kk in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ijk_conserv[kk,kj],kj] += j3c_tmp[kk,kj]
    #        => order_KJ = [ijk_conserv[kk,kj],kj]
    order_KJ = (ijk_conserv * nkpts + cp.arange(nkpts)).ravel()
    order_KJ = cp.asnumpy(order_KJ)

    def sr_int3c2e():
        eval_j3c, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
            aux_batch_size=batch_size, cart=True)
        aux_batches = len(aux_offsets) - 1

        aux0 = aux1 = 0
        j3c_full = cp.zeros((nao*bvk_ncells*nao,blksize,nkpts), dtype=np.complex128)
        max_aux_batch = int(np.diff(aux_offsets).max())
        buf = cp.empty((bvk_ncells*max_aux_batch, n_compact_pairs))
        buf1 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
        buf2 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
        # Compute the occ-occ block of j3c, should be identical to
        #:j3c = int3c2e.sr_aux_e2(cell.cell, auxcell.cell, omega, kpts)
        #:j3c_oo = cp.einsum('IJpqr,Ipi,Jqj->rIJij', j3c, dm_factor_r, dm_factor_l)
        j3c_oo = cp.empty((2, naux, nkpts, nkpts, nocc, nocc), dtype=np.complex128)
        for kbatch in range(aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
            compressed = contract('tLr,LKz->trKz', compressed, expLk_conjz)
            compressed = compressed.view(np.complex128)[:,:,:,0]
            # *.5 because diagonal blocks are accessed twice
            compressed[compact_diag] *= .5
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                # TODO: decompress the j3c tensor using rsdf_builder._unpack_cderi_v2
                j3c = j3c_full[:,:dk]
                j3c[compact_idx] = compressed[:,k0:k1]
                j3c = j3c.reshape(nao, bvk_ncells, nao, dk, nkpts)

                # Construct j3c_ij in crystal AOs
                #:j3c_ij = cp.empty((nkpts, nkpts, nao, nao, dk), dtype=np.complex128)
                #:j3c_I = contract('jLikK,LI->KIijk', j3c, expLk.conj())
                #:j3c_J = contract('iLjkK,LJ->KJijk', j3c, expLk)
                #:for ki in range(nkpts):
                #:    for kj in range(nkpts):
                #:        kk = ijk_conserv[ki,kj]
                #:        j3c_ij[ki,kj] = j3c_I[kk,ki] + j3c_J[kk,kj]
                # The indices (kk*nkpts+ki) and (kk*nkpts+kj) are precomputed and
                # provided by order_KI and order_KJ
                j3c_ij = ndarray((nkpts*nkpts, nao*nao*dk), dtype=np.complex128, buffer=buf1)
                j3c_tmp = ndarray((nkpts,nkpts, nao,nao,dk), dtype=np.complex128, buffer=buf2)
                j3c_tmp = contract('jLikK,LI->KIijk', j3c, expLk_conj, out=j3c_tmp)
                j3c_ij[order_KI] = j3c_tmp.reshape(nkpts**2,-1)
                j3c_tmp = contract('iLjkK,LJ->KJijk', j3c, expLk, out=j3c_tmp)
                #:j3c_ij[order_KJ] += j3c_tmp.reshape(nkpts**2,-1)
                j3c_ij = scatter_add(j3c_ij, order_KJ, j3c_tmp.reshape(nkpts**2,-1))
                j3c_ij = j3c_ij.reshape(nkpts, nkpts, nao, nao, dk)

                tmp = ndarray((nkpts, nkpts, nocc, nao, dk), dtype=np.complex128, buffer=buf2)
                contract('IJpqr,Ipi->IJiqr', j3c_ij, dm_factor_r[0], out=tmp)
                contract('IJiqr,Jqj->rIJij', tmp, dm_factor_l[0], out=j3c_oo[0,aux0:aux1])
                contract('IJpqr,Ipi->IJiqr', j3c_ij, dm_factor_r[1], out=tmp)
                contract('IJiqr,Jqj->rIJij', tmp, dm_factor_l[1], out=j3c_oo[1,aux0:aux1])
            compressed = None
        j3c_full = buf = buf1 = buf2 = eval_j3c = None
        compressed = j3c = j3c_tmp = j3c_ij = tmp = None
        return j3c_oo

    if n_compact_pairs > 0:
        j3c_oo = sr_int3c2e()
        t0 = log.timer_debug1('contract dm', *t0)
    else:
        j3c_oo = cp.zeros((2, naux, nkpts, nkpts, nocc, nocc), dtype=np.complex128)

    kpt_iters = list(kk_adapted_iter(int3c2e_opt.bvk_kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    nkpts_uniq = len(uniq_kpts)

    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    rcut = _estimate_sr_2c2e_rcut(auxcell, int3c2e_opt.omega, precision)
    with lib.temporary_env(auxcell, rcut=rcut):
        int2c2e_opt = Int2c2eOpt(auxcell, int3c2e_opt.bvk_kmesh)
    j2c = int2c2e_opt.int2c2e(
        uniq_kpts, sort_output=False, omega=-int3c2e_opt.omega)
    if j2c.dtype == np.float64:
        j2c = j2c.astype(np.complex128)

    ################################
    # LR part 0th order
    mesh = int3c2e_opt.mesh
    log.debug('mesh for LR coulG %s', mesh)
    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    assert ft_opt.permutation_symmetry

    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
    mesh = int3c2e_opt.mesh
    Gv, _, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    wcoulG_LR0 = cp.empty((nkpts_uniq, ngrids))
    wcoulG_LR1 = cp.empty((nkpts_uniq, 3, 3, ngrids))
    for k, kpt in enumerate(uniq_kpts):
        Gk = Gv + kpt
        wcoulG_LR0[k], wcoulG_LR1[k] = get_wcoulG(cell, Gk, int3c2e_opt.omega)
        if omega != 0:
            wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gk, omega)
            wcoulG_LR0[k] -= wcoulG_0
            wcoulG_LR1[k] -= wcoulG_1
    # The removed G=0 short-range contribution only belongs to q=0.
    wcoulG_SR_at_G0 = np.pi / int3c2e_opt.omega**2 * kws
    wcoulG_LR0[0, 0] -= wcoulG_SR_at_G0
    wcoulG_LR1[0, :, :, 0] += wcoulG_SR_at_G0 * cp.eye(3)

    eval_compact = None
    if n_compact_pairs > 0:
        eval_compact = ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]
    if separated_dd:
        wcoulG_FR0 = cp.empty_like(wcoulG_LR0)
        wcoulG_FR1 = cp.empty_like(wcoulG_LR1)
        for k, kpt in enumerate(uniq_kpts):
            wcoulG_FR0[k], wcoulG_FR1[k] = get_wcoulG(cell, Gv + kpt, -omega)
        eval_dd = dd_ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]

    def eval_ft(Gv, out=None):
        result = ndarray((nao_pair, len(Gv)), dtype=np.complex128, buffer=out)
        if n_compact_pairs > 0:
            eval_compact(Gv, out=result[:n_compact_pairs])
        if separated_dd:
            eval_dd(Gv, out=result[n_compact_pairs:])
        return result

    conj_mapping = cp.asarray(
        conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh), dtype=np.int32)

    def unpack_ft(pqG_compressed, kj_idx):
        # Full primitive diagonal blocks need half weights before addition.
        pqG_compressed[diag_idx] *= .5
        pqG = _unpack_cderi_v2(
            pqG_compressed.T, pair_addresses, kj_idx, conj_mapping,
            expLk, nao, axis=0)
        return pqG.transpose(0, 2, 3, 1)

    def lr_3c2e(j3c_oo):
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        Gblksize = _get_lr_block_size(
            nao, nocc, naux, nkpts, nkpts_uniq, ngrids, nspin=2)
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, (Gv[p0:p1]+uniq_kpts[:,None]).reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)
            auxGw = auxG.conj()
            auxGw *= wcoulG_LR0[:,p0:p1]
            contract('iKG,jKG->Kij', auxGw, auxG, beta=1, out=j2c)
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                pqG_compressed = eval_ft(Gv[p0:p1] + kpts[kp])
                pqG_compressed[:n_compact_pairs] *= wcoulG_LR0[j2c_idx,p0:p1]
                if separated_dd:
                    pqG_compressed[n_compact_pairs:] *= wcoulG_FR0[j2c_idx,p0:p1]
                pqG = unpack_ft(pqG_compressed, kj_idx)
                tmp = contract('kpqG,skpi->skiqG', pqG, dm_factor_r)
                ijG = contract('skiqG,skqj->skijG', tmp, dm_factor_l[:,kj_idx])
                j3c_oo[:,:,ki_idx,kj_idx] += contract(
                    'rG,skijG->srkij', auxG[:,j2c_idx].conj(), ijG)
                tmp = ijG = None
                if kp != kp_conj:
                    tmp = contract('kqpG,skpi->skiqG', pqG.conj(), dm_factor_r[:,kj_idx])
                    ijG = contract('skiqG,skqj->skijG', tmp, dm_factor_l)
                    j3c_oo[:,:,kj_idx,ki_idx] += contract(
                        'rG,skijG->srkij', auxG[:,j2c_idx], ijG)
                pqG = tmp = ijG = None
            auxG = auxGw = None
        return j3c_oo
    j3c_oo = lr_3c2e(j3c_oo)

    ################################
    # (d/dX P|Q) contributions
    j2c = auxcell.apply_CT_mat_C(j2c)
    j_factor /= nkpts**2
    k_factor /= nkpts**2
    aux_coeff = cp.asarray(auxcell.ctr_coeff)
    dm_oo = j3c_oo
    buf = cp.empty((2, naux, nkpts, nocc, nocc), dtype=np.complex128)
    buf1 = cp.empty((2, naux, nkpts, nocc, nocc), dtype=np.complex128)
    dm_aux = cp.empty((nkpts_uniq, naux, naux), dtype=np.complex128)
    # Contractions for kp and kp_conj are complex conjugated.
    # A factor of 2 is applied due to this symmetry.
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        j2c_k = j2c[j2c_idx]
        if kp == kp_conj:
            j2c_k = j2c_k.real
        solve_j2c = _gen_metric_solver(
            j2c_k, linear_dep_threshold, auxcell.dimension)
        metric = aux_coeff.dot(solve_j2c(aux_coeff.T))
        j3c_oo_k = j3c_oo[:,:,ki_idx,kj_idx]
        dm_oo_k = contract('uv,svnij->sunij', metric, j3c_oo_k, out=buf)
        dm_oo[:,:,ki_idx,kj_idx] = dm_oo_k
        if kp == 0:
            dm_oo_kconj = dm_oo_k
        elif kp == kp_conj:
            # for kp == kp_conj != 0, dm_oo_kconj and dm_oo_k correspond to
            # the same blocks in dm_oo, which has been updated previously
            dm_oo_kconj = dm_oo[:,:,kj_idx,ki_idx]
        else:
            j3c_oo_k = j3c_oo[:,:,kj_idx,ki_idx]
            dm_oo_kconj = contract('vu,svnij->sunij', metric, j3c_oo_k, out=buf1)
            dm_oo[:,:,kj_idx,ki_idx] = dm_oo_kconj

        beta = 0
        if j_factor != 0 and kp == 0:
            dm_sorted = contract('skpi,skqi->kpq', dm_factor_l, dm_factor_r)
            assert all(ki_idx == kj_idx)
            auxvec = cp.einsum('sunii->u', dm_oo_k)
            cp.multiply(auxvec[:,None], auxvec.conj(), out=dm_aux[j2c_idx])
            beta = j_factor

        contract('urkij,uskji->rs', dm_oo_k, dm_oo_kconj,
                 alpha=-k_factor, beta=beta, out=dm_aux[j2c_idx])
        # Contractions for kp and kp_conj are complex conjugated.
        # A factor of 2 is applied due to this time-reversal symmetry.
        if kp != kp_conj:
            dm_aux[j2c_idx] *= 2
        metric = j3c_oo_k = dm_oo_k = dm_oo_kconj = None
    ejk_sigma = int2c2e_opt.energy_derivatives(
        dm_aux, uniq_kpts, omega=-int3c2e_opt.omega)
    ejk_sigma = cp.asarray(-ejk_sigma)
    j2c = j3c_oo = None
    aux_coeff = buf = buf1 = None
    t0 = log.timer_debug1('contract int2c2e_deriv', *t0)

    ################################
    # LR part response
    def lr_3c2e_response():
        Gk = (asarray(Gv) + asarray(uniq_kpts)[:,None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(auxcell, Gk, cp.zeros(3), mesh)
        Gk = Gk.reshape(nkpts_uniq, ngrids, 3)

        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = \
                aft_jk._shl_pairs_for_derivative_kernel(ft_opt)
        if separated_dd:
            dd_bas_ij_idx, dd_bas_ij_img_idx, dd_shl_pair_offsets = \
                    aft_jk._shl_pairs_for_derivative_kernel(dd_ft_opt)
            bas_ij_idx = cp.hstack([bas_ij_idx, dd_bas_ij_idx])
            bas_ij_img_idx = cp.hstack([bas_ij_img_idx, dd_bas_ij_img_idx])
            shl_pair_offsets = cp.hstack([
                shl_pair_offsets[:-1], shl_pair_offsets[-1] + dd_shl_pair_offsets])
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        i_addr, j_addr = divmod(pair_addresses, bvk_ncells*nao)
        # CUDA reads [image, j, i], while compressed FT stores [i, image, j].
        response_idx = j_addr * nao + i_addr
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        Gblksize = _get_lr_block_size(nao, nocc, naux, nkpts, nkpts_uniq, ngrids, nspin=2)
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        kern = libpbc.PBC_ft_aopair_ek_deriv
        kern_auxG = libpbc.PBC_ft_ao_deriv
        ejk_sigma_lr = cp.zeros([cell.natm+3, 3])
        sigma_G = cp.zeros((3, 3))
        aux_ft_envs = RysIntEnvVars.new(
            auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
            _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)
        null_ptr = lib.c_null_ptr()
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gk[:,p0:p1].reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)

            # (ij|r)^{[0]} * metric * (r|G)^{[1]} (ji|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                dm_oo_k = dm_oo[:,:,kj_idx,ki_idx]
                if kp != kp_conj:
                    dm_oo_k *= 2

                # Orbital response: form the unweighted BvK density first.
                # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
                auxG_conj = auxG[:,j2c_idx].conj()
                dm_ooG = contract('srkji,rG->skijG', dm_oo_k, auxG_conj)
                tmp = contract('skijG,skpi->skpjG', dm_ooG, dm_factor_r)
                dm_vG = contract('skpjG,skqj->kpqG', tmp, dm_factor_l[:,kj_idx], -k_factor)
                LpqG = contract('Lk,kpqG->LqpG', expLk[:,kj_idx], dm_vG)
                if ft_opt.permutation_symmetry:
                    contract('Lk,kpqG->LpqG', expLk_conj, dm_vG, beta=1, out=LpqG)
                if j_factor != 0 and kp == 0:
                    vG = auxvec.dot(auxG_conj) * j_factor
                    if ft_opt.permutation_symmetry:
                        vG *= 2
                    bvk_dm = contract('Lk,kpq->Lpq', expLk, dm_sorted)
                    contract('Lpq,G->LpqG', bvk_dm, vG, beta=1, out=LpqG)
                dm_vG = cp.asarray(LpqG, order='C')
                dm_vG_compressed = dm_vG[response_idx]
                if n_compact_pairs > 0:
                    indexed_scale(dm_vG, response_idx[:n_compact_pairs], wcoulG_LR0[j2c_idx,p0:p1])
                if separated_dd:
                    indexed_scale(dm_vG, response_idx[n_compact_pairs:], wcoulG_FR0[j2c_idx,p0:p1])
                GvT = cp.asarray((Gv[p0:p1]+kpts[kp]).T.ravel())
                err = kern(
                    ctypes.cast(ejk_sigma_lr[:-3].data.ptr, ctypes.c_void_p),
                    ctypes.cast(ejk_sigma_lr[-3:].data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    ctypes.byref(aft_envs),
                    ctypes.c_int(nbatches_shl_pair),
                    ctypes.c_int(nGv),
                    ctypes.c_int(shm_size),
                    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(ft_opt.permutation_symmetry))
                if err != 0:
                    raise RuntimeError('PBC_ft_aopair_ek_deriv failed')
                LpqG = None

                pqG_compressed = eval_ft(Gv[p0:p1] + kpts[kp])
                dm_vG_compressed[diag_idx] *= .5
                vG = cp.einsum('pg,pg->g', pqG_compressed[:n_compact_pairs],
                               dm_vG_compressed[:n_compact_pairs]).real
                # LpqG already includes both exchanged orbital orientations.
                sigma_G += cp.einsum('g,xyg->xy', vG, wcoulG_LR1[j2c_idx,:,:,p0:p1])
                if separated_dd:
                    vG = cp.einsum('pg,pg->g', pqG_compressed[n_compact_pairs:],
                                   dm_vG_compressed[n_compact_pairs:]).real
                    sigma_G += cp.einsum('g,xyg->xy', vG, wcoulG_FR1[j2c_idx,:,:,p0:p1])
                dm_vG = dm_ooG = dm_vG_compressed = tmp = None

                # Auxiliary response uses the weighted compact and DD columns.
                pqG_compressed[:n_compact_pairs] *= wcoulG_LR0[j2c_idx,p0:p1]
                if separated_dd:
                    pqG_compressed[n_compact_pairs:] *= wcoulG_FR0[j2c_idx,p0:p1]
                pqG = unpack_ft(pqG_compressed, kj_idx)

                beta = 0
                dm_auxG = ndarray((naux,nGv), dtype=np.complex128, buffer=buf2)
                if j_factor != 0 and kp == 0:
                    rhoGz = cp.einsum('kpqG,kqp->G', pqG, dm_sorted)
                    cp.multiply(auxvec[:,None], rhoGz, out=dm_auxG)
                    beta = j_factor
                # einsum('pqG,pi,qj,rij,Gx,rG->rx', pqG, c, c, dm_oo, 1j*Gv, conj(auxG))
                tmp = contract('kpqG,skpi->skiqG', pqG, dm_factor_r)
                ijG = contract('skiqG,skqj->skijG', tmp, dm_factor_l[:,kj_idx])
                # (ji|r)^{[0]} * metric * (r|G)^{[1]} (G|ij)^{[0]}
                # contracting all [0] order terms -> dm_auxG
                contract('srkji,skijG->rG', dm_oo_k, ijG, -k_factor, beta, out=dm_auxG)

                # (ji|r)^{[0]} * metric * -J2c^{[1]} * metric * (ij|s)^{[0]}
                # = -(ji|r)^{[0]} * metric * (r|G)^{[1]} (G|s)^{[0]} * metric * (ij|s)^{[0]}
                dm_auxG1 = contract('sr,sG->rG', dm_aux[j2c_idx], auxG[:,j2c_idx])
                vG = cp.einsum('rg,rg->g', dm_auxG1, auxG_conj).real
                sigma_G -= .5 * cp.einsum('g,xyg->xy', vG, wcoulG_LR1[j2c_idx,:,:,p0:p1])
                dm_auxG1 *= wcoulG_LR0[j2c_idx,p0:p1]
                dm_auxG -= dm_auxG1
                dm_auxG = dm_auxG.view(np.float64)
                GkT = cp.asarray(Gk[j2c_idx,p0:p1].T.ravel())
                err = kern_auxG(
                    ctypes.cast(ejk_sigma_lr[:-3].data.ptr, ctypes.c_void_p),
                    ctypes.cast(ejk_sigma_lr[-3:].data.ptr, ctypes.c_void_p),
                    null_ptr,
                    ctypes.cast(dm_auxG.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GkT.data.ptr, ctypes.c_void_p),
                    ctypes.byref(aux_ft_envs), ctypes.c_int(nGv))
                if err != 0:
                    raise RuntimeError('ft_ao_deriv failed')
                pqG = pqG_compressed = tmp = ijG = dm_auxG1 = dm_oo_k = None

        ejk_sigma_lr[-3:] += sigma_G
        return ejk_sigma_lr

    ejk_sigma += lr_3c2e_response()
    log.timer_debug1('LR coulomb', *t0)
    ft_opt = eval_compact = eval_ft = eval_dd = None
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

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
        l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
            l_ctr_aux_offsets, auxcell.uniq_l_ctr, batch_size)

        ksh_offsets_cpu = l_ctr_aux_offsets
        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

        nksh_per_batch = ksh_offsets_cpu[1:] - ksh_offsets_cpu[:-1]
        shl_pair_batch_size = _get_shl_pair_batch_size(
            nksh_per_batch, bvk_ncells)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
            int3c2e_opt.bas_ij_cache, nsp_per_block=shl_pair_batch_size)
        ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache, cart=True)
        aux_loc = auxcell.ao_loc

        diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
        diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
        log_cutoff = math.log(int3c2e_opt.cutoff)

        order_KI = (ijk_conserv.T * nkpts + cp.arange(nkpts)[:,None]).ravel()
        ejk_sigma_sr = cp.zeros([cell.natm+3, 3])
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        int3c2e_envs = int3c2e_opt.int3c2e_envs
        kern = libpbc.PBCsr_ejk_int3c2e_deriv
        aux0 = aux1 = 0
        max_aux_batch = int(np.diff(aux_loc[ksh_offsets_cpu]).max())
        buf = cp.empty(n_compact_pairs*max_aux_batch*bvk_ncells)
        buf1 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
        buf2 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
        for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
            aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
            naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
            compressed = ndarray((n_compact_pairs, bvk_ncells, naux_in_batch), buffer=buf)
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                # The contraction with first order derivative integrals are
                #:for ki in range(nkpts):
                #:    for kj in range(nkpts):
                #:        einsum('li,ijp,jk,qp,xklq->x', dm[ki], j3c[ki,kj], dm[kj],
                #:               metric[kk_conserv[ki,kj]], j3c_ip1[kj,ki])
                # dm_tensor stores the contraction 'li,ijp,jk,qp->lkq'.
                dm_tensor = ndarray((nkpts,nkpts,nao,nao,dk), dtype=np.complex128, buffer=buf2)
                tmp = ndarray((nkpts,nkpts,nocc,nao,dk), dtype=np.complex128, buffer=buf1)
                contract('rIJij,Jqj->IJiqr', dm_oo[0,aux0:aux1],
                         dm_factor_r[0], -k_factor, out=tmp)
                contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_l[0], out=dm_tensor)
                contract('rIJij,Jqj->IJiqr', dm_oo[1,aux0:aux1],
                         dm_factor_r[1], -k_factor, out=tmp)
                contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_l[1],
                         beta=1, out=dm_tensor)
                # j3c_ip1 (xklq) is first evaluated in real space, then l and q
                # are transformed to k-adpated indices. kpt for l is associated with
                # the first index of dm_tensor.
                # To match the kpt indexing of j3c_ip, dm_tensor's orbital k-indices
                # JI needs to be transformed to abs-obs mixed k-indices KI.
                #:dm_tensor_swap = cp.zeros_like(dm_tensor)
                #:for ki in range(nkpts):
                #:    for kj in range(nkpts):
                #:        kk = ijk_conserv[kj,ki]
                #:        dm_tensor_swap[kk,ki] = dm_tensor[ki,kj]
                dm_tensor_swap = ndarray((nkpts*nkpts,nao,nao,dk), dtype=np.complex128, buffer=buf1)
                dm_tensor_swap[order_KI] = dm_tensor.reshape(nkpts**2,nao,nao,dk)
                dm_tensor_swap = dm_tensor_swap.reshape(nkpts,nkpts,nao,nao,dk)
                if j_factor != 0:
                    dm_tensor_swap[0] += j_factor * auxvec[aux0:aux1] * dm_sorted[:,:,:,None]

                tmp = ndarray((nkpts,nao,nao,bvk_ncells,dk), dtype=np.complex128, buffer=buf2)
                tmp1 = ndarray((nao,bvk_ncells,nao,bvk_ncells,dk), dtype=np.complex128, buffer=buf1)
                dm_tensor = contract('KJpqr,LK->JpqLr', dm_tensor_swap, expLk_conj, out=tmp)
                dm_tensor = contract('JpqLr,NJ->qNpLr', dm_tensor, expLk, out=tmp1)
                dm_tensor = dm_tensor.reshape(-1,bvk_ncells,dk).real
                #:compressed[:,:,k0:k1] = dm_tensor[compact_idx]
                cp.take(dm_tensor, compact_idx, axis=0, out=compressed[:,:,k0:k1])
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
        ejk_sigma += ejk_sigma_sr * 2
        t0 = log.timer_debug1('contract int3c2e_ejk_deriv', *t0)

    ejk_sigma = ejk_sigma.get()

    if (exxdiv == 'ewald' and
        (cell.dimension == 3 or
         (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
        bvk_kmesh = int3c2e_opt.bvk_kmesh
        s0 = int1e.int1e_ovlp(cell, kpts, bvk_kmesh)
        k_dm = contract('nkpq,kqr->nkpr', dm, s0)
        k_dm = contract('nkpr,nkrs->kps', k_dm, dm)
        # The k_factor was previously scaled by 1/nkpts^2.
        k_factor *= nkpts**2
        de_ewald = int1e.ovlp_derivatives(cell, k_dm, kpts, bvk_kmesh)
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell.cell, kpts, -omega)
        de_ewald *= -k_factor * exx_0 / nkpts
        ejk_sigma += de_ewald

        ek_G0 = float(cp.einsum('kij,kji->', s0, k_dm).real.get()) / nkpts**2
        # *.5 for the factor 1/2 in Coulomb operator
        ejk_sigma[-3:] -= k_factor * .5 * ek_G0 * exx_1
    return ejk_sigma

def _jk_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, omega=None, verbose=None,
                        linear_dep_threshold=LINEAR_DEP_THR):
    '''Compatibility wrapper returning only the atomic J/K derivatives.'''
    return _get_ejk_derivatives(
        int3c2e_opt, dm, kpts, hermi, j_factor, k_factor, exxdiv, omega, verbose,
        linear_dep_threshold)[:-3]
