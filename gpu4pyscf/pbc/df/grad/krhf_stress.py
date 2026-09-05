#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Functions for computing stress tensor and strain derivatives
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
    contract, asarray, ndarray, transpose_sum, get_avail_mem, empty_aligned)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.pbc.df.int3c2e import libpbc, POOL_SIZE, MAX_IMGS_PER_TASK
from gpu4pyscf.pbc.df.rsdf_builder import LINEAR_DEP_THR
from gpu4pyscf.pbc.tools.pbc import madelung, _Gv_wrap_around
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.grad import rhf_stress as rhf
from gpu4pyscf.pbc.df.grad.rhf import factorize_dm
from gpu4pyscf.pbc.df.grad.rhf import _split_l_ctr_pattern, get_ao_pair_loc
from gpu4pyscf.pbc.df.int3c2e import int3c2e_scheme
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
from gpu4pyscf.gto.mole import RysIntEnvVars, _scale_sp_ctr_coeff
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.gto.cell import get_Gv_weights
from gpu4pyscf.pbc.lib.kpts_helper import fft_matrix, kk_adapted_iter


def _get_ejk_strain_deriv(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, omega=None, verbose=None,
                        linear_dep_threshold=LINEAR_DEP_THR):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    from gpu4pyscf.pbc.grad.rks_stress import (
        _get_weighted_coulG_strain_derivatives as get_wcoulG)
    if kpts is None:
        assert dm.ndim == 2
        return rhf._get_ejk_strain_deriv(
            int3c2e_opt, dm, hermi, j_factor, k_factor, exxdiv, omega, verbose,
            linear_dep_threshold)

    if hermi == 2:
        j_factor = 0

    if k_factor == 0:
        ej, sigma = _get_ej_strain_deriv(
            int3c2e_opt, dm, kpts, hermi, omega, verbose,
            linear_dep_threshold)
        return ej * j_factor, sigma * j_factor

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
    dm_factor_l = cell.apply_C_dot(dm_factor_l, axis=1)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l.conj()
    else:
        dm_factor_r = cell.apply_C_dot(dm_factor_r, axis=1)
    nkpts, nao, nocc = dm_factor_l.shape

    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)
    i_addr, j_addr = divmod(pair_addresses, bvk_ncells*nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    assert nkpts == len(kpts)
    expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conj = expLk.conj()
    expLk_conjz = expLk_conj.view(np.float64).reshape(bvk_ncells,nkpts,2)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    buffer_size = mem_free // 4
    batch_size = max(1, min(naux, buffer_size // (nao_pair*8*bvk_ncells)))
    assert batch_size < POOL_SIZE
    eval_j3c, _, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, buffer_size // ((nao*bvk_ncells)**2*8)))
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

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao*bvk_ncells*nao,blksize,nkpts), dtype=np.complex128)
    buf = cp.empty((bvk_ncells*batch_size, nao_pair))
    buf1 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    buf2 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    # Compute the occ-occ block of j3c, should be identical to
    #:j3c = int3c2e.sr_aux_e2(cell.cell, auxcell.cell, omega, kpts)
    #:j3c_oo = cp.einsum('IJpqr,Ipi,Jqj->rIJij', j3c, dm_factor_r, dm_factor_l)
    j3c_oo = cp.empty((naux, nkpts, nkpts, nocc, nocc), dtype=np.complex128)
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        compressed = contract('tLr,LKz->trKz', compressed, expLk_conjz)
        compressed = compressed.view(np.complex128)[:,:,:,0]
        # *.5 because diagonal blocks are accessed twice
        compressed[diag_idx] *= .5
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            # TODO: decompress the j3c tensor using rsdf_builder._unpack_cderi_v2
            j3c = j3c_full[:,:dk]
            j3c[pair_addresses] = compressed[:,k0:k1]
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
            j3c_ij[order_KJ] += j3c_tmp.reshape(nkpts**2,-1)
            j3c_ij = j3c_ij.reshape(nkpts, nkpts, nao, nao, dk)

            tmp = ndarray((nkpts, nkpts, nocc, nao, dk), dtype=np.complex128, buffer=buf2)
            contract('IJpqr,Ipi->IJiqr', j3c_ij, dm_factor_r, out=tmp)
            contract('IJiqr,Jqj->rIJij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
    j3c_full = buf = buf1 = buf2 = eval_j3c = None
    compressed = j3c = j3c_tmp = j3c_ij = tmp = None
    t0 = log.timer_debug1('contract dm', *t0)

    kpt_iters = list(kk_adapted_iter(int3c2e_opt.bvk_kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    nkpts_uniq = len(uniq_kpts)

    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, int3c2e_opt.omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell, int3c2e_opt.bvk_kmesh)
    j2c = int2c2e_opt.int2c2e(uniq_kpts, sort_output=False)
    if j2c.dtype == np.float64:
        j2c = j2c.astype(np.complex128)

    ################################
    # LR part 0th order
    mesh = int3c2e_opt.mesh
    log.debug('mesh for LR coulG %s', mesh)
    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    assert ft_opt.permutation_symmetry
    ft_kern = ft_opt.gen_ft_kernel(transform_ao=False)

    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
    with_long_range = omega < int3c2e_opt.omega
    if with_long_range:
        mesh = int3c2e_opt.mesh
    else:
        assert cell.dimension == 3
        mesh = [1] * 3
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

    def lr_3c2e(j3c_oo):
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        Gblksize = int(mem_avail*.8//((nao*2+nocc)*nao*16*nkpts))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
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
                Gpq = ft_kern(Gv[p0:p1], kpts[kp], kpts, kj_idx)
                pqG, Gpq = Gpq.transpose(0,2,3,1)[kj_idx], None
                tmp = contract('kpqG,kpi->kiqG', pqG, dm_factor_r)
                ijG = contract('kiqG,kqj->kijG', tmp, dm_factor_l[kj_idx])
                j3c_oo[:,ki_idx,kj_idx] += contract(
                    'rG,kijG->rkij', auxGw[:,j2c_idx], ijG)
                if kp != kp_conj:
                    tmp = contract('kqpG,kpi->kiqG', pqG.conj(), dm_factor_r[kj_idx])
                    ijG = contract('kiqG,kqj->kijG', tmp, dm_factor_l)
                    j3c_oo[:,kj_idx,ki_idx] += contract(
                        'rG,kijG->rkij', auxGw[:,j2c_idx].conj(), ijG)
                pqG = None
        return j3c_oo
    j3c_oo = lr_3c2e(j3c_oo)

    ################################
    # (d/dX P|Q) contributions
    j2c = auxcell.apply_CT_mat_C(j2c)
    j_factor /= nkpts**2
    k_factor /= nkpts**2
    aux_coeff = cp.asarray(auxcell.ctr_coeff)
    dm_oo = j3c_oo
    buf = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
    buf1 = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
    dm_aux = cp.empty((nkpts_uniq, naux, naux), dtype=np.complex128)
    # Contractions for kp and kp_conj are complex conjugated.
    # A factor of 2 is applied due to this symmetry.
    time_reversal_sym_weights = cp.full(nkpts_uniq, 2.)
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        j2c_k = j2c[j2c_idx]
        if kp == kp_conj:
            j2c_k = j2c_k.real
        solve_j2c = rhf._gen_metric_solver(
            j2c_k, linear_dep_threshold, auxcell.dimension)
        metric = aux_coeff.dot(solve_j2c(aux_coeff.T))
        j3c_oo_k = j3c_oo[:,ki_idx,kj_idx]
        dm_oo_k = contract('uv,vnij->unij', metric, j3c_oo_k, out=buf)
        dm_oo[:,ki_idx,kj_idx] = dm_oo_k
        if kp == 0:
            dm_oo_kconj = dm_oo_k
        elif kp == kp_conj:
            # for kp == kp_conj != 0, dm_oo_kconj and dm_oo_k correspond to
            # the same blocks in dm_oo, which has been updated previously
            dm_oo_kconj = dm_oo[:,kj_idx,ki_idx]
        else:
            j3c_oo_k = j3c_oo[:,kj_idx,ki_idx]
            dm_oo_kconj = contract('vu,vnij->unij', metric, j3c_oo_k, out=buf1)
            dm_oo[:,kj_idx,ki_idx] = dm_oo_kconj

        beta = 0
        if j_factor != 0 and kp == 0:
            dm_sorted = contract('kpi,kqi->kpq', dm_factor_l, dm_factor_r)
            assert all(ki_idx == kj_idx)
            auxvec = cp.einsum('unii->u', dm_oo_k)
            cp.multiply(auxvec[:,None], auxvec.conj(), out=dm_aux[j2c_idx])
            beta = j_factor

        dm_aux_k = contract('rkij,skji->rs', dm_oo_k, dm_oo_kconj,
                            alpha=-.5*k_factor, beta=beta, out=dm_aux[j2c_idx])
        if kp == kp_conj:
            time_reversal_sym_weights[j2c_idx] = 1
        metric = j3c_oo_k = dm_oo_k = dm_oo_kconj = dm_aux_k = None
    ejk, sigma = int2c2e_opt.energy_derivatives(
        dm_aux * time_reversal_sym_weights[:,None,None], uniq_kpts,
        omega=-int3c2e_opt.omega)
    ejk = cp.asarray(-ejk)
    sigma = cp.asarray(-sigma)
    j2c = j3c_oo = None
    aux_coeff = buf = buf1 = None
    t0 = log.timer_debug1('contract int2c2e_deriv', *t0)

    ################################
    # LR part response
    def lr_3c2e_response():
        Gk = (asarray(Gv) + asarray(uniq_kpts)[:,None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(auxcell, Gk, cp.zeros(3), mesh)
        Gk = Gk.reshape(nkpts_uniq, ngrids, 3)

        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        Gblksize = int(mem_avail*.8//((nao*2+nocc)*nao*16*nkpts))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        kern = libpbc.PBC_ft_aopair_ek_deriv
        kern_auxG = libpbc.PBC_ft_ao_deriv
        ejk_lr = cp.zeros((cell.natm, 3))
        ejk_aux = cp.zeros((cell.natm, 3))
        sigma_lr = cp.zeros((3, 3))
        sigma_aux = cp.zeros((3, 3))
        sigma_G = cp.zeros((3, 3))
        aux_ft_envs = RysIntEnvVars.new(
            auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
            _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)
        null_ptr = lib.c_null_ptr()
        vG = cp.empty(ngrids, dtype=np.complex128)
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gk[:,p0:p1].reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)

            # (ij|r)^{[0]} * metric * (r|G)^{[1]} (ji|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                Gpq = ft_kern(Gv[p0:p1], kpts[kp], kpts, kj_idx)
                pqG, Gpq = Gpq.transpose(0,2,3,1)[kj_idx], None

                beta = 0
                dm_auxG = ndarray((naux,nGv), dtype=np.complex128, buffer=buf2)
                if j_factor != 0 and kp == 0:
                    rhoGz = cp.einsum('kpqG,kqp->G', pqG, dm_sorted)
                    cp.multiply(auxvec[:,None], rhoGz, out=dm_auxG)
                    beta = j_factor
                # einsum('pqG,pi,qj,rij,Gx,rG->rx', pqG, c, c, dm_oo, 1j*Gv, conj(auxG))
                tmp = contract('kpqG,kpi->kiqG', pqG, dm_factor_r)
                ijG = contract('kiqG,kqj->kijG', tmp, dm_factor_l[kj_idx])
                # (ji|r)^{[0]} * metric * (r|G)^{[1]} (G|ij)^{[0]}
                # contracting all [0] order terms -> dm_auxG
                dm_oo_k = dm_oo[:,kj_idx,ki_idx]
                contract('rkji,kijG->rG', dm_oo_k, ijG, -.5*k_factor, beta, out=dm_auxG)

                # (ji|r)^{[0]} * metric * -J2c^{[1]} * metric * (ij|s)^{[0]}
                # = -(ji|r)^{[0]} * metric * (r|G)^{[1]} (G|s)^{[0]} * metric * (ij|s)^{[0]}
                dm_auxG1 = contract('sr,sG->rG', dm_aux[j2c_idx], auxG[:,j2c_idx])
                vG_metric = cp.einsum(
                    'rg,rg->g', dm_auxG1, auxG[:,j2c_idx].conj()).real
                vG_total = cp.einsum(
                    'rg,rg->g', dm_auxG, auxG[:,j2c_idx].conj()).real * 2
                vG_total -= vG_metric
                sym_fac = 2 if kp != kp_conj else 1
                sigma_G += .5 * sym_fac * cp.einsum(
                    'g,xyg->xy', vG_total, wcoulG_LR1[j2c_idx,:, :,p0:p1])
                dm_auxG -= dm_auxG1
                dm_auxG *= wcoulG_LR0[j2c_idx, p0:p1]
                if kp != kp_conj:
                    dm_auxG *= 2
                dm_auxG = dm_auxG.view(np.float64)

                # contract to (r|G)^{[1]} = einsum('ag,ag->a', (iG IFT(aux)), dm_auxG)
                # when kp != kp_conj, contributions of kp_conj are identical to the kp part.
                #:if kp != kp_conj:
                #:    tmp = contract('kqpG,kpi->kiqG', pqG.conj(), dm_factor_r[kj_idx])
                #:    ijG = contract('kiqG,kqj->kijG', tmp, dm_factor_l)
                #:    dm_auxG = contract('rkji,kijG->rG', dm_oo[:,ki_idx,kj_idx], ijG, -.5*k_factor)
                #:    dm_auxG -= contract('rs,sG->rG', dm_aux[j2c_idx], auxG[:,j2c_idx].conj())
                #:    dm_auxG *= coulG_LR[j2c_idx, p0:p1]
                #:    dm_auxG = dm_auxG.view(np.float64)
                #:    for i in range(3):
                #:        ip_auxG = auxG[:,j2c_idx].conj() * (1j*Gk[j2c_idx,p0:p1,i])
                #:        partial_daux[i] += cp.einsum('ag,ag->a', ip_auxG.view(np.float64), dm_auxG)
                GkT = cp.asarray(Gk[j2c_idx,p0:p1].T.ravel())
                err = kern_auxG(
                    ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sigma_aux.data.ptr, ctypes.c_void_p),
                    null_ptr,
                    ctypes.cast(dm_auxG.data.ptr, ctypes.c_void_p),
                    ctypes.cast(GkT.data.ptr, ctypes.c_void_p),
                    ctypes.byref(aux_ft_envs), ctypes.c_int(nGv))
                if err != 0:
                    raise RuntimeError('ft_ao_deriv failed')

                # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
                auxG_conj = auxG[:,j2c_idx].conj()
                auxG_conj *= wcoulG_LR0[j2c_idx,p0:p1]
                # Note: PBC_ft_aopair_ek_deriv kernel only processes the tril part.
                # dm_oo must be symmetric
                dm_ooG = contract('rkji,rG->kijG', dm_oo_k, auxG_conj)
                tmp = contract('kijG,kpi->kpjG', dm_ooG, dm_factor_r)
                dm_vG = contract('kpjG,kqj->kpqG', tmp, dm_factor_l[kj_idx], -.5*k_factor)
                LpqG = contract('Lk,kpqG->LqpG', expLk[:,kj_idx], dm_vG)
                if ft_opt.permutation_symmetry:
                    #TODO: This transformation is likely identical to the
                    # previous one. Scale LpqG by a factor of two instead.
                    LpqG += contract('Lk,kpqG->LpqG', expLk.conj(), dm_vG)

                if j_factor != 0 and kp == 0:
                    vG = auxvec.dot(auxG_conj) * j_factor
                    if ft_opt.permutation_symmetry:
                        vG *= 2
                    bvk_dm = contract('Lk,kpq->Lpq', expLk, dm_sorted)
                    LpqG += bvk_dm[:,:,:,None] * vG

                if kp != kp_conj:
                    # The contribution of the kp_conj can be computed using the
                    # following code. Their contribution is identical to the kp part.
                    LpqG *= 2
                    #:auxG1 = ft_ao.ft_ao(auxcell, (Gv+kpts[kp_conj])).T
                    #:auxG_conj = auxG1.conj()
                    #:auxG_conj *= _weighted_coulG_LR(auxcell, Gv, omega, kws, kpts[kp_conj])
                    #:dm_oo_k = dm_oo[:,ki_idx,kj_idx]
                    #:dm_ooG = contract('rkji,rG->kijG', dm_oo_k, auxG_conj)
                    #:tmp = contract('kijG,kpi->kpjG', dm_ooG, dm_factor_r[kj_idx])
                    #:kpqG = contract('kpjG,kqj->kpqG', tmp, dm_factor_l, -.5*k_factor)
                    #:dm_vG = contract('Lk,kpqG->LqpG', expLk, kpqG)
                    #:dm_vG += contract('Lk,kpqG->LpqG', expLk[:,kj_idx].conj(), kpqG)
                    #:dm_vG = cp.asarray(dm_vG, order='C')
                    #:GvT = cp.asarray((Gv[p0:p1]+kpts[kp_conj]).T.ravel())
                    #:err = kern(
                    #:    ctypes.cast(ejk_lr.data.ptr, ctypes.c_void_p),
                    #:    ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                    #:    ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                    #:    ctypes.byref(aft_envs),
                    #:    ctypes.c_int(nbatches_shl_pair),
                    #:    ctypes.c_int(nGv),
                    #:    ctypes.c_int(shm_size),
                    #:    ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                    #:    ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                    #:    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                    #:    ctypes.c_int(ft_opt.permutation_symmetry))
                    #:if err != 0:
                    #:    raise RuntimeError('PBC_ft_aopair_ek_deriv failed')
                dm_vG = cp.asarray(LpqG, order='C')

                GvT = cp.asarray((Gv[p0:p1]+kpts[kp]).T.ravel())
                err = kern(
                    ctypes.cast(ejk_lr.data.ptr, ctypes.c_void_p),
                    ctypes.cast(sigma_lr.data.ptr, ctypes.c_void_p),
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
                dm_oo_k = dm_ooG = tmp = dm_vG = LpqG = None

        ejk_lr += ejk_aux
        sigma_lr += sigma_aux + sigma_G
        return ejk_lr, sigma_lr

    ejk_lr, sigma_lr = lr_3c2e_response()
    ejk += ejk_lr
    sigma += sigma_lr
    log.timer_debug1('LR coulomb', *t0)
    ft_opt = ft_kern = None
    dm_aux = None

    ################################
    # SR int3c2e response
    # contract the derivatives and the pseudo DM/rho
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
    shl_pair_batch_size = rhf._get_shl_pair_batch_size(
        nksh_per_batch, bvk_ncells)
    bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block=shl_pair_batch_size)
    ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache, cart=True)
    aux_loc = auxcell.ao_loc

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    order_KI = (ijk_conserv.T * nkpts + cp.arange(nkpts)[:,None]).ravel()
    ejk_sr = cp.zeros((cell.natm, 3))
    ejk_aux_sr = cp.zeros((cell.natm, 3))
    sigma_sr = cp.zeros((3, 3))
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
    head = pool[-1:]
    task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_deriv
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size*bvk_ncells))
    buf1 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    buf2 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
        compressed = ndarray((nao_pair, bvk_ncells, naux_in_batch), buffer=buf)
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
            contract('rIJij,Jqj->IJiqr', dm_oo[aux0:aux1], dm_factor_r, -.5*k_factor, out=tmp)
            contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_l, out=dm_tensor)
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
            #:compressed[:,:,k0:k1] = dm_tensor[cgto_pair_addresses]
            cp.take(dm_tensor, pair_addresses, axis=0, out=compressed[:,:,k0:k1])
        err = kern(
            ctypes.cast(ejk_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma_sr.data.ptr, ctypes.c_void_p),
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
    ejk_sr += ejk_aux_sr
    ejk += ejk_sr * 2
    sigma += sigma_sr * 2
    t0 = log.timer_debug1('contract int3c2e_ejk_deriv', *t0)

    ejk = ejk.get()
    sigma = sigma.get()

    if (exxdiv == 'ewald' and
        (cell.dimension == 3 or
         (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
        bvk_kmesh = int3c2e_opt.bvk_kmesh
        s0 = int1e.int1e_ovlp(cell, kpts, bvk_kmesh)
        s1 = int1e.int1e_ipovlp(cell, kpts, bvk_kmesh)
        k_dm = contract('kpq,kqr->kpr', dm, s0)
        k_dm = contract('kpr,krs->kps', k_dm, dm)
        # The cell object reorders the AOs. s1 and k_dm are stored in the order
        # of the original cell. It's necessary to pass the original cell to
        # contract_h1e_dm
        ejk_ewald = contract_h1e_dm(cell.cell, s1, k_dm, hermi=1)
        weighted_coulG_at_G0 = madelung(cell, kpts, omega=-omega)
        # The k_factor was previously scaled by 1/nkpts^2. The ewald term
        # requires a factor of 1/nkpts. Rescale k_factor by nkpts
        ewald_k_factor = k_factor * nkpts
        # Note the additional minus sign for nabla_A ovlp = -nabla ovlp
        ejk_ewald *= .5 * ewald_k_factor * weighted_coulG_at_G0
        ejk += ejk_ewald

        ek_G0 = float(cp.einsum('kij,kji->', s0, k_dm).real.get()) / nkpts
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell.cell, kpts, -omega)
        # *.5 for the factor 1/2 in Coulomb operator; second *.5 for J-K/2 in RHF
        fac = ewald_k_factor * .5 * .5
        sigma -= fac * exx_1 * ek_G0
        # *2 due to (d/dX ij|kl) + (ij|d/dX kl)
        sigma -= 2 * fac * exx_0 * int1e.ovlp_strain_deriv(cell.cell, k_dm, kpts)
    return ejk, sigma

def _get_ej_strain_deriv(int3c2e_opt, dm, kpts=None, hermi=0, omega=None,
                       verbose=None, linear_dep_threshold=LINEAR_DEP_THR):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    from gpu4pyscf.pbc.grad.rks_stress import (
        _get_weighted_coulG_strain_derivatives as get_wcoulG)
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    dm = cell.apply_C_mat_CT(dm)
    if hermi != 1:
        dm = transpose_sum(dm, inplace=True)
        dm[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dm, kpts, hermi=1)
    t0 = log.timer_debug1('contract dm', *t0)

    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    aux_loc = auxcell.ao_loc
    nao = dm.shape[-1]
    naux = int(aux_loc[-1])

    if kpts is None or is_zero(kpts):
        dm = cp.asarray(dm.real, order='C')
        nkpts = 1
    else:
        assert len(int3c2e_opt.bvkmesh_Ls) == len(kpts)
        nkpts = len(kpts)
        #:expLk = cp.exp(1j*asarray(int3c2e_opt.bvkmesh_Ls).dot(asarray(kpts).T))
        expLk = fft_matrix(int3c2e_opt.bvk_kmesh)
        dm = contract('Lk,kpq->Lpq', expLk, dm)
        dm = cp.asarray(dm.real, order='C')
        dm *= 1./nkpts

    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, int3c2e_opt.omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)

    ################################
    # LR part 0th order
    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
    with_long_range = omega < int3c2e_opt.omega
    if with_long_range:
        mesh = int3c2e_opt.mesh
        log.debug('mesh for LR coulG %s', mesh)
        Gv, _, kws = get_Gv_weights(cell, mesh)
        ngrids = len(Gv)
        wcoulG_LR0, wcoulG_LR1 = get_wcoulG(
            cell, Gv, int3c2e_opt.omega)
        if omega != 0:
            wcoulG_0, wcoulG_1 = get_wcoulG(cell, Gv, omega)
            wcoulG_LR0 -= wcoulG_0
            wcoulG_LR1 -= wcoulG_1
        wcoulG_SR_at_G0 = np.pi / int3c2e_opt.omega**2 * kws
        wcoulG_LR0[0] -= wcoulG_SR_at_G0
        wcoulG_LR1[:,:,0] += wcoulG_SR_at_G0 * cp.eye(3)
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
    else:
        assert cell.dimension == 3

    def lr_3c2e():
        eval_ft = ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]
        pair_addresses, diag_idx = ft_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)
        # To fold the upper triangular part of dm[i(0),j(L)] into the lower
        # triangular part, the transformations are
        # dm_tril = contract('LK,Kji->iLj', expLk, dm)
        # dm_triu = contract('LK,Kji->jLi', expLk.conj(), dm)
        # (dm_tril+dm_triu).real.ravel()[pair_addresses]
        # Notice dm_triu == contract('LK,Kji->jLi', expLk, dm.T).conj()
        #                == contract('LK,Kji->iLj', expLk, dm).conj()
        #                == dm_tril.conj()
        # (dm_tril+dm_triu).real is identical to 2*dm.transpose(2,0,1).real
        i_addr, j_addr = divmod(pair_addresses, bvk_ncells * nao)
        dm_tril = dm.reshape(bvk_ncells*nao, nao).real[j_addr, i_addr]
        dm_tril[diag_idx] *= .5
        dm_tril *= 2

        mem_avail = get_avail_mem(exclude_memory_pool=True)
        nao_pair = len(dm_tril)
        Gblksize = int(mem_avail*.8//((nao_pair+naux*2)*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)

        auxvec_LR = cp.zeros(naux)
        rhoG = cp.empty(ngrids, dtype=np.complex128)
        buf  = cp.empty(max(nao_pair,naux)*Gblksize, dtype=np.complex128)
        buf1 = cp.empty((naux,Gblksize), dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            pqG = eval_ft(Gv[p0:p1], out=buf)
            rhoGz = cp.einsum('pG,p->G', pqG.view(np.float64), dm_tril)
            rhoG[p0:p1] = rhoGz.view(np.complex128)

            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            auxGw = ndarray((naux, nGv), dtype=np.complex128, buffer=buf1)
            cp.multiply(auxG, wcoulG_LR0[p0:p1], out=auxGw)
            auxGw = auxGw.view(np.float64)
            contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)
            auxvec_LR += auxGw.dot(rhoGz)
        return auxvec_LR, rhoG

    if with_long_range:
        auxvec_LR, rhoG = lr_3c2e()
        auxvec += auxcell.apply_CT_dot(auxvec_LR)
        auxvec_LR = None
        t0 = log.timer_debug1('lr_int2c2e via aft', *t0)

    ################################
    # (d/dX P|Q) contributions
    j2c = auxcell.apply_CT_mat_C(j2c)
    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        auxvec = rhf._gen_metric_solver(
            j2c, linear_dep_threshold, auxcell.dimension)(auxvec)
    auxvec = auxcell.C_dot_mat(auxvec)
    j2c = None

    dm_aux = auxvec[:,None] * auxvec
    ej, sigma = int2c2e_opt.energy_derivatives(dm_aux, omega=-int3c2e_opt.omega)
    ej = cp.asarray(-ej)
    sigma = cp.asarray(-sigma)
    dm_aux = None
    t0 = log.timer_debug1('contract int2c2e_deriv', *t0)

    #########################
    # LR part response
    def lr_3c2e_response():
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        Gblksize = int(mem_avail*.8//(naux*2*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        rho_auxG = cp.empty(ngrids, dtype=np.complex128)
        buf = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            rho_auxG[p0:p1] = auxvec.dot(
                auxG.view(np.float64)).view(np.complex128)

        vG = (rhoG - rho_auxG) * wcoulG_LR0
        GvT = cp.asarray(Gv.T.ravel())
        ej_aux = cp.zeros((cell.natm, 3))
        sigma_aux = cp.zeros((3, 3))
        aux_ft_envs = RysIntEnvVars.new(
            auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
            _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)
        err = libpbc.PBC_ft_ao_deriv(
            ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.cast(vG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aux_ft_envs), ctypes.c_int(ngrids))
        if err != 0:
            raise RuntimeError('ft_ao_deriv failed')

        ej_lr = cp.zeros((cell.natm, 3))
        sigma_lr = cp.zeros((3, 3))
        vG_conj = rho_auxG.conj() * wcoulG_LR0
        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        err = libpbc.PBC_ft_aopair_ej_deriv(
            ctypes.cast(ej_lr.data.ptr, ctypes.c_void_p),
            ctypes.cast(sigma_lr.data.ptr, ctypes.c_void_p),
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

        ej_lr *= 2
        ej_lr += ej_aux
        sigma_lr *= 2
        sigma_lr += sigma_aux
        sigma_lr += .5 * cp.einsum(
            'g,g,xyg->xy', rho_auxG, (rhoG*2-rho_auxG).conj(),
            wcoulG_LR1).real
        return ej_lr, sigma_lr

    if with_long_range:
        ej_lr, sigma_lr = lr_3c2e_response()
        ej += ej_lr
        sigma += sigma_lr
        t0 = log.timer_debug1('lr_int3c2e_deriv via aft', *t0)
        ft_opt = None

    ################################
    # SR int3c2e response
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        gout_width=54, deriv=(1,0,0))
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxcell.uniq_l_ctr, POOL_SIZE)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

    nksh_per_batch = ksh_offsets_cpu[1:] - ksh_offsets_cpu[:-1]
    shl_pair_batch_size = rhf._get_shl_pair_batch_size(
        nksh_per_batch, bvk_ncells)
    bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block=shl_pair_batch_size)

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    ej_sr = cp.zeros((cell.natm, 3))
    ej_aux_sr = cp.zeros((cell.natm, 3))
    sigma_sr = cp.zeros((3, 3))
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
    head = pool[-1:]
    task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_deriv
    err = kern(
        ctypes.cast(ej_sr.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux_sr.data.ptr, ctypes.c_void_p),
        ctypes.cast(sigma_sr.data.ptr, ctypes.c_void_p),
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
    ej_sr += ej_aux_sr
    ej += ej_sr * 2
    sigma += sigma_sr * 2
    t0 = log.timer_debug1('contract int3c2e_ejk_deriv', *t0)
    return ej.get(), sigma.get()
