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

import math
import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ATOM_OF
from pyscf.pbc.tools.k2gamma import double_translation_indices
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, unpack_tril, transpose_sum, get_avail_mem)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, diffuse_exps_by_atom, _aggregate_bas_idx, POOL_SIZE)
from gpu4pyscf.pbc.tools.pbc import get_coulG, _Gv_wrap_around
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.grad import rhf
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern, get_ao_pair_loc, int3c2e_scheme, factorize_dm)
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_LR
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.tools.pbc import madelung
from gpu4pyscf.pbc.lib.kpts_helper import (
    fft_matrix, kk_adapted_iter, conj_images_in_bvk_cell)


def _jk_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, with_long_range=True, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if kpts is None:
        assert dm.ndim == 2
        return rhf._jk_energy_per_atom(
            int3c2e_opt, dm, hermi, j_factor, k_factor, exxdiv, with_long_range, verbose)

    if hermi == 2:
        j_factor = 0

    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm, kpts, hermi, with_long_range,
                                  verbose) * j_factor

    # Must be symmetric density matrices, otherwise, dm_tensor needs to be
    # symmetrized since PBCsr_ejk_int3c2e_ip1 only handles the tril pairs
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
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, buffer_size // ((nao*bvk_ncells)**2*8)))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    # k=ijk_conserv[i,j] provides: -i + j - k = 2n\pi
    # therefore, i=ijk_conserv[k,j]
    ijk_conserv = double_translation_indices(int3c2e_opt.bvk_kmesh)
    #for ki in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ki,kj] += j3c_tmp[ijk_conserv[ki,kj],ki]
    #        => order_KI = argsort([ki,ijk_conserv[ki,kj]])
    order_KI = cp.asarray(
        np.argsort((ijk_conserv * nkpts + np.arange(nkpts)[:,None]).ravel()))
    #for kk in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ijk_conserv[kk,kj],kj] += j3c_tmp[kk,kj]
    #        => order_KJ = [ijk_conserv[kk,kj],kj]
    order_KJ = cp.asarray((ijk_conserv * nkpts + np.arange(nkpts)).ravel())

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
        compressed = contract('trL,LKz->trKz', compressed, expLk_conjz)
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
    omega = abs(int3c2e_opt.omega)
    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell, int3c2e_opt.bvk_kmesh)
    j2c = int2c2e_opt.int2c2e(uniq_kpts, sort_output=False)
    if j2c.dtype == np.float64:
        j2c = j2c.astype(np.complex128)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        log.debug('mesh for LR coulG %s', mesh)
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        assert ft_opt.permutation_symmetry
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=False)
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        ngrids = Gv.shape[0]

        # The _weighted_coulG_LR() for multiple k-points.
        # To ensure the symmetry between conjugated k-points, it is important to
        # wrap around the high-freq Gv.
        assert Gv[0].dot(Gv[0]) == 0
        Gk = (asarray(Gv) + asarray(uniq_kpts)[:,None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(auxcell, Gk, cp.zeros(3), mesh)
        coulG_LR = get_coulG(auxcell, Gv=Gk, omega=omega).reshape(-1, ngrids)
        coulG_LR *= kws
        coulG_LR[0,0] -= np.pi / omega**2 / auxcell.vol
        Gk = Gk.reshape(nkpts_uniq, ngrids, 3)

        mem_avail = get_avail_mem(exclude_memory_pool=True)
        Gblksize = int(mem_avail//((nao*2+nocc)*nao*16*nkpts))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, (Gv[p0:p1]+uniq_kpts[:,None]).reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)
            auxGw = auxG.conj()
            auxGw *= coulG_LR[:,p0:p1]
            contract('iKG,jKG->Kij', auxGw, auxG, beta=1, out=j2c)

            permuted_auxGw = auxG
            permuted_auxGw[aux_sorting] = auxGw
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                Gpq = ft_kern(Gv[p0:p1], kpts[kp], kpts, kj_idx)
                pqG, Gpq = Gpq.transpose(0,2,3,1)[kj_idx], None
                tmp = contract('kpqG,kpi->kiqG', pqG, dm_factor_r)
                ijG = contract('kiqG,kqj->kijG', tmp, dm_factor_l[kj_idx])
                j3c_oo[:,ki_idx,kj_idx] += contract(
                    'rG,kijG->rkij', permuted_auxGw[:,j2c_idx], ijG)
                if kp != kp_conj:
                    tmp = contract('kqpG,kpi->kiqG', pqG.conj(), dm_factor_r[kj_idx])
                    ijG = contract('kiqG,kqj->kijG', tmp, dm_factor_l)
                    j3c_oo[:,kj_idx,ki_idx] += contract(
                        'rG,kijG->rkij', permuted_auxGw[:,j2c_idx].conj(), ijG)
                pqG = None
        tmp = ijG = auxG = auxGw = permuted_auxGw = None

    j2c = auxcell.apply_CT_mat_C(j2c)
    j2c_ip1 = auxcell.pbc_intor('int2c2e_ip1', kpts=uniq_kpts)

    aux_coeff = cp.asarray(auxcell.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    j_factor /= nkpts**2
    k_factor /= nkpts**2
    dm_oo = j3c_oo
    ejk = np.zeros((cell.natm, 3))
    buf = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
    buf1 = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
    dm_aux = cp.empty((nkpts_uniq, naux, naux), dtype=np.complex128)
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        metric = aux_coeff.dot(cp.linalg.solve(j2c[j2c_idx], aux_coeff.T))
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
        j2c_k = asarray(j2c_ip1[j2c_idx])
        dm_aux_k = dm_aux_k[aux_sorting[:,None],aux_sorting]
        if kp == kp_conj:
            ejk += contract_h1e_dm(auxcell, j2c_k, dm_aux_k, hermi=0)
        else:
            # The following contractions for kp and kp_conj are complex conjugated.
            # A factor of 2 is applied due to this symmetry.
            #:ejk += contract_h1e_dm(auxcell, j2c_k, dm_aux_k, hermi=0)
            #:_dm_aux = contract('rkij,skji->rs', dm_oo_kconj, dm_oo_k, alpha=-.5*k_factor)
            #:ejk += contract_h1e_dm(
            #:    auxcell, j2c_k.conj(), _dm_aux[aux_sorting[:,None],aux_sorting], hermi=0)
            ejk += 2 * contract_h1e_dm(auxcell, j2c_k, dm_aux_k, hermi=0)
        metric = j3c_oo_k = dm_oo_k = dm_oo_kconj = dm_aux_k = j2c_k = None
    ejk *= .5
    j2c = j2c_ip1 = j3c_oo = None
    aux_coeff = buf = buf1 = None
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    if with_long_range:
        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        kern = libpbc.PBC_ft_aopair_ek_ip1
        ejk_lr = cp.zeros((cell.natm, 3))
        partial_daux = cp.zeros((3, naux))
        vG = cp.empty(ngrids, dtype=np.complex128)
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            # the auxliary dimension of dm_oo and dm_aux are regrouped and
            # permuted. Instead of sorting dm_oo (dm_oo[aux_sorting]) and
            # dm_aux, we reorder auxG here.
            auxG = ft_ao.ft_ao(auxcell, (Gv[p0:p1]+uniq_kpts[:,None]).reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)
            permuted_auxG = cp.empty_like(auxG)
            permuted_auxG[aux_sorting] = auxG
            auxG, permuted_auxG = permuted_auxG, None

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
                contract('sr,sG->rG', dm_aux[j2c_idx], auxG[:,j2c_idx], -1, 1, out=dm_auxG)
                dm_auxG *= coulG_LR[j2c_idx, p0:p1]
                dm_auxG = dm_auxG.view(np.float64)

                # contract to (r|G)^{[1]} = einsum('ag,ag->a', (iG IFT(aux)), dm_auxG)
                # when kp != kp_conj, contributions of kp_conj are identical to
                # the kp part.
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
                for i in range(3):
                    ip_auxG = auxG[:,j2c_idx] * (-1j*Gk[j2c_idx,p0:p1,i])
                    ip_auxG = ip_auxG.view(np.float64)
                    if kp != kp_conj:
                        partial_daux[i] += 2 * cp.einsum('ag,ag->a', ip_auxG, dm_auxG)
                    else:
                        partial_daux[i] += cp.einsum('ag,ag->a', ip_auxG, dm_auxG)

                # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
                auxG_conj = auxG[:,j2c_idx].conj()
                auxG_conj *= coulG_LR[j2c_idx,p0:p1]
                # Note: PBC_ft_aopair_ek_ip1 kernel only processes the tril part.
                # dm_oo must be symmetric
                dm_ooG = contract('rkji,rG->kijG', dm_oo_k, auxG_conj)
                tmp = contract('kijG,kpi->kpjG', dm_ooG, dm_factor_r)
                dm_vG = contract('kpjG,kqj->kpqG', tmp, dm_factor_l[kj_idx], -.5*k_factor)
                LpqG = contract('Lk,kpqG->LqpG', expLk[:,kj_idx], dm_vG)
                if ft_opt.permutation_symmetry:
                    #TODO: This transformation is likely identical to the
                    # previous one. Scale LpqG a factor of two instead.
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
                    #:    raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
                dm_vG = cp.asarray(LpqG, order='C')

                GvT = cp.asarray((Gv[p0:p1]+kpts[kp]).T.ravel())
                err = kern(
                    ctypes.cast(ejk_lr.data.ptr, ctypes.c_void_p),
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
                    raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
                dm_oo_k = dm_ooG = tmp = dm_vG = LpqG = None
        pqG = ijG = tmp = dm_auxG = ip_auxG = None
        auxG = permuted_auxG = auxG_conj = None
        buf = buf1 = buf2 = None
        ft_opt = ft_kern = None

        dims = aux_loc[1:] - aux_loc[:-1]
        atm_id_for_aux = np.repeat(auxcell._bas[:,ATOM_OF], dims)
        partial_daux = partial_daux.T[aux_sorting].get()
        ejk_aux = groupby(atm_id_for_aux, partial_daux, op='sum')
        if len(ejk_aux) < cell.natm:
            ejk[np.unique(atm_id_for_aux)] += ejk_aux
        else:
            ejk += ejk_aux
        ejk += ejk_lr.get()
        log.timer_debug1('LR coulomb', *t0)

    dm_aux = None

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, 1000000)[0]
    ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0],
                                  int3c2e_opt.bas_ij_cache, cart=True)
    aux_loc = auxcell.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxcell.uniq_l_ctr, batch_size)
    ksh_idx = _aggregate_bas_idx(
        l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, auxcell.nbas)[1]
    ksh_idx += int3c2e_opt.bvkcell.nbas
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    order_KI = cp.asarray((ijk_conserv.T * nkpts + np.arange(nkpts)[:,None]).ravel())
    ejk_sr = cp.zeros((cell.natm, 3))
    ejk_aux_sr = cp.zeros((cell.natm, 3))
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size*bvk_ncells))
    buf1 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    buf2 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
        compressed = ndarray((nao_pair, naux_in_batch, bvk_ncells), buffer=buf)
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

            tmp = ndarray((nkpts,nao,nao,dk,bvk_ncells), dtype=np.complex128, buffer=buf2)
            tmp1 = ndarray((nao,bvk_ncells,nao,dk,bvk_ncells), dtype=np.complex128, buffer=buf1)
            dm_tensor = contract('KJpqr,LK->JpqrL', dm_tensor_swap, expLk_conj, out=tmp)
            dm_tensor = contract('JpqrL,NJ->qNprL', dm_tensor, expLk, out=tmp1)
            dm_tensor = dm_tensor.reshape(-1,dk,bvk_ncells).real
            #:compressed[:,k0:k1] = dm_tensor[cgto_pair_addresses]
            cp.take(dm_tensor, pair_addresses, axis=0, out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_int(1),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(naux_in_batch * bvk_ncells),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')
    ejk_sr += ejk_aux_sr
    ejk += ejk_sr.get() * 2
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

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
        # the madelung function by default read the value of cell.omega.
        # cell.omega is not 0, which can lead to incorrect correction for
        # full-range ewald probe charge correction.
        if with_long_range:
            weighted_coulG_at_G0 = madelung(cell, kpts, omega=0)
        else:
            weighted_coulG_at_G0 = madelung(cell, kpts, omega=-omega)
        # The k_factor was previously scaled by 1/nkpts^2. The ewald term
        # requires a factor of 1/nkpts. Rescale k_factor by nkpts
        k_factor *= nkpts
        # Note the additional minus sign for nabla_A ovlp = -nabla ovlp
        ejk_ewald *= .5 * k_factor * weighted_coulG_at_G0
        ejk += ejk_ewald
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0,
                       with_long_range=True, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
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

    omega = abs(int3c2e_opt.omega)
    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        log.debug('mesh for LR coulG %s', mesh)
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft = ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws)
        ngrids = Gv.shape[0]
        Gv = asarray(Gv)

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
        Gblksize = int(mem_avail//((nao_pair+naux*2)*16))//32*32
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
            cp.multiply(auxG, coulG_LR[p0:p1], out=auxGw)
            auxGw = auxGw.view(np.float64)
            contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)
            auxvec_LR += auxGw.dot(rhoGz)
        auxvec += auxcell.apply_CT_dot(auxvec_LR)
        buf = buf1 = pqG = rhoGz = auxvec_LR = auxG = auxGw = None
        eval_ft = dm_tril = None
        t0 = log.timer_debug1('lr_int2c2e via aft', *t0)

    j2c = auxcell.apply_CT_mat_C(j2c)

    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        auxvec = cp.linalg.solve(j2c, auxvec)
    auxvec = auxcell.C_dot_mat(auxvec)
    j2c = None

    # (d/dX P|Q) contributions
    dm_aux = auxvec[:,None] * auxvec
    ej = -int2c2e_opt.energy_ip1_per_atom(dm_aux)
    dm_aux = None
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    if with_long_range:
        Gblksize = int(mem_avail//(naux*2*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        partial_daux = cp.zeros((3, naux))
        vG = cp.empty(ngrids, dtype=np.complex128)
        buf = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            rho_auxG = auxvec.dot(auxG.view(np.float64))
            vG[p0:p1] = rho_auxG.view(np.complex128)
            vG[p0:p1] *= coulG_LR[p0:p1]

            # (ii|r)^{[0]} * metric * (r|G)^{[1]} (jj|G)^{[0]}
            # = auxvec * (r|G)^{[1]} (jj|G)^{[0]}
            # IFT(nabla_A aux) = IFT(-nabla aux) = (iG IFT(aux)) = (iG conj(FT(aux)))
            ip_vG = rhoG[p0:p1] * coulG_LR[p0:p1] * 1j * Gv[p0:p1].T
            # (ii|r)^{[0]} * metric * -J2c^{[1]} * metric * (jj|r)^{[0]}
            # = auxvec * J2c^{[1]} * auxvec
            ip_vG -= vG[p0:p1] * 1j * Gv[p0:p1].T
            partial_daux += cp.einsum('xg,ag->xa', ip_vG.view(np.float64),
                                      auxG.view(np.float64))
        partial_daux *= auxvec
        buf = auxG = None

        dims = aux_loc[1:] - aux_loc[:-1]
        atm_id_for_aux = np.repeat(auxcell._bas[:,ATOM_OF], dims)
        partial_daux = partial_daux.T.real.get()
        ej_aux = groupby(atm_id_for_aux, partial_daux, op='sum')
        if len(ej_aux) < cell.natm:
            ej[np.unique(atm_id_for_aux)] += ej_aux
        else:
            ej += ej_aux
        t0 = log.timer_debug1('lr_int2c2e_ip1 via aft', *t0)

        ej_lr = cp.zeros((cell.natm, 3))
        vG = vG.conj()
        GvT = cp.asarray(Gv.T.ravel())
        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        err = libpbc.PBC_ft_aopair_ej_ip1(
            ctypes.cast(ej_lr.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.cast(vG.data.ptr, ctypes.c_void_p),
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
            raise RuntimeError('PBC_ft_aopair_ej_ip1 failed')
        ej += ej_lr.get() * 2
        t0 = log.timer_debug1('lr_int3c2e_ip1 via aft', *t0)
        ft_opt = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, 1000000)[0]

    uniq_l_ctr_aux = auxcell.uniq_l_ctr
    l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    ksh_idx = _aggregate_bas_idx(
        l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, auxcell.nbas)[1]
    ksh_idx += int3c2e_opt.bvkcell.nbas
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    ej_sr = cp.zeros((cell.natm, 3))
    ej_aux_sr = cp.zeros((cell.natm, 3))
    err = kern(
        ctypes.cast(ej_sr.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux_sr.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_envs),
        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(bas_ij_idx)),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(int3c2e_opt.img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(int3c2e_opt.img_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.c_int(naux),
        ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
        ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff))
    if err != 0:
        raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')
    ej_sr += ej_aux_sr
    ej += ej_sr.get() * 2
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ej
