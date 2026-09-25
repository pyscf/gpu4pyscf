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
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df import aft as aft_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, transpose_sum, get_avail_mem, empty_aligned)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.gto.mole import RysIntEnvVars, _scale_sp_ctr_coeff
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, POOL_SIZE, MAX_IMGS_PER_TASK, int3c2e_scheme, SRInt3c2eOpt,
    _get_shl_pair_per_block, _counts_to_offsets)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.df.rsdf_builder import LINEAR_DEP_THR, _unpack_cderi_v2
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.grad import rhf
from gpu4pyscf.pbc.df.grad.rhf import (
    factorize_dm, get_ao_pair_loc, _split_l_ctr_pattern, indexed_scale)
from gpu4pyscf.pbc.grad.krks_stress import (
    _get_weighted_coulG_strain_derivatives as get_wcoulG)
from gpu4pyscf.pbc.tools.pbc import madelung, _Gv_wrap_around
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.gto.cell import get_Gv_weights
from gpu4pyscf.pbc.lib.kpts_helper import (
    fft_matrix, kk_adapted_iter, conj_images_in_bvk_cell)
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega, estimate_omega_for_ke_cutoff


def _get_ejk_derivatives(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                         exxdiv=None, omega=None, verbose=None,
                         linear_dep_threshold=LINEAR_DEP_THR):
    '''
    Computes the first-order derivatives (nuclear gradients and strain
    derivatives) of the energy contributions from J and K terms per atom.
    '''
    if kpts is None or kpts.ndim == 1:
        assert dm.ndim == 2
        assert dm.dtype == np.float64
        if kpts is not None:
            assert is_zero(kpts)
        return rhf._get_ejk_derivatives(
            int3c2e_opt, dm, hermi, j_factor, k_factor, exxdiv, omega, verbose,
            linear_dep_threshold)

    if hermi == 2:
        j_factor = 0

    if k_factor == 0:
        ej_sigma = _get_ej_derivatives(
            int3c2e_opt, dm, kpts, hermi, omega, verbose, linear_dep_threshold)
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
    dm_factor_l = cell.apply_C_dot(dm_factor_l, axis=1)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l.conj()
    else:
        dm_factor_r = cell.apply_C_dot(dm_factor_r, axis=1)
    nkpts, nao, nocc = dm_factor_l.shape

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

    kpt_iters = kk_adapted_iter(int3c2e_opt.bvk_kmesh)
    kpt_iters = [(kp, kp_conj, cp.asarray(ki_idx, dtype=np.int32),
                  cp.asarray(kj_idx, dtype=np.int32))
                 for kp, kp_conj, ki_idx, kj_idx in kpt_iters]
    uniq_kpts_idx = np.array([x[0] for x in kpt_iters])
    uniq_kpts = kpts[uniq_kpts_idx]
    nkpts_uniq = len(uniq_kpts)

    conj_mapping = cp.asarray(
        conj_images_in_bvk_cell(int3c2e_opt.bvk_kmesh), dtype=np.int32)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    batch_size = blksize = 0
    if n_compact_pairs > 0:
        word_avail = mem_free // 8
        word_avail -= naux*nkpts**2*nocc**2 * 2  # j3c_oo
        batch_words = n_compact_pairs*nkpts_uniq * 2  # compressed j3c
        # Conservatively count both integral and contraction uses of buf1.
        block_words = max(nao**2*bvk_ncells, nkpts*nao*nocc) * 2
        block_words += nao**2*nkpts * 2  # ao_buf
        block_words += nkpts*nocc**2 * 2  # occupied-occupied result
        batch_size = min(naux, int(word_avail*.75/batch_words))
        blksize = min(batch_size, int(word_avail*.2/block_words))
        if batch_size < int(np.diff(aux_loc).max()) or blksize < 1:
            raise RuntimeError('Insufficient GPU memory for GDF gradient buffers')

    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    def sr_int3c2e():
        eval_j3c, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
            aux_batch_size=None if batch_size >= naux else batch_size, cart=True)
        aux_batches = len(aux_offsets) - 1

        max_aux_batch = int(np.diff(aux_offsets).max())
        buf = cp.empty(nkpts_uniq*max_aux_batch*n_compact_pairs*2)
        work = cp.empty(max(block_words*blksize,
                            bvk_ncells*max_aux_batch*n_compact_pairs))
        ao_buf, buf1 = _allocate(nao**2*nkpts*blksize*2, work)
        j3c_oo = cp.empty((naux, nkpts*nkpts, nocc, nocc), dtype=np.complex128)
        aux_start = 0
        for kbatch in range(aux_batches):
            j3c = eval_j3c(aux_batch_id=kbatch, out=work)
            naux_in_batch = j3c.shape[-1]
            compressed = ndarray((nkpts_uniq, naux_in_batch, n_compact_pairs, 2), buffer=buf)
            contract('tLr,Lkz->krtz', j3c, expLk_conjz[:,uniq_kpts_idx], out=compressed)
            compressed = compressed.view(np.complex128)[:,:,:,0]
            for k_j2c, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                compressed_k = compressed[k_j2c]
                # Primitive diagonal blocks are added twice when unpacking.
                compressed_k[:,compact_diag] *= .5
                for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                    aux0, aux1 = aux_start + k0, aux_start + k1
                    j3c = _unpack_cderi_v2(compressed_k[k0:k1],
                        compact_idx, kj_idx, conj_mapping, expLk, nao,
                        axis=0, buf=buf1, out=ao_buf).transpose(0, 2, 3, 1)
                    tmp = ndarray((nkpts,nocc,nao,k1-k0), dtype=np.complex128, buffer=buf1)
                    contract('kpqr,kpi->kiqr', j3c, dm_factor_r, out=tmp)
                    j3c_oo[aux0:aux1,ki_idx*nkpts+kj_idx] = contract(
                        'kiqr,kqj->rkij', tmp, dm_factor_l[kj_idx])
                    if kp != kp_conj:
                        tmp = ndarray((nkpts,nocc,nao,k1-k0), dtype=np.complex128, buffer=buf1)
                        j3c.imag *= -1 # j3c.conj() inplace
                        contract('kqpr,kpi->kiqr', j3c, dm_factor_r[kj_idx], out=tmp)
                        j3c_oo[aux0:aux1,kj_idx*nkpts+ki_idx] = contract(
                            'kiqr,kqj->rkij', tmp, dm_factor_l)
            aux_start += naux_in_batch
        return j3c_oo

    if n_compact_pairs > 0:
        j3c_oo = sr_int3c2e()
        t0 = log.timer_debug1('contract sr_int3c2e dm', *t0)
    else:
        j3c_oo = cp.zeros((naux, nkpts*nkpts, nocc, nocc), dtype=np.complex128)

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

    def unpack_ft(pqG_compressed, kj_idx, work=None, out=None):
        # Full primitive diagonal blocks need half weights before addition.
        pqG_compressed[diag_idx] *= .5
        pqG = _unpack_cderi_v2(
            pqG_compressed.T, pair_addresses, kj_idx, conj_mapping,
            expLk, nao, axis=0, buf=work, out=out)
        return pqG.transpose(0, 2, 3, 1)

    def lr_3c2e(j3c_oo):
        mem_free = get_avail_mem()
        word_avail = mem_free // 8
        word_avail -= naux*nkpts*nocc**2 * 2 # result; accumulation needs no copy
        aux_size = naux*nkpts_uniq
        ao_size = nkpts*nao**2
        unpack_size = nao_pair + bvk_ncells*nao**2
        mo_size = nkpts*nocc*(nao+nocc) + naux
        Gsize = aux_size + max(aux_size, ao_size + max(unpack_size, mo_size))
        Gblksize = min(ngrids, int(word_avail*.8//(Gsize*2))//32*32)
        if Gblksize < 1:
            raise RuntimeError('Insufficient GPU memory for GDF Fourier buffers')
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_free*1e-9, Gblksize)
        buf = cp.empty(Gsize*Gblksize*2)
        work1, buf1 = _allocate(aux_size*Gblksize*2, buf)
        work2, buf2 = _allocate(ao_size*Gblksize*2, buf1)
        work3, buf3 = _allocate(nao_pair*Gblksize*2, buf2)
        result = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            Gk = Gv[p0:p1] + uniq_kpts[:,None]
            auxG = ft_ao.ft_ao(auxcell, Gk.reshape(-1,3), out=work1).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)
            auxGw = ndarray(auxG.shape, dtype=np.complex128, buffer=buf1)
            cp.conjugate(auxG, out=auxGw)
            auxGw *= wcoulG_LR0[:,p0:p1]
            contract('iKG,jKG->Kij', auxGw, auxG, beta=1, out=j2c)
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                pqG_compressed = eval_ft(Gv[p0:p1] + kpts[kp], out=work3)
                pqG_compressed[:n_compact_pairs] *= wcoulG_LR0[j2c_idx,p0:p1]
                if separated_dd:
                    pqG_compressed[n_compact_pairs:] *= wcoulG_FR0[j2c_idx,p0:p1]
                pqG = unpack_ft(pqG_compressed, kj_idx, work=buf3, out=work2)
                kiqG, kijG = _allocate((nkpts, nocc, nao, nGv*2), buf2)
                kiqG = kiqG.view(np.complex128)
                kijG, aux_work = _allocate((nkpts, nocc, nocc, nGv*2), kijG)
                kijG = kijG.view(np.complex128)
                auxG_k = ndarray((naux,nGv), dtype=np.complex128, buffer=aux_work)
                cp.take(auxG, j2c_idx, axis=1, out=auxG_k)
                auxG_k.imag *= -1 # auxG.conj() inplace
                contract('kpqG,kpi->kiqG', pqG, dm_factor_r, out=kiqG)
                contract('kiqG,kqj->kijG', kiqG, dm_factor_l[kj_idx], out=kijG)
                contract('rG,kijG->rkij', auxG_k, kijG, out=result)
                #:j3c_oo[:,ki_idx,kj_idx] += result
                _add_j3c_oo(j3c_oo, result, ki_idx*nkpts+kj_idx)
                if kp != kp_conj:
                    pqG.imag *= -1 # pqG.conj() inplace
                    contract('kqpG,kpi->kiqG', pqG, dm_factor_r[kj_idx], out=kiqG)
                    contract('kiqG,kqj->kijG', kiqG, dm_factor_l, out=kijG)
                    cp.take(auxG, j2c_idx, axis=1, out=auxG_k)
                    contract('rG,kijG->rkij', auxG_k, kijG, out=result)
                    #:j3c_oo[:,kj_idx,ki_idx] += result
                    _add_j3c_oo(j3c_oo, result, kj_idx*nkpts+ki_idx)
        return j3c_oo
    j3c_oo = lr_3c2e(j3c_oo)
    t0 = log.timer_debug1('contract lr_int3c2e dm', *t0)

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
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        j2c_k = j2c[j2c_idx]
        if kp == kp_conj:
            j2c_k = j2c_k.real
        solve_j2c = rhf._gen_metric_solver(
            j2c_k, linear_dep_threshold, auxcell.dimension)
        metric = aux_coeff.dot(solve_j2c(aux_coeff.T))
        j3c_oo_k = j3c_oo[:,ki_idx*nkpts+kj_idx]
        dm_oo_k = contract('uv,vnij->unij', metric, j3c_oo_k, out=buf)
        dm_oo[:,ki_idx*nkpts+kj_idx] = dm_oo_k
        if kp == 0:
            dm_oo_kconj = dm_oo_k
        elif kp == kp_conj:
            # for kp == kp_conj != 0, dm_oo_kconj and dm_oo_k correspond to
            # the same blocks in dm_oo, which has been updated previously
            dm_oo_kconj = dm_oo[:,kj_idx*nkpts+ki_idx]
        else:
            j3c_oo_k = j3c_oo[:,kj_idx*nkpts+ki_idx]
            dm_oo_kconj = contract('vu,vnij->unij', metric, j3c_oo_k, out=buf1)
            dm_oo[:,kj_idx*nkpts+ki_idx] = dm_oo_kconj

        beta = 0
        if j_factor != 0 and kp == 0:
            dm_sorted = contract('kpi,kqi->kpq', dm_factor_l, dm_factor_r)
            assert all(ki_idx == kj_idx)
            auxvec = cp.einsum('unii->u', dm_oo_k)
            cp.multiply(auxvec[:,None], auxvec.conj(), out=dm_aux[j2c_idx])
            beta = j_factor

        contract('rkij,skji->rs', dm_oo_k, dm_oo_kconj,
                 alpha=-.5*k_factor, beta=beta, out=dm_aux[j2c_idx])
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
        mem_avail = get_avail_mem()
        mem_avail -= naux*nkpts*nocc**2 * 16  # dm_oo_k = dm_oo[:,kj_idx*nkpts+ki_idx]
        # Complex elements per G-vector. The two scratch banks alternate
        # between MO contractions, BvK density, and AO unpacking. Keep the
        # compact FT and auxiliary densities separate from those banks.
        aux_size = naux*nkpts_uniq
        ao_size = nkpts*nao**2
        scratch1_size = max(nkpts*nocc**2, nao_pair, bvk_ncells*nao**2, naux)
        scratch2_size = max(nkpts*nao*nocc, bvk_ncells*nao**2)
        Gsize = aux_size + 2*naux + nao_pair + ao_size + scratch1_size + scratch2_size
        Gblksize = min(ngrids, int(mem_avail*.8//(Gsize*16))//32*32)
        if Gblksize < 1:
            raise RuntimeError('Insufficient GPU memory for GDF Fourier buffers')
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
        buf = cp.empty(Gsize*Gblksize*2)
        aux_buf, work = _allocate(aux_size*Gblksize*2, buf)
        conj_buf, work = _allocate(naux*Gblksize*2, work)
        density_buf, work = _allocate(naux*Gblksize*2, work)
        compact_buf, work = _allocate(nao_pair*Gblksize*2, work)
        ao_buf, work = _allocate(ao_size*Gblksize*2, work)
        scratch1, scratch2 = _allocate(scratch1_size*Gblksize*2, work)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gk[:,p0:p1].reshape(-1,3), out=aux_buf).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)

            # (ij|r)^{[0]} * metric * (r|G)^{[1]} (ji|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                dm_oo_k = dm_oo[:,kj_idx*nkpts+ki_idx]
                if kp != kp_conj:
                    dm_oo_k *= 2

                # Orbital response: form the unweighted BvK density first.
                # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
                auxG_conj = ndarray((naux,nGv), dtype=np.complex128, buffer=conj_buf)
                cp.conjugate(auxG[:,j2c_idx], out=auxG_conj)
                dm_ooG = ndarray((nkpts,nocc,nocc,nGv), dtype=np.complex128, buffer=scratch1)
                tmp = ndarray((nkpts,nao,nocc,nGv), dtype=np.complex128, buffer=scratch2)
                dm_vG = ndarray((nkpts,nao,nao,nGv), dtype=np.complex128, buffer=ao_buf)
                contract('rkji,rG->kijG', dm_oo_k, auxG_conj, out=dm_ooG)
                contract('kijG,kpi->kpjG', dm_ooG, dm_factor_r, out=tmp)
                contract('kpjG,kqj->kpqG', tmp, dm_factor_l[kj_idx], -.5*k_factor, out=dm_vG)
                LpqG = ndarray((bvk_ncells,nao,nao,nGv), dtype=np.complex128, buffer=scratch2)
                contract('Lk,kpqG->LqpG', expLk[:,kj_idx], dm_vG, out=LpqG)
                if ft_opt.permutation_symmetry:
                    contract('Lk,kpqG->LpqG', expLk_conj, dm_vG, beta=1, out=LpqG)
                if j_factor != 0 and kp == 0:
                    vG = auxvec.dot(auxG_conj) * j_factor
                    if ft_opt.permutation_symmetry:
                        vG *= 2
                    bvk_dm = contract('Lk,kpq->Lpq', expLk, dm_sorted)
                    contract('Lpq,G->LpqG', bvk_dm, vG, beta=1, out=LpqG)
                    bvk_dm = None
                dm_vG = LpqG.reshape(-1, nGv)
                # Save the unweighted density before indexed_scale modifies it.
                dm_vG_compressed = ndarray((nao_pair,nGv), dtype=np.complex128, buffer=scratch1)
                cp.take(dm_vG, response_idx, axis=0, out=dm_vG_compressed)
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

                pqG_compressed = eval_ft(Gv[p0:p1] + kpts[kp], out=compact_buf)
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
                pqG = unpack_ft(pqG_compressed, kj_idx, work=scratch1, out=ao_buf)

                beta = 0
                dm_auxG = ndarray((naux,nGv), dtype=np.complex128, buffer=density_buf)
                if j_factor != 0 and kp == 0:
                    rhoGz = cp.einsum('kpqG,kqp->G', pqG, dm_sorted)
                    cp.multiply(auxvec[:,None], rhoGz, out=dm_auxG)
                    beta = j_factor
                # einsum('pqG,pi,qj,rij,Gx,rG->rx', pqG, c, c, dm_oo, 1j*Gv, conj(auxG))
                tmp = ndarray((nkpts,nocc,nao,nGv), dtype=np.complex128, buffer=scratch2)
                ijG = ndarray((nkpts,nocc,nocc,nGv), dtype=np.complex128, buffer=scratch1)
                contract('kpqG,kpi->kiqG', pqG, dm_factor_r, out=tmp)
                contract('kiqG,kqj->kijG', tmp, dm_factor_l[kj_idx], out=ijG)
                # (ji|r)^{[0]} * metric * (r|G)^{[1]} (G|ij)^{[0]}
                # contracting all [0] order terms -> dm_auxG
                contract('rkji,kijG->rG', dm_oo_k, ijG, -.5*k_factor, beta, out=dm_auxG)

                # (ji|r)^{[0]} * metric * -J2c^{[1]} * metric * (ij|s)^{[0]}
                # = -(ji|r)^{[0]} * metric * (r|G)^{[1]} (G|s)^{[0]} * metric * (ij|s)^{[0]}
                dm_auxG1 = ndarray((naux,nGv), dtype=np.complex128, buffer=scratch1)
                contract('sr,sG->rG', dm_aux[j2c_idx], auxG[:,j2c_idx], out=dm_auxG1)
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
    t0 = log.timer_debug1('lr_int3c2e_deriv', *t0)
    ft_opt = eval_compact = eval_dd = None
    dm_aux = None

    ################################
    # SR int3c2e response
    # contract the derivatives and the pseudo DM/rho
    if len(int3c2e_opt.img_idx) > 0:
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
        ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache, cart=True)
        aux_loc = auxcell.ao_loc

        diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
        diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
        log_cutoff = math.log(int3c2e_opt.cutoff)

        # The response uses K = ki - kj, opposite to the zero-order integral.
        response_kpts = []
        for kp, kp_conj, ki_idx, kj_idx in kpt_iters:
            response_kpts.append((kp, kj_idx, ki_idx))
            if kp != kp_conj:
                response_kpts.append((kp_conj, ki_idx, kj_idx))
        ejk_sigma_sr = cp.zeros([cell.natm+3, 3])
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE*(MAX_IMGS_PER_TASK+2) + 1, dtype=np.uint32)
        head = pool[-1:]
        task_pool = empty_aligned((workers, POOL_SIZE*16), np.int32, alignment=128)
        int3c2e_envs = int3c2e_opt.int3c2e_envs
        kern = libpbc.PBCsr_ejk_int3c2e_deriv
        max_aux_batch = int(np.diff(aux_loc[ksh_offsets_cpu]).max())
        mem_free = get_avail_mem(exclude_memory_pool=True)
        word_avail = mem_free // 8
        word_avail -= nkpts*n_compact_pairs*max_aux_batch * 2  # compressed
        compressed_real_size = n_compact_pairs*bvk_ncells*max_aux_batch
        if compressed_real_size > word_avail * 0.9:
            raise RuntimeError('Insufficient GPU memory for GDF gradient response buffers')
        blk_unit = (nkpts*nao**2 + max(nkpts*nocc*nao, bvk_ncells*nao**2)) * 2
        blksize = min(max_aux_batch, int(word_avail*.9/blk_unit))
        if blksize < 1:
            raise RuntimeError('Insufficient GPU memory for GDF gradient response buffers')
        buf = cp.empty(nkpts*n_compact_pairs*max_aux_batch, dtype=np.complex128)
        buf1 = cp.empty(max(nkpts*nocc*nao*blksize*2 + nkpts*nao**2*blksize*2,
                            bvk_ncells*nao**2*blksize*2 + nkpts*nao**2*blksize*2,
                            n_compact_pairs*bvk_ncells*max_aux_batch))
        work1, work2 = _allocate(nkpts*nao**2*blksize*2, buf1)
        for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
            aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
            naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
            compressed = ndarray((nkpts, n_compact_pairs, naux_in_batch),
                                 dtype=np.complex128, buffer=buf)
            for kp, ki_idx, kj_idx in response_kpts:
                for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                    dk = k1 - k0
                    aux0, aux1 = aux_ao_offset + k0, aux_ao_offset + k1
                    tmp = ndarray((nkpts,nocc,nao,dk), dtype=np.complex128, buffer=work2)
                    dm_tensor = ndarray((nkpts,nao,nao,dk), dtype=np.complex128, buffer=work1)
                    dm_oo_blk = ndarray((dk,nkpts,nocc,nocc), dtype=np.complex128, buffer=work1)
                    cp.take(dm_oo[aux0:aux1], ki_idx*nkpts+kj_idx, axis=1, out=dm_oo_blk)
                    contract('rkij,kqj->kiqr', dm_oo_blk, dm_factor_r[kj_idx],
                             -.5*k_factor, out=tmp)
                    contract('kiqr,kpi->kpqr', tmp, dm_factor_l[ki_idx], out=dm_tensor)
                    if j_factor != 0 and kp == 0:
                        contract('r,kpq->kpqr', auxvec[aux0:aux1], dm_sorted[ki_idx],
                                 j_factor, beta=1, out=dm_tensor)

                    # Transform the first orbital k-point, then select compact
                    # AO pairs before expanding the auxiliary image dimension.
                    dm_realspace = ndarray((nao,bvk_ncells,nao,dk),
                                           dtype=np.complex128, buffer=work2)
                    contract('kpqr,Nk->qNpr', dm_tensor, expLk[:,ki_idx], out=dm_realspace)
                    cp.take(dm_realspace.reshape(-1,dk), compact_idx, axis=0,
                            out=compressed[kp,:,k0:k1])
            compressed_real = ndarray((n_compact_pairs, bvk_ncells, naux_in_batch), buffer=buf1)
            contract('ktr,Lk->tLr', compressed.real, expLk_conj.real,
                     out=compressed_real)
            contract('ktr,Lk->tLr', compressed.imag, expLk_conj.imag,
                     alpha=-1, beta=1, out=compressed_real)
            err = kern(
                ctypes.cast(ejk_sigma_sr[:-3].data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_sigma_sr[-3:].data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.cast(compressed_real.data.ptr, ctypes.c_void_p),
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
        t0 = log.timer_debug1('contract sr_int3c2e_ejk_deriv', *t0)

    ejk_sigma = ejk_sigma.get()

    if (exxdiv == 'ewald' and
        (cell.dimension == 3 or
         (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
        bvk_kmesh = int3c2e_opt.bvk_kmesh
        s0 = int1e.int1e_ovlp(cell, kpts, bvk_kmesh)
        k_dm = contract('kpq,kqr->kpr', dm, s0)
        k_dm = contract('kpr,krs->kps', k_dm, dm)
        # The k_factor was previously scaled by 1/nkpts^2.
        k_factor *= nkpts**2
        de_ewald = int1e.ovlp_derivatives(cell, k_dm, kpts, bvk_kmesh)
        exx_0, exx_1 = aft_jk._exxdiv_ewald_strain_deriv(cell.cell, kpts, -omega)
        de_ewald *= -.5 * k_factor * exx_0 / nkpts
        ejk_sigma += de_ewald

        ek_G0 = float(cp.einsum('kij,kji->', s0, k_dm).real.get()) / nkpts**2
        # *.5 for the factor 1/2 in Coulomb operator; second *.5 for J-K/2 in RHF
        ejk_sigma[-3:] -= k_factor * .5 * .5 * ek_G0 * exx_1
    return ejk_sigma

def _get_ej_derivatives(int3c2e_opt, dm, kpts=None, hermi=0, omega=None,
                        verbose=None, linear_dep_threshold=LINEAR_DEP_THR):
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
    has_compact = len(int3c2e_opt.img_idx) > 0
    if has_compact:
        auxvec = int3c2e_opt.contract_dm(dm, kpts, hermi=1)
    else:
        auxvec = cp.zeros(auxcell.cell.nao)
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
    rcut = _estimate_sr_2c2e_rcut(auxcell, int3c2e_opt.omega, precision)
    with lib.temporary_env(auxcell, rcut=rcut):
        int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False, omega=-int3c2e_opt.omega)

    ################################
    # LR part 0th order
    if omega is None:
        omega = 0
    else:
        omega = abs(omega)
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
    dd_ft_opt = int3c2e_opt.dd_ft_opt
    separated_dd = dd_ft_opt is not None

    def lr_3c2e(ft_opt, wcoulG0, update_metric):
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
        # Conservative sum: buf reuses storage for pqG and auxG.
        unit = nao_pair  # pqG capacity in buf
        unit += naux * 2 # auxG, auxGw
        Gblksize = int(mem_avail*.8//(unit*16))//32*32
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
            cp.multiply(auxG, wcoulG0[p0:p1], out=auxGw)
            auxGw = auxGw.view(np.float64)
            if update_metric:
                contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)
            auxvec_LR += auxGw.dot(rhoGz)
        return auxvec_LR, rhoG

    auxvec_LR, rhoG_LR = lr_3c2e(ft_opt, wcoulG_LR0, True)
    if separated_dd:
        wcoulG_FR0, wcoulG_FR1 = get_wcoulG(cell, Gv, -omega)
        auxvec_FR, rhoG_FR = lr_3c2e(dd_ft_opt, wcoulG_FR0, False)
        auxvec_LR += auxvec_FR
    auxvec += auxcell.apply_CT_dot(auxvec_LR)
    auxvec_LR = None
    t0 = log.timer_debug1('contract dm', *t0)

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
        unit = naux * 2 # buf: auxG
        Gblksize = int(mem_avail*.8//(unit*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        rho_auxG = cp.empty(ngrids, dtype=np.complex128)
        buf = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            rho_auxG[p0:p1] = auxvec.dot(auxG.view(np.float64)).view(np.complex128)

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
            'g,g,xyg->xy', rho_auxG, rhoG_LR.conj(), wcoulG_LR1).real
        if separated_dd:
            ej_sigma_lr[-3:] += cp.einsum(
                'g,g,xyg->xy', rho_auxG, rhoG_FR.conj(), wcoulG_FR1).real
        ej_sigma_lr[-3:] -= .5 * cp.einsum(
            'g,g,xyg->xy', rho_auxG, rho_auxG.conj(), wcoulG_LR1).real
        return ej_sigma_lr

    ej_sigma += lr_3c2e_response()
    t0 = log.timer_debug1('lr_int3c2e_deriv', *t0)
    ft_opt = None

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

def _allocate(shape, buf):
    a = ndarray(shape, buffer=buf)
    return a, buf[a.size:]

_add_j3c_oo_kernel = cp.RawKernel(r'''
extern "C" __global__
void add_j3c_oo(double2* out, const double2* values, const int* pair_idx,
                unsigned long long nkpairs, unsigned long long npairs,
                unsigned long long noo)
{
    unsigned long long pair = blockIdx.x % npairs;
    unsigned long long aux = blockIdx.x / npairs;
    unsigned long long src = (unsigned long long)blockIdx.x * noo;
    unsigned long long dest = (aux * nkpairs + pair_idx[pair]) * noo;
    for (unsigned long long ij = threadIdx.x; ij < noo; ij += blockDim.x) {
        double2 a = out[dest + ij];
        double2 b = values[src + ij];
        out[dest + ij] = make_double2(a.x + b.x, a.y + b.y);
    }
}
''', 'add_j3c_oo')

def _add_j3c_oo(j3c_oo, values, pair_idx):
    '''Add values to unique flattened k-point pairs: j3c_oo[:,pair_idx] += values.

    Both tensors must be C-contiguous complex128 arrays. Each block handles
    one auxiliary function and one pair; pair indices must be unique and in range.
    '''
    assert j3c_oo.dtype == values.dtype == np.complex128
    assert j3c_oo.flags.c_contiguous and values.flags.c_contiguous
    assert j3c_oo.ndim == values.ndim == 4
    assert j3c_oo.shape[0] == values.shape[0] and j3c_oo.shape[2:] == values.shape[2:]
    pair_idx = cp.asarray(pair_idx, dtype=np.int32, order='C')
    assert pair_idx.ndim == 1 and len(pair_idx) == values.shape[1]
    if values.size == 0:
        return
    _add_j3c_oo_kernel(
        (values.shape[0]*len(pair_idx),), (512,),
        (j3c_oo, values, pair_idx,
         np.uint64(j3c_oo.shape[1]), np.uint64(values.shape[1]),
         np.uint64(values.shape[2]*values.shape[3])))

def _jk_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, omega=None, verbose=None,
                        linear_dep_threshold=LINEAR_DEP_THR):
    '''Compatibility wrapper returning only the atomic J/K derivatives.'''
    return _get_ejk_derivatives(
        int3c2e_opt, dm, kpts, hermi, j_factor, k_factor, exxdiv, omega, verbose,
        linear_dep_threshold)[:-3]

def _j_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, omega=None,
                       verbose=None, linear_dep_threshold=LINEAR_DEP_THR):
    '''Compatibility wrapper returning only the atomic Coulomb derivatives.'''
    return _get_ej_derivatives(
        int3c2e_opt, dm, kpts, hermi, omega, verbose, linear_dep_threshold)[:-3]

def get_pp_loc_part1_grad(cell, dm, kpts=None, hermi=0, with_pseudo=True, verbose=None):
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    is_single_kpt = kpts is not None and kpts.ndim == 1
    is_gamma_point = kpts is None or is_zero(kpts)
    if is_single_kpt:
        kpts = kpts.reshape(1, 3)
    if is_gamma_point:
        bvk_kmesh = np.ones(3, dtype=int)
    else:
        bvk_kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)

    # Guess range-separation parameter based on system size
    omega = 0.4
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    nGv = np.prod(mesh)
    ke_cutoff *= (3e3/nGv)**(2./3)
    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff)
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    log.debug('get_pp_loc_part1_grad: omega = %g Ecut = %s mesh = %s',
              omega, ke_cutoff, mesh)

    fakenuc = aft_cpu._fake_nuc(cell, with_pseudo=with_pseudo)
    int3c2e_opt = SRInt3c2eOpt(cell, fakenuc, omega=-omega, bvk_kmesh=bvk_kmesh).build()
    charges = -cp.asarray(cell.atom_charges(), dtype=np.float64)

    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell

    dm = cp.asarray(dm)
    dm = cell.apply_C_mat_CT(dm)
    if hermi != 1:
        dm = transpose_sum(dm, inplace=True)
        dm[:] *= .5

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

    Gv, _, kws = get_Gv_weights(cell, mesh)
    ngrids = len(Gv)
    wcoulG_LR0, wcoulG_LR1 = get_wcoulG(
        cell, Gv, int3c2e_opt.omega)
    wcoulG_SR_at_G0 = np.pi / int3c2e_opt.omega**2 * kws
    wcoulG_LR0[0] -= wcoulG_SR_at_G0
    wcoulG_LR1[:,:,0] += wcoulG_SR_at_G0 * cp.eye(3)
    ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)

    if with_pseudo:
        assert (cell.dimension == 3 or
                (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))
        exps = cp.asarray(np.hstack(fakenuc.bas_exps()))
        pp_G0_term = -charges.dot(np.pi/exps) * kws
    else:
        pp_G0_term = 0

    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    aux_loc = auxcell.ao_loc
    nao = dm.shape[-1]
    naux = int(aux_loc[-1])

    eval_ft = ft_opt.ft_evaluator(
        compressing=True, cart=True, original_ao_order=False)[0]
    pair_addresses, diag_idx = ft_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)
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

    rhoG = cp.empty(ngrids, dtype=np.complex128)
    buf  = cp.empty(max(nao_pair,naux)*Gblksize, dtype=np.complex128)
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        nGv = p1 - p0
        # conj((r|G)^{[0]}) (ij|G)^{[0]}
        pqG = eval_ft(Gv[p0:p1], out=buf)
        rhoGz = cp.einsum('pG,p->G', pqG.view(np.float64), dm_tril)
        rhoG[p0:p1] = rhoGz.view(np.complex128)

    aft_envs = ft_opt.aft_envs
    shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    Gblksize = int(mem_avail*.8//(naux*2*16))//32*32
    Gblksize = min(Gblksize, ngrids)
    rho_nucG = cp.empty(ngrids, dtype=np.complex128)
    buf = cp.empty(naux*Gblksize, dtype=np.complex128)
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
        rho_nucG[p0:p1] = charges.dot(
            auxG.view(np.float64)).view(np.complex128)

    vG = rhoG * wcoulG_LR0
    GvT = cp.asarray(Gv.T.ravel())
    ej_sigma_aux = cp.zeros([cell.natm+3, 3])
    aux_ft_envs = RysIntEnvVars.new(
        auxcell.natm, auxcell.nbas, auxcell._atm, auxcell._bas,
        _scale_sp_ctr_coeff(auxcell), auxcell.ao_loc)
    err = libpbc.PBC_ft_ao_deriv(
        ctypes.cast(ej_sigma_aux[:-3].data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_sigma_aux[-3:].data.ptr, ctypes.c_void_p),
        ctypes.cast(charges.data.ptr, ctypes.c_void_p),
        ctypes.cast(vG.data.ptr, ctypes.c_void_p),
        ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
        ctypes.byref(aux_ft_envs), ctypes.c_int(ngrids))
    if err != 0:
        raise RuntimeError('ft_ao_deriv failed')

    ej_sigma_lr = cp.zeros([cell.natm+3, 3])
    vG_conj = rho_nucG.conj() * wcoulG_LR0
    vG_conj[0] += pp_G0_term
    bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = \
        aft_jk._shl_pairs_for_derivative_kernel(ft_opt)
    nbatches_shl_pair = len(shl_pair_offsets) - 1
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

    ej_sigma_lr *= 2
    ej_sigma_lr += ej_sigma_aux
    ej_sigma_lr[-3:] += cp.einsum(
        'g,g,xyg->xy', rho_nucG, rhoG.conj(), wcoulG_LR1).real
    ej_sigma_lr[-3:] -= cp.eye(3) * (rhoG[0] * pp_G0_term).real

    ej_sigma = ej_sigma_lr
    t0 = log.timer_debug1('lr_int3c2e_deriv via aft', *t0)
    ft_opt = None

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
        ctypes.cast(charges.data.ptr, ctypes.c_void_p),
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

    t0 = log.timer_debug1('contract int3c2e_ejk_deriv', *t0)
    return ej_sigma.get()

def get_nuc(cell, dm, kpts=None, hermi=1):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    nuc = get_pp_loc_part1_grad(cell, dm, kpts, hermi, with_pseudo=False, verbose=log)
    log.timer('get_nuc gradient', *t0)
    return nuc

def get_pp_loc(cell, dm, kpts=None, hermi=1):
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    depp = get_pp_loc_part1_grad(cell, dm, kpts, hermi, with_pseudo=True, verbose=log)
    raise NotImplementedError("get_pp_loc_part2_grad not implemented yet")
    # depp += get_pp_loc_part2_grad(cell, dm, kpts, hermi)
    log.timer('get_pp_loc gradient', *t0)
    return depp
