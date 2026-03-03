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
Build GDF tensor using the range-separation integral algorithm.
'''

import os
import math
import ctypes
import warnings
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
from pyscf.pbc.df import aft as aft_cpu
from pyscf.pbc.tools.k2gamma import (
    translation_vectors_for_kmesh, double_translation_indices)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, asarray, sandwich_dot, empty_mapped, ndarray)
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.pbc.df import ft_ao
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter, conj_images_in_bvk_cell
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.tools.pbc import get_coulG, _Gv_wrap_around
from gpu4pyscf.gto.mole import extract_pgto_params, SortedGTO
from gpu4pyscf.pbc.df.int3c2e import libpbc, fill_triu_bvk, SRInt3c2eOpt
from gpu4pyscf.pbc.df.int2c2e import int2c2e

OMEGA_MIN = 0.25

# In the ED of the j2c2e metric, the default LINEAR_DEP_THR setting in pyscf-2.8
# is too loose. The linear dependency truncation often leads to serious errors.
# PBC GDF very differs to the molecular GDF approximation where diffuse
# functions typically have insignificant contributions. The diffuse auxiliary
# crystal orbitals have large impacts on the accuracy of Coulomb integrals. A
# tight linear dependency threshold have to be applied to control the error,
# even this may cause more numerical stability issues.
LINEAR_DEP_THR = 1e-11
# Use eigenvalue decomposition in decompose_j2c
PREFER_ED = False

THREADS = 256

def build_cderi(cell, auxcell, kpts=None, kmesh=None, j_only=False,
                omega=None, linear_dep_threshold=LINEAR_DEP_THR,
                compress=False):
    '''
    Create density fitting integral tensor
    '''
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension >= 2
    with_long_range = cell.omega == 0
    if with_long_range:
        if omega is None:
            cell_exps, cs = extract_pgto_params(cell, 'diffuse')
            omega = min(OMEGA_MIN, (cell_exps.min()*.5)**.5)
            logger.debug(cell, 'omega guess in rsdf_builder = %g', omega)
        omega = abs(omega)
    else:
        assert cell.omega < 0
        # Not supporting a custom omega for SR CDERI
        assert omega is None or omega == abs(cell.omega)
        omega = abs(cell.omega)

    is_gamma_point = kpts is None or is_zero(kpts)
    if is_gamma_point:
        cderi, cderip, cderi_idx = compressed_cderi_gamma_point(
            cell, auxcell, omega, with_long_range, linear_dep_threshold)
        kpts = np.zeros((1, 3))
        kmesh = np.array([1, 1, 1])
    elif j_only:
        # Coulomb integrals can be converged within a smaller bvk cell.
        kmesh = kpts_to_kmesh(cell, kpts)
        cderi, cderip, cderi_idx = compressed_cderi_j_only(
            cell, auxcell, kmesh, omega, with_long_range, linear_dep_threshold)
    else:
        # Remote images may contribute to certain k-point mesh, contributing
        # to the finite-size effects in HFX. For sufficiently large number of
        # kpts, the truncation radius cell.rcut may cause finite-size errors.
        # Use a large radius to generate MP kmesh.
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut*10, bound_by_supmol=False)
        cderi, cderip, cderi_idx = compressed_cderi_kk(
            cell, auxcell, kpts, kmesh, omega, with_long_range, linear_dep_threshold)
    if compress:
        return cderi, cderip, cderi_idx

    kpt_iters = list(kk_adapted_iter(kmesh))
    if not (is_gamma_point or j_only):
        assert len(kpt_iters) == len(cderi)

    pair_address = cp.asarray(cderi_idx[0], dtype=np.int32)
    conj_mapping = conj_images_in_bvk_cell(kmesh)
    bvkmesh_Ls = cp.asarray(translation_vectors_for_kmesh(cell, kmesh, True))
    expLk = cp.exp(1j*bvkmesh_Ls.dot(cp.asarray(kpts).T))
    nao = cell.nao
    for kp, kp_conj, ki_idx, kj_idx in kpt_iters:
        if kp in cderi:
            cderi_k = _unpack_cderi_v2(cderi.pop(kp), pair_address, kj_idx,
                                       conj_mapping, expLk, nao)
            for (ki, kj) in zip(ki_idx, kj_idx):
                cderi[ki, kj] = cderi_k[ki]
        if cderip is not None and kp in cderip:
            cderi_k = _unpack_cderi_v2(cderip.pop(kp), pair_address, kj_idx,
                                       conj_mapping, expLk, nao)
            for (ki, kj) in zip(ki_idx, kj_idx):
                cderip[ki, kj] = cderi_k[ki]
    return cderi, cderip

def _weighted_coulG_LR(cell, Gv, omega, kws, kpt=np.zeros(3)):
    coulG = get_coulG(cell, kpt, exx=False, Gv=Gv, omega=abs(omega))
    coulG *= kws
    if is_zero(kpt):
        assert Gv[0].dot(Gv[0]) == 0
        coulG[0] -= np.pi / omega**2 / cell.vol
    return asarray(coulG)

def decompose_j2c(j2c, prefer_ed=PREFER_ED, linear_dep_threshold=LINEAR_DEP_THR):
    if not prefer_ed:
        try:
            return cholesky_decomposed_metric(j2c)
        except LinearDepencyError:
            # Restore to ED if the j2c metric is found to be linearly dependent
            pass
    return eigenvalue_decomposed_metric(j2c, linear_dep_threshold)

def cholesky_decomposed_metric(j2c):
    '''Return L for j2c = L L^T'''
    j2c_negative = None
    j2ctag = 'CD'
    # Cupy cholesky does not check positive-definite, seems returning nan in the
    # resultant CD matrix silently.
    j2c = cp.asarray(j2c, order='C')
    j2c = cp.linalg.cholesky(j2c)
    if cp.isnan(j2c[-1,-1]):
        raise LinearDepencyError('j2c is not positive definite')
    return j2c, j2c_negative, j2ctag

def eigenvalue_decomposed_metric(j2c, linear_dep_threshold=LINEAR_DEP_THR):
    j2c = cp.asarray(j2c, order='C')
    w, v = cp.linalg.eigh(j2c)
    mask = w > linear_dep_threshold
    # Note this implementation is different to the one in PySCF-2.10. In PySCf,
    # j2c at a wrong k-point is passed to this function. v.conj() is called.
    v1 = v[:,mask]
    v1 *= w[mask]**-.5
    j2c = v1
    # linear_dep_threshold for negative eigenvalues are too tight. Small errors
    # in 2c2e metric would lead to small negative eigenvalues. They can be
    # safely filtered.
    #idx = cp.where(w < -linear_dep_threshold)[0]
    idx = cp.where(w < -1e-4)[0]
    j2c_negative = None
    if len(idx) > 0:
        j2c_negative = (v[:,idx] * (-w[idx])**-.5)
    j2ctag = 'ED'
    return j2c, j2c_negative, j2ctag

def _get_2c2e(auxcell, uniq_kpts, omega, with_long_range=True, bvk_kmesh=None):
    # Compute SR Coulomb 2c2e
    if uniq_kpts is not None:
        assert uniq_kpts.ndim == 2
    with auxcell.with_short_range_coulomb(-omega):
        j2c = int2c2e(auxcell, kpts=uniq_kpts, bvk_kmesh=bvk_kmesh)
    j2c = cp.asarray(j2c)

    if not with_long_range:
        return j2c

    # Compute LR Coulomb 2c2e
    precision = auxcell.precision * 1e-3
    ke = estimate_ke_cutoff_for_omega(auxcell, omega, precision)
    mesh = auxcell.cutoff_to_mesh(ke)
    mesh = auxcell.symmetrize_mesh(mesh)
    logger.debug(auxcell, 'Set 2c2e integrals precision %g, mesh %s', precision, mesh)

    Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
    ngrids = Gv.shape[0]
    naux = auxcell.nao
    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem = mem_free - naux**2 * 16
    mem *= .5 # the temporary .conj() consumes another half mem
    blksize = int(mem//(16*naux*2))
    logger.debug2(auxcell, 'max_memory %s (MB)  blocksize %s', mem_free, blksize)

    if uniq_kpts is None:
        coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1])
            auxG_conj = auxG.conj()
            auxG_conj *= coulG_LR[p0:p1,None]
            j2c += auxG_conj.T.dot(auxG).real
            auxG = auxG_conj = None
    else:
        for k, kpt in enumerate(uniq_kpts):
            coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws, kpt)
            is_gamma_point = is_zero(kpt)
            for p0, p1 in lib.prange(0, ngrids, blksize):
                auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], kpt=kpt)
                auxG_conj = auxG.conj()
                auxG_conj *= coulG_LR[p0:p1,None]
                v = auxG_conj.T.dot(auxG)
                if is_gamma_point:
                    v = v.real
                j2c[k] += v
                auxG = auxG_conj = v = None
    return j2c

def compressed_cderi_gamma_point(cell, auxcell, omega=OMEGA_MIN, with_long_range=True,
                                 linear_dep_threshold=LINEAR_DEP_THR):
    kmesh = np.array([1, 1, 1])
    return compressed_cderi_j_only(cell, auxcell, kmesh, omega, with_long_range,
                                   linear_dep_threshold)

def compressed_cderi_j_only(cell, auxcell, kmesh, omega=OMEGA_MIN,
                            with_long_range=True, linear_dep_threshold=LINEAR_DEP_THR):
    assert kmesh is not None
    log = logger.new_logger(cell)
    t1 = log.init_timer()

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)

    ft_opt = ft_ao.FTOpt(cell, kmesh)
    ft_opt.__dict__.update(int3c2e_opt.__dict__)
    ft_opt._aft_envs = int3c2e_opt._int3c2e_envs

    log.debug('Generate auxcell 2c2e integrals')
    cd_j2c_cache, negative_metric_size = _precontract_j2c_aux_coeff(
        auxcell, None, omega, with_long_range, linear_dep_threshold)
    naux_cart, naux = cd_j2c_cache[0].shape

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    nao_pairs = len(cderi_idx[0])

    omega = abs(int3c2e_opt.omega)
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    coulG = _weighted_coulG_LR(auxcell, Gv, omega, kws)

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_free -= cd_j2c_cache[0].nbytes # cd_j2c_cache
    mem_free -= ngrids * naux * 16 # auxG_cache
    log.debug('Avail GPU mem = %s B', mem_free)
    # To ensure tasks consistently distributed to each processor, the same batch
    # size should be used for int3c2e_evaluator for each processor.
    batch_size = min(nao_pairs, mem_free // (naux_cart*bvk_ncells*16*4))

    log.debug('Required %.6g GB mapped memory on host', naux*nao_pairs*8e-9)
    cderi = empty_mapped((naux, nao_pairs))

    tasks = iter(range(nao_pairs))
    def proc():
        nsp_per_block = ft_ao.ft_ao_scheme()[0]
        bas_ij_aggregated = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

        eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
            ao_pair_batch_size=batch_size, bas_ij_aggregated=bas_ij_aggregated)[:3]
        shl_pair_batches = len(ao_pair_offsets) - 1
        aux_coeff = cp.asarray(cd_j2c_cache[0])

        if with_long_range:
            eval_ft, _ao_pair_offsets = ft_opt.ft_evaluator(
                batch_size, bas_ij_aggregated=bas_ij_aggregated)
            assert np.array_equal(ao_pair_offsets, _ao_pair_offsets)

            log.debug1('cache auxG')
            auxG_conj = ft_ao.ft_ao(auxcell, Gv, sort_cell=False).T.conj()
            auxG_conj = aux_coeff.T.dot(auxG_conj)
            auxG_conj *= asarray(coulG)

            avail_mem = mem_free - naux_cart*batch_size*16*2
            Gblksize = int(avail_mem//(16*(batch_size+naux*2))) // 32 * 32
            if Gblksize == 0:
                raise RuntimeError('Insufficient GPU memory')
            Gblksize = min(Gblksize, ngrids)
            log.debug1('ngrids = %d Gblksize = %d naux=%d max_pair_size=%d',
                       ngrids, Gblksize, naux, batch_size)
            buf2 = cp.empty(batch_size*Gblksize, dtype=np.complex128)

        aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
        aux_coeff[aux_sorting] = tmp
        tmp = None

        buf0 = cp.empty(naux*batch_size)
        buf1 = cp.empty(batch_size*naux_cart*bvk_ncells, dtype=np.complex128)
        for batch_id in tasks:
            if batch_id >= shl_pair_batches:
                break
            log.debug1('batch %d/%d', batch_id, shl_pair_batches)
            j3c = eval_j3c(shl_pair_batch_id=batch_id, out=buf1)
            if j3c.size == 0:
                continue

            pair_size = j3c.shape[0]
            j3c_buf = ndarray((naux, pair_size), buffer=buf0)
            if kmesh is None:
                j3c = aux_coeff.T.dot(j3c[:,:,0].T, out=j3c_buf)
            else:
                j3c = aux_coeff.T.dot(j3c.sum(axis=2).T, out=j3c_buf)

            if with_long_range:
                j3c_buf = ndarray(j3c.shape, dtype=np.complex128, buffer=buf1)
                for p0, p1 in lib.prange(0, ngrids, Gblksize):
                    auxG_c = asarray(auxG_conj[:,p0:p1])
                    pqG = eval_ft(Gv[p0:p1], batch_id, out=buf2)
                    # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                    # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux
                    # functions |P> are assumed to be real
                    j3c += auxG_c.dot(pqG.T, out=j3c_buf).real

            p0 = ao_pair_offsets[batch_id]
            p1 = ao_pair_offsets[batch_id+1]
            #:cderi[:,p0:p1] = j3c.get()
            libpbc.store_col_segment(
                cderi.ctypes,
                ctypes.cast(j3c.data.ptr, ctypes.c_void_p),
                ctypes.c_int(naux), ctypes.c_int(nao_pairs),
                ctypes.c_int(p0), ctypes.c_int(p1))
            j3c = None

    multi_gpu.run(proc, non_blocking=True)

    cderip = None
    for k, nauxp in negative_metric_size.items():
        # For low-dimensional systems, CDERI has negative eigenvectors
        cderip, cderi = cderi[-nauxp:], cderi[:-nauxp]
    # Follow the output format in compressed_cderi_kk
    cderi = {0: cderi}
    if cderip is not None:
        cderip = {0: cderip}
    t1 = log.timer_debug1('build cderi', *t1)
    return cderi, cderip, cderi_idx

def compressed_cderi_kk(cell, auxcell, kpts, kmesh=None, omega=OMEGA_MIN,
                        with_long_range=True, linear_dep_threshold=LINEAR_DEP_THR):
    log = logger.new_logger(cell)
    t1 = log.init_timer()

    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts, rcut=cell.rcut*10, bound_by_supmol=False)
    kpts = kpts.reshape(-1, 3)
    bvk_ncells = np.prod(kmesh)
    assert len(kpts) == bvk_ncells
    kpt_iters = list(kk_adapted_iter(kmesh))
    # uniq_kpts corresponds to the k-conserved k_aux = -(kj-ki)
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    nkpts = len(uniq_kpts)

    int3c2e_opt = SRInt3c2eOpt(cell, auxcell, omega=-omega, bvk_kmesh=kmesh).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell

    ft_opt = ft_ao.FTOpt(cell, kmesh)
    ft_opt.__dict__.update(int3c2e_opt.__dict__)
    ft_opt._aft_envs = int3c2e_opt._int3c2e_envs

    log.debug('Generate auxcell 2c2e integrals')
    cd_j2c_cache, negative_metric_size = _precontract_j2c_aux_coeff(
        auxcell, kpts, omega, with_long_range, linear_dep_threshold, kmesh)
    naux_cart = cd_j2c_cache[0].shape[0]
    naux_max = max(x.shape[1] for x in cd_j2c_cache)

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    nao_pairs = len(cderi_idx[0])

    omega = abs(int3c2e_opt.omega)
    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = len(Gv)
    # To ensure the symmetry between conjugated k-points, it is important to
    # wrap around the high-freq Gv.
    assert Gv[0].dot(Gv[0]) == 0
    Gk = (Gv + uniq_kpts[:,None]).reshape(-1, 3)
    Gk = _Gv_wrap_around(cell, Gk, cp.zeros(3), mesh)
    coulG = get_coulG(cell, Gv=Gk, omega=abs(omega)).reshape(nkpts, ngrids)
    coulG *= kws
    coulG[0,0] -= np.pi / omega**2 / cell.vol

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_free -= cd_j2c_cache[0].nbytes * nkpts # cd_j2c_cache
    mem_free -= ngrids * naux_max * 16 * nkpts # auxG_conj
    log.debug('Avail GPU mem = %s B', mem_free)
    # To ensure tasks consistently distributed to each processor, the same batch
    # size should be used for int3c2e_evaluator for each processor.
    batch_size = min(nao_pairs, mem_free//(nkpts*naux_cart*16*4)+225)

    log.debug('Required %.6g GB mapped memory on host',
              len(cd_j2c_cache)*naux_max*nao_pairs*16e-9)
    cderi = {}
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        naux = cd_j2c_cache[j2c_idx].shape[1]
        cderi[kp] = empty_mapped((naux,nao_pairs), dtype=np.complex128)

    tasks = iter(range(nao_pairs))
    def proc():
        nsp_per_block = ft_ao.ft_ao_scheme()[0]
        bas_ij_aggregated = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

        eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
            ao_pair_batch_size=batch_size, bas_ij_aggregated=bas_ij_aggregated)[:3]
        shl_pair_batches = len(ao_pair_offsets) - 1

        aux_coeffs = []
        for x in cd_j2c_cache:
            aux_coeff = cp.empty_like(x)
            aux_coeff[aux_sorting] = cp.asarray(x)
            aux_coeffs.append(aux_coeff)
        aux_coeff = x = None

        expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(uniq_kpts.T)))
        expLk_conjz = expLk.conj().view(np.float64).reshape(bvk_ncells,nkpts,2)
        expLk = None

        if with_long_range:
            eval_ft, _ao_pair_offsets = ft_opt.ft_evaluator(
                batch_size, bas_ij_aggregated=bas_ij_aggregated)
            # To ensure the same subsets of orbital paris (ao_pair_offsets) are
            # evaluated in int3c2e_evaluator and ft_evaluator, the bas_ij_idx
            # and shl_pair_offsets (bas_ij_aggregated) must be shared
            # by the two evaluators
            assert np.array_equal(ao_pair_offsets, _ao_pair_offsets)

            log.debug1('cache auxG')
            auxG = ft_ao.ft_ao(auxcell, Gk, sort_cell=False).T.reshape(naux_cart,nkpts,ngrids)
            # Note: in the case of ft_ao, auxG[kp].conj() != auxG[kp_conj]
            for k in range(nkpts):
                auxG[aux_sorting,k] = auxG[:,k].conj()
            # auxG_conj at -(kj-ki) = conj(kp)
            auxG_conj, auxG = auxG, None
            auxG_conj *= cp.asarray(coulG)

            avail_mem = mem_free - nkpts*naux_cart*batch_size*16*2
            Gblksize = int(avail_mem//(16*batch_size)) // 32 * 32
            if Gblksize == 0:
                raise RuntimeError('Insufficient GPU memory')
            Gblksize = min(Gblksize, ngrids)
            log.debug1('ngrids = %d Gblksize = %d naux=%d max_pair_size=%d',
                       ngrids, Gblksize, naux_max, batch_size)
            buf2 = cp.empty(batch_size*Gblksize, dtype=np.complex128)

        buf0 = cp.empty(nkpts*batch_size*naux_cart, dtype=np.complex128)
        buf1 = cp.empty(naux_max*batch_size*bvk_ncells, dtype=np.complex128)
        for batch_id in tasks:
            if batch_id >= shl_pair_batches:
                break
            log.debug1('batch %d/%d', batch_id, shl_pair_batches)
            j3c = eval_j3c(shl_pair_batch_id=batch_id, out=buf1)
            if j3c.size == 0:
                continue

            pair_size = j3c.shape[0]
            j3c_buf = ndarray((nkpts, naux_cart, pair_size, 2), buffer=buf0)
            j3c = contract('prL,LKz->Krpz', j3c, expLk_conjz, out=j3c_buf)
            j3c = j3c.view(np.complex128)[:,:,:,0]

            if with_long_range:
                for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                    for p0, p1 in lib.prange(0, ngrids, Gblksize):
                        auxG_c = auxG_conj[:,j2c_idx,p0:p1]
                        pqG = eval_ft(Gv[p0:p1] + kpts[kp], batch_id, out=buf2)
                        # \sum_G coulG * ints(ij * exp(-i G * r)) * ints(P * exp(i G * r))
                        # = \sum_G FT(ij, G) conj(FT(aux, G)) , where aux functions |P>
                        # are assumed to be real
                        contract('rG,pG->rp', auxG_c, pqG, beta=1., out=j3c[j2c_idx])

            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                aux_coeff = aux_coeffs[j2c_idx] # at -(kj-ki)
                naux = aux_coeff.shape[1]
                cderi_k = ndarray((naux, pair_size), dtype=np.complex128, buffer=buf1)
                cderi_k = aux_coeff.T.dot(j3c[j2c_idx], out=cderi_k)
                p0 = ao_pair_offsets[batch_id]
                p1 = ao_pair_offsets[batch_id+1]
                #:cderi[kp][:,p0:p1] = cderi_k.get()
                libpbc.store_col_segment(
                    cderi[kp].ctypes,
                    ctypes.cast(cderi_k.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(naux),
                    # *2 for complex number
                    ctypes.c_int(nao_pairs*2),
                    ctypes.c_int(p0*2), ctypes.c_int(p1*2))
                cp.cuda.get_current_stream().synchronize()
            j3c = None

    multi_gpu.run(proc, non_blocking=True)

    cderip = None
    if negative_metric_size:
        cderip = {}
        for j2c_idx, nauxp in negative_metric_size.items():
            kp = kpt_iters[j2c_idx][0]
            cderip[kp] = cderi[kp][-nauxp:]
            cderi [kp] = cderi[kp][:-nauxp]
    t1 = log.timer_debug1('build cderi', *t1)
    return cderi, cderip, cderi_idx

def _precontract_j2c_aux_coeff(auxcell, kpts, omega, with_long_range,
                               linear_dep_threshold, kmesh=None):
    auxcell = SortedGTO.from_cell(auxcell)
    if kmesh is None:
        j2c = _get_2c2e(auxcell, kpts, omega, with_long_range, kmesh)
        if j2c.ndim == 2:
            j2c = j2c[None]
    else:
        assert len(kpts) == np.prod(kmesh)
        kpt_iters = list(kk_adapted_iter(kmesh))
        # uniq_kpts corresponds to (kj-ki)
        uniq_kpts = kpts[[x[0] for x in kpt_iters]]
        j2c = _get_2c2e(auxcell, uniq_kpts, omega, with_long_range, kmesh)
        # DF metric for self-conjugated k-point should be real
        j2c = [j2c_k.real if kp == kp_conj else j2c_k
               for j2c_k, (kp, kp_conj, _, _) in zip(j2c, kpt_iters)]

    aux_coeff = asarray(auxcell.ctr_coeff)
    prefer_ed = PREFER_ED
    if auxcell.dimension == 2:
        prefer_ed = True
    cd_j2c_cache = []
    negative_metric_size = {}
    for j2c_idx, j2c_k in enumerate(j2c):
        # The three-index tensor to construct is
        # cd_j2c^{-1} aux_cart2sph.T (aux[-(kj-ki)]|i,j)
        # The first two terms (cd_j2c^{-1} and aux_cart2sph.T).T can be
        # precomputed and cached.
        cd_j2c, cd_j2c_negative, j2ctag = decompose_j2c(
            j2c_k, prefer_ed, linear_dep_threshold)

        if cd_j2c_negative is not None:
            # concatenate the ED eigenvectors so that the transformation for the two
            # vectors can be processed together
            assert auxcell.dimension == 2
            cd_j2c = cp.hstack(cd_j2c, cd_j2c_negative)
            negative_metric_size[j2c_idx] = cd_j2c_negative.shape[1]

        if j2ctag == 'ED':
            # For ED, cd_j2c^{-1} ~ (ED_eigenvectors * eigvals^{-.5})^\dagger
            cd_j2c = aux_coeff.dot(cd_j2c.conj())
        else:
            #:cd_j2c = aux_coeff.dot(cp.linalg.inv(cd_j2c.T))
            cd_j2c = solve_triangular(cd_j2c, aux_coeff.T, lower=True).T
        cd_j2c_cache.append(cd_j2c)
    return cd_j2c_cache, negative_metric_size

def unpack_cderi(cderi_compressed, cderi_idx, k_idx, kk_conserv, expLk, nao,
                 axis=0, buf=None, out=None):
    r'''
    Constructs a dense cderi tensor from a partially compressed cderi at a
    specific k-point on the auxiliary dimension. The resulting tensor has the
    shape [Nk, naux, nao, nao]. The first dimension corresponds to the sorted
    kpts for orbital i in (ij|aux).

    Args:
        cderi_compressed :
            Compressed cderi tensor, with shape [naux, npair], where the
            orbital-pair is compressed.
        cderi_idx :
            (pari_addresses, and diag_addresses) for the compressed orbital pairs.
        k_idx (int):
            The index of the k-point = kpt_j - kpt_i
        kk_conserv (ndarray):
            kk = kk_conserv[ki,kj] satisfies kpts[kk] = kpts[kj] - kpts[ki] + 2n\pi
            This table can be created by k2gamma.double_translation_indices(kmesh)
            (kk_conserv == k_idx) gives all the ki,kj pairs that can produce k_idx.
        axis (int):
            which index to apply the real-space to k-index transformation.
            If axis=0, transform i in (ij|k) with conj(exp(L*k)). If axis=1,
            transform j in (ij|k) with exp(L*k)
    '''
    pair_address, diag_idx = cderi_idx
    naux = cderi_compressed.shape[0]
    nL, nkpts = expLk.shape
    cderi_tril = ndarray((naux, nao*nL*nao), cderi_compressed.dtype, buffer=buf)
    cderi_tril.fill(0.)
    cderi_tril[:,pair_address] = cderi_compressed
    # diagonal blocks are accessed twice
    cderi_tril[:,pair_address[diag_idx]] *= .5
    cderi_tril = cderi_tril.reshape(naux, nao, nL, nao)
    if expLk.size == 1: # gamma point
        out = ndarray((naux,nao,nao), cderi_compressed.dtype, buffer=out)
        out[:] = cderi_tril[:,:,0,:]
        out += cderi_tril[:,:,0,:].transpose(0,2,1)
        return out.reshape(1,naux,nao,nao)

    assert expLk.dtype == np.complex128
    # Searching adapted k indices for (aux|ij)
    if kk_conserv is None:
        ki_idx = kj_idx = slice(None)
    else:
        ki_idx, kj_idx = np.where(kk_conserv == k_idx)
    if axis == 0:
        expLk_i = expLk.conj()
        # Make kpt_j in expLk_j correspond to the sorted kpt_i
        expLk_j = expLk[:,kj_idx]
    else:
        expLk_i = cp.empty_like(expLk)
        expLk_i[:,kj_idx] = expLk.conj()
        expLk_j = expLk
    if cderi_tril.dtype == np.complex128:
        out = ndarray((nkpts,naux,nao,nao), dtype=np.complex128, buffer=out)
        # p and q in (p q+L|r) are real orbitals: (p+L q|r) = (q p+L|r).
        # The k-adpated tensor (pq|r) can be derived by two types of
        # transformations: transforming index q for (p q+L|r) and transforming p
        # for (p+L q |r).
        out = contract('kjLi,LK->Kkij', cderi_tril, expLk_i, out=out)
        out = contract('kiLj,LK->Kkij', cderi_tril, expLk_j, beta=1., out=out)
    else:
        out = ndarray((nkpts,naux,nao,nao,2), dtype=np.float64, buffer=out)
        expLk_iz = expLk_i.view(np.float64).reshape(nL,nkpts,2)
        expLk_jz = expLk_j.view(np.float64).reshape(nL,nkpts,2)
        out = contract('kjLi,LKz->Kkijz', cderi_tril, expLk_iz, out=out)
        out = contract('kiLj,LKz->Kkijz', cderi_tril, expLk_jz, beta=1., out=out)
        out = out.view(np.complex128)[:,:,:,:,0]
    return out

def _unpack_cderi_v2(cderi_compressed, pair_address, kj_idx, conj_mapping,
                     expLk, nao, axis=0, buf=None, out=None):
    r'''
    Constructs a dense cderi tensor from a partially compressed cderi at a
    specific k-point on the auxiliary dimension. The resulting tensor has the
    shape [Nk, naux, nao, nao]. The first dimension corresponds to the sorted
    kpts for orbital i in (ij|aux). This version does the same thing as
    unpack_cderi in a more efficient way.

    Args:
        cderi_compressed :
            Compressed cderi tensor, with shape [naux, npair], where the
            orbital-pair is compressed.
        kj_idx (ndarray):
            Indices to sort k-points associated with orbital j.
            These indices are obtained from k-point conservation table
            kk_conserv = k2gamma.double_translation_indices(kmesh).
            This table encodes k-point relationships for (ij|k) 3c2e integrals.
            kk = kk_conserv[ki,kj] satisfies kpts[kk] = kpts[kj] - kpts[ki] + 2n\pi
            The indices can be extracted via
            ki_idx, kj_idx = np.where(kk_conserv == k_idx)
        conj_mapping (ndarray):
            Given image index k in BvK cell, conj_mapping[k] shows the
            associated (-k) image in BvK cell. This table can be created by
            the pbc.lib.kpts_helper.conj_images_in_bvk_cell(kmesh) function.
        axis (int):
            which index to apply the real-space to k-index transformation.
            If axis=0, transform i in (ij|k) with conj(exp(L*k)). If axis=1,
            transform j in (ij|k) with exp(L*k)
    '''
    pair_address = cp.asarray(pair_address, dtype=np.int32)
    nao_pairs = len(pair_address)
    naux = cderi_compressed.shape[0]
    nL, nkpts = expLk.shape
    is_gamma_point = expLk.size == 1 and cderi_compressed.dtype == np.float64
    if is_gamma_point:
        buf = out # write to the output directly
    cderi = ndarray((nao*nL*nao, naux), cderi_compressed.dtype, buffer=buf)
    cderi.fill(0.)
    on_host = not isinstance(cderi_compressed, cp.ndarray)
    if cderi_compressed.flags.c_contiguous:
        if cderi_compressed.dtype == np.float64:
            kern = libpbc.decompress_and_transpose
        else:
            kern = libpbc.z_decompress_and_transpose
        if on_host:
            j3c_ptr = cderi_compressed.ctypes
        else:
            j3c_ptr = ctypes.cast(cderi_compressed.data.ptr, ctypes.c_void_p)
        if is_gamma_point:
            fill_triu = True
        else:
            fill_triu = False
        kern(ctypes.cast(cderi.data.ptr, ctypes.c_void_p), j3c_ptr,
             ctypes.cast(pair_address.data.ptr, ctypes.c_void_p),
             ctypes.c_int(nao_pairs), ctypes.c_int(nL*nao), ctypes.c_int(naux),
             ctypes.c_int(0), ctypes.c_int(naux),
             ctypes.c_int(fill_triu), ctypes.c_int(on_host))
    else:
        assert not on_host
        assert cderi_compressed.flags.f_contiguous
        cderi[pair_address] = cderi_compressed.T
        if is_gamma_point:
            tril_idx = pair_address
            conj_ki_order = cp.zeros(1, dtype=np.int32)
            libpbc.fill_indexed_triu(
                ctypes.cast(cderi.data.ptr, ctypes.c_void_p),
                ctypes.cast(tril_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(conj_ki_order.data.ptr, ctypes.c_void_p),
                ctypes.c_int(len(tril_idx)), ctypes.c_int(1),
                ctypes.c_int(nao), ctypes.c_int(naux))
    cderi = cderi.reshape(nao, nL, nao, naux)
    if is_gamma_point:
        return cderi.transpose(1,3,0,2)

    assert nkpts == len(conj_mapping)
    assert nkpts == len(kj_idx)
    assert expLk.dtype == np.complex128
    if axis == 0:
        # j is reordered so that the corresponding index i is sorted
        expLk_j = expLk[:,kj_idx]
        # index j in out has been transformed to an order corresponding to index
        # i in [0...Nk] order. The original kpt for each transformed j-index is
        # provided by the kj_idx.
        conj_ki_order = conj_mapping[kj_idx]
    else:
        expLk_j = expLk
        conj_ki_order = np.empty(nkpts, dtype=np.int32)
        # index j in out has been transformed to the order [0...Nk]
        # The associated index i must be reordered to the argsort(kj_idx)
        # The conj_mapping corresponds to conj(expLk) for transforming index i
        conj_ki_order[kj_idx] = conj_mapping # conj_mapping[ki_idx]
    conj_ki_order = cp.asarray(conj_ki_order, dtype=np.int32)

    if cderi.dtype == np.complex128:
        out = ndarray((nkpts,nao,nao,naux), dtype=np.complex128, buffer=out)
        out = contract('iLjk,LK->Kijk', cderi, expLk_j, out=out)
    else:
        out = ndarray((nkpts,nao,nao,naux,2), dtype=np.float64, buffer=out)
        expLkz = expLk_j.view(np.float64).reshape(nL,nkpts,2)
        out = contract('iLjk,LKz->Kijkz', cderi, expLkz, out=out)
        out = out.view(np.complex128)[:,:,:,:,0]

    # tril_idx in the reference cell associated to the pair_address.
    # Note indices within this array does not guarantee i>=j. It only indicates
    # the unique pairs for each unit cell.
    mask = cp.zeros(nao*nL*nao, dtype=bool)
    mask[pair_address] = True
    mask = cp.any(mask.reshape(nao, nL, nao), axis=1)
    tril_idx = cp.asarray(cp.where(mask.ravel())[0], dtype=np.int32)

    libpbc.fill_indexed_triu(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(tril_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(conj_ki_order.data.ptr, ctypes.c_void_p),
        ctypes.c_int(len(tril_idx)), ctypes.c_int(nkpts),
        ctypes.c_int(nao),
        # *2 for complex number
        ctypes.c_int(naux*2))
    return out.transpose(0,3,1,2)

def get_pp_loc_part1(cell, kpts=None, with_pseudo=True, verbose=None):
    log = logger.new_logger(cell, verbose)
    cell_exps, cs = extract_pgto_params(cell, 'diffuse')
    omega = 0.2
    log.debug('omega guess in get_pp_loc_part1 = %g', omega)

    is_single_kpt = kpts is not None and kpts.ndim == 1
    is_gamma_point = kpts is None or is_zero(kpts)
    if is_gamma_point:
        bvk_kmesh = np.ones(3, dtype=int)
        bvk_ncells = 1
    else:
        bvk_kmesh = kpts_to_kmesh(cell, kpts, bound_by_supmol=True)
        bvk_ncells = np.prod(bvk_kmesh)
    if is_single_kpt:
        kpts = kpts.reshape(1, 3)

    fakenuc = aft_cpu._fake_nuc(cell, with_pseudo=with_pseudo)
    int3c2e_opt = SRInt3c2eOpt(cell, fakenuc, omega=-omega, bvk_kmesh=bvk_kmesh).build()
    charges = -cp.asarray(cell.atom_charges(), dtype=np.float64)
    nuc = int3c2e_opt.contract_auxvec(charges, kpts)

    ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
    mesh = cell.cutoff_to_mesh(ke_cutoff)
    mesh = cell.symmetrize_mesh(mesh)
    Gv, (basex, basey, basez), kws = cell.get_Gv_weights(mesh)
    if with_pseudo:
        #TODO: call multigrid.eval_vpplocG after removing its part2 contribution
        ZG = ft_ao.ft_ao(fakenuc, Gv).conj()
        ZG = ZG.dot(charges)
        ZG *= _weighted_coulG_LR(cell, Gv, omega, kws)
        if ((cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            exps = cp.asarray(np.hstack(fakenuc.bas_exps()))
            ZG[0] -= charges.dot(np.pi/exps) / cell.vol
    else:
        basex = cp.asarray(basex)
        basey = cp.asarray(basey)
        basez = cp.asarray(basez)
        b = cell.reciprocal_vectors()
        coords = cell.atom_coords()
        rb = cp.asarray(coords.dot(b.T))
        SIx = cp.exp(-1j*rb[:,0,None] * basex)
        SIy = cp.exp(-1j*rb[:,1,None] * basey)
        SIz = cp.exp(-1j*rb[:,2,None] * basez)
        SIx *= cp.asarray(-cell.atom_charges())[:,None]
        ZG = cp.einsum('qx,qy,qz->xyz', SIx, SIy, SIz).ravel().conj()
        ZG *= _weighted_coulG_LR(cell, Gv, omega, kws)

    ft_opt = ft_ao.FTOpt(cell, bvk_kmesh=bvk_kmesh).build()
    cell = ft_opt.cell
    pair_address = ft_opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]
    nao_pairs = len(pair_address)

    eval_ft = ft_opt.ft_evaluator(cart=True, original_ao_order=False)[0]

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    avail_mem = mem_free * .8
    ngrids = len(Gv)
    Gblksize = int(avail_mem/(16*nao_pairs)) // 32 * 32
    if Gblksize == 0:
        raise RuntimeError('Insufficient GPU memory')
    log.debug2('ft_ao_iter ngrids = %d Gblksize = %d', ngrids, Gblksize)
    buf = cp.empty(nao_pairs*Gblksize, dtype=np.complex128)
    nuc_compressed = 0
    for p0, p1 in lib.prange(0, ngrids, Gblksize):
        pqG = eval_ft(Gv[p0:p1], out=buf)
        nuc_compressed += contract('pG,G->p', pqG, ZG[p0:p1]).real
    buf = None

    nao = cell.nao
    nuc_raw = cp.zeros((nao * bvk_ncells * nao))
    nuc_raw[pair_address] = nuc_compressed
    nuc_raw = nuc_raw.reshape(nao, bvk_ncells, nao).transpose(1,0,2)
    nuc_raw = fill_triu_bvk(cp.asarray(nuc_raw, order='C'), nao, bvk_kmesh)
    nuc_raw = cell.apply_CT_mat_C(nuc_raw)

    if is_gamma_point:
        nuc_raw = nuc_raw[0]
    else:
        bvkmesh_Ls = translation_vectors_for_kmesh(cell, bvk_kmesh, True)
        expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
        nuc_raw = contract('lk,lpq->kpq', expLk, nuc_raw)

    nuc += nuc_raw
    if is_single_kpt and nuc.ndim == 3:
        nuc = nuc[0]
    return nuc

def get_nuc(cell, kpts=None):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.
    '''
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    nuc = get_pp_loc_part1(cell, kpts, with_pseudo=False, verbose=log)
    log.timer('get_nuc', *t0)
    return nuc

def get_pp(cell, kpts=None):
    '''Get the periodic pseudopotential nuc-el ao matrix, with G=0 removed.
    '''
    from pyscf.pbc.gto import pseudo
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    is_single_kpt = kpts is not None and kpts.ndim == 1
    pp2builder = aft_cpu._IntPPBuilder(cell, kpts)
    vpp  = cp.asarray(pp2builder.get_pp_loc_part2())
    t1 = log.timer_debug1('get_pp_loc_part2', *t0)
    vpp += cp.asarray(pseudo.pp_int.get_pp_nl(cell, kpts))
    t1 = log.timer_debug1('get_pp_nl', *t1)

    vpp += get_pp_loc_part1(cell, kpts, with_pseudo=True, verbose=log)
    if is_single_kpt and vpp.ndim == 3:
        vpp = vpp[0]
    t1 = log.timer_debug1('get_pp_loc_part1', *t1)
    log.timer('get_pp', *t0)
    return vpp

class LinearDepencyError(RuntimeError):
    pass
