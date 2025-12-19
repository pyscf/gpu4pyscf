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
import warnings
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.tools.k2gamma import double_translation_indices
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, unpack_tril
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.scf.jk import (
    apply_coeff_C_mat_CT, apply_coeff_C_mat, _nearest_power2, SHM_SIZE)
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, sr_int2c2e, LMAX, L_AUX_MAX, THREADS, PAGES_PER_BLOCK, PAGE_SIZE)
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.pbc.df.grad.rhf import _split_l_ctr_pattern, int3c2e_scheme
from gpu4pyscf.pbc.grad.krhf import _contract_h1e_dm
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, mo_coeff, mo_occ, kpts=None, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    mo_coeff = asarray(mo_coeff)
    mo_occ = asarray(mo_occ)

    if k_factor == 0:
        c = mo_coeff * mo_occ[:,None,:]
        dm = contract('kpi,kqi->kpq', mo_coeff, c.conj())
        return _j_energy_per_atom(int3c2e_opt, dm, kpts, verbose) * j_factor

    cell = int3c2e_opt.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    auxcell = int3c2e_opt.auxcell
    sorted_cell = int3c2e_opt.sorted_cell
    sorted_auxcell = int3c2e_opt.sorted_auxcell
    bvk_ncells = np.prod(int3c2e_opt.bvk_kmesh)

    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    img_idx_cache = int3c2e_opt.make_img_idx_cache(cutoff)
    log_cutoff = math.log(cutoff)

    # transform the mo_coeff to the AO order in sorted_cell
    mo_coeff = apply_coeff_C_mat(
        mo_coeff, cell, sorted_cell, int3c2e_opt.uniq_l_ctr,
        int3c2e_opt.l_ctr_offsets, int3c2e_opt.ao_idx)
    nocc = cp.count_nonzero(mo_occ > 0, axis=-1).max()
    dm_factor = mo_coeff[:,:,:nocc] * cp.sqrt(mo_occ[:,None,:nocc])
    dm_factor_conj = dm_factor.conj()
    nao, nocc = dm_factor.shape[1:]

    nsp_per_block, gout_stride, shm_size = int3c2e.int3c2e_scheme(gout_width=54)
    lmax = int3c2e_opt.uniq_l_ctr[:,0].max()
    laux = int3c2e_opt.uniq_l_ctr_aux[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    shl_pair_offsets, bas_ij_idx, img_idx, img_offsets = \
            int3c2e._aggregate_shl_pairs(img_idx_cache, np.full((lmax+1,lmax+1), 8))

    # For each primitive shell-pair in bas_ij_idx, ao_pair_loc points to the
    # addresses of first element for the contracted pair-GTOs. In each
    # shell-pair, there are nfij elements. Note, the nfij elements are
    # sorted as [nfj,nfi] (in F-order).
    l = np.arange(max(lmax, laux)+1)
    nf = (l + 1) * (l + 2) // 2
    p0 = p1 = 0
    ao_pair_loc = []
    for li, lj in img_idx_cache:
        p2c_pair_mapping, c_pair_idx = img_idx_cache[li,lj][3:5]
        nfij = nf[li] * nf[lj]
        p0, p1 = p1, p1 + nfij * len(c_pair_idx)
        ao_pair_loc.append(p0 + nfij * p2c_pair_mapping)
    ao_pair_loc = cp.asarray(cp.hstack(ao_pair_loc), dtype=np.int32)
    nao_pair = p1

    nkpts = len(kpts)
    expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conj = expLk.conj()
    expLk_conjz = expLk_conj.view(np.float64).reshape(bvk_ncells,nkpts,2)

    cgto_pair_addresses, diag_idx = int3c2e_opt._pair_and_diag_indices(
        img_idx_cache, for_sorted_cell=True)

    # int3c2e integrals are generated in batches. To avoid the integral
    # temporaries using too large memory, split the auxiliary dimension into
    # small chunks.
    buffer_size = 4e9
    batch_size = max(1, int(buffer_size / (nao_pair*8*bvk_ncells)))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        int3c2e_opt.l_ctr_aux_offsets, int3c2e_opt.uniq_l_ctr_aux, batch_size)

    nbas_aux = sorted_auxcell.nbas
    ksh_offsets, ksh_idx, _ = int3c2e._aggregate_bas_idx(
        l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, nbas_aux, 65536)
    ksh_idx += int3c2e_opt.bvkcell.nbas
    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)

    # The auxiliary functions are sorted to
    # [s,s,s,...,px,px,px,...,py,py,py,...,pz,pz,pz,...] than the
    # conventional order [s,s,...,px,py,pz,px,py,pz,pz,...].
    # aux_sorting maps the addresses of the two storge formats.
    nksh = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]
    batch_aux_from_beginning = cp.zeros(1, dtype=np.int32)

    # To address the density matrix (dm) represented in contracted GTOs,
    # the ao_loc for prim_cell should point to the corresponding offsets of
    # contracted shells.
    ao_loc = sorted_cell.ao_loc
    p2c_ao_loc = ao_loc[int3c2e_opt.prim_to_ctr_mapping]
    ao_loc = np.arange(bvk_ncells, dtype=np.int32)[:,None] * nao + p2c_ao_loc
    aux_loc = sorted_auxcell.ao_loc
    naux = aux_loc[-1]
    bvk_aux_loc = int3c2e_opt.bvk_auxcell.ao_loc
    ao_loc = np.hstack([ao_loc.ravel(), bvk_ncells*nao+bvk_aux_loc])
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    int3c2e_envs.ao_loc = ao_loc.data.ptr

    # k=ijk_conserv[i,j] provides: -i + j - k = 2n\pi
    # therefore, i=ijk_conserv[k,j]
    ijk_conserv = double_translation_indices(int3c2e_opt.bvk_kmesh)
    #for ki in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ki,kj] += j3c_tmp[ijk_conserv[ki,kj],ki]
    #        => order_KI = argsort([ki,ijk_conserv[ki,kj]])
    order_KI = cp.empty(nkpts**2, dtype=int)
    order_KI[(ijk_conserv * nkpts + np.arange(nkpts)).ravel()] = cp.arange(nkpts**2)
    #for kk in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ijk_conserv[kk,kj],kj] += j3c_tmp[kk,kj]
    #        => order_KJ = [ijk_conserv[kk,kj],kj]
    order_KJ = cp.asarray((ijk_conserv * nkpts + np.arange(nkpts)).ravel())

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers,PAGES_PER_BLOCK,PAGE_SIZE), dtype=np.int8)
    kern = libpbc.PBCsr_int3c2e_latsum23_bdiv

    blksize = max(nf.max(), min(int(buffer_size / ((nao*bvk_ncells)**2*8)), naux))

    aux0 = aux1 = 0
    j3c_full = cp.zeros(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    buf1 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    buf2 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    j3c_oo = cp.empty((naux, nkpts, nkpts, nocc, nocc), dtype=np.complex128)
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * nksh[kbatch]
        compressed = cp.zeros((nao_pair, nf[lk], bvk_ncells, nksh[kbatch]))
        err = kern(
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(batch_aux_from_beginning.data.ptr, ctypes.c_void_p),
            ctypes.c_int(bvk_ncells*naux_in_batch),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        compressed = compressed.transpose(3,1,0,2).reshape(naux_in_batch, nao_pair, bvk_ncells)

        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            compressed_p = contract('kfL,LKz->kfKz', compressed[k0:k1], expLk_conjz)
            compressed_p = compressed_p.view(np.complex128)[:,:,:,0]
            # decompress the j3c tensor using the rsdf_builder.unpack_tril algorithm
            j3c = ndarray((dk, nao*bvk_ncells*nao, nkpts), dtype=np.complex128, buffer=j3c_full)
            j3c[:,cgto_pair_addresses] = compressed_p
            # *.5 because diagonal blocks are accessed twice
            j3c[:,cgto_pair_addresses[diag_idx]] *= .5
            j3c = j3c.reshape(dk, nao, bvk_ncells, nao, nkpts)

            j3c_ij = ndarray((dk, nkpts*nkpts, nao*nao), dtype=np.complex128, buffer=buf1)
            j3c_tmp = ndarray((dk, nkpts,nkpts, nao,nao), dtype=np.complex128, buffer=buf2)
            j3c_tmp = contract('kjLiK,LI->kKIij', j3c, expLk_conj, out=j3c_tmp)
            j3c_ij[:,order_KI] = j3c_tmp.reshape(dk,nkpts**2,-1)
            j3c_tmp = contract('kiLjK,LJ->kKJij', j3c, expLk, out=j3c_tmp)
            j3c_ij[:,order_KJ] += j3c_tmp.reshape(dk,nkpts**2,-1)
            j3c_ij = j3c_ij.reshape(dk, nkpts, nkpts, nao, nao)

            tmp = ndarray((dk, nkpts, nkpts, nocc, nao), dtype=np.complex128, buffer=buf2)
            contract('rIJpq,Ipi->rIJiq', j3c_ij, dm_factor_conj, out=tmp)
            contract('rIJiq,Jqj->rIJij', tmp, dm_factor, out=j3c_oo[aux0:aux1])
    buf = buf1 = buf2 = None

    t0 = log.timer_debug1('contract dm', *t0)

    kpt_iters = list(kk_adapted_iter(int3c2e_opt.bvk_kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    omega = int3c2e_opt.omega
    auxcell = int3c2e_opt.auxcell
    j2c = sr_int2c2e(auxcell, -omega, uniq_kpts, int3c2e_opt.bvk_kmesh)
    # Adjust the rcut as cell.rcut is estimated based on overlap integrals
    sorted_auxcell.rcut = int3c2e._estimate_sr_2c2e_rcut(auxcell, omega)
    j2c_ip1 = asarray(sorted_auxcell.pbc_intor('int2c2e_ip1', kpts=uniq_kpts))

    j_factor /= nkpts**2
    k_factor /= nkpts**2
    dm_oo = j3c_oo
    ejk = np.zeros((cell.natm, 3))
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    dm_aux = cp.empty((naux,naux), dtype=np.complex128)
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        metric = aux_coeff.dot(cp.linalg.solve(j2c[j2c_idx], aux_coeff.T))
        dm_oo_k = cp.einsum('uv,vnij->unij', metric, j3c_oo[:,ki_idx,kj_idx])
        dm_oo[:,ki_idx,kj_idx] = dm_oo_k
        if kp != kp_conj:
            dm_oo_kconj = cp.einsum('uv,vnij->unij', metric.conj(), j3c_oo[:,kj_idx,ki_idx])
            dm_oo[:,kj_idx,ki_idx] = dm_oo_kconj
        elif kp == 0:
            dm_oo_kconj = dm_oo_k
        else:
            dm_oo_kconj = dm_oo_k[:,kj_idx]

        beta = 0
        if j_factor != 0 and kp == 0:
            assert all(ki_idx == kj_idx)
            auxvec = dm_oo_k.trace(axis1=2, axis2=3).sum(axis=1)
            dm_aux = cp.multiply(auxvec[:,None], auxvec.conj(), out=dm_aux)
            beta = j_factor

        dm_aux = contract('rkij,skji->rs', dm_oo_k, dm_oo_kconj,
                          alpha=-.5*k_factor, beta=beta, out=dm_aux)
        ejk += _contract_h1e_dm(sorted_auxcell, j2c_ip1[j2c_idx], dm_aux)
        if kp != kp_conj:
            dm_aux = contract('rkij,skji->rs', dm_oo_kconj, dm_oo_k,
                              alpha=-.5*k_factor, out=dm_aux)
            ejk += _contract_h1e_dm(sorted_auxcell, j2c_ip1[j2c_idx].conj(), dm_aux)
    j2c = j2c_ip1 = dm_aux = j3c_oo = metric = None
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme()
    shl_pair_offsets, bas_ij_idx, img_idx, img_offsets = \
            int3c2e._aggregate_shl_pairs(img_idx_cache, nsp_per_block[0]*4)

    if j_factor != 0:
        dm = contract('kpi,kqi->kpq', dm_factor, dm_factor_conj)

    ejk = asarray(ejk)
    kern = libpbc.PBCsr_int3c2e_ejk_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    buf1 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    for kbatch, lk, in enumerate(int3c2e_opt.uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * nksh[kbatch]
        compressed = cp.empty((nao_pair, nf[lk], bvk_ncells, nksh[kbatch]))
        for k0, k1 in lib.prange(0, nksh[kbatch], max(1, blksize//nf[lk])):
            dk = (k1 - k0) * nf[lk]
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nkpts,nkpts,nao,nao,dk), dtype=np.complex128, buffer=buf)
            tmp = ndarray((nkpts,nkpts,nocc,nao,dk), dtype=np.complex128, buffer=buf1)
            # Note the commutation of the indices due to the exchange term (ij|ji)
            contract('rJIji,Jqj->IJiqr', dm_oo[aux0:aux1], dm_factor,
                     -.5*k_factor, out=tmp)
            contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_conj, out=dm_tensor)
            dm_tensor = dm_tensor.reshape(nkpts**2,nao,nao,dk)
            dm_tensor = dm_tensor[order_KJ].reshape(nkpts,nkpts,nao,nao,dk)
            if j_factor != 0:
                dm_tensor[0] += j_factor * auxvec[aux0:aux1] * dm[:,:,:,None]
            tmp = ndarray((nkpts,nao,nao,bvk_ncells,dk), dtype=np.complex128, buffer=buf)
            tmp1 = ndarray((nao,bvk_ncells,nao,bvk_ncells,dk), dtype=np.complex128, buffer=buf1)
            dm_tensor = contract('LK,KJpqr->JpqLr', expLk_conj, dm_tensor, out=tmp)
            dm_tensor = contract('NJ,JpqLr->pNqLr', expLk, dm_tensor, out=tmp1)
            dm_tensor = dm_tensor.reshape(-1,bvk_ncells,k1-k0,nf[lk]).real
            compressed[:,:,:,k0:k1] = dm_tensor[cgto_pair_addresses].transpose(0,3,1,2)

        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size.max()),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_int(1),
            ctypes.cast(ksh_offsets[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(batch_aux_from_beginning.data.ptr, ctypes.c_void_p),
            ctypes.c_int(bvk_ncells*naux_in_batch),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_int3c2e_ejk_ip1 failed')
    # TODO: Add long-range
    buf = buf1 = None
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    ejk = ejk.get()
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, kpts=None, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    cell = int3c2e_opt.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    auxcell = int3c2e_opt.auxcell
    sorted_cell = int3c2e_opt.sorted_cell
    sorted_auxcell = int3c2e_opt.sorted_auxcell
    bvk_ncells = np.prod(int3c2e_opt.bvk_kmesh)

    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    img_idx_cache = int3c2e_opt.make_img_idx_cache(cutoff)
    log_cutoff = math.log(cutoff)

    auxvec = int3c2e_opt.contract_dm(dm, kpts, img_idx_cache=img_idx_cache,
                                     cutoff=cutoff, verbose=log)
    t0 = log.timer_debug1('contract dm', *t0)

    omega = int3c2e_opt.omega
    auxcell = int3c2e_opt.auxcell
    j2c = sr_int2c2e(auxcell, -omega)[0]
    # TODO: Add long-range
    auxvec = cp.linalg.solve(j2c, auxvec)
    auxvec = cp.asarray(int3c2e_opt.aux_coeff).dot(auxvec)
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme()
    shl_pair_offsets, bas_ij_idx, img_idx, img_offsets = \
            int3c2e._aggregate_shl_pairs(img_idx_cache, nsp_per_block[0]*4)
    l_ctr_aux_offsets = int3c2e_opt.l_ctr_aux_offsets
    uniq_l_ctr_aux = int3c2e_opt.uniq_l_ctr_aux
    nbas_aux = sorted_auxcell.nbas
    ksh_offsets, ksh_idx, _ = int3c2e._aggregate_bas_idx(
        l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, nbas_aux, 65536)
    ksh_idx += int3c2e_opt.bvkcell.nbas
    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)

    # To address the density matrix (dm) represented in contracted GTOs,
    # the ao_loc for prim_cell should point to the corresponding offsets of
    # contracted shells.
    ao_loc = sorted_cell.ao_loc
    aux_loc = sorted_auxcell.ao_loc
    nao = ao_loc[-1]
    naux = aux_loc[-1]
    p2c_ao_loc = ao_loc[int3c2e_opt.prim_to_ctr_mapping]
    ao_loc = np.arange(bvk_ncells, dtype=np.int32)[:,None] * nao + p2c_ao_loc

    # .dot(expLk) will perform a summation over all BvK cells for auxiliary
    # dimension. This summation can be performed in advance by shifting aux_loc.
    aux_loc = np.repeat(bvk_ncells*nao + aux_loc[None,:-1], bvk_ncells, axis=0)
    ao_loc = np.hstack([ao_loc.ravel(), aux_loc.ravel(), bvk_ncells*nao+naux])
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    int3c2e_envs.ao_loc = ao_loc.data.ptr

    dm = apply_coeff_C_mat_CT(
        dm, int3c2e_opt.cell, int3c2e_opt.sorted_cell, int3c2e_opt.uniq_l_ctr,
        int3c2e_opt.l_ctr_offsets, int3c2e_opt.ao_idx)
    if kpts is None or is_zero(kpts):
        assert dm.dtype == np.float64
        nkpts = 1
    else:
        expLk = cp.exp(1j*asarray(int3c2e_opt.bvkmesh_Ls).dot(asarray(kpts).T))
        dm = contract('Lk,kpq->Lpq', expLk, dm)
        dm = cp.asarray(dm.real, order='C')
        nkpts = len(kpts)

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers,PAGES_PER_BLOCK,PAGE_SIZE), dtype=np.int8)
    kern = libpbc.PBCsr_int3c2e_ejk_ip1
    ej = cp.zeros((cell.natm, 3))

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_envs),
        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
        ctypes.c_int(shm_size.max()),
        ctypes.c_int(len(bas_ij_idx)),
        ctypes.c_int(len(ksh_offsets) - 1),
        ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(), lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff))
    if err != 0:
        raise RuntimeError('PBCsr_int3c2e_ejk_ip1 failed')
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    # Adjust the rcut as cell.rcut is estimated based on overlap integrals
    sorted_auxcell.rcut = int3c2e._estimate_sr_2c2e_rcut(auxcell, omega)
    j2c_ip1 = asarray(sorted_auxcell.pbc_intor('int2c2e_ip1'))
    dm_aux = auxvec[:,None] * auxvec
    ej = ej.get() / nkpts
    ej += _contract_h1e_dm(sorted_auxcell, j2c_ip1, dm_aux)
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

class Gradients(krhf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_df', 'auxbasis_response'}

    def check_sanity(self):
        from gpu4pyscf.pbc.srdf import SRGDF
        assert isinstance(self.base.with_df, SRGDF)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        raise NotImplementedError

    def get_stress(self):
        raise NotImplementedError

Grad = Gradients
