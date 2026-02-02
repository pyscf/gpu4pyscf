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
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, diffuse_exps_by_atom, _aggregate_bas_idx, POOL_SIZE)
from gpu4pyscf.pbc.grad import krhf as krhf_grad
from gpu4pyscf.pbc.df.grad.rhf import (
    _split_l_ctr_pattern, get_ao_pair_loc, int3c2e_scheme, factorize_dm)
from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt
from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter, conj_images_in_bvk_cell

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if kpts is None:
        kpts = np.zeros((1, 3))
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm, kpts, hermi, verbose) * j_factor

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
    nao, nocc = dm_factor_l.shape[1:]
    naux = auxcell.nao

    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)
    nao_pair = len(pair_addresses)

    nkpts = len(kpts)
    expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conj = expLk.conj()
    expLk_conjz = expLk_conj.view(np.float64).reshape(bvk_ncells,nkpts,2)

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    buffer_size = mem_free // 4
    batch_size = max(1, min(naux, buffer_size // (nao_pair*8*bvk_ncells)))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, buffer_size // ((nao*bvk_ncells)**2*8)))

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

    aux0 = aux1 = 0
    j3c_full = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    buf = cp.empty((bvk_ncells*batch_size, nao_pair))
    buf1 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    buf2 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
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
            # decompress the j3c tensor using the rsdf_builder.unpack_tril algorithm
            j3c = ndarray((nao*bvk_ncells*nao, dk, nkpts), dtype=np.complex128, buffer=j3c_full)
            j3c[:] = 0
            j3c[pair_addresses] = compressed[:,k0:k1]
            j3c = j3c.reshape(nao, bvk_ncells, nao, dk, nkpts)

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
    compressed = tmp = j3c_tmp = j3c_ij = None
    t0 = log.timer_debug1('contract dm', *t0)

    kpt_iters = list(kk_adapted_iter(int3c2e_opt.bvk_kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    int2c2e_opt = Int2c2eOpt(auxcell, int3c2e_opt.bvk_kmesh).build()
    j2c = int2c2e_opt.int2c2e(uniq_kpts)
    j2c_ip1 = auxcell.pbc_intor('int2c2e_ip1', kpts=uniq_kpts)

    j_factor /= nkpts**2
    k_factor /= nkpts**2
    dm_oo = j3c_oo
    ejk = np.zeros((cell.natm, 3))
    aux_coeff = auxcell.ctr_coeff
    buf = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
    buf1 = cp.empty((naux, nkpts, nocc, nocc), dtype=np.complex128)
    dm_aux = cp.empty((naux, naux), dtype=np.complex128)
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        metric = aux_coeff.dot(cp.linalg.solve(j2c[j2c_idx], aux_coeff.T))
        j3c_oo_k = j3c_oo[aux_sorting[:,None],ki_idx,kj_idx]
        dm_oo_k = contract('uv,vnij->unij', metric, j3c_oo_k, out=buf)
        dm_oo[aux_sorting[:,None],ki_idx,kj_idx] = dm_oo_k
        if kp == 0:
            dm_oo_kconj = dm_oo_k
        elif kp == kp_conj:
            # for kp == kp_conj != 0, dm_oo_kconj and dm_oo_k correspond to
            # the same blocks in dm_oo, which has been updated previously
            dm_oo_kconj = dm_oo_k[:,kj_idx]
            dm_oo[aux_sorting[:,None],kj_idx,ki_idx] = dm_oo_kconj
        else:
            j3c_oo_k = j3c_oo[aux_sorting[:,None],kj_idx,ki_idx]
            dm_oo_kconj = contract('uv,vnij->unij', metric.conj(), j3c_oo_k, out=buf1)
            dm_oo[aux_sorting[:,None],kj_idx,ki_idx] = dm_oo_kconj

        beta = 0
        if j_factor != 0 and kp == 0:
            assert all(ki_idx == kj_idx)
            auxvec = dm_oo_k.trace(axis1=2, axis2=3).sum(axis=1)
            dm_aux = cp.multiply(auxvec[:,None], auxvec.conj(), out=dm_aux)
            beta = j_factor

        dm_aux = contract('rkij,skji->rs', dm_oo_k, dm_oo_kconj,
                          alpha=-.5*k_factor, beta=beta, out=dm_aux)
        j2c_k = asarray(j2c_ip1[j2c_idx])
        ejk += contract_h1e_dm(auxcell, j2c_k, dm_aux, hermi=1) * .5
        if kp != kp_conj:
            dm_aux = contract('rkij,skji->rs', dm_oo_kconj, dm_oo_k,
                              alpha=-.5*k_factor, out=dm_aux)
            ejk += contract_h1e_dm(auxcell, j2c_k.conj(), dm_aux, hermi=1) * .5
    j2c = j2c_ip1 = dm_aux = j3c_oo = metric = j3c_oo_k = j2c_k = buf = buf1 = None
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx = SortedMole.aggregate_shl_pairs(cell, int3c2e_opt.bas_ij_cache, 1000000)[0]
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

    if j_factor != 0:
        dm = contract('kpi,kqi->kpq', dm_factor_l, dm_factor_r)
        auxvec, tmp = cp.empty_like(auxvec), auxvec
        auxvec[aux_sorting] = tmp
        tmp = None

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    ejk = asarray(ejk)
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
            dm_tensor = ndarray((nkpts,nkpts,nao,nao,dk), dtype=np.complex128, buffer=buf2)
            tmp = ndarray((nkpts,nkpts,nocc,nao,dk), dtype=np.complex128, buffer=buf1)
            # Note the commutation of the indices due to the exchange term (ij|ji)
            contract('rJIji,Jqj->IJiqr', dm_oo[aux0:aux1], dm_factor_l,
                     -.5*k_factor, out=tmp)
            contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_r, out=dm_tensor)
            dm_tensor = dm_tensor.reshape(nkpts**2,nao,nao,dk)
            dm_tensor = dm_tensor[order_KJ].reshape(nkpts,nkpts,nao,nao,dk)
            if j_factor != 0:
                dm_tensor[0] += j_factor * auxvec[aux0:aux1] * dm[:,:,:,None]
            tmp = ndarray((nkpts,nao,nao,dk,bvk_ncells), dtype=np.complex128, buffer=buf2)
            tmp1 = ndarray((nao,bvk_ncells,nao,dk,bvk_ncells), dtype=np.complex128, buffer=buf1)
            dm_tensor = contract('KJpqr,LK->JpqrL', dm_tensor, expLk_conj, out=tmp)
            dm_tensor = contract('JpqrL,NJ->pNqrL', dm_tensor, expLk, out=tmp1)
            dm_tensor = dm_tensor.reshape(-1,dk,bvk_ncells).real
            #:compressed[:,k0:k1] = dm_tensor[cgto_pair_addresses]
            cp.take(dm_tensor, pair_addresses, axis=0, out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
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
    buf = buf1 = buf2 = None
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    ejk = ejk.get()
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    dm = cell.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, kpts, hermi=hermi)
    t0 = log.timer_debug1('contract dm', *t0)

    int2c2e_opt = Int2c2eOpt(auxcell).build()
    j2c = int2c2e_opt.int2c2e()
    # TODO: Add long-range
    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        auxvec = cp.linalg.solve(j2c, auxvec)
    auxvec = auxcell.C_dot_mat(auxvec)
    naux = len(auxvec)
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx = SortedMole.aggregate_shl_pairs(cell, int3c2e_opt.bas_ij_cache, 1000000)[0]

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

    if kpts is None or is_zero(kpts):
        dm == cp.asarray(dm.real, order='C')
        nkpts = 1
    else:
        expLk = cp.exp(1j*asarray(int3c2e_opt.bvkmesh_Ls).dot(asarray(kpts).T))
        dm = contract('Lk,kpq->Lpq', expLk, dm)
        dm = cp.asarray(dm.real, order='C')
        nkpts = len(kpts)

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    ej = cp.zeros((cell.natm, 3))
    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
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
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    ej /= nkpts

    # (d/dX P|Q) contributions
    dm_aux = auxvec[:,None] * auxvec
    ej += cp.asarray(int2c2e_opt.energy_ip1_per_atom(dm_aux)) * -.5
    ej = ej.get()
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
