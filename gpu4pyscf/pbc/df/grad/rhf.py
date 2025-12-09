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
from pyscf.lib.misc import _blocksize_partition
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, unpack_tril
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.scf.jk import (
    apply_coeff_C_mat_CT, apply_coeff_C_mat, _nearest_power2, SHM_SIZE)
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, sr_int2c2e, LMAX, L_AUX_MAX, THREADS, PAGES_PER_BLOCK, PAGE_SIZE,
    _aggregate_shl_pairs, _aggregate_bas_idx)
from gpu4pyscf.pbc.grad import rhf as rhf_grad
from gpu4pyscf.pbc.grad.krhf import _contract_h1e_dm

def _jk_energy_per_atom(int3c2e_opt, mo_coeff, mo_occ, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    mo_coeff = asarray(mo_coeff)
    mo_occ = asarray(mo_occ)

    if k_factor == 0:
        mask = mo_occ > 0
        orbo = mo_coeff[:,mask]
        orbo *= cp.sqrt(mo_occ[mask])
        dm = orbo.dot(orbo.T)
        return _j_energy_per_atom(int3c2e_opt, dm, verbose) * j_factor

    cell = int3c2e_opt.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    auxcell = int3c2e_opt.auxcell
    sorted_cell = int3c2e_opt.sorted_cell
    sorted_auxcell = int3c2e_opt.sorted_auxcell
    bvkcell = int3c2e_opt.bvkcell
    l_ctr_aux_offsets = int3c2e_opt.l_ctr_aux_offsets

    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    img_idx_cache = int3c2e_opt.make_img_idx_cache(cutoff)
    log_cutoff = math.log(cutoff)

    mo_coeff = apply_coeff_C_mat(mo_coeff, cell, sorted_cell,
                                 int3c2e_opt.uniq_l_ctr, int3c2e_opt.ao_idx)
    mask = mo_occ > 0
    dm_factor = mo_coeff[:,mask]
    dm_factor *= cp.sqrt(mo_occ[mask])
    nao, nocc = dm_factor.shape
    aux_loc = sorted_auxcell.ao_loc
    naux = aux_loc[-1]

    l = np.arange(LMAX+1)
    nf = (l + 1) * (l + 2) // 2
    p0 = p1 = 0
    ao_pair_offsets = {}
    for (li, lj), img_idx in img_idx_cache.items():
        npairs = nf[li] * nf[lj] * len(img_idx[4])
        ao_pair_offsets[li, lj] = p0, p1 = p1, p1 + npairs
    nao_pairs = p1

    ao_pair_mapping, diag_addresses = int3c2e_opt._pair_and_diag_indices(
        img_idx_cache, cart=True)

    evaluate = int3c2e_opt.int3c2e_evaluator(img_idx_cache, verbose=log)
    oo_tril_mask = cp.arange(nocc)[:,None] >= cp.arange(nocc)
    noo_tril = nocc*(nocc+1)//2
    j3c_oo = cp.empty((naux, noo_tril), dtype=np.complex128)
    for k in range(len(int3c2e_opt.uniq_l_ctr_aux)):
        ksh0, ksh1 = l_ctr_aux_offsets[k:k+2]
        k0, k1 = aux_loc[ksh0], aux_loc[ksh1]
        compressed = cp.empty((k1-k0, nao_pairs))
        for li, lj in img_idx_cache:
            p0, p1 = ao_pair_offsets[li, lj]
            c_pair_idx, tmp = evaluate(li, lj, k)
            # kjiuLv->Lvkuji
            compressed[:,p0:p1] = tmp.transpose(4,5,0,3,1,2).reshape(k1-k0,-1)

        j3c = cp.zeros((k1-k0, nao, nao))
        i, j = divmod(ao_pair_mapping, nao)
        j3c[:,i,j] = compressed
        j3c[:,j,i] = compressed
        roo = contract('rpq,qj->rpj', j3c, dm_factor)
        roo = contract('rpj,pi->rij', roo, dm_factor)
        j3c_oo[k0:k1] = roo[:,oo_tril_mask]

    i = np.arange(nocc)
    oo_diag_idx = i*(i+1)//2 + i
    if j_factor != 0:
        auxvec = j3c_oo[:,oo_diag_idx].sum(axis=1)

    t0 = log.timer_debug1('contract dm', *t0)

    omega = int3c2e_opt.omega
    auxcell = int3c2e_opt.auxcell
    with auxcell.with_range_coulomb(omega):
        int2c = sr_int2c2e(auxcell, -omega)[0]
        # TODO: Add long-range

    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme()
    shl_pair_offsets, bas_ij_idx, img_idx, img_offsets = \
            _aggregate_shl_pairs(img_idx_cache, nsp_per_block)
    bvk_ncells = np.prod(int3c2e_opt.bvk_kmesh)
    nbas_aux = int3c2e_opt.sorted_auxcell.nbas
    ksh_offsets, ksh_idx = _aggregate_bas_idx(l_ctr_aux_offsets,
                                              bvk_ncells, nbas_aux)
    ksh_idx += bvkcell.nbas
    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)

    # To address the density matrix (dm) represented in contracted GTOs,
    # the ao_loc for prim_cell should point to the corresponding offsets of
    # contracted shells.
    ao_loc = sorted_cell.ao_loc
    aux_loc = int3c2e_opt.sorted_auxcell.ao_loc
    nao = ao_loc[-1]
    naux = aux_loc[-1]
    p2c_ao_loc = ao_loc[int3c2e_opt.prim_to_ctr_mapping]
    ao_loc = np.arange(bvk_ncells, dtype=np.int32)[:,None] * nao + p2c_ao_loc

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers,PAGES_PER_BLOCK,PAGE_SIZE), dtype=np.int8)
    kern = libpbc.PBCsr_int3c2e_ejk_ip1
    ejk = cp.zeros((cell.natm, 3))

    metric = aux_coeff.dot(cp.linalg.solve(int2c, aux_coeff.T))
    dm_oo = metric.dot(j3c_oo)
    if j_factor != 0:
        auxvec = metric.dot(auxvec)
    auxvec_ptr = lib.c_null_ptr()
    int2c = j3c_oo = metric = None

    aux_loc = np.arange(bvk_ncells, dtype=np.int32)[:,None] * naux + aux_loc[:-1]
    ao_loc = np.hstack([ao_loc.ravel(), aux_loc.ravel(), bvk_ncells*(nao+naux)])
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    int3c2e_envs.ao_loc = ao_loc.data.ptr

    nao, nocc = dm_factor.shape
    mem_size = 5e9
    blksize = int(mem_size / (nao**2*8))
    assert all(ksh_idx[1:] - ksh_idx[:-1] == 1)
    kbatch_partitions = _blocksize_partition(aux_loc[ksh_offsets], blksize)
    # kbatch_sizes saves the number of AOs within each batch
    kbatch_ao_offsets = aux_loc[ksh_offsets[kbatch_partitions]]
    kbatch_sizes = kbatch_ao_offsets[1:] - kbatch_ao_offsets[:-1]

    blksize = kbatch_sizes.max()
    buf = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))

    for k, batch_naux in enumerate(kbatch_sizes):
        kb0 = kbatch_partitions[k]
        kb1 = kbatch_partitions[k+1]
        nbatches_ksh = kb1 - kb0
        k0 = kbatch_ao_offsets[k]
        k1 = kbatch_ao_offsets[k+1]
        dm_tensor = buf[:batch_naux]
        beta = 0
        if j_factor != 0:
            cp.multiply(auxvec[k0:k1,None,None], dm, out=dm_tensor)
            beta = j_factor
        tmp = unpack_tril(dm_oo[k0:k1])
        tmp = contract('rij,pi->rpj', tmp, dm_factor, -.5*k_factor, out=buf1[:batch_naux])
        contract('rpj,qj->rpq', tmp, dm_factor, beta=beta, out=dm_tensor)
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_tensor.data.ptr, ctypes.c_void_p),
            auxvec_ptr,
            ctypes.byref(int3c2e_envs),
            ctypes.cast(pool.data.ptr, ctypes.c_void_p),
            ctypes.c_int(shm_size.max()),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_int(nbatches_ksh),
            ctypes.cast(ksh_offsets[kb0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(img_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_int3c2e_ejk_ip1 failed')
    # TODO: Add long-range
    buf = buf1 = None
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    sorted_auxcell = int3c2e_opt.sorted_auxcell
    int2c_e1 = asarray(sorted_auxcell.pbc_intor('int2c2e_ip1'))
    dm_aux = auxvec[:,None] * auxvec
    dm_oo[:,oo_diag_idx] *= .5
    dm_aux = contract('ijr,ijs->rs', dm_oo, dm_oo,
                      alpha=-2*k_factor, beta=j_factor, out=dm_aux)
    ejk = ejk.get()
    ejk += _contract_h1e_dm(sorted_auxcell, int2c_e1, dm_aux)
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    cell = int3c2e_opt.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    auxcell = int3c2e_opt.auxcell
    sorted_cell = int3c2e_opt.sorted_cell
    sorted_auxcell = int3c2e_opt.sorted_auxcell
    bvkcell = int3c2e_opt.bvkcell
    l_ctr_aux_offsets = int3c2e_opt.l_ctr_aux_offsets

    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    img_idx_cache = int3c2e_opt.make_img_idx_cache(cutoff)
    log_cutoff = math.log(cutoff)

    auxvec = int3c2e_opt.contract_dm(dm, img_idx_cache=img_idx_cache,
                                     cutoff=cutoff, verbose=log)
    t0 = log.timer_debug1('contract dm', *t0)

    omega = int3c2e_opt.omega
    auxcell = int3c2e_opt.auxcell
    with auxcell.with_range_coulomb(omega):
        int2c = sr_int2c2e(auxcell, -omega)[0]
        # TODO: Add long-range

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme()
    shl_pair_offsets, bas_ij_idx, img_idx, img_offsets = \
            _aggregate_shl_pairs(img_idx_cache, nsp_per_block)
    bvk_ncells = np.prod(int3c2e_opt.bvk_kmesh)
    nbas_aux = int3c2e_opt.sorted_auxcell.nbas
    ksh_offsets, ksh_idx = _aggregate_bas_idx(l_ctr_aux_offsets,
                                              bvk_ncells, nbas_aux)
    ksh_idx += bvkcell.nbas
    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)

    # To address the density matrix (dm) represented in contracted GTOs,
    # the ao_loc for prim_cell should point to the corresponding offsets of
    # contracted shells.
    ao_loc = sorted_cell.ao_loc
    aux_loc = int3c2e_opt.sorted_auxcell.ao_loc
    nao = ao_loc[-1]
    naux = aux_loc[-1]
    p2c_ao_loc = ao_loc[int3c2e_opt.prim_to_ctr_mapping]
    ao_loc = np.arange(bvk_ncells, dtype=np.int32)[:,None] * nao + p2c_ao_loc

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers,PAGES_PER_BLOCK,PAGE_SIZE), dtype=np.int8)
    kern = libpbc.PBCsr_int3c2e_ejk_ip1
    ej = cp.zeros((cell.natm, 3))

    dm = apply_coeff_C_mat_CT(
        dm, int3c2e_opt.cell, int3c2e_opt.sorted_cell, int3c2e_opt.uniq_l_ctr,
        int3c2e_opt.l_ctr_offsets, int3c2e_opt.ao_idx)
    auxvec = cp.linalg.solve(int2c, auxvec)
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    auxvec = aux_coeff.dot(asarray(auxvec))
    int2c = aux_coeff = None

    # .dot(expLk) will perform a summation over all BvK cells for auxiliary
    # dimension. This summation can be performed in advance by shifting aux_loc.
    aux_loc = np.repeat(bvk_ncells*nao + aux_loc[None,:-1], bvk_ncells, axis=0)
    ao_loc = np.hstack([ao_loc.ravel(), aux_loc.ravel(), bvk_ncells*nao+naux])
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    int3c2e_envs.ao_loc = ao_loc.data.ptr

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
        ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff))
    if err != 0:
        raise RuntimeError('PBCsr_int3c2e_ejk_ip1 failed')
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    sorted_auxcell = int3c2e_opt.sorted_auxcell
    int2c_e1 = asarray(sorted_auxcell.pbc_intor('int2c2e_ip1'))
    dm_aux = auxvec[:,None] * auxvec
    ej = ej.get()
    ej += _contract_h1e_dm(sorted_auxcell, int2c_e1, dm_aux)
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def int3c2e_scheme(shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    order = li + lj + lk + 1
    nroots = (order//2 + 1) * 2
    g_size = (li+2)*(lj+1)*(lk+2)
    unit = g_size*3 + nroots*2 + 7
    nsp_max = shm_size // (unit*8)
    nsp_max = _nearest_power2(nsp_max)
    nsp_per_block = np.where(nsp_max < THREADS, nsp_max, THREADS)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    return nsp_per_block, gout_stride, shm_size

class Gradients(rhf_grad.Gradients):
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
