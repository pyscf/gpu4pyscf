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
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, unpack_tril
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.scf.jk import (
    apply_coeff_C_mat_CT, apply_coeff_C_mat, _nearest_power2, SHM_SIZE)
from gpu4pyscf.df.int3c2e_bdiv import _split_l_ctr_pattern
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, sr_int2c2e, LMAX, L_AUX_MAX, THREADS, PAGES_PER_BLOCK, PAGE_SIZE)
from gpu4pyscf.pbc.grad import rhf as rhf_grad
from gpu4pyscf.pbc.grad.krhf import _contract_h1e_dm

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, mo_coeff, mo_occ, exxdiv=None,
                        j_factor=1., k_factor=1., verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    mo_coeff = asarray(mo_coeff)
    mo_occ = asarray(mo_occ)
    assert mo_coeff.dtype == np.float64

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
    bvk_ncells = np.prod(int3c2e_opt.bvk_kmesh)

    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    img_idx_cache = int3c2e_opt.make_img_idx_cache(cutoff)
    log_cutoff = math.log(cutoff)

    # transform the mo_coeff to the AO order in sorted_cell
    mo_coeff = apply_coeff_C_mat(
        mo_coeff, cell, sorted_cell, int3c2e_opt.uniq_l_ctr,
        int3c2e_opt.l_ctr_offsets, int3c2e_opt.ao_idx)
    mask = mo_occ > 0
    dm_factor = mo_coeff[:,mask]
    dm_factor *= cp.sqrt(mo_occ[mask])
    nao, nocc = dm_factor.shape

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

    cgto_pair_addresses = int3c2e_opt._pair_and_diag_indices(
        img_idx_cache, for_sorted_cell=True)[0]
    i_addr, j_addr = divmod(cgto_pair_addresses, nao)

    # int3c2e integrals are generated in batches. To avoid the integral
    # temporaries using too large memory, split the auxiliary dimension into
    # small chunks.
    # TODO: move into the SRInt3c2eOpt.build() method
    buffer_size = 5e9
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
    aux0 = aux1 = 0
    aux_sorting = []
    nksh = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]
    for k, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux0, aux1 = aux1, aux1 + nf[lk] * nksh[k]
        aux_sorting.append(cp.arange(aux0, aux1).reshape(nf[lk], nksh[k]).T.ravel())
    aux_sorting = cp.hstack(aux_sorting)
    naux = aux1
    batch_aux_from_beginning = cp.zeros(1, dtype=np.int32)

    # To address the density matrix (dm) represented in contracted GTOs,
    # the ao_loc for prim_cell should point to the corresponding offsets of
    # contracted shells.
    ao_loc = sorted_cell.ao_loc
    p2c_ao_loc = ao_loc[int3c2e_opt.prim_to_ctr_mapping]
    ao_loc = np.arange(bvk_ncells, dtype=np.int32)[:,None] * nao + p2c_ao_loc
    bvk_aux_loc = int3c2e_opt.bvk_auxcell.ao_loc
    ao_loc = np.hstack([ao_loc.ravel(), bvk_ncells*nao+bvk_aux_loc])
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    int3c2e_envs.ao_loc = ao_loc.data.ptr

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers,PAGES_PER_BLOCK,PAGE_SIZE), dtype=np.int8)
    kern = libpbc.PBCsr_int3c2e_latsum23_bdiv

    buffer_size = 5e9
    blksize = max(1, min(int(buffer_size / (nao**2*8)), naux))

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * nksh[kbatch]
        compressed = cp.zeros((nao_pair, naux_in_batch))
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
            ctypes.c_int(naux_in_batch),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))

        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:k1-k0]
            j3c[j_addr,i_addr] = compressed[:,k0:k1]
            j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = contract('pqr,pi->riq', j3c, dm_factor, out=buf1[:dk])
            contract('riq,qj->rij', tmp, dm_factor, out=j3c_oo[aux0:aux1])
    j3c_full = buf1 = None
    j3c_oo = j3c_oo[aux_sorting]

    t0 = log.timer_debug1('contract dm', *t0)

    omega = int3c2e_opt.omega
    auxcell = int3c2e_opt.auxcell
    j2c = sr_int2c2e(auxcell, -omega)[0]
    # TODO: Add long-range
    aux_coeff = cp.asarray(int3c2e_opt.aux_coeff)
    metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    # (d/dX P|Q) contributions
    # Adjust the rcut as cell.rcut is estimated based on overlap integrals
    sorted_auxcell.rcut = int3c2e._estimate_sr_2c2e_rcut(auxcell, omega)
    j2c_ip1 = asarray(sorted_auxcell.pbc_intor('int2c2e_ip1'))
    if j_factor == 0:
        dm_aux = None
    else:
        dm_aux = auxvec[:,None] * auxvec
    dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                      alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
    ejk = asarray(_contract_h1e_dm(sorted_auxcell, j2c_ip1, dm_aux))
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    # Reorder the auxiliary index for better memory access efficiency
    j3c_oo[aux_sorting] = dm_oo
    dm_oo = j3c_oo
    j2c = j2c_ip1 = dm_aux = j3c_oo = metric = None

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme()
    shl_pair_offsets, bas_ij_idx, img_idx, img_offsets = \
            int3c2e._aggregate_shl_pairs(img_idx_cache, nsp_per_block[0]*4)

    if j_factor != 0:
        auxvec, auxvec_tmp = cp.empty_like(auxvec), auxvec
        auxvec[aux_sorting] = auxvec_tmp
        dm = dm_factor.dot(dm_factor.T)

    kern = libpbc.PBCsr_int3c2e_ejk_ip1
    aux0 = aux1 = 0
    buf = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))
    for kbatch, lk, in enumerate(int3c2e_opt.uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * nksh[kbatch]
        compressed = cp.empty((nao_pair, naux_in_batch))
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf)
            tmp = ndarray((nocc,nao,dk), buffer=buf1)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            contract('rij,qj->iqr', dm_oo[aux0:aux1], dm_factor, out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor, -.5*k_factor, beta, out=dm_tensor)
            compressed[:,k0:k1] = dm_tensor.reshape(-1,dk)[cgto_pair_addresses]

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
            ctypes.c_int(naux_in_batch),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_int3c2e_ejk_ip1 failed')
    # TODO: Add long-range
    buf = buf1 = None
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ejk.get()

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
    bvk_ncells = np.prod(int3c2e_opt.bvk_kmesh)

    cutoff = int3c2e_opt.estimate_cutoff_with_penalty()
    img_idx_cache = int3c2e_opt.make_img_idx_cache(cutoff)
    log_cutoff = math.log(cutoff)

    auxvec = int3c2e_opt.contract_dm(dm, img_idx_cache=img_idx_cache,
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
    ej = ej.get()
    ej += _contract_h1e_dm(sorted_auxcell, j2c_ip1, dm_aux)
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
