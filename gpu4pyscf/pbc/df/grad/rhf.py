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
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, get_ao_pair_loc, _nearest_power2,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS)
from gpu4pyscf.df.grad.rhf import factorize_dm, int3c2e_scheme
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, diffuse_exps_by_atom, _aggregate_bas_idx, POOL_SIZE)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt
from gpu4pyscf.pbc.grad import rhf as rhf_grad
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, dm, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm, hermi, verbose) * j_factor

    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
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
    nao, nocc = dm_factor_l.shape
    naux = auxcell.nao

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    buffer_size = mem_free // 4
    batch_size = max(1, min(naux, buffer_size // (nao_pair*8*bvk_ncells)))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, buffer_size // (nao**2*8)))
    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)[:,:,0]
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = tmp = compressed = None
    j3c_oo = j3c_oo[aux_sorting]
    t0 = log.timer_debug1('contract dm', *t0)

    int2c2e_opt = Int2c2eOpt(auxcell).build()
    j2c = int2c2e_opt.int2c2e()
    # TODO: Add long-range
    aux_coeff = cp.asarray(auxcell.ctr_coeff)
    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    # (d/dX P|Q) contributions
    if j_factor == 0:
        dm_aux = None
    else:
        dm_aux = auxvec[:,None] * auxvec
    dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                      alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
    # ejk = .5 * contract_h1e_dm(auxcell, auxcell.pbc_intor('int2c2e_ip1'), dm_aux)
    ejk = cp.asarray(int2c2e_opt.energy_ip1_per_atom(dm_aux)) * -.5
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

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

    # Reorder the auxiliary index for better memory access efficiency
    j3c_oo[aux_sorting] = dm_oo
    dm_oo = j3c_oo
    j2c = dm_aux = j3c_oo = metric = None

    if j_factor != 0:
        auxvec, tmp = cp.empty_like(auxvec), auxvec
        auxvec[aux_sorting] = tmp
        tmp = None
        dm = dm_factor_l.dot(dm_factor_r.T)

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf2 = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf2)
            tmp = ndarray((nocc,nao,dk), buffer=buf1)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            contract('rij,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
            cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0, out=compressed[:,k0:k1])
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
            ctypes.c_int(naux_in_batch),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')
    buf = buf1 = buf2 = None
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ejk.get()

def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    dm = cell.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, hermi=hermi)
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

    # (d/dX P|Q) contributions
    dm_aux = auxvec[:,None] * auxvec
    ej += cp.asarray(int2c2e_opt.energy_ip1_per_atom(dm_aux)) * -.5
    ej = ej.get()
    # TODO: Add long-range
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

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
