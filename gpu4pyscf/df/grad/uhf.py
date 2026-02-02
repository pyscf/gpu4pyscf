# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf.df.grad.rhf import (
    int3c2e_scheme, _j_energy_per_atom, factorize_dm, _gen_metric_solver)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys, Int3c2eOpt, int2c2e)
from gpu4pyscf.df import df_jk

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=0,
                        auxbasis_response=True, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    assert dm.ndim == 3
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm[0]+dm[1], hermi,
                                  auxbasis_response, verbose) * j_factor

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = factorize_dm(dm, hermi)
    # transform to the AO order in sorted_cell
    dm_factor_l = mol.apply_C_dot(dm_factor_l, axis=1)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l
    else:
        dm_factor_r = mol.apply_C_dot(dm_factor_r, axis=1)
    nao, nocc = dm_factor_l.shape[1:]
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_avail = mem_free - 2*naux*nocc**2*8 - nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.4/(nao*(nao+2*nocc)*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((2, naux, nocc, nocc))
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            contract('pqr,pi->iqr', j3c, dm_factor_r[0], out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l[0], out=j3c_oo[0,aux0:aux1])
            contract('pqr,pi->iqr', j3c, dm_factor_r[1], out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l[1], out=j3c_oo[1,aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    aux_coeff = cp.asarray(auxmol.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    j2c = int2c2e(auxmol)
    if mol.omega <= 0 and not auxmol.mol.cart:
        metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    else:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
    j2c = aux_coeff = None
    dm_oo = contract('uv,nvij->nuij', metric, j3c_oo)
    metric = j3c_oo = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=2, axis2=3).sum(axis=0)

    # (d/dX P|Q) contributions
    if auxbasis_response:
        if j_factor == 0:
            dm_aux = None
        else:
            dm_aux = auxvec[:,None] * auxvec
        if hasattr(dm, 'mo_coeff'):
            dm_aux = contract('nrij,nsij->rs', dm_oo, dm_oo,
                              alpha=-k_factor, beta=j_factor, out=dm_aux)
        else:
            dm_aux = contract('nrij,nsji->rs', dm_oo, dm_oo,
                              alpha=-k_factor, beta=j_factor, out=dm_aux)
        dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
        ejk_aux = cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
        ejk_aux *= -.5
        t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
        ejk_aux_ptr = ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p)
        dm_aux = None
    else:
        ejk_aux_ptr = lib.c_null_ptr()

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, 54)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    l_ctr_aux_counts = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]

    if j_factor != 0:
        dm = mol.apply_C_mat_CT(dm[0]+dm[1])

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.ejk_int3c2e_ip1
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf2 = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((2, blksize, nao, nocc))
    ejk = cp.zeros((mol.natm, 3))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf2)
            tmp = ndarray((2,nocc,nao,dk), buffer=buf1)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            contract('nrji,nqj->niqr', dm_oo[:,aux0:aux1], dm_factor_l, out=tmp)
            contract('niqr,npi->pqr', tmp, dm_factor_r, -k_factor, beta, out=dm_tensor)
            cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0, out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p), ejk_aux_ptr,
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(1),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao_pair),
            ctypes.c_int(naux_in_batch))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
    if auxbasis_response:
        ejk += ejk_aux
    ejk = ejk.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ejk

class Gradients(uhf_grad.Gradients):
    '''Unrestricted density-fitting Hartree-Fock gradients'''

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df_jk._DFHF)

    def get_veff(self, mol=None, dm=None, verbose=None):
        '''
        Computes the first-order derivatives of the energy contributions from
        Veff per atom, corresponding to contracting dm with Veff:
        [np.einsum('xpq,pq->x', veff[:,AO_idx_for_atom], dm[AO_idx_for_atom]) for all atoms]
        This contraction is equal to 1/2 of the nuclear derivatives of the
        two-electron potential.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        '''
        if mol is None: mol = self.mol
        mf = self.base
        mf.with_df.reset() # Release GPU memory
        if dm is None: dm = mf.make_rdm1()
        int3c2e_opt = Int3c2eOpt(mf.mol, mf.with_df.auxmol).build()
        return _jk_energy_per_atom(
            int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=1,
            auxbasis_response=self.auxbasis_response, verbose=verbose) * .5

Grad = Gradients
