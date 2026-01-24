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
from gpu4pyscf.df.grad.rhf import (
    _split_l_ctr_pattern, get_ao_pair_loc, libvhf_rys, Int3c2eOpt, int2c2e,
    int3c2e_scheme, _gen_metric_solver)
from gpu4pyscf.df import df
from gpu4pyscf.df.df_jk import factorize_dm
from gpu4pyscf.tdscf import rhf as tdrhf
from gpu4pyscf.grad import tdrhf as tdrhf_grad

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, dms, j_factor=None, k_factor=None, hermi=0,
                        verbose=None):
    '''
    Computes the first-order derivatives of J/K contributions from multiple
    density matrices.
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    if k_factor is None:
        return _j_energy_per_atom(int3c2e_opt, dms, j_factor, hermi, verbose)

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    if not isinstance(dms, cp.ndarray):
        dms = cp.asarray(dms)
    if dms.ndim == 2:
        dms = dms[None]

    dm_factor_l, dm_factor_r = factorize_dm(dms, hermi)
    # transform to the AO order in sorted_cell
    dm_factor_l = mol.apply_C_dot(dm_factor_l, axis=1)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l
    else:
        dm_factor_r = mol.apply_C_dot(dm_factor_r, axis=1)
    n_dm, nao, nocc = dm_factor_l.shape
    # TODO: if nocc is large, memory might not be enough to store a tensor of
    # shape (n_dm, naux, nocc, nocc). Split dms into several sub tensors and
    # process separately.
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_avail = mem_free - n_dm*naux*nocc**2*8 - n_dm*nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.4/(nao*(nao+nocc)*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = [cp.empty((naux, nocc, nocc)) for i in range(n_dm)]
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            for i in range(n_dm):
                contract('pqr,pi->iqr', j3c, dm_factor_r[i], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[i], out=j3c_oo[i][aux0:aux1])
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
    dm_oo = []
    buf = None
    for i in range(n_dm):
        dm_oo.append(contract('uv,vij->uij', metric, j3c_oo[i], out=buf))
        buf = j3c_oo[i]
    metric = j3c_oo = buf = None
    if j_factor is not None:
        auxvec = cp.empty((n_dm, naux))
        for i in range(n_dm):
            dm_oo[i].trace(axis1=1, axis2=2, out=auxvec[i])

    # (d/dX P|Q) contributions
    if j_factor is None:
        dm_aux = cp.zeros((naux,naux))
    else:
        auxvec_jfac = cp.asarray(j_factor)[:,None] * auxvec
        dm_aux = auxvec.T.dot(auxvec_jfac)
    for i in range(n_dm):
        contract('rij,sji->rs', dm_oo[i], dm_oo[i], -.5*k_factor[i], 1, out=dm_aux)
    dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
    ejk_aux = cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux)) * -.5
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    dm_aux = None

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

    if j_factor is not None:
        dms = mol.apply_C_mat_CT(dms)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.ejk_int3c2e_ip1
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf1 = cp.empty((blksize, nao, nocc))
    buf2 = cp.empty((blksize, nao, nao))
    ejk = cp.zeros((mol.natm, 3))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf2)
            tmp = ndarray((nocc,nao,dk), buffer=buf1)
            if j_factor is None:
                dm_tensor[:] = 0
            else:
                contract('npq,nr->pqr', dms, auxvec_jfac[:,aux0:aux1], out=dm_tensor)
            for i in range(n_dm):
                contract('rji,qj->iqr', dm_oo[i][aux0:aux1], dm_factor_l[i], out=tmp)
                contract('iqr,pi->pqr', tmp, dm_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
            cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0, out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
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
    ejk += ejk_aux
    ejk = ejk.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ejk

def _j_energy_per_atom(int3c2e_opt, dms, j_factor, hermi=0, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    if dms.ndim == 2:
        dms = dms[None]
    dms = mol.apply_C_mat_CT(dms)
    auxvec = int3c2e_opt.contract_dm(dms, hermi)
    auxvec = auxmol.apply_CT_dot(auxvec, axis=1)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    n_dm = len(dms)
    assert len(j_factor) == n_dm
    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = cp.linalg.solve(j2c, auxvec.T).T
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxvec.T).T
    auxvec = cp.asarray(auxmol.apply_C_dot(auxvec, axis=1), order='C')
    auxvec_jfac = auxvec * cp.asarray(j_factor)[:,None]
    naux = auxvec.shape[1]
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, 54)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.ejk_int3c2e_ip1
    ej = cp.zeros((mol.natm, 3))
    ej_aux = cp.zeros_like(ej)

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec_jfac.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(), ctypes.c_int(0),
        ctypes.c_int(0), ctypes.c_int(naux))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
    dm_aux = auxvec.T.dot(auxvec_jfac)
    ej_aux -= .5 * cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

class Gradients(tdrhf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)

    def get_veff(self, mol=None, dm=None, j_factor=1, k_factor=1, omega=0,
                 hermi=0, verbose=None):
        from gpu4pyscf.df.grad.rhf import _jk_energy_per_atom
        if mol is None:
            mol = self.mol
        mf = self.base._scf
        if dm is None:
            dm = mf.make_rdm1()
        auxmol = mf.with_df.auxmol
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
            return _jk_energy_per_atom(
                int3c2e_opt, dm, j_factor, k_factor, hermi,
                auxbasis_response=self.auxbasis_response, verbose=verbose) * .5

    def jk_energy_per_atom(self, dms, j_factor=None, k_factor=None, omega=0,
                           hermi=0, verbose=None):
        assert self.auxbasis_response
        mol = self.mol
        mf = self.base._scf
        auxmol = mf.with_df.auxmol
        mf.with_df._cderi = None # Release memory
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
            return _jk_energy_per_atom(
                int3c2e_opt, dms, j_factor, k_factor, hermi, verbose=verbose)


Grad = Gradients
