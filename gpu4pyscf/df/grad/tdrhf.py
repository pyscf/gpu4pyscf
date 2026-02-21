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
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, transpose_sum
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
    density matrices and adds up the results.
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    if k_factor is None:
        return _j_energy_per_atom(int3c2e_opt, dms, j_factor, hermi, verbose)

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = _factorize_multiple_dm(mol, dms, hermi)
    n_dm, nao, nocc = dm_factor_l.shape
    assert len(k_factor) == n_dm
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

    blksize = max(1, min(naux, int(mem_avail*.4/(nao*nao*2*8))//8*8))
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
    ejk_aux = -cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
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
        dms = contract('npi,nqi->npq', dm_factor_l, dm_factor_r)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((blksize, nao, nao))
    ejk = cp.zeros((mol.natm, 3))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf1)
            tmp = ndarray((nocc,nao,dk), buffer=buf2)
            if j_factor is None:
                dm_tensor[:] = 0
            else:
                contract('npq,nr->pqr', dms, auxvec_jfac[:,aux0:aux1], out=dm_tensor)
            for i in range(n_dm):
                contract('rji,qj->iqr', dm_oo[i][aux0:aux1], dm_factor_l[i], out=tmp)
                contract('iqr,pi->pqr', tmp, dm_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
            if hermi == 1:
                cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
                compressed[:] *= 2.
            else:
                dm_tensor1 = ndarray((nao,nao,dk), buffer=buf2)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
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
    if hermi != 1:
        dms = transpose_sum(dms, inplace=True)
        dms[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dms, hermi=1)
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
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
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
    ej *= 2
    ej_aux *= 2
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
    dm_aux = auxvec.T.dot(auxvec_jfac)
    ej_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _jk_energies_per_atom(int3c2e_opt, dm_pairs, j_factor=None, k_factor=None, hermi=None,
                          verbose=None):
    '''
    Computes a set of first-order derivatives of J/K contributions for each
    element (density matrix or a pair of density matrices) in dm_pairs.
    This method is similar _jk_energy_per_atom, but instead of summing all
    contributions into a single result, it returns derivatives for each
    individual set.

    This function supports evaluating multiple sets of energy derivatives in a
    single call. Additionally, for each set, the two density matrices for the
    four-index Coulomb integrals can be different.

    Args:
        dm_pairs:
            A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
            Each element corresponds to one set of energy derivative.
        j_factor:
            A list of factors for Coulomb (J) term
        k_factor:
            A list of factors for Coulomb (K) term
        hermi:
            A list of integer to indicate whether the density matrices are
            symmetric for each set 

    Returns:
        An numpy ndarray of shape (*, Natm, 3)
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    n_dm = len(dm_pairs)
    assert j_factor is None or len(j_factor) == n_dm
    assert k_factor is None or len(k_factor) == n_dm
    if k_factor is None or all(x == 0 for x in k_factor):
        return _j_energies_per_atom(int3c2e_opt, dm_pairs, j_factor, hermi, verbose)

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    if isinstance(hermi, int):
        hermi = [hermi] * n_dm
    elif hermi is None:
        hermi = [0] * n_dm
    else:
        assert len(hermi) == n_dm

    dm1_factor_l = []
    dm1_factor_r = []
    dm2_factor_l = []
    dm2_factor_r = []
    noccs = []
    for dm1_dm2, h in zip(dm_pairs, hermi):
        factor_l, factor_r = _factorize_multiple_dm(mol, dm1_dm2, h)
        noccs.append(factor_l.shape[2])
        dm1_factor_l.append(factor_l[0])
        dm1_factor_r.append(factor_r[0])
        if len(factor_l) == 1: # for two identical dms
            dm2_factor_l.append(factor_l[0])
            dm2_factor_r.append(factor_r[0])
        else:
            dm2_factor_l.append(factor_l[1])
            dm2_factor_r.append(factor_r[1])
    nao = mol.nao
    nocc_max = max(noccs)
    log.debug1('nao=%d noccs=%s', nao, noccs)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    if j_factor is not None:
        dm1 = cp.empty((n_dm, nao, nao))
        dm2 = cp.empty((n_dm, nao, nao))
        for i in range(n_dm):
            dm1_factor_l[i].dot(dm1_factor_r[i].T, out=dm1[i])
            dm2_factor_l[i].dot(dm2_factor_r[i].T, out=dm2[i])
        auxvec1 = cp.empty((n_dm, naux))
        auxvec2 = cp.empty((n_dm, naux))

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_avail = mem_free - 2*n_dm*naux*nocc_max**2*8 - 2*n_dm*nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(n_dm*nao_pair*8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.45/(nao*nao*2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc_max, nao))
    j3c_o2o1 = [cp.empty((naux, nocc, nocc)) for nocc in noccs]
    j3c_o1o2 = [cp.empty((naux, nocc, nocc)) for nocc in noccs]
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            for i, nocc in enumerate(noccs):
                tmp = ndarray((nocc, nao, dk), buffer=buf1)
                contract('pqr,pi->iqr', j3c, dm2_factor_r[i], out=tmp)
                contract('iqr,qj->rij', tmp, dm1_factor_l[i], out=j3c_o2o1[i][aux0:aux1])
                contract('pqr,pi->iqr', j3c, dm1_factor_r[i], out=tmp)
                contract('iqr,qj->rij', tmp, dm2_factor_l[i], out=j3c_o1o2[i][aux0:aux1])
            if j_factor is not None:
                auxvec1[:,aux0:aux1] = cp.einsum('pqr,nqp->nr', j3c, dm1)
                auxvec2[:,aux0:aux1] = cp.einsum('pqr,nqp->nr', j3c, dm2)
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
    for i in range(n_dm):
        j3c_o2o1[i] = contract('uv,vij->uij', metric, j3c_o2o1[i])
        j3c_o1o2[i] = contract('uv,vij->uij', metric, j3c_o1o2[i])

    if j_factor is not None:
        j_factor = cp.asarray(j_factor)
        auxvec1 = cp.einsum('uv,nv->nu', metric, auxvec1)
        auxvec2 = cp.einsum('uv,nv->nu', metric, auxvec2)
        auxvec1_jfac = j_factor[:,None] * auxvec1
        auxvec2_jfac = j_factor[:,None] * auxvec2
    metric = None

    # (d/dX P|Q) contributions
    dm_aux = cp.empty((naux,naux))
    ejk_aux = []
    for i in range(n_dm):
        if j_factor is None:
            beta = 0
        else:
            cp.multiply(auxvec1[i,:,None], auxvec2_jfac[i], out=dm_aux)
            beta = 1
        contract('rij,sji->rs', j3c_o1o2[i], j3c_o2o1[i], -.5*k_factor[i],
                 beta, out=dm_aux)
        # needs to scale by *.5, applied at the end of this function
        dm_aux = transpose_sum(dm_aux, inplace=True)
        dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
        ejk_aux.append(-int2c2e_ip1_per_atom(auxmol, dm_aux))
    ejk_aux = cp.array(ejk_aux)
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    auxvec1 = auxvec2 = dm_aux = None

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

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.ejk_int3c2e_ip1
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf = cp.empty((n_dm*nao_pair*batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((blksize, nao, nao))
    ejk = cp.zeros((n_dm, mol.natm, 3))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((n_dm, nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf2)
            dm_tensor1 = ndarray((nao,nao,dk), buffer=buf1)
            for i in range(n_dm):
                if j_factor is None:
                    dm_tensor[:] = 0
                else:
                    cp.multiply(dm1[i][:,:,None], auxvec2_jfac[i,None,None,aux0:aux1], out=dm_tensor)
                    cp.multiply(dm2[i][:,:,None], auxvec1_jfac[i,None,None,aux0:aux1], out=dm_tensor1)
                    dm_tensor += dm_tensor1
                tmp = ndarray((noccs[i],nao,dk), buffer=buf1)
                contract('rji,qj->iqr', j3c_o1o2[i][aux0:aux1], dm1_factor_l[i], out=tmp)
                contract('iqr,pi->pqr', tmp, dm2_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
                contract('rji,qj->iqr', j3c_o2o1[i][aux0:aux1], dm2_factor_l[i], out=tmp)
                contract('iqr,pi->pqr', tmp, dm1_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[i,:,k0:k1])
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(n_dm),
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
            ctypes.c_int(nao), ctypes.c_int(nao_pair),
            ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
    ejk += ejk_aux
    ejk *= .5
    ejk = ejk.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ejk

def _j_energies_per_atom(int3c2e_opt, dm_pairs, j_factor, hermi=None, verbose=None):
    '''
    Computes first-order derivatives of Coulomb energy for multiple sets of
    density matrix pairs.
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    n_dm = len(dm_pairs)
    assert len(j_factor) == n_dm
    nao = mol.mol.nao
    dms = cp.empty((2, n_dm, nao, nao))
    for i, dm1_dm2 in enumerate(dm_pairs):
        if dm1_dm2.ndim == 2:
            dms[0,i] = dms[1,i] = dm1_dm2
        else:
            dms[:,i] = dm1_dm2
    dms = mol.apply_C_mat_CT(dms.reshape(2*n_dm,nao,nao))
    dms = transpose_sum(dms, inplace=True)
    dms[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dms, hermi=1)
    auxvec = auxmol.apply_CT_dot(auxvec, axis=1)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = cp.linalg.solve(j2c, auxvec.T).T
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxvec.T).T
    auxvec = cp.asarray(auxmol.apply_C_dot(auxvec, axis=1), order='C')
    naux = auxvec.shape[1]
    # Swap the output of dm1 and dm2 in auxvec. They are cross-contracted in
    # ejk_int3c2e_ip1, i.e. dm1*auxvec2 + dm2*auxvec1
    auxvec = auxvec.reshape(2, n_dm, naux)
    auxvec21 = auxvec[[1, 0]].reshape(2*n_dm, naux)
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
    ej = cp.zeros((2, n_dm, mol.natm, 3))
    ej_aux = cp.zeros_like(ej)

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec21.data.ptr, ctypes.c_void_p),
        ctypes.c_int(2*n_dm),
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
    ej = ej[0] + ej[1]
    ej_aux = ej_aux[0] + ej_aux[1]
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
    for i in range(n_dm):
        dm_aux  = auxvec[0,i,:,None] * auxvec[1,i]
        dm_aux += auxvec[1,i,:,None] * auxvec[0,i]
        ej_aux[i] -= .5 * cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    ej += ej_aux.get()
    ej *= np.array(j_factor)[:,None,None]
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _factorize_multiple_dm(mol, dm1, hermi):
    if not isinstance(dm1, cp.ndarray):
        dm1 = cp.asarray(dm1)
    if dm1.ndim == 2:
        dm1 = dm1[None]
    dm1_factor_l, dm1_factor_r = factorize_dm(dm1, hermi)
    # transform to the AO order in sorted_cell
    dm1_factor_l = mol.apply_C_dot(dm1_factor_l, axis=1)
    if dm1_factor_r is None:
        dm1_factor_r = dm1_factor_l
    else:
        dm1_factor_r = mol.apply_C_dot(dm1_factor_r, axis=1)
    return dm1_factor_l, dm1_factor_r

class Gradients(tdrhf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)

    def get_veff(self, mol, dm, j_factor=1, k_factor=1, omega=0,
                 hermi=0, verbose=None):
        ejk = self.jk_energy_per_atom(
            dm, j_factor, k_factor, omega, hermi, verbose)
        return ejk * .5

    def jk_energy_per_atom(self, dms, j_factor=None, k_factor=None, omega=0,
                           hermi=0, verbose=None):
        '''
        Computes the sum of first-order derivatives of J/K contributions for
        multiple density matrices.

        Args:
            dms:
                A list of density-matrices
            j_factor :
                A list of factors for Coulomb (J) term
            k_factor :
                A list of factors for Coulomb (K) term
            hermi :
                An overall symmetry code for all density matrices

	Returns:
            An array of shape (Natm, 3).
        '''
        return self.jk_energies_per_atom(dms, j_factor, k_factor, omega,
                                         sum_results=True, verbose=verbose)

    def jk_energies_per_atom(self, dm_list, j_factor=None, k_factor=None, omega=0,
                             hermi=0, sum_results=False, verbose=None):
        '''
        Computes a set of first-order derivatives of J/K contributions for each
        element (density matrix or a pair of density matrices) in dm_pairs.

        This function supports evaluating multiple sets of energy derivatives in a
        single call. Additionally, for each set, the two density matrices for the
        four-index Coulomb integrals can be different.

        Args:
            dm_list :
                A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
                Each element corresponds to one set of energy derivative.
            j_factor :
                A list of factors for Coulomb (J) term
            k_factor :
                A list of factors for Coulomb (K) term
            hermi :
                An integer or a list of integer to indicate whether the density
                matrices are symmetric for each set . If an integer is specified,
                the same symmetry code is applied to all density matrices.
	    sum_results : bool
		If True, aggregate all sets of derivatives into a single result.

	Returns:
            An array of shape (*, Natm, 3) if sum_results is False; otherwise,
            an array of shape (Natm, 3).
        '''
        assert self.auxbasis_response
        mf = self.base._scf
        mol = mf.with_df.mol
        auxmol = mf.with_df.auxmol
        mf.with_df.reset() # Release GPU memory
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol).build()

        if (sum_results and
            # When the input is a list, each density matrix is applied twice in
            # a symmetric manner for computing the J and K contributions.
            isinstance(dm_list, cp.ndarray) and dm_list.ndim < 4):
            if not isinstance(hermi, int):
                hermi = all(x == 1 for x in hermi)
            return _jk_energy_per_atom(
                int3c2e_opt, dm_list, j_factor, k_factor, hermi, verbose=verbose)

        ejk = _jk_energies_per_atom(
            int3c2e_opt, dm_list, j_factor, k_factor, hermi, verbose=verbose)
        if sum_results:
            ejk = ejk.sum(axis=0)
        return ejk

Grad = Gradients
