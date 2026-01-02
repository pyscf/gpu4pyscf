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
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, cholesky, eigh
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad.rhf import contract_h1e_dm
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, _int3c2e_scheme,
    _nearest_power2, SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys,
    Int3c2eOpt_v2, int2c2e)
from gpu4pyscf.df import df

__all__ = ['Gradients']

def _gen_metric_solver(int2c, decompose_j2c='CD', lindep=df.LINEAR_DEP_THR):
    ''' generate a solver to solve Ax = b, RHS must be in (n,....) '''
    if decompose_j2c.upper() == 'CD':
        try:
            j2c = cholesky(int2c)
            def j2c_solver(b):
                out = solve_triangular(j2c, b.reshape(j2c.shape[0],-1), lower=True,
                                        overwrite_b=False).reshape(b.shape)
                return cp.asarray(out, order='A')
            return j2c_solver
        except RuntimeError:
            pass

    w, v = eigh(int2c)
    mask = w > lindep
    v1 = v[:,mask]
    j2c = (v1/w[mask]).dot(v1.conj().T)
    def j2c_solver(b): # noqa: F811
        return j2c.dot(b.reshape(j2c.shape[0],-1)).reshape(b.shape)
    return j2c_solver

def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=0,
                        auxbasis_response=True, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm, hermi, auxbasis_response,
                                  verbose) * j_factor

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    mo_coeff = None
    if hasattr(dm, 'mo_coeff'):
        mo_coeff = asarray(dm.mo_coeff)
        assert mo_coeff.dtype == np.float64
        mo_occ = asarray(dm.mo_occ)
        # transform the mo_coeff to the AO order in sorted_cell
        mo_coeff = mol.apply_C_dot(mo_coeff)
        mask = mo_occ > 0
        dm_factor = mo_coeff[:,mask]
        dm_factor *= cp.sqrt(mo_occ[mask])
        dm_factor_l = dm_factor_r = dm_factor
    else:
        dm_factor_l, dm_factor_r = _decompose_rdm1_svd(dm, hermi)
        dm_factor_l = mol.apply_C_dot(dm_factor_l, axis=0)
        dm_factor_r = mol.apply_C_dot(dm_factor_r, axis=0)
    nao, nocc = dm_factor_l.shape

    nsp_per_block, gout_stride, shm_size = _int3c2e_scheme(mol.omega, gout_width=54)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)
    ao_pair_loc = get_ao_pair_loc(mol, int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    buffer_size = 4e9
    batch_size = max(1, int(buffer_size / (nao_pair*8)))
    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    uniq_l_ctr_aux = auxmol.uniq_l_ctr
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, uniq_l_ctr_aux, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    l_ctr_aux_counts = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]
    aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
    naux = len(aux_sorting)
    reorder_aux = 1
    to_sph = 0

    shl_pair_blocks = len(shl_pair_offsets) - 1
    ksh_blocks = len(ksh_offsets_cpu) - 1
    log.debug1('sp_blocks = %d, ksh_blocks = %d', shl_pair_blocks, ksh_blocks)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.fill_int3c2e
    buffer_size = 2e9
    blksize = max(1, min(int(buffer_size / (nao**2*8)), naux))
    l = np.arange(max(lmax, laux)+1)
    nf = (l + 1) * (l + 2) // 2

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = cp.empty((nao_pair, naux_in_batch))
        err = kern(
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(0), ctypes.c_int(aux_ao_offset),
            ctypes.c_int(naux_in_batch),
            ctypes.c_int(reorder_aux), ctypes.c_int(to_sph))

        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:k1-k0]
            j3c[j_addr,i_addr] = compressed[:,k0:k1]
            j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = contract('pqr,pi->riq', j3c, dm_factor_r, out=buf1[:dk])
            contract('riq,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
    j3c_full = buf1 = None
    j3c_oo = j3c_oo[aux_sorting]
    t0 = log.timer_debug1('contract dm', *t0)

    j2c = int2c2e(auxmol.mol)
    aux_coeff = cp.asarray(auxmol.ctr_coeff)
    if mol.omega <= 0 and not auxmol.mol.cart:
        metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    else:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    # (d/dX P|Q) contributions
    if auxbasis_response:
        if j_factor == 0:
            dm_aux = None
        else:
            dm_aux = auxvec[:,None] * auxvec
        if mo_coeff is not None:
            dm_aux = contract('rij,sij->rs', dm_oo, dm_oo,
                              alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
        else:
            dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                              alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
        #ejk_aux = .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
        ejk_aux = cp.asarray(_int2c2e_ip1_per_atom(auxmol, dm_aux))
        ejk_aux *= -.5
        t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
        ejk_aux_ptr = ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p)
    else:
        ejk_aux_ptr = lib.c_null_ptr()

    # Reorder the auxiliary index for better memory access efficiency
    j3c_oo[aux_sorting] = dm_oo
    dm_oo = j3c_oo
    j2c = dm_aux = j3c_oo = metric = None

    if j_factor != 0:
        auxvec[aux_sorting] = auxvec
        dm = dm_factor_l.dot(dm_factor_r.T)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip1(mol.omega)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    kern = libvhf_rys.ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))
    ejk = cp.zeros((mol.natm, 3))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
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
            contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
            compressed[:,k0:k1] = dm_tensor.reshape(-1,dk)[pair_addresses]
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p), ejk_aux_ptr,
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
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
            ctypes.c_int(naux_in_batch))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
    buf = buf1 = None
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    if auxbasis_response:
        ejk += ejk_aux
    return ejk.get()

def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, auxbasis_response=True, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm = mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, hermi)
    t0 = log.timer_debug1('contract dm', *t0)

    j2c = int2c2e(auxmol.mol)
    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = cp.linalg.solve(j2c, auxmol.CT_dot_mat(auxvec))
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxmol.CT_dot_mat(auxvec))
    auxvec = auxmol.C_dot_mat(auxvec)
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip1(mol.omega)
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
    if auxbasis_response:
        ej_aux = cp.zeros_like(ej)
        ej_aux_ptr = ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p)
    else:
        ej_aux_ptr = lib.c_null_ptr()

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p), ej_aux_ptr,
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(), lib.c_null_ptr(),
        ctypes.c_int(0))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    ej = ej.get()

    # (d/dX P|Q) contributions
    if auxbasis_response:
        #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
        dm_aux = auxvec[:,None] * auxvec
        ej_aux -= .5 * cp.asarray(_int2c2e_ip1_per_atom(auxmol, dm_aux))
        ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _int2c2e_ip1_per_atom(mol, dm):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.gto.mole import PBCIntEnvVars, _scale_sp_ctr_coeff
    from gpu4pyscf.pbc.df.ft_ao import libpbc
    libpbc.e_int2c2e_ip1.restype = ctypes.c_int
    is_sorted_mol = isinstance(mol, SortedMole)
    if is_sorted_mol:
        dm = cp.asarray(dm)
    else:
        mol = SortedMole.from_mol(mol)
        dm = mol.apply_C_mat_CT(dm)
    lmax = mol.uniq_l_ctr[:,0].max()
    assert lmax <= L_AUX_MAX

    li = np.arange(L_AUX_MAX+1)[:,None]
    lj = np.arange(L_AUX_MAX+1)
    order = li + lj + 1
    nroots = order//2 + 1
    if mol.omega < 0:
        nroots *= 2 # for short-range
    g_size = (li+2)*(lj+2)
    unit = g_size*3 + nroots*2 + 4
    nsp_max = _nearest_power2(SHM_SIZE // (unit*8))
    nsp_per_block = np.where(nsp_max < THREADS, nsp_max, THREADS)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    shm_size_max = shm_size[:lmax+1,:lmax+1].max()

    ao_loc = mol.ao_loc
    Ls = cp.zeros((1, 3))
    _env = _scale_sp_ctr_coeff(mol)
    rys_envs = PBCIntEnvVars.new(
        mol.natm, mol.nbas, 1, 1, mol._atm, mol._bas, _env, ao_loc, Ls)
    nbas = mol.nbas
    mask = cp.ones((nbas, nbas), dtype=bool)
    bas_ij_cache = mol.generate_shl_pairs(mask=mask, hermi=1)
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        bas_ij_cache, nsp_per_block)

    nbatches_shl_pair = len(shl_pair_offsets) - 1
    out = cp.zeros((mol.natm, 3))
    err = libpbc.e_int2c2e_ip1(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('fill_int2c2e_ip1 failed')
    return out

def _decompose_rdm1_svd(dm, hermi=0):
    '''Decompose density matrix as U.Vh using SVD

    Args:
        dm : ndarray or sequence of ndarrays of shape (*,nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    if hermi == 1:
        s, u = cp.linalg.eigh(cp.asarray(dm))
        idx = abs(s) > 1e-8
        if dm.ndim == 2:
            c = u[:,idx]
            return c, contract('i,pi->pi', s[idx], c)
        else:
            assert dm.shape[0] == 2
            idx = idx[0] | idx[1]
            c = u[:,:,idx]
            return c, contract('si,spi->spi', s[:,idx], c)

    u, s, vh = cp.linalg.svd(cp.asarray(dm))
    idx = s > 1e-8
    if dm.ndim == 2:
        return u[:,idx], contract('i,ip->pi', s[idx], vh[idx])
    else:
        assert dm.shape[0] == 2
        idx = idx[0] | idx[1]
        return u[:,:,idx], contract('si,sip->spi', s[:,idx], vh[:,idx])

def int3c2e_scheme_ip1(omega=0, gout_width=None, shm_size=SHM_SIZE):
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

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df.df_jk._DFHF)

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
        int3c2e_opt = Int3c2eOpt_v2(mol, mf.with_df.auxmol).build()
        return _jk_energy_per_atom(
            int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=1,
            auxbasis_response=self.auxbasis_response, verbose=verbose) * .5

Grad = Gradients
