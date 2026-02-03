# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#  Modified by:   Xiaojie Wu <wxj6000@gmail.com>

'''
Non-relativistic RHF analytical Hessian with density-fitting approximation
Ref:
[1] Efficient implementation of the analytic second derivatives of
    Hartree-Fock and hybrid DFT energies: a detailed analysis of different
    approximations.  Dmytro Bykov, Taras Petrenko, Robert Izsak, Simone
    Kossmann, Ute Becker, Edward Valeev, Frank Neese. Mol. Phys. 113, 1961 (2015)
'''

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ATOM_OF
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, condense, transpose_sum)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, _nearest_power2,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys, Int3c2eOpt, int2c2e,
    int2c2e_ip1)
from gpu4pyscf.df import df
from gpu4pyscf.df.df_jk import factorize_dm
from gpu4pyscf.df.grad.rhf import _gen_metric_solver
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import rhf as rhf_hess

def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm, verbose) * j_factor

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l = dm_factor_r = _factorize_dm(mol, dm)
    nao, nocc = dm_factor_l.shape

    natm = mol.natm
    naux = auxmol.nao
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    buffer_size = mem_free // 4
    batch_size = max(1, min(naux, buffer_size // (nao_pair*8)))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, buffer_size // (nao**2*8)))
    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    original_auxmol = auxmol.mol
    j2c = int2c2e(original_auxmol)
    w, j2c_factor = cp.linalg.eigh(j2c)
    j2c_factor *= w**-.5
    j2c_factor = auxmol.apply_C_dot(j2c_factor)
    if mol.omega > 0 or original_auxmol.cart:
        j2c_factor = j2c_factor[:,w>df.LINEAR_DEP_THR]
    j2c = w = None
    j2c_factor, tmp = cp.empty_like(j2c_factor), j2c_factor
    j2c_factor[aux_sorting] = tmp
    tmp = None

    metric = j2c_factor.dot(j2c_factor.T)
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    j3c_oo = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    # (00|0)(2|0)(0|00)
    if j_factor == 0:
        dm_aux = None
    else:
        dm_aux = auxvec[:,None] * auxvec
    dm_aux = contract('rij,sij->rs', dm_oo, dm_oo,
                      alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
    ejk_aux = cp.asarray(_int2c2e_ip2_per_atom(
        auxmol, dm_aux[aux_sorting[:,None], aux_sorting]))
    t0 = log.timer_debug1('contract int2c2e_ip2', *t0)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2(mol.omega)
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
    # assert cp.array_equal(aux_sorting, argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux))
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    l_ctr_aux_counts = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]

    if j_factor != 0:
        dm = dm_factor_l.dot(dm_factor_r.T)

    # (20|0)(0|0)(0|00) + (10|1)(0|0)(0|00)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern_ip2 = libvhf_rys.ejk_int3c2e_ip2
    ejk = cp.zeros((natm, natm, 3, 3))
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf2 = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
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
            contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
            cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0, out=compressed[:,k0:k1])
        err = kern_ip2(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
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
            raise RuntimeError('ejk_int3c2e_ip2 failed')
    ejk = ejk + ejk.transpose(1,0,3,2)
    # *2 for i>=j, *2 for ij <-> kl, *.5 from Coulomb operator
    ejk *= 2 * 2 * .5
    ejk -= ejk_aux
    ejk = ejk.get()
    buf = buf1 = buf2 = ejk_aux = None
    t0 = log.timer_debug1('contract ejk_int3c2e_ip2', *t0)

    # (00|1)(0|1)(0|00)
    # (00|1)(1|0)(0|00)
    # (00|1)(0|0)(1|00)
    # ...
    eval_ipaux = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ipaux')[0]
    eval_ip1 = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ip1')[0]

    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((3, nao_pair, batch_size))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo1 = cp.empty((3, naux, nocc, nocc)) # = (1|00)
    aux0 = aux1 = 0
    for kbatch in range(aux_batches):
        compressed = eval_ipaux(kbatch, out=buf)
        naux_in_batch = compressed.shape[-1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            for i in range(3):
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[i,:,k0:k1]
                contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
                # Note d/dX = -d/dr, apply alpha=-1
                contract('iqr,qj->rij', tmp, dm_factor_l, alpha=-1,
                         out=j3c_oo1[i,aux0:aux1])
    j3c_full = buf = buf1 = tmp = compressed = None
    t0 = log.timer_debug1('fill_int3c2e_ipaux', *t0)

    j2c_inv = metric
    # note int2c2e_ip1 computs d/dr and d/dX = -d/dr
    j2c_10 = int2c2e_ip1(auxmol, sort_output=False)
    j2c_10 *= -1
    j2c_10, tmp = cp.empty_like(j2c_10), j2c_10
    j2c_10[:,aux_sorting[:,None], aux_sorting] = tmp
    j2c_10v = contract('xrs,st->xrt', j2c_10, j2c_inv)
    # (00|0)(1|0)(0|1)(0|00)
    j2c_ip2 = contract('xrs,yts->rtxy', j2c_10v, j2c_10)
    j2c_ip2 *= dm_aux[:,:,None,None]
    h_aux = j2c_ip2
    dm_aux = None

    # j3c_oo1p = (1|0)(0|00) + (1|00)
    j3c_oo1p = contract('xuv,vij->xuij', j2c_10, dm_oo, alpha=-1, beta=1, out=j3c_oo1)
    # (00|0)(1|0)(1|00) + (00|0)(1|0)(1|0)(0|00)
    if j_factor == 0:
        dm_aux1 = None
    else:
        auxvec_ipauxp = cp.einsum('xrii->xr', j3c_oo1p)
        dm_aux1 = cp.einsum('xr,t->xrt', auxvec_ipauxp, auxvec)
    dm_aux1 = contract('xrij,sij->xrs', j3c_oo1p, dm_oo, -.5*k_factor,
                       beta=j_factor, out=dm_aux1)
    w10_100 = cp.einsum('xrt,ytr->rtxy', j2c_10v, dm_aux1)
    h_aux -= w10_100
    h_aux -= w10_100.transpose(1,0,3,2) # swap the asymetric di,dj indices
    dm_aux1 = j2c_10v = w10_100 = None

    # (00|1)(1|0)(0|00) + (00|1)(0|0)(1|00) +
    # (00|0)(0|1)(1|00) + (00|0)(0|1)(1|0)(0|00)
    # = 001p * 001p
    if j_factor == 0:
        dm_aux11 = None
    else:
        dm_aux11 = cp.einsum('xr,ys->rsxy', auxvec_ipauxp, auxvec_ipauxp)
    dm_aux11 = contract('xrij,ysij->rsxy', j3c_oo1p, j3c_oo1p, -.5*k_factor,
                        beta=j_factor, out=dm_aux11)
    dm_aux11 *= j2c_inv[:,:,None,None]
    h_aux += dm_aux11
    # swap the differentiation order
    # (00|0)(1|0)(1|00) + (00|0)(1|0)(1|0)(0|00)
    h_aux = h_aux + h_aux.transpose(1,0,3,2)
    j2c_inv = metric = None

    aux_idx, aux_slices = _argsort_aux_by_atom(auxmol, aux_sorting)
    pqxy = h_aux[aux_idx[:,None], aux_idx].get()
    ejk_aux = np.zeros_like(ejk)
    for i, (p0, p1) in enumerate(aux_slices):
        for j, (q0, q1) in enumerate(aux_slices):
            ejk_aux[i,j] = pqxy[p0:p1,q0:q1].sum(axis=(0,1))
    ejk += ejk_aux * .5
    pqxy = None
    dm_aux11 = j3c_oo1 = h_aux = None
    t1 = t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    j2c_10_fac = contract('yrs,st->ytr', j2c_10, j2c_factor)
    j2c_10 = None

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_avail = int(mem_free * .6)
    aux_batch_size = max(1, min(naux, mem_avail // (nao*nocc*8)))
    mem_avail = int(mem_free * .2)
    aux_blksize = mem_avail // (3*natm*nocc*nocc)

    # 3c integrals are computed in Cartesian bases, and sorted in the original
    # AO order.
    original_mol = mol.mol
    ao_loc = original_mol.ao_loc_nr(cart=True)
    aoslices = original_mol.aoslice_by_atom(ao_loc=ao_loc)

    h_ao_aux = cp.zeros((natm,naux,3,3))
    ejk_ao = cp.zeros((natm,natm,3,3))

    j3c_buf = cp.empty((3,aux_batch_size,nao,nocc))
    j3c_full = cp.zeros((nao, nao, blksize))
    metric_size = j2c_factor.shape[1]
    # TODO: for small molecules, merge the kern_ipaux to the previouse case.
    for v0, v1 in lib.prange(0, metric_size, aux_batch_size):
        dv = v1 - v0
        j3c_100 = ndarray((3, dv, nao, nocc), buffer=j3c_buf)
        j3c_100[:] = 0.
        buf0 = cp.empty((3, nao_pair, batch_size))
        buf1 = cp.empty((3, nao_pair, batch_size))
        buf2 = cp.empty((blksize, nocc, nao))
        aux0 = aux1 = 0
        for kbatch in range(aux_batches):
            compressed_di = eval_ip1(kbatch, out=buf0)
            compressed_dk = eval_ipaux(kbatch, out=buf1)
            naux_in_batch = compressed_di.shape[-1]
            # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
            compressed_dk += compressed_di
            compressed_dj = compressed_dk # ~ d/dX on j
            compressed_di *= -1           # ~ d/dX on i
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:,:,:dk]
                tmp = ndarray((dk, nao, nocc), buffer=buf2)
                for i in range(3):
                    j3c[j_addr,i_addr] = compressed_dj[i,:,k0:k1]
                    j3c[i_addr,j_addr] = compressed_di[i,:,k0:k1]
                    tmp = contract('pqr,qi->rpi', j3c, dm_factor_r, out=tmp)
                    contract('rs,rpi->spi', j2c_factor[aux0:aux1,v0:v1], tmp,
                             beta=1, out=j3c_100[i])
        buf0 = buf1 = buf2 = compressed_di = compressed_dj = compressed_dk = None
        t0 = log.timer_debug1(f'fill_int3c2e_ip1 {v0}:{v1}', *t0)

        # (10|0)(0|0)(0|01) + (10|0)(0|0)(0|10)
        # (01|0)(0|0)(0|01) + (01|0)(0|0)(0|10)
        for k0, k1 in lib.prange(0, dv, aux_blksize):
            j3c_oo_atm = cp.empty((3,natm,k1-k0,nocc,nocc))
            for i, (p0, p1) in enumerate(aoslices[:,2:]):
                contract('xrpj,pi->xrij', j3c_100[:,k0:k1,p0:p1],
                         dm_factor_l[p0:p1], out=j3c_oo_atm[:,i])
            # di/dX + dj/dX
            transpose_sum(j3c_oo_atm.reshape(-1,nocc,nocc), inplace=True)
            contract('xprij,yqrij->pqxy', j3c_oo_atm, j3c_oo_atm, -.5*k_factor,
                     beta=1, out=ejk_ao)
            if j_factor != 0:
                auxvec_100_atm = cp.einsum('xprii->xpr', j3c_oo_atm)
                contract('xpr,yqr->pqxy', auxvec_100_atm, auxvec_100_atm,
                         j_factor, beta=1, out=ejk_ao)

            j2c_factor_part = j2c_factor[:,v0+k0:v0+k1]
            j2c_10_part = j2c_10_fac[:,v0+k0:v0+k1,:]
            if j_factor != 0:
                # (10|0)(1|00) + (10|0)(1|0)(0|00)
                tmp = contract('yr,rt->ytr', auxvec_ipauxp, j2c_factor_part)
                # (10|0)(0|1)(0|00)
                tmp -= j2c_10_part * auxvec
                contract('xpt,ytr->prxy', auxvec_100_atm, tmp, j_factor, 1, h_ao_aux)
                tmp = None

            # (10|0)(1|0)(0|00) + (10|0)(0|0)(1|00)
            #:h_ao_aux += einsum('xspj,rs,pi,yrji->prxy',
            #:                   j3c_100, j2c_factor, dm_factor_l, j3c_oo1p)
            # TODO: to reduce memory footprint, maybe loop over p
            # TODO: pre-allocated buffer
            tmp = contract('xprij,ysij->prsxy', j3c_oo_atm, j3c_oo1p)
            contract('prsxy,sr->psxy', tmp, j2c_factor_part, -.5*k_factor, 1, h_ao_aux)
            tmp = None

            # (10|0)(0|1)(0|00)
            #:h_ao_aux -= einsum('xtpj,yrs,st,pi,rji->prxy',
            #:                   j3c_100, j2c_10, j2c_factor, dm_factor_l, dm_oo)
            tmp = contract('xprij,sij->xprs', j3c_oo_atm, dm_oo)
            contract('xprs,yrs->psxy', tmp, j2c_10_part, .5*k_factor, 1, h_ao_aux)
            tmp = None
        t1 = log.timer_debug1(f'contract int3c2e_ip1 {v0}:{v1}', *t1)
    t0 = log.timer_debug1('int3c2e_ipaux and int2c2e_ip1 cross term', *t0)

    h_ao_aux = h_ao_aux[:,aux_idx].get()
    ejk_ao_aux = np.zeros_like(ejk)
    for i, (p0, p1) in enumerate(aux_slices):
        ejk_ao_aux[:,i] = h_ao_aux[:,p0:p1].sum(axis=1)
    ejk += ejk_ao_aux
    ejk += ejk_ao_aux.transpose(1,0,3,2)

    # scale ejk_ao: *2 for swaping (di/dX j|dk/dY l) -> (di/dY j|dk/dX l)
    # *.5 from Coulomb operator
    ejk += ejk_ao.get()
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l = dm_factor_r = _factorize_dm(mol, dm)
    nao, nocc = dm_factor_l.shape

    dm = mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, hermi=1)
    naux = len(auxvec)
    t0 = log.timer_debug1('contract dm', *t0)

    original_auxmol = auxmol.mol
    j2c = int2c2e(original_auxmol)
    w, j2c_factor = cp.linalg.eigh(j2c)
    j2c_factor *= w**-.5
    j2c_factor = auxmol.apply_C_dot(j2c_factor)
    if mol.omega > 0 or original_auxmol.cart:
        j2c_factor = j2c_factor[:,w>df.LINEAR_DEP_THR]
    j2c = w = None

    metric = j2c_factor.dot(j2c_factor.T)
    auxvec = metric.dot(auxvec)

    # (00|0)(2|0)(0|00)
    dm_aux = auxvec[:,None] * auxvec
    ej_aux = cp.asarray(_int2c2e_ip2_per_atom(auxmol, dm_aux))
    t0 = log.timer_debug1('contract int2c2e_ip2', *t0)

    # (20|0)(0|0)(0|00) + (10|1)(0|0)(0|00)
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2(mol.omega)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern_ip2 = libvhf_rys.ejk_int3c2e_ip2
    ej = cp.zeros_like(ej_aux)
    err = kern_ip2(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
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
        ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(0),
        ctypes.c_int(naux))
    if err != 0:
        raise RuntimeError('ejk_int3c2e_ip2 failed')
    ej = ej + ej.transpose(1,0,3,2)
    # *2 for i>=j, *2 for ij <-> kl, *.5 from Coulomb operator
    ej *= 2 * 2 * .5
    ej -= ej_aux
    ej = ej.get()
    ej_aux = None
    t0 = log.timer_debug1('contract ejk_int3c2e_ip2', *t0)

    natm = mol.natm
    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    # dm must be reconstructed from the dm_factor, because the pair_addresses
    # are sorted in an order corresponding to the Cartesian GTOs in the original
    # basis order, while the AO indices in mol.apply_C_mat_CT(dm) are grouped
    # and reordered based on angular momentum.
    dm = dm_factor_l.dot(dm_factor_r.T)
    dm_compressed = dm[i_addr,j_addr]
    dm_compressed[diag_idx] *= .5
    dm_compressed *= 2

    # (00|1)(0|1)(0|00)
    # (00|1)(1|0)(0|00)
    # (00|1)(0|0)(1|00)
    # ...
    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_avail = mem_free // 4
    batch_size = max(1, min(naux, mem_avail // (nao_pair*8)))
    blksize = max(1, min(naux, mem_avail // (nao**2*8)))

    eval_ipaux, aux_sorting, aux_offsets = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ipaux')
    eval_ip1 = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ip1')[0]
    aux_batches = len(aux_offsets) - 1

    # 3c integrals are computed in Cartesian bases, and sorted in the original
    # AO order.
    original_mol = mol.mol
    ao_loc = original_mol.ao_loc_nr(cart=True)
    aoslices = original_mol.aoslice_by_atom(ao_loc=ao_loc)

    auxvec_ipaux = cp.empty((3, naux))
    auxvec_100_atm = cp.empty((3, natm, naux))
    j3c_full = cp.zeros((3, nao, nao, blksize))
    buf0 = cp.empty((3, nao_pair, batch_size))
    buf1 = cp.empty((3, nao_pair, batch_size))
    buf2 = cp.empty((3, batch_size, nao, nocc))
    aux0 = aux1 = 0
    p0 = p1 = 0
    for kbatch in range(aux_batches):
        compressed_di = eval_ip1(kbatch, out=buf0)
        compressed_dk = eval_ipaux(kbatch, out=buf1)
        _aux0, _aux1 = aux1, aux1 + compressed_dk.shape[-1]
        auxvec_ipaux[:,_aux0:_aux1] = contract('xpr,p->xr', compressed_dk, dm_compressed)

        # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
        compressed_dk += compressed_di
        compressed_dj = compressed_dk # ~ d/dX on j
        compressed_di *= -1           # ~ d/dX on i
        naux_in_batch = compressed_di.shape[-1]
        j3c_100 = ndarray((3, naux_in_batch, nao, nocc), buffer=buf2)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:,:dk]
            tmp = ndarray((dk, nao, nocc), buffer=buf2)
            j3c[:,j_addr,i_addr] = compressed_dj[:,:,k0:k1]
            j3c[:,i_addr,j_addr] = compressed_di[:,:,k0:k1]
            for i, (p0, p1) in enumerate(aoslices[:,2:]):
                contract('xpqr,pq->xr', j3c[:,p0:p1], dm[p0:p1],
                         out=auxvec_100_atm[:,i,_aux0:_aux1])
    auxvec_ipaux = auxvec_ipaux[:,aux_sorting]
    auxvec_100_atm = auxvec_100_atm[:,:,aux_sorting]
    auxvec_100_atm *= 2 # di/dX + dj/dX
    buf0 = buf1 = buf2 = compressed_di = compressed_dj = compressed_dk = None
    t0 = log.timer_debug1('fill_int3c2e_ip1 and fill_int3c2e_ipaux', *t0)

    j2c_inv, metric = metric, None
    # note int2c2e_ip1 computs d/dr and d/dX = -d/dr
    j2c_10 = int2c2e_ip1(auxmol, sort_output=False)
    j2c_10 *= -1
    j2c_10v = contract('xrs,st->xrt', j2c_10, j2c_inv)
    # (00|0)(1|0)(0|1)(0|00)
    j2c_ip2 = contract('xrs,yts->rtxy', j2c_10v, j2c_10)
    j2c_ip2 *= dm_aux[:,:,None,None]
    h_aux = j2c_ip2
    tmp = dm_aux = None

    # d/dX = -d/dr
    auxvec_ipaux *= -1
    # (1|0)(0|00) + (1|00)
    auxvec_ipauxp = contract('xuv,v->xu', j2c_10, auxvec, alpha=-1, beta=1,
                             out=auxvec_ipaux)
    # (00|0)(1|0)(1|00) + (00|0)(1|0)(1|0)(0|00)
    dm_aux1 = cp.einsum('xr,t->xrt', auxvec_ipauxp, auxvec)
    w10_100 = cp.einsum('xrt,ytr->rtxy', j2c_10v, dm_aux1)
    h_aux -= w10_100
    h_aux -= w10_100.transpose(1,0,3,2) # swap the asymetric di,dj indices
    dm_aux1 = j2c_10v = w10_100 = None

    # (00|1)(1|0)(0|00) + (00|1)(0|0)(1|00) +
    # (00|0)(0|1)(1|00) + (00|0)(0|1)(1|0)(0|00)
    # = 001p * 001p
    dm_aux11 = cp.einsum('xr,ys->rsxy', auxvec_ipauxp, auxvec_ipauxp)
    dm_aux11 *= j2c_inv[:,:,None,None]
    h_aux += dm_aux11
    # swap the differentiation order
    # (00|0)(1|0)(1|00) + (00|0)(1|0)(1|0)(0|00)
    h_aux = h_aux + h_aux.transpose(1,0,3,2)

    aux_idx, aux_slices = _argsort_aux_by_atom(auxmol, aux_sorting)
    pqxy = h_aux[aux_idx[:,None], aux_idx].get()
    ej_aux = np.zeros_like(ej)
    for i, (p0, p1) in enumerate(aux_slices):
        for j, (q0, q1) in enumerate(aux_slices):
            ej_aux[i,j] = pqxy[p0:p1,q0:q1].sum(axis=(0,1))
    ej += ej_aux * .5
    dm_aux11 = h_aux = None
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    # (10|0)(0|0)(0|01) + (10|0)(0|0)(0|10)
    # (01|0)(0|0)(0|01) + (01|0)(0|0)(0|10)
    auxvec_100v_atm = contract('xpr,rs->xps', auxvec_100_atm, j2c_inv)
    ej_ao = contract('xpr,yqr->pqxy', auxvec_100v_atm, auxvec_100_atm)
    # scale ej_ao: *2 for swaping (di/dX j|dk/dY l) -> (di/dY j|dk/dX l)
    # *.5 from Coulomb operator
    ej += ej_ao.get()

    if 0:
        auxvec_100_atm = contract('xpr,rs->xps', auxvec_100_atm, j2c_factor)
        # (10|0)(1|00) + (10|0)(1|0)(0|00)
        tmp = contract('yr,rt->ytr', auxvec_ipauxp, j2c_factor)
        # (10|0)(0|1)(0|00)
        j2c_10_fac = contract('yrs,st->ytr', j2c_10, j2c_factor)
        tmp -= j2c_10_fac * auxvec
        h_ao_aux = contract('xpt,ytr->prxy', auxvec_100_atm, tmp)
        tmp = None
        t0 = log.timer_debug1('int3c2e_ipaux and int2c2e_ip1 cross term', *t0)

    # (10|0)(1|00) + (10|0)(1|0)(0|00)
    h_ao_aux = contract('xpr,yr->prxy', auxvec_100v_atm, auxvec_ipauxp)
    j2c_10 *= auxvec[:,None] # Overwrite j2c_10 ~ (0|1)(0|00)
    # (10|0)(0|1)(0|00)
    contract('xpt,yrt->prxy', auxvec_100v_atm, j2c_10, alpha=-1, beta=1, out=h_ao_aux)
    t0 = log.timer_debug1('int3c2e_ipaux and int2c2e_ip1 cross term', *t0)

    h_ao_aux = h_ao_aux[:,aux_idx].get()
    ej_ao_aux = np.zeros_like(ej)
    for i, (p0, p1) in enumerate(aux_slices):
        ej_ao_aux[:,i] = h_ao_aux[:,p0:p1].sum(axis=1)
    ej += ej_ao_aux
    ej += ej_ao_aux.transpose(1,0,3,2)
    return ej

def _int3c2e_ip1_evaluator(int3c2e_opt, scheme, batch_size,
                           kern='fill_int3c2e_ip1'):
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    nsp_per_block, gout_stride, shm_size = scheme
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    nao_pair = int(ao_pair_loc[-1].get())

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    uniq_l_ctr_aux = auxmol.uniq_l_ctr
    aux_loc = auxmol.ao_loc
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, uniq_l_ctr_aux, batch_size)
    aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
    # assert cp.array_equal(aux_sorting, argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux))

    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    aux_splits = range(len(ksh_offsets_cpu))
    aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
    kern = getattr(libvhf_rys, kern)
    int3c2e_envs = int3c2e_opt.int3c2e_envs

    def evaluate_j3c(batch_id, out=None):
        aux_split0 = aux_splits[batch_id]
        aux_split1 = aux_splits[batch_id+1]
        ksh0 = ksh_offsets_cpu[aux_split0]
        ksh1 = ksh_offsets_cpu[aux_split1]
        aux_ao_offset = aux_loc[ksh0]
        naux = aux_loc[ksh1] - aux_ao_offset
        out = ndarray((3, nao_pair, naux), buffer=out)
        if out.size == 0:
            return out

        err = kern(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(0),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao_pair),
            ctypes.c_int(naux))
        if err != 0:
            raise RuntimeError(f'{kern} failed')
        return out
    return evaluate_j3c, aux_sorting, aux_offsets

def _argsort_aux_by_atom(auxmol, aux_sorting=None):
    # groupby atom Id
    aux_idx = auxmol.get_ao_idx()
    if aux_sorting is not None:
        aux_idx = cp.asnumpy(aux_sorting)[aux_idx]
    original_auxmol = auxmol.mol
    aux_loc = original_auxmol.ao_loc_nr(cart=True)
    aux_slices = original_auxmol.aoslice_by_atom(ao_loc=aux_loc)
    return aux_idx, aux_slices[:,2:]

def _factorize_dm(mol, dm):
    '''Symmetric factorization'''
    dm_factor_l, dm_factor_r = factorize_dm(dm)
    assert dm_factor_r is None
    # transform to the AO order in sorted_cell
    dm_factor_l = mol.apply_C_dot(dm_factor_l, axis=0)

    # dm_factor are sorted and grouped based on atom_id
    ao_loc = mol.ao_loc_nr(cart=True)
    nao = ao_loc[-1]
    ao_idx = np.split(np.arange(nao), ao_loc[1:-1])
    inv_sorted = np.empty_like(mol.sorted_idx)
    inv_sorted[mol.sorted_idx] = np.arange(len(mol.sorted_idx))
    ao_idx = np.hstack([ao_idx[i] for i in inv_sorted])
    dm_factor_l = dm_factor_l[ao_idx]
    return dm_factor_l

def get_veff(int3c2e_opt, dm, j_factor=1, k_factor=1, verbose=None):
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    assert hasattr(dm, 'mo_occ')
    ao_idx = mol.get_ao_idx()
    mo_coeff = mol.apply_C_dot(dm.mo_coeff, axis=0)
    mo_coeff = mo_coeff[ao_idx]
    orbo = mo_coeff[:,dm.mo_occ>0]
    nao, nocc = orbo.shape

    natm = mol.natm
    naux = auxmol.nao
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    vhf_atm = cp.zeros((natm,3,nao,nocc))
    mem_sufficient = 0
    if mem_sufficient:
        vhf_atm_ao = cp.zeros((natm,3,nao,nao))

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    mem_avail = mem_free - naux*nocc*nao * 8 # cache dm_aux ~ (aux,i,a)
    mem_avail -= naux*nocc**2 * 8 # cache dm_oo ~ (aux,i,j)
    mem_avail -= 3*nao*nao * 8 # cache vhf1 = <di/dR|Veff|j>
    assert mem_avail > 0, 'Insufficient GPU memory'

    # size for caching a tensor with the shape (:,nao_pair) or (:,nocc*nao)
    pair_size_max = max(nao_pair, nocc*nao)
    _unit = 3*(nao_pair+pair_size_max)*8
    batch_size = max(1, min(naux, int(mem_avail*.7) // _unit))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = min(naux, int(mem_avail*.9 - batch_size*_unit) // (nao**2*8))
    assert blksize > 0, 'Insufficient GPU memory'
    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    j3c_00 = cp.empty((naux, nocc, nao))
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            contract('pqr,pi->riq', j3c, orbo, out=j3c_00[aux0:aux1])
    j3c_full = buf = eval_j3c = j3c = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    aux_idx, aux_slices = _argsort_aux_by_atom(auxmol, aux_sorting)

    counts = aux_slices[:,1] - aux_slices[:,0]
    atm_id_for_aux = np.empty(naux, dtype=int)
    atm_id_for_aux[aux_idx] = np.repeat(np.arange(auxmol.natm), counts)

    # aux_filling_order points to address where to write the aux function for the
    # aux-index generated by the _int3c2e_ip1_evaluator
    aux_filling_order = np.empty(naux, dtype=int)
    aux_filling_order[aux_idx] = np.arange(naux)

    # 3c integrals are computed in Cartesian bases, and sorted in the original
    # AO order.
    original_mol = mol.mol
    ao_loc = original_mol.ao_loc_nr(cart=True)
    aoslices = original_mol.aoslice_by_atom(ao_loc=ao_loc)

    original_auxmol = auxmol.mol
    j2c = int2c2e(original_auxmol)
    w, j2c_factor = cp.linalg.eigh(j2c)
    j2c_factor *= w**-.5
    j2c_factor = auxmol.apply_C_dot(j2c_factor)
    if mol.omega > 0 or original_auxmol.cart:
        j2c_factor = j2c_factor[:,w>df.LINEAR_DEP_THR]
    j2c = w = None
    j2c_factor, tmp = cp.empty_like(j2c_factor), j2c_factor
    j2c_factor[aux_sorting] = tmp
    tmp = None
    # metric is not symmetric. Its first index is sorted according to the
    # associated atoms. The sorted indices can be sliced by aux_slices.
    metric = j2c_factor.dot(j2c_factor.T)
    # TODO: update inplace
    dm_3c = cp.einsum('rs,siq->riq', metric, j3c_00)
    j3c_00 = metric = None

    dm_oo = cp.einsum('riq,qj->rij', dm_3c, orbo)
    if j_factor != 0:
        auxvec = cp.einsum('rii->r', dm_oo)
        auxvec_orig_order = auxvec[aux_idx]
        dm = orbo.dot(orbo.T)

    # (00|0)(1|0)(0|00)
    # int2c2e_ip1 computs d/dr. d/dX = -d/dr, the derivative of metric
    # introduces another -1. The overall factor is 1.
    j2c_10 = int2c2e_ip1(auxmol, sort_output=False)

    # The first AO indices in j2c_10 should be grouped by atoms; The second
    # indices are sorted to match the aux indices in dm_3c
    inv_aux_sorting = cp.empty(naux, dtype=int)
    inv_aux_sorting[aux_sorting] = cp.arange(naux)
    j2c_10 = j2c_10[:,inv_aux_sorting[aux_idx,None],inv_aux_sorting]
    naux_in_atm = (aux_slices[:,1] - aux_slices[:,0]).max()
    buf = cp.empty((3,naux_in_atm,nocc,nao))
    for i, (p0, p1) in enumerate(aux_slices):
        if mem_sufficient:
            dm_3c_atm = dm_3c[aux_idx[p0:p1]]
            tmp = ndarray((3,p1-p0,nocc,nao), buffer=buf)
            j3c_1 = contract('xrs,siq->xriq', j2c_10[:,p0:p1], dm_3c, out=tmp)
            contract('xrip,riq->xpq', j3c_1, dm_3c_atm, -.5*k_factor, beta=1, out=vhf_atm_ao[i])
        else:
            dm_3c_atm = dm_3c[aux_idx[p0:p1]]
            dm_oo_atm = dm_oo[aux_idx[p0:p1]]
            tmp = ndarray((3,p1-p0,nocc,nocc), buffer=buf)
            tmp = contract('xrs,sij->xrij', j2c_10[:,p0:p1], dm_oo, out=tmp)
            contract('riq,xrij->xqj', dm_3c_atm, tmp, -.5*k_factor, beta=1, out=vhf_atm[i])

            tmp = ndarray((3,p1-p0,nocc,nao), buffer=buf)
            j3c_1 = contract('xrs,siq->xriq', j2c_10[:,p0:p1], dm_3c, out=tmp)
            contract('xriq,rij->xqj', j3c_1, dm_oo_atm, -.5*k_factor, beta=1, out=vhf_atm[i])

        if j_factor != 0:
            contract('xriq,r->xqi', j3c_1, auxvec_orig_order[p0:p1], j_factor,
                     beta=1, out=vhf_atm[i])
            tmp = cp.einsum('xrs,s->xr', j2c_10[:,p0:p1], auxvec)
            contract('riq,xr->xqi', dm_3c_atm, tmp, j_factor, beta=1, out=vhf_atm[i])
    buf = tmp = j3c_1 = dm_3c_atm = dm_oo_atm = None
    j2c_10 = None

    # (10|0)(0|0)(0|00)
    eval_ip1 = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ip1')[0]
    eval_ipaux = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ipaux')[0]

    vhf1 = cp.zeros((3, nao, nao))
    j3c_full = cp.zeros((nao, nao, blksize))
    buf0 = cp.empty((3, pair_size_max, batch_size))
    buf1 = cp.empty((3, pair_size_max, batch_size))
    buf2 = cp.empty((blksize, nocc, nao))
    aux0 = aux1 = 0
    for kbatch in range(aux_batches):
        compressed_dk = eval_ipaux(kbatch, out=buf1)
        compressed_di = eval_ip1(kbatch, out=buf0)
        naux_in_batch = compressed_dk.shape[-1]
        _aux0, _aux1 = aux1, aux1 + naux_in_batch

        # (10|0)(0|0)(0|00)
        # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
        compressed_dk += compressed_di
        compressed_dj = compressed_dk # ~ d/dX on j
        compressed_di *= -1           # ~ d/dX on i
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            tmp = ndarray((nao, dk, nocc), buffer=buf2)
            for x in range(3):
                j3c[j_addr,i_addr] = compressed_dj[x,:,k0:k1]
                j3c[i_addr,j_addr] = compressed_di[x,:,k0:k1]
                contract('pqr,qi->pri', j3c, orbo, out=tmp)
                contract('pri,riq->pq', tmp, dm_3c[aux0:aux1], -.5*k_factor,
                         beta=1, out=vhf1[x])
                if mem_sufficient:
                    for i, (p0, p1) in enumerate(aoslices[:,2:]):
                        contract('pqr,pi->qri', j3c[p0:p1], orbo[p0:p1], out=tmp)
                        contract('pri,riq->pq', tmp, dm_3c[aux0:aux1], -.5*k_factor,
                                beta=1, out=vhf_atm_ao[i,x])
                else:
                    for i, (p0, p1) in enumerate(aoslices[:,2:]):
                        contract('pqr,pi->qri', j3c[p0:p1], orbo[p0:p1], out=tmp)
                        contract('pri,rij->pj', tmp, dm_oo[aux0:aux1], -.5*k_factor,
                                beta=1, out=vhf_atm[i,x])
                        j3c_oo = contract('qri,qj->jri', tmp, orbo)
                        contract('riq,jri->qj', dm_3c[aux0:aux1], j3c_oo, -.5*k_factor,
                                beta=1, out=vhf_atm[i,x])

                if j_factor != 0:
                    contract('pqr,r->pq', j3c, auxvec[aux0:aux1], j_factor,
                             beta=1, out=vhf1[x])
                    for i, (p0, p1) in enumerate(aoslices[:,2:]):
                        auxvec1 = cp.einsum('pqr,pq->r', j3c[p0:p1], dm[p0:p1])
                        contract('riq,r->qi', dm_3c[aux0:aux1], auxvec1,
                                 2*j_factor, beta=1, out=vhf_atm[i,x])

        # (00|1)(0|0)(0|00)
        compressed_dk += compressed_di
        j3c_aux_tmp = ndarray((3,naux_in_batch,nocc,nao), buffer=buf0)
        aux1 = _aux0
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            tmp = ndarray((nao, dk, nocc), buffer=buf1)
            for i in range(3):
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed_dk[i,:,k0:k1]
                # Note d/dX = -d/dr, apply alpha=-1
                contract('pqr,pi->riq', j3c, orbo, alpha=-1, out=j3c_aux_tmp[i,k0:k1])

        # In a batch of auxiliary basis, sort the aux index based on their atom
        # Id and stored the tensor in compressed_sorted_aux.
        idx = np.argsort(aux_filling_order[_aux0:_aux1])
        j3c_aux = ndarray((3,naux_in_batch,nocc,nao), buffer=buf1)
        j3c_aux = cp.take(j3c_aux_tmp, idx, axis=1, out=j3c_aux)
        dm_3c_batch = ndarray((naux_in_batch,nocc,nao), buffer=buf0[0])
        dm_oo_batch = ndarray((naux_in_batch,nocc,nocc), buffer=buf0[1])
        dm_3c_batch = cp.take(dm_3c[_aux0:_aux1], idx, axis=0, out=dm_3c_batch)
        dm_oo_batch = cp.take(dm_oo[_aux0:_aux1], idx, axis=0, out=dm_oo_batch)
        if j_factor != 0:
            auxvec_batch = auxvec[_aux0:_aux1][idx]
        counts = np.bincount(atm_id_for_aux[_aux0:_aux1])

        p0 = p1 = 0
        for i, count in enumerate(counts):
            if count == 0:
                continue
            p0, p1 = p1, p1 + count
            if mem_sufficient:
                auxvec_ipaux = contract('xrip,pi->xr', j3c_aux[:,p0:p1], orbo)
                contract('xrip,riq->xpq', j3c_aux[:,p0:p1], dm_3c_batch[p0:p1],
                         -.5*k_factor, beta=1, out=vhf_atm_ao[i])
            else:
                j3c_1 = contract('xriq,qj->xrij', j3c_aux[:,p0:p1], orbo)
                auxvec_ipaux = cp.einsum('xrii->xr', j3c_1)
                contract('riq,xrij->xqj', dm_3c_batch[p0:p1], j3c_1,
                         -.5*k_factor, beta=1, out=vhf_atm[i])
                contract('xriq,rij->xqj', j3c_aux[:,p0:p1], dm_oo_batch[p0:p1],
                         -.5*k_factor, beta=1, out=vhf_atm[i])
            if j_factor != 0:
                contract('riq,xr->xqi', dm_3c_batch[p0:p1], auxvec_ipaux,
                         j_factor, beta=1, out=vhf_atm[i])
                contract('xriq,r->xqi', j3c_aux[:,p0:p1], auxvec_batch[p0:p1],
                         j_factor, beta=1, out=vhf_atm[i])
    j3c_full = buf0 = buf1 = buf2 = None
    dm_oo = dm_3c = None
    t0 = log.timer_debug1('fill_int3c2e_ip1 and fill_int3c2e_ipaux', *t0)

    if mem_sufficient:
        vhf_atm_ao = vhf_atm_ao + vhf_atm_ao.transpose(0,1,3,2)
        contract('nxpq,qj->nxpj', vhf_atm_ao, orbo, beta=1, out=vhf_atm)

    # (10|0)(0|0)(0|00)
    # Distribute <d/dR i|Veff|j> to derivatives on atoms
    for i, (p0, p1) in enumerate(aoslices[:,2:]):
        contract('xpq,pi->xqi', vhf1[:,p0:p1], orbo[p0:p1], beta=1, out=vhf_atm[i])
        contract('xpq,qi->xpi', vhf1[:,p0:p1], orbo, beta=1, out=vhf_atm[i,:,p0:p1])

    # *2 for double occupancy
    vhf_atm *= 2

    vhf_atm = contract('nxpj,pi->nxij', vhf_atm, mo_coeff)
    return vhf_atm

def int3c2e_scheme_ip2(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    order = li + lj + lk + 2
    nroots = (order//2 + 1)
    if omega < 0:
        nroots *= 2
    g_size = (li+2)*(lj+2)*(lk+3)
    unit = g_size*3 + nroots*2 + 7
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        nfi = (li + 1) * (li + 2) // 2
        nfj = (lj + 1) * (lj + 2) // 2
        nfk = (lk + 1) * (lk + 2) // 2
        gout_size = nfi * nfj * nfk
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    return nsp_per_block, gout_stride, shm_size

def int3c2e_scheme_ip1(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    order = li + lj + lk + 1
    nroots = (order//2 + 1)
    if omega < 0:
        nroots *= 2
    g_size = (li+2)*(lj+2)*(lk+1)
    unit = g_size*3 + nroots*2 + 7
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        nfi = (li + 1) * (li + 2) // 2
        nfj = (lj + 1) * (lj + 2) // 2
        nfk = (lk + 1) * (lk + 2) // 2
        gout_size = nfi * nfj * nfk
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    return nsp_per_block, gout_stride, shm_size

def int3c2e_scheme_ipaux(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    order = li + lj + lk + 1
    nroots = (order//2 + 1)
    if omega < 0:
        nroots *= 2
    g_size = (li+1)*(lj+1)*(lk+2)
    unit = g_size*3 + nroots*2 + 7
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        nfi = (li + 1) * (li + 2) // 2
        nfj = (lj + 1) * (lj + 2) // 2
        nfk = (lk + 1) * (lk + 2) // 2
        gout_size = nfi * nfj * nfk
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    return nsp_per_block, gout_stride, shm_size

def _int2c2e_ip2_per_atom(mol, dm):
    '''Second order nuclear derivatives of 2c2e Coulomb integrals.
    '''
    from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt
    opt = Int2c2eOpt(mol).build()
    mol = opt.cell
    li = np.arange(L_AUX_MAX+1)[:,None]
    lj = np.arange(L_AUX_MAX+1)
    order = li + lj + 2
    nroots = order//2 + 1
    if mol.omega < 0:
        nroots *= 2 # for short-range
    g_size = (li+3)*(lj+3)
    unit = g_size*3 + nroots*2 + 4
    nsp_max = _nearest_power2(SHM_SIZE // (unit*8))
    nsp_per_block = np.where(nsp_max < THREADS, nsp_max, THREADS)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    lmax = mol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        opt.bas_ij_cache, nsp_per_block)

    nbatches_shl_pair = len(shl_pair_offsets) - 1
    rys_envs = opt._rys_envs
    natm = mol.natm
    ejk = cp.zeros((natm, natm, 3, 3))
    libvhf_rys.e_int2c2e_ip2.restype = ctypes.c_int
    err = libvhf_rys.e_int2c2e_ip2(
        ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.byref(rys_envs), ctypes.c_int(shm_size_max),
        ctypes.c_int(nbatches_shl_pair),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('e_int2c2e_ip2 failed')
    ejk = ejk + ejk.transpose(1,0,3,2)
    # *2 for i>=j, *.5 from Coulomb operator
    ejk *= 2 * .5
    return ejk

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    ejk = _jk_energy_per_atom(int3c2e_opt, dm)
    return e1 + ejk

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None,
                      max_memory=None, verbose=None, with_j=True, with_k=True, omega=None):
    '''Partial derivative
    '''
    log = logger.new_logger(hessobj, verbose)
#    time0 = t1 = log.init_timer()
    mem_free = cp.cuda.runtime.memGetInfo()[0]
#    mem_avail = int(mem_free * .6)
    log.debug('Partial Hessian with density fitting approximation')
#    log.debug(f'Memory available {mem_avail/GB} GB')

    mol = hessobj.mol
    mf = hessobj.base
    mf.with_df._cderi = None
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mo_coeff = asarray(mo_coeff, order='C')
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
#    mocc_2 = mocc * mo_occ[mo_occ>0]**.5
    dm0 = cupy.dot(mocc, mocc.T) * 2

    auxmol = df.addons.make_auxmol(mol, auxbasis=mf.with_df.auxbasis)
#    auxslices = auxmol.aoslice_by_atom()
#    aoslices = mol.aoslice_by_atom()

    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    ejk = _jk_energy_per_atom(int3c2e_opt, dm0)
    # Energy weighted density matrix
    # pi,qi,i->pq
#    dme0 = cupy.dot(mocc, (mocc * mo_energy[mo_occ>0] * 2).T)
#    de_hcore = rhf_hess._e_hcore_generator(hessobj, dm0)
#    t1 = log.timer_debug1('hcore generate', *t1)
#
#    # ------------------------------------
#    #      overlap matrix contributions
#    # ------------------------------------
#    s1aa, s1ab, _ = rhf_hess.get_ovlp(mol)
#    s1aa = asarray(s1aa, order='C')
#    s1ab = asarray(s1ab, order='C')
#    h1aa = 2.0*contract('xypq,pq->pxy', s1aa, dme0)
#    h1ab = 2.0*contract('xypq,pq->pqxy', s1ab, dme0)
#    s1aa = s1ab = dme0 = None
#    # -----------------------------------------
#    #        collecting all
#    # -----------------------------------------
#    natm = len(atmlst)
#    e1 = cupy.zeros([natm,natm,3,3])
#    ej = hj_ipip
#    ek = hk_ipip
#
#    for i0, ia in enumerate(atmlst):
#        shl0, shl1, p0, p1 = aoslices[ia]
#        e1[i0,i0] -= cupy.sum(h1aa[p0:p1], axis=0)
#        for j0, ja in enumerate(atmlst[:i0+1]):
#            q0, q1 = aoslices[ja][2:]
#            e1[i0,j0] -= cupy.sum(h1ab[p0:p1,q0:q1], axis=[0,1])
#            if with_j:
#                ej[i0,j0] += cupy.sum(hj_ao_ao[p0:p1,q0:q1], axis=[0,1])
#            if with_k:
#                ek[i0,j0] += cupy.sum(hk_ao_ao[p0:p1,q0:q1], axis=[0,1])
#            e1[i0,j0] += de_hcore(ia, ja)
#        #
#        # The first order RI basis response
#        #
#        if hessobj.auxbasis_response:
#            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
#                if with_j:
#                    _ej = cupy.sum(hj_ao_aux[p0:p1,q0:q1], axis=[0,1])
#                    if hessobj.auxbasis_response > 1:
#                        ej[i0,j0] += _ej * 2
#                        ej[j0,i0] += _ej.T * 2
#                    else:
#                        ej[i0,j0] += _ej
#                        ej[j0,i0] += _ej.T
#                if with_k:
#                    _ek = cupy.sum(hk_ao_aux[p0:p1,q0:q1], axis=[0,1])
#                    if hessobj.auxbasis_response > 1:
#                        ek[i0,j0] += _ek
#                        ek[j0,i0] += _ek.T
#                    else:
#                        ek[i0,j0] += _ek * .5
#                        ek[j0,i0] += _ek.T * .5
#        #
#        # The second order RI basis response
#        #
#        if hessobj.auxbasis_response > 1:
#            shl0, shl1, p0, p1 = auxslices[ia]
#            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
#                if with_j:
#                    _ej = cupy.sum(hj_aux_aux[p0:p1,q0:q1], axis=[0,1])
#                    ej[i0,j0] += _ej
#                    ej[j0,i0] += _ej.T
#                if with_k:
#                    _ek = cupy.sum(hk_aux_aux[p0:p1,q0:q1], axis=[0,1])
#                    ek[i0,j0] += _ek * .5
#                    ek[j0,i0] += _ek.T * .5
#    for i0, ia in enumerate(atmlst):
#        for j0 in range(i0):
#            e1[j0,i0] = e1[i0,j0].T
#            if with_j:
#                ej[j0,i0] = ej[i0,j0].T
#            if with_k:
#                ek[j0,i0] = ek[i0,j0].T
#    t1 = log.timer_debug1('hcore contribution', *t1)
#
#    aux2atom = int3c2e.get_aux2atom(intopt, auxslices)
#
#    natm = mol.natm
#    idx = range(natm)
#    # Diagonal contributions
#    if hessobj.auxbasis_response > 1:
#        if with_j:
#            ej[idx, idx] += contract('ia,ixy->axy', aux2atom, hj_aux_diag)
#        if with_k:
#            ek[idx, idx] += contract('ia,ixy->axy', aux2atom, hk_aux_diag)
#
#    log.timer('RHF partial hessian', *time0)
#    return e1, ejk, 0
    return ejk


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst ==range(natm)
    vj, vk = _get_jk_ip(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose, True)
    # h1mo = h1 + vj - 0.5 * vk
    h1mo = vk
    h1mo *= -.5
    h1mo += vj
    h1mo += rhf_grad.get_grad_hcore(hessobj.base.nuc_grad_method())
    return h1mo

def _get_jk_ip(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None,
            verbose=None, with_j=True, with_k=True, omega=None):
    '''
    Derivatives of J, K matrices in MO bases
    '''
#    return vj1_int3c, vk1_int3c

def _get_jk_mo(hessobj, mol, dms, mo_coeff, mocc,
           hermi=1, with_j=True, with_k=True, omega=None):
    mf = hessobj.base
    dfobj = mf.with_df
    if omega is None:
        return jk.get_jk(dfobj, dms, mo_coeff, mocc,
                         hermi=hermi, with_j=with_j, with_k=with_k)

    # A temporary treatment for RSH-DF integrals
    key = '%.6f' % omega
    if key in dfobj._rsh_df:
        rsh_df = dfobj._rsh_df[key]
    else:
        rsh_df = dfobj._rsh_df[key] = dfobj.copy().reset()
        logger.info(dfobj, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

    with rsh_df.mol.with_range_coulomb(omega):
        return jk.get_jk(rsh_df, dms, mo_coeff, mocc,
                         hermi=hermi, with_j=with_j, with_k=with_k, omega=omega)


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    _keys = {'auxbasis_response',}

    auxbasis_response = 2
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    get_jk_mo = _get_jk_mo
