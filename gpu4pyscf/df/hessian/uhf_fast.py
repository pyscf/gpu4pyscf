# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

'''
Non-relativistic UHF analytical Hessian with density-fitting approximation
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
    contract, asarray, ndarray, transpose_sum, get_avail_mem)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, _nearest_power2,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys, Int3c2eOpt, int2c2e,
    int2c2e_ip1)
from gpu4pyscf.df import df
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import uhf as uhf_hess
from gpu4pyscf.df.hessian import jk
from gpu4pyscf.df.hessian import rhf_fast as df_rhf_hess
from gpu4pyscf.df.hessian.rhf_fast import (
    int3c2e_scheme_ip2, int3c2e_scheme_ip1, int3c2e_scheme_ipaux,
    _int3c2e_ip1_evaluator, _auxbas_atom_labels, _aggregate_to_atoms,
    _factorize_j2c, _argsort_aux_by_atom, _allocate)

def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    assert dm.ndim == 3
    if k_factor == 0:
        from gpu4pyscf.df.hessian.rhf_fast import _j_energy_per_atom
        return _j_energy_per_atom(int3c2e_opt, dm[0]+dm[1], verbose) * j_factor

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l = dm_factor_r = df_rhf_hess._factorize_dm(mol, dm)
    nao, nocc = dm_factor_l.shape[1:]

    natm = mol.natm
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxmol.ao_loc
    naux = int(aux_loc[-1])
    largest_shell_nao = int((aux_loc[1:] - aux_loc[:-1]).max())

    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    word_avail -= 2 * 3 * naux * nocc**2 # j3c_oo1
    batch_size = int(word_avail * 0.04) // nao_pair
    assert batch_size > largest_shell_nao, 'Insufficient GPU memory'
    batch_size = min(batch_size, max(largest_shell_nao, naux))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))
    aux_batches = len(aux_offsets) - 1

    blksize = min(naux, int(word_avail * 0.5) // (nao**2*8) * 8)
    assert blksize > 1, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    log.debug1('mem_avail=%.3f MB, aux_batches=%d, batch_size=%d, blksize=%d',
                mem_avail*1e-6, aux_batches, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, 2, nocc, nocc))
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
            contract('iqr,qj->rij', tmp, dm_factor_l[0], out=j3c_oo[aux0:aux1,0])
            contract('pqr,pi->iqr', j3c, dm_factor_r[1], out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l[1], out=j3c_oo[aux0:aux1,1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, mol.omega > 0)
    dm_oo = cp.einsum('uv,unij->vnij', metric_v, j3c_oo)
    dm_oo /= metric_w[:,None,None,None]
    dm_oo = contract('uv,vnij->unij', metric_v, dm_oo, out=j3c_oo)
    j3c_oo = None
    if j_factor != 0:
        auxvec = cp.einsum('rnii->r', dm_oo)

    # (00|0)(2|0)(0|00)
    if j_factor == 0:
        dm_aux = None
    else:
        dm_aux = auxvec[:,None] * auxvec
    dm_aux = contract('rnij,snij->rs', dm_oo, dm_oo,
                      alpha=-k_factor, beta=j_factor, out=dm_aux)
    ejk_aux = cp.asarray(df_rhf_hess._int2c2e_ip2_per_atom(
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

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    # assert cp.array_equal(aux_sorting, argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux))
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    aux_batches = len(aux_offsets) - 1

    if j_factor != 0:
        dm = contract('npi,nqi->pq', dm_factor_l, dm_factor_r)

    # (20|0)(0|0)(0|00) + (10|1)(0|0)(0|00)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern_ip2 = libvhf_rys.ejk_int3c2e_ip2
    ejk = cp.zeros((natm, natm, 3, 3))
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf2 = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))
    for kbatch in range(aux_batches):
        naux_in_batch = aux_offsets[kbatch+1] - aux_offsets[kbatch]
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
            contract('rji,qj->iqr', dm_oo[aux0:aux1,0], dm_factor_l[0], out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_r[0], -k_factor, beta, out=dm_tensor)
            contract('rji,qj->iqr', dm_oo[aux0:aux1,1], dm_factor_l[1], out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_r[1], -k_factor, 1., out=dm_tensor)
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
    buf = buf1 = buf2 = ejk_aux = compressed = dm_tensor = tmp = None
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
    j3c_oo1 = cp.empty((3, naux, 2, nocc, nocc)) # = (1|00)
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
                contract('pqr,pi->iqr', j3c, dm_factor_r[0], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[0], alpha=-1,
                         out=j3c_oo1[i,aux0:aux1,0])
                contract('pqr,pi->iqr', j3c, dm_factor_r[1], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[1], alpha=-1,
                         out=j3c_oo1[i,aux0:aux1,1])
    j3c_full = buf = buf1 = tmp = compressed = None
    t0 = log.timer_debug1('fill_int3c2e_ipaux', *t0)

    # note int2c2e_ip1 computs d/dr and d/dX = -d/dr
    j2c_10 = int2c2e_ip1(auxmol, sort_output=False)
    j2c_10 *= -1
    j2c_10, tmp = cp.empty_like(j2c_10), j2c_10
    j2c_10[:,aux_sorting[:,None], aux_sorting] = tmp
    j2c_10v = contract('xrs,st->xrt', j2c_10, metric_v)
    j2c_10v_w = j2c_10v / metric_w
    # (00|0)(1|0)(0|1)(0|00)
    j2c_ip2 = contract('xrs,yts->rtxy', j2c_10v_w, j2c_10v)
    j2c_ip2 *= dm_aux[:,:,None,None]
    h_aux = j2c_ip2
    dm_aux = j2c_10v = None

    # j3c_oo1p = (1|0)(0|00) + (1|00)
    j3c_oo1p = contract('xuv,vnij->xunij', j2c_10, dm_oo, alpha=-1, beta=1, out=j3c_oo1)
    # (00|0)(1|0)(1|00) + (00|0)(1|0)(1|0)(0|00)
    if j_factor == 0:
        dm_aux1 = None
    else:
        auxvec_ipauxp = cp.einsum('xrnii->xr', j3c_oo1p)
        dm_aux1 = cp.einsum('xr,t->xrt', auxvec_ipauxp, auxvec)
    dm_aux1 = contract('xrnij,snij->xrs', j3c_oo1p, dm_oo, -k_factor,
                       beta=j_factor, out=dm_aux1)
    tmp = cp.einsum('xrt,st->xrs', j2c_10v_w, metric_v)
    w10_100 = cp.einsum('xrt,ytr->rtxy', tmp, dm_aux1)
    h_aux -= w10_100
    h_aux -= w10_100.transpose(1,0,3,2) # swap the asymetric di,dj indices
    dm_aux1 = j2c_10 = w10_100 = tmp = None

    # (00|1)(1|0)(0|00) + (00|1)(0|0)(1|00) +
    # (00|0)(0|1)(1|00) + (00|0)(0|1)(1|0)(0|00)
    # = 001p * 001p
    if j_factor == 0:
        dm_aux11 = None
    else:
        dm_aux11 = cp.einsum('xr,ys->rsxy', auxvec_ipauxp, auxvec_ipauxp)
    dm_aux11 = contract('xrnij,ysnij->rsxy', j3c_oo1p, j3c_oo1p, -k_factor,
                        beta=j_factor, out=dm_aux11)
    j2c_inv = (metric_v / metric_w).dot(metric_v.T)
    dm_aux11 *= j2c_inv[:,:,None,None]
    h_aux += dm_aux11
    # swap the differentiation order
    # (00|0)(1|0)(1|00) + (00|0)(1|0)(1|0)(0|00)
    h_aux = h_aux + h_aux.transpose(1,0,3,2)
    j2c_inv = None

    atm_labels = _auxbas_atom_labels(auxmol, aux_sorting)
    ejk_aux = _aggregate_to_atoms(h_aux, natm, atm_labels, axis=(0,1))
    ejk += ejk_aux * .5
    dm_aux11 = j3c_oo1 = h_aux = None
    t1 = t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    # 3c integrals are computed in Cartesian bases, and sorted in the original
    # AO order.
    original_mol = mol.mol
    ao_loc = original_mol.ao_loc_nr(cart=True)
    aoslices = original_mol.aoslice_by_atom(ao_loc=ao_loc)
    nao_on_atom = int((aoslices[:,3] - aoslices[:,2]).max())

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    word_avail -= 3 * batch_size * nao_pair * 2 # eval_ip1, eval_ipaux
    aux_unit1 = 3*2*nao*nocc + 3*naux # j3c_100
    aux_batch_size = (min(int(word_avail*.8), word_avail-12*naux*nao_on_atom*nocc)
                      // (aux_unit1 * 8) * 8)
    assert aux_batch_size > 0, 'Insufficient GPU memory'
    metric_size = metric_v.shape[1]
    aux_batch_size = min(aux_batch_size, metric_size)
    word_avail -= aux_unit1 * aux_batch_size
    aux_unit2 = natm*3*2*nocc*nocc * 2 # j3c_oo_atm
    aux_blksize = word_avail // (aux_unit2 * 8) * 8
    assert aux_blksize > 0, 'Insufficient GPU memory'
    aux_blksize = min(aux_blksize, aux_batch_size)
    blksize = word_avail // ((nocc*2+nao)*nao * 8) * 8
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    log.debug1('mem_avail=%.3f MB, aux_batch_size=%d, aux_blksize=%d, blksize=%d',
               mem_avail*1e-6, aux_batch_size, aux_blksize, blksize)

    h_ao_aux = cp.zeros((natm,naux,3,3))
    ejk_ao = cp.zeros((natm,natm,3,3))

    j3c_buf = cp.empty((3,aux_batch_size,2,nao,nocc))
    work = cp.empty(
        max(aux_unit2*aux_blksize,
            3*nao_pair*batch_size * 2 + (nocc*2+nao)*nao * blksize,
            3*naux*2*nao_on_atom*nocc * 2,
            3*naux*2*nao_on_atom*nocc + 3*aux_batch_size*naux))
    # TODO: for small molecules, merge the kern_ipaux to the previouse case.
    for v0, v1 in lib.prange(0, metric_size, aux_batch_size):
        dv = v1 - v0
        j3c_100 = ndarray((3, dv, 2, nao, nocc), buffer=j3c_buf)
        j3c_100[:] = 0.
        j3c_full, work1 = _allocate((nao, nao, blksize), work)
        j3c_full[:] = 0.
        buf0, work1 = _allocate((3, nao_pair, batch_size), work1)
        buf1, work1 = _allocate((3, nao_pair, batch_size), work1)
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
                tmp = _allocate((dk, 2, nao, nocc), work1)[0]
                for i in range(3):
                    j3c[j_addr,i_addr] = compressed_dj[i,:,k0:k1]
                    j3c[i_addr,j_addr] = compressed_di[i,:,k0:k1]
                    tmp = contract('pqr,nqi->rnpi', j3c, dm_factor_r, out=tmp)
                    contract('rs,rnpi->snpi', metric_v[aux0:aux1,v0:v1], tmp,
                             beta=1, out=j3c_100[i])
        t1 = log.timer_debug1(f'fill_int3c2e_ip1 {v0}:{v1}', *t1)

        # (10|0)(0|0)(0|01) + (10|0)(0|0)(0|10)
        # (01|0)(0|0)(0|01) + (01|0)(0|0)(0|10)
        for k0, k1 in lib.prange(0, dv, aux_blksize):
            j3c_oo_atm, work1 = _allocate((natm,3,k1-k0,2,nocc,nocc), work)
            for i, (p0, p1) in enumerate(aoslices[:,2:]):
                contract('xrnuj,nui->xrnij', j3c_100[:,k0:k1,:,p0:p1],
                         dm_factor_l[:,p0:p1], out=j3c_oo_atm[i])
            # di/dX + dj/dX
            w_part = metric_w[v0+k0:v0+k1]
            transpose_sum(j3c_oo_atm.reshape(-1,nocc,nocc), inplace=True)
            j3c_oo_atm_w = ndarray(j3c_oo_atm.shape, buffer=work1)
            cp.divide(j3c_oo_atm, w_part[:,None,None,None], out=j3c_oo_atm_w)
            contract('pxrnij,qyrnij->pqxy', j3c_oo_atm_w, j3c_oo_atm, -k_factor,
                     beta=1, out=ejk_ao)

            if j_factor != 0:
                auxvec_100_atm = cp.einsum('pxrnii->pxr', j3c_oo_atm)
                contract('pxr,qyr->pqxy', auxvec_100_atm/w_part, auxvec_100_atm,
                         j_factor, beta=1, out=ejk_ao)

                # (10|0)(1|00) + (10|0)(1|0)(0|00)
                tmp = ndarray((3,naux,k1-k0), buffer=work)
                cp.multiply(j2c_10v_w[:,:,v0+k0:v0+k1], auxvec[:,None], out=tmp)
                contract('ys,sr->ysr', auxvec_ipauxp, metric_v[:,v0+k0:v0+k1]/w_part,
                         beta=-1, out=tmp)
                # (10|0)(0|1)(0|00)
                contract('pxr,ysr->psxy', auxvec_100_atm, tmp, j_factor, 1, h_ao_aux)
                tmp = None

        for i, (p0, p1) in enumerate(aoslices[:,2:]):
            # (10|0)(1|0)(0|00) + (10|0)(0|0)(1|00)
            #:h_ao_aux += einsum('xrnpj,sr,npi,ysnji->prxy',
            #:                   j3c_100, metric_v, dm_factor_l, j3c_oo1p)
            tmp0, work1 = _allocate((3,naux,2,p1-p0,nocc), work)
            tmp1, work1 = _allocate((3,naux,2,p1-p0,nocc), work1)
            contract('xrnuj,sr->xsnuj', j3c_100[:,:,:,p0:p1],
                     metric_v[:,v0:v1]/metric_w[v0:v1], out=tmp0)
            contract('ysnij,nui->ysnuj', j3c_oo1p, dm_factor_l[:,p0:p1], out=tmp1)
            # *2 corresponds to to di/dX + dj/dX on j3c_100
            contract('xsnuj,ysnuj->sxy', tmp0, tmp1, -k_factor*2, 1, h_ao_aux[i])

            # (10|0)(0|1)(0|00)
            #:h_ao_aux -= einsum('xrnpj,ysr,npi,snji->prxy',
            #:                   j3c_100, j2c_10v_w, dm_factor_l, dm_oo)
            tmp0, work1 = _allocate((naux,2,p1-p0,nocc), work)
            tmp1, work1 = _allocate((3,v1-v0,naux), work1)
            contract('snij,nui->snuj', dm_oo, dm_factor_l[:,p0:p1], out=tmp0)
            contract('xrnuj,snuj->xrs', j3c_100[:,:,:,p0:p1], tmp0, out=tmp1)
            contract('xrs,ysr->sxy', tmp1, j2c_10v_w[:,:,v0:v1], k_factor*2, 1,
                     h_ao_aux[i])
        t1 = log.timer_debug1(f'contract int3c2e_ip1 {v0}:{v1}', *t1)
    t0 = log.timer_debug1('int3c2e_ipaux and int2c2e_ip1 cross term', *t0)

    ejk_ao_aux = _aggregate_to_atoms(h_ao_aux, natm, atm_labels, axis=1)
    ejk += ejk_ao_aux
    ejk += ejk_ao_aux.transpose(1,0,3,2)

    # scale ejk_ao: *2 for swaping (di/dX j|dk/dY l) -> (di/dY j|dk/dX l)
    # *.5 from Coulomb operator
    ejk += ejk_ao
    return ejk

def _get_veff(int3c2e_opt, mo_coeff, mo_occ, j_factor=1, k_factor=1, verbose=None):
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    ao_idx = mol.get_ao_idx()
    mo_coeff = mol.apply_C_dot(mo_coeff, axis=1)
    mo_coeff = mo_coeff[:,ao_idx]
    mask = mo_occ > 0
    mask_sf = mask[0] | mask[1]
    orbo = mo_coeff[:,:,mask_sf]
    orbo *= cp.sqrt(mo_occ[:,None,mask_sf])
    nao, nocc = orbo.shape[1:]

    natm = mol.natm
    aux_loc = auxmol.ao_loc
    naux = int(aux_loc[-1])
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    vhf_atm = cp.zeros((natm,3,2,nocc,nao))
    vhf1 = cp.zeros((3, 2, nao, nao))

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    word_avail -= 2 * naux*nocc*nao # dm_3c ~ (aux,i,a)
    word_avail -= naux**2 * 3 # metric_v
    word_avail -= 2 * naux*nocc**2 # dm_oo ~ (aux,i,j)
    assert word_avail > 0, 'Insufficient GPU memory'

    mem_sufficient = natm*3*2*nao**2 < word_avail * .6
    if mem_sufficient:
        vhf_atm_ao = cp.zeros((natm,3,2,nao,nao))
        word_avail -= vhf_atm_ao.size

    # size for caching a tensor with the shape (:,nao_pair) or (:,nocc*nao)
    pair_size_max = max(nao_pair, nocc*nao)
    _unit = 6*pair_size_max
    largest_shell_nao = int((aux_loc[1:] - aux_loc[:-1]).max())
    batch_size = int(word_avail*.8 / _unit)
    if batch_size < largest_shell_nao:
        mem_req = (largest_shell_nao-batch_size) *_unit * 1.5 * 1e-6
        raise MemoryError(f'Insufficient GPU memory. Need {mem_req} MB more.')
    batch_size = min(batch_size, max(largest_shell_nao, naux))

    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))
    aux_batches = len(aux_offsets) - 1

    # 3c integrals are computed in Cartesian bases, and sorted in the original
    # AO order.
    original_mol = mol.mol
    ao_loc = original_mol.ao_loc_nr(cart=True)
    aoslices = original_mol.aoslice_by_atom(ao_loc=ao_loc)

    nao_on_atom = int((aoslices[:,3] - aoslices[:,2]).max())
    blksize = int((word_avail*.95 - batch_size*_unit) /
                  (nao**2 + (nao+nocc*2)*max(nao_on_atom, nocc))) // 8 * 8
    assert blksize > 1, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    log.debug1('mem_avail=%.3f MB, mem_sufficient=%s, batch_size=%d, blksize=%d',
               mem_avail*1e-6, mem_sufficient, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    j3c_00 = cp.empty((naux, 2, nocc, nao))
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            contract('pqr,npi->rniq', j3c, orbo, out=j3c_00[aux0:aux1])
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

    metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, mol.omega > 0)

    nw = metric_v.shape[1]
    work = cp.empty((nw, 2, nocc, nao))
    for i0, i1 in lib.prange(0, nocc, 8):
        tmp = ndarray((nw, 2, i1-i0, nao), buffer=work)
        contract('rs,rniq->sniq', metric_v, j3c_00[:,:,i0:i1], out=tmp)
        tmp /= metric_w[:,None,None,None]
        contract('rs,sniq->rniq', metric_v, tmp, out=j3c_00[:,:,i0:i1])
    dm_3c = j3c_00
    j3c_00 = tmp = work = metric_v = metric_w = None

    dm_oo = cp.einsum('rniq,nqj->rnij', dm_3c, orbo)
    if j_factor != 0:
        auxvec = cp.einsum('rnii->r', dm_oo)
        auxvec_by_atm = auxvec[aux_idx]
        dm = contract('npi,nqi->pq', orbo, orbo)

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
    buf = cp.empty((3,naux_in_atm,2,nocc,nao))
    buf1 = cp.empty((naux_in_atm,2,nocc,nao))
    for i, (p0, p1) in enumerate(aux_slices):
        if mem_sufficient:
            dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0, out=buf1[:p1-p0])
            tmp = ndarray((3,p1-p0,2,nocc,nao), buffer=buf)
            j3c_1 = contract('xrs,sniq->xrniq', j2c_10[:,p0:p1], dm_3c, out=tmp)
            contract('xrnip,rniq->xnpq', j3c_1, dm_3c_atm, -k_factor, beta=1, out=vhf_atm_ao[i])
        else:
            dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0, out=buf1[:p1-p0])
            dm_oo_atm = dm_oo[aux_idx[p0:p1]]
            tmp = ndarray((3,p1-p0,2,nocc,nocc), buffer=buf)
            tmp = contract('xrs,snij->xrnij', j2c_10[:,p0:p1], dm_oo, out=tmp)
            contract('rniq,xrnij->xnjq', dm_3c_atm, tmp, -k_factor, beta=1, out=vhf_atm[i])

            tmp = ndarray((3,p1-p0,2,nocc,nao), buffer=buf)
            j3c_1 = contract('xrs,sniq->xrniq', j2c_10[:,p0:p1], dm_3c, out=tmp)
            contract('xrniq,rnij->xnjq', j3c_1, dm_oo_atm, -k_factor, beta=1, out=vhf_atm[i])

        if j_factor != 0:
            contract('xrniq,r->xniq', j3c_1, auxvec_by_atm[p0:p1], j_factor,
                     beta=1, out=vhf_atm[i])
            tmp = cp.einsum('xrs,s->xr', j2c_10[:,p0:p1], auxvec)
            contract('rniq,xr->xniq', dm_3c_atm, tmp, j_factor, beta=1, out=vhf_atm[i])
    tmp = j3c_1 = dm_3c_atm = dm_oo_atm = None
    buf = buf1 = j2c_10 = None
    t1 = t0 = log.timer_debug1('contract j2c_10', *t0)

    # (10|0)(0|0)(0|00)
    eval_ip1 = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ip1')[0]
    eval_ipaux = _int3c2e_ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(mol.omega, 27), batch_size,
        kern='fill_int3c2e_ipaux')[0]

    work = cp.empty(max(
        (nao**2 + 3*pair_size_max + 6*nocc*nao) * batch_size,
        6*nocc*nao*batch_size * 2,
        6*pair_size_max*batch_size +
        (nao*nao + (nao+nocc)*2 * max(nao_on_atom, nocc)) * blksize))
    aux0 = aux1 = 0
    for kbatch in range(aux_batches):
        naux_in_batch = aux_offsets[kbatch+1] - aux_offsets[kbatch]
        aux0, aux1 = aux1, aux1 + naux_in_batch

        j3c_full, work1 = _allocate((nao, nao, blksize), work)
        j3c_full[:] = 0.

        buf0, work1 = _allocate((3, nao_pair, batch_size), work1)
        buf1, work1 = _allocate((3, nao_pair, batch_size), work1)
        compressed_dk = eval_ipaux(kbatch, out=buf0)
        compressed_di = eval_ip1(kbatch, out=buf1)

        # (10|0)(0|0)(0|00)
        # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
        compressed_dk += compressed_di
        compressed_dj = compressed_dk # ~ d/dX on j
        compressed_di *= -1           # ~ d/dX on i
        _aux1 = aux0
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            _aux0, _aux1 = _aux1, _aux1 + dk
            j3c = j3c_full[:,:,:dk]
            if j_factor != 0:
                auxvec_tmp = cp.repeat(auxvec[None,_aux0:_aux1], 2, axis=0)
            for x in range(3):
                j3c[j_addr,i_addr] = compressed_dj[x,:,k0:k1]
                j3c[i_addr,j_addr] = compressed_di[x,:,k0:k1]
                tmp = ndarray((2, nao, dk, nocc), buffer=work1)
                j3c_pri = contract('pqr,nqi->npri', j3c, orbo, out=tmp)
                contract('npri,rniq->npq', j3c_pri, dm_3c[_aux0:_aux1], -k_factor,
                         beta=1, out=vhf1[x])
                if mem_sufficient:
                    for i, (p0, p1) in enumerate(aoslices[:,2:]):
                        if p1 - p0 < nocc:
                            tmp = ndarray((2, p1-p0, dk, nao), buffer=work1)
                            contract('nui,rniq->nurq', orbo[:,p0:p1], dm_3c[_aux0:_aux1], out=tmp)
                            contract('upr,nurq->npq', j3c[p0:p1], tmp,
                                     -k_factor, beta=1, out=vhf_atm_ao[i,x])
                        else:
                            tmp = ndarray((nao, dk, 2, nocc), buffer=work1)
                            contract('pqr,npi->qrni', j3c[p0:p1], orbo[:,p0:p1], out=tmp)
                            contract('prni,rniq->npq', tmp, dm_3c[_aux0:_aux1],
                                     -k_factor, beta=1, out=vhf_atm_ao[i,x])
                else:
                    for i, (p0, p1) in enumerate(aoslices[:,2:]):
                        tmp, buf2 = _allocate((2, p1-p0, dk, nocc), work1)
                        tmp1 = ndarray((2, p1-p0, dk, nao), buffer=buf2)
                        contract('nui,rnij->nurj', orbo[:,p0:p1], dm_oo[_aux0:_aux1], out=tmp)
                        contract('uqr,nurj->njq', j3c[p0:p1], tmp,
                                 -k_factor, beta=1, out=vhf_atm[i,x])

                        contract('uqr,nqj->nurj', j3c[p0:p1], orbo, out=tmp)
                        contract('nui,rniq->nurq', orbo[:,p0:p1], dm_3c[_aux0:_aux1], out=tmp1)
                        contract('nurj,nurq->njq', tmp, tmp1,
                                 -k_factor, beta=1, out=vhf_atm[i,x])

                if j_factor != 0:
                    contract('pqr,nr->npq', j3c, auxvec_tmp, j_factor, beta=1,
                             out=vhf1[x])
                    for i, (p0, p1) in enumerate(aoslices[:,2:]):
                        auxvec1 = cp.einsum('pqr,pq->r', j3c[p0:p1], dm[p0:p1])
                        contract('rniq,r->niq', dm_3c[_aux0:_aux1], auxvec1,
                                 2*j_factor, beta=1, out=vhf_atm[i,x])

        # (00|1)(0|0)(0|00)
        compressed_dk += compressed_di
        j3c_aux, work1 = _allocate((3,naux_in_batch,2,nocc,nao), work)
        off = max(j3c_full.size + compressed_dk.size, j3c_aux.size)
        j3c_aux_tmp = ndarray(j3c_aux.shape, buffer=work[off:])
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            j3c = j3c_full[:,:,:dk]
            for i in range(3):
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed_dk[i,:,k0:k1]
                # Note d/dX = -d/dr, apply alpha=-1
                contract('pqr,npi->rniq', j3c, orbo, alpha=-1, out=j3c_aux_tmp[i,k0:k1])

        # In a batch of auxiliary basis, sort the aux index based on their atom
        # Id and stored the tensor in compressed_sorted_aux.
        idx = np.argsort(aux_filling_order[aux0:aux1])
        j3c_aux = cp.take(j3c_aux_tmp, idx, axis=1, out=j3c_aux)

        dm_3c_batch, work1 = _allocate((naux_in_batch,2,nocc,nao), work1)
        dm_oo_batch, work1 = _allocate((naux_in_batch,2,nocc,nocc), work1)
        # the memory spaces of dm_3c_batch and j3c_aux_tmp overlap,
        # dm_3c_batch must be accessed after copying j3c_aux_tmp to j3c_aux
        dm_3c_batch = cp.take(dm_3c[aux0:aux1], idx, axis=0, out=dm_3c_batch)
        dm_oo_batch = cp.take(dm_oo[aux0:aux1], idx, axis=0, out=dm_oo_batch)
        if j_factor != 0:
            auxvec_batch = auxvec[aux0:aux1][idx]

        counts = np.bincount(atm_id_for_aux[aux0:aux1])
        # when nocc/nao is large and the system contains only 2-3 atoms,
        # the remaining space in work  might be insufficient to cache a tensor
        # of the shape (3,counts.max(),nocc,nocc)
        oo_buf = None
        if work1.size > 6*counts.max()*nocc**2:
            oo_buf = work1

        p0 = p1 = 0
        for i, count in enumerate(counts):
            if count == 0:
                continue
            p0, p1 = p1, p1 + count
            if mem_sufficient:
                auxvec_ipaux = contract('xrnip,npi->xr', j3c_aux[:,p0:p1], orbo)
                contract('xrnip,rniq->xnpq', j3c_aux[:,p0:p1], dm_3c_batch[p0:p1],
                         -k_factor, beta=1, out=vhf_atm_ao[i])
            else:
                tmp = ndarray((3,p1-p0,2,nocc,nocc), buffer=oo_buf)
                j3c_1 = contract('xrniq,nqj->xrnij', j3c_aux[:,p0:p1], orbo, out=tmp)
                auxvec_ipaux = cp.einsum('xrnii->xr', j3c_1)
                contract('rniq,xrnij->xnjq', dm_3c_batch[p0:p1], j3c_1,
                         -k_factor, beta=1, out=vhf_atm[i])
                contract('xrniq,rnij->xnjq', j3c_aux[:,p0:p1], dm_oo_batch[p0:p1],
                         -k_factor, beta=1, out=vhf_atm[i])
            if j_factor != 0:
                contract('rniq,xr->xniq', dm_3c_batch[p0:p1], auxvec_ipaux,
                         j_factor, beta=1, out=vhf_atm[i])
                contract('xrniq,r->xniq', j3c_aux[:,p0:p1], auxvec_batch[p0:p1],
                         j_factor, beta=1, out=vhf_atm[i])
        t1 = log.timer_debug1(f'vhf_atm batch {kbatch}', *t1)
    dm_oo = dm_3c = None
    t0 = log.timer_debug1('fill_int3c2e_ip1 and fill_int3c2e_ipaux', *t0)

    if mem_sufficient:
        transpose_sum(vhf_atm_ao.reshape(-1,nao,nao), inplace=True)
        contract('nxspq,sqj->nxsjp', vhf_atm_ao, orbo, beta=1, out=vhf_atm)

    # (10|0)(0|0)(0|00)
    # Distribute <d/dR i|Veff|j> to derivatives on atoms
    for i, (p0, p1) in enumerate(aoslices[:,2:]):
        contract('xnpq,npi->xniq', vhf1[:,:,p0:p1], orbo[:,p0:p1], beta=1, out=vhf_atm[i])
        contract('xnpq,nqi->xnip', vhf1[:,:,p0:p1], orbo, beta=1, out=vhf_atm[i,:,:,:,p0:p1])

    vhf_atm = contract('nxsjp,spi->snxij', vhf_atm, mo_coeff)
    vhfa = vhf_atm[0][:,:,:,mask[0,mask_sf]]
    vhfb = vhf_atm[1][:,:,:,mask[1,mask_sf]]
    return [vhfa, vhfb]

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = log.init_timer()

    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    mol = mf.with_df.mol
    auxmol = mf.with_df.auxmol
    mf.with_df.reset() # Release GPU memory
    int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    ejk = _jk_energy_per_atom(int3c2e_opt, dm0, verbose=log)
    t1 = log.timer_debug1('two-electron contribution', *time0)

    # Energy weighted density matrix
    dme0_sf  = (mo_coeff[0] * (mo_occ[0]*mo_energy[0])).dot(mo_coeff[0].T)
    dme0_sf += (mo_coeff[1] * (mo_occ[1]*mo_energy[1])).dot(mo_coeff[1].T)
    dm0_sf = dm0[0] + dm0[1]

    e1 = df_rhf_hess._hcore_energy(hessobj, dm0_sf, dme0_sf)
    log.timer_debug1('hcore contribution', *t1)
    log.timer('RHF partial hessian', *time0)
    return e1 + ejk

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mf = hessobj.base
    mol = mf.with_df.mol
    auxmol = mf.with_df.auxmol
    mf.with_df.reset() # Release GPU memory
    natm = mol.natm
    assert atmlst is None or len(atmlst) == natm
    # h1mo = h1 + vj - 0.5 * vk
    opt = Int3c2eOpt(mol, auxmol).build()
    h1mo = _get_veff(opt, mo_coeff, mo_occ, verbose=verbose)
    gobj = hessobj.base.nuc_grad_method()
    h1mo[0] += rhf_grad.get_grad_hcore(gobj, mo_coeff[0], mo_occ[0])
    h1mo[1] += rhf_grad.get_grad_hcore(gobj, mo_coeff[1], mo_occ[1])
    return h1mo

class Hessian(uhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    _keys = {'auxbasis_response',}

    auxbasis_response = 2
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1
    get_jk_mo = df_rhf_hess._get_jk_mo
