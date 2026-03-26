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
from pyscf.gto import ATOM_OF
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.gto.eval_gto import _estimate_rcut
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, transpose_sum, get_avail_mem)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, get_ao_pair_loc, _nearest_power2,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS)
from gpu4pyscf.df.grad.rhf import factorize_dm
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, diffuse_exps_by_atom, _aggregate_bas_idx, POOL_SIZE)
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_LR
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.pbc.tools.pbc import madelung
from gpu4pyscf.__config__ import props as gpu_specs


def _jk_energy_per_atom(int3c2e_opt, dm, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, with_long_range=True, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm, hermi, with_long_range,
                                  verbose) * j_factor

    assert hermi == 1 or hermi == 2
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
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - naux*nocc**2*8 - nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8*bvk_ncells))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.4/(nao**2*2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

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
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    aux_coeff = cp.asarray(auxcell.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    # Adjust the rcut because the default cell.rcut is estimated based on
    # overlap integrals
    omega = abs(int3c2e_opt.omega)
    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        log.debug('mesh for LR coulG %s', mesh)
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft = ft_opt.ft_evaluator(
            compressing=False, cart=True, original_ao_order=False)[0]
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws)
        ngrids = Gv.shape[0]
        Gv = asarray(Gv)

        Gblksize = int(mem_avail//((nao+nocc)*nao*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)
        buf  = cp.empty(max(nao**2,naux)*Gblksize, dtype=np.complex128)
        buf1 = cp.empty(max(nao*nocc,naux)*Gblksize, dtype=np.complex128)
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf1).T
            auxGw = ndarray((naux, nGv), dtype=np.complex128, buffer=buf2)
            cp.multiply(auxG, coulG_LR[p0:p1], out=auxGw)
            auxGw = auxGw.view(np.float64)
            contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)

            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            pqG = eval_ft(Gv[p0:p1], out=buf)
            pqG = pqG.view(np.float64).reshape(nao,nao,nGv*2)
            pqG[j_addr, i_addr] = pqG[i_addr, j_addr]
            tmp = ndarray((nocc,nao,nGv*2), buffer=buf1)
            ijG = ndarray((nocc,nocc,nGv*2), buffer=buf)
            contract('pqG,pi->iqG', pqG, dm_factor_r, out=tmp)
            contract('iqG,qj->ijG', tmp, dm_factor_l, out=ijG)
            permuted_auxGw = ndarray((naux, nGv*2), buffer=buf1)
            permuted_auxGw[aux_sorting] = auxGw
            contract('rG,ijG->rij', permuted_auxGw, ijG, beta=1, out=j3c_oo)
        pqG = tmp = ijG = auxG = auxGw = permuted_auxGw = None

    j2c = auxcell.apply_CT_mat_C(j2c)

    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    j2c = aux_coeff = None
    dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
    metric = j3c_oo = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)
        dm_sorted = dm_factor_l.dot(dm_factor_r.T)

    # (d/dX P|Q) contributions
    if j_factor == 0:
        dm_aux = None
    else:
        dm_aux = auxvec[:,None] * auxvec
    # dm_aux should be symmetric
    dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                      alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
    # ejk = .5 * contract_h1e_dm(auxcell, auxcell.pbc_intor('int2c2e_ip1'), dm_aux)
    ejk = -int2c2e_opt.energy_ip1_per_atom(dm_aux[aux_sorting[:,None], aux_sorting])
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    if with_long_range:
        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        log.debug1('bas_ij_idx=%d shm_size=%d blksize=%d',
                   len(bas_ij_idx), shm_size, Gblksize)

        kern = libpbc.PBC_ft_aopair_ek_ip1
        ejk_lr = cp.zeros((cell.natm, 3))
        partial_daux = cp.zeros((3, naux))
        vG = cp.empty(ngrids, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            # (ij|r)^{[0]} * metric * (r|G)^{[1]} (ji|G)^{[0]}
            pqG = eval_ft(Gv[p0:p1], out=buf)
            pqG = pqG.view(np.float64).reshape(nao,nao,nGv*2)
            pqG[j_addr, i_addr] = pqG[i_addr, j_addr]
            beta = 0
            dm_auxG = ndarray((naux,nGv*2), buffer=buf2)
            if j_factor != 0:
                rhoGz = cp.einsum('pqG,qp->G', pqG, dm_sorted)
                cp.multiply(auxvec[:,None], rhoGz, out=dm_auxG)
                beta = j_factor
            # einsum('pqG,pi,qj,rij,Gx,rG->rx', pqG, c, c, dm_oo, 1j*Gv, conj(auxG))
            tmp = ndarray((nocc,nao,nGv*2), buffer=buf1)
            ijG = ndarray((nocc,nocc,nGv*2), buffer=buf)
            contract('pqG,pi->iqG', pqG, dm_factor_r, out=tmp)
            contract('iqG,qj->ijG', tmp, dm_factor_l, out=ijG)
            # (ji|r)^{[0]} * metric * (r|G)^{[1]} (G|ij)^{[0]}
            # contracting all [0] order terms -> dm_auxG
            contract('rji,ijG->rG', dm_oo, ijG, -.5*k_factor, beta, out=dm_auxG)

            # the auxliary dimension of dm_oo and dm_aux are regrouped and
            # permuted. Instead of sorting dm_oo (dm_oo[aux_sorting]) and
            # dm_aux, we reorder auxG here.
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            permuted_auxG = ndarray((naux, nGv), dtype=np.complex128, buffer=buf1)
            permuted_auxG[aux_sorting] = auxG
            auxG = permuted_auxG

            # (ji|r)^{[0]} * metric * -J2c^{[1]} * metric * (ij|s)^{[0]}
            # = -(ji|r)^{[0]} * metric * (r|G)^{[1]} (G|s)^{[0]} * metric * (ij|s)^{[0]}
            contract('sr,sG->rG', dm_aux, auxG.view(np.float64), -1, 1, out=dm_auxG)
            dm_auxG = dm_auxG.view(np.complex128)
            dm_auxG *= coulG_LR[p0:p1]
            dm_auxG = dm_auxG.view(np.float64)

            # contract to (r|G)^{[1]}.
            # (r|G)^{[1]} = IFT(nabla_A aux) = IFT(-nabla aux) = (iG IFT(aux))
            # Contributions to derivatives are
            # 1/2 * einsum('ag,ag->a', (iG IFT(aux)), dm_auxG).real
            # = 1/2 * einsum('ag,ag->a', conj(-iG FT(aux)), dm_auxG).real
            # = 1/2 *(einsum('ag,ag->a', Re(-iG FT(aux)), Re(dm_auxG))
            #        +einsum('ag,ag->a', Im(-iG FT(aux)), Im(dm_auxG)))
            # The derivatives also include a term that is contracted to (G|r)^{[1]},
            # which is complex conjugated to this term. The overall
            # contributions are
            # 1/2 * einsum('ag,ag->a', (iG IFT(aux)), dm_auxG) + c.c
            # = (einsum('ag,ag->a', Re(-iG FT(aux)), Re(dm_auxG))
            #   +einsum('ag,ag->a', Im(-iG FT(aux)), Im(dm_auxG)))
            ip_auxG = ndarray((naux, nGv), dtype=np.complex128, buffer=buf)
            for i in range(3):
                cp.multiply(auxG, -1j*Gv[p0:p1,i], out=ip_auxG)
                partial_daux[i] += cp.einsum('ag,ag->a', ip_auxG.view(np.float64), dm_auxG)

            # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
            auxG_conj = ndarray((naux, nGv), dtype=np.complex128, buffer=buf2)
            auxG_conj = cp.conj(auxG, out=auxG_conj)
            auxG_conj *= coulG_LR[p0:p1]
            auxG_conj = auxG_conj.view(np.float64)

            # Note: PBC_ft_aopair_ek_ip1 kernel only processes the tril part.
            # dm_oo must be symmetric
            dm_vG = ndarray((nao,nao,nGv*2), buffer=buf)
            dm_ooG = ndarray((nocc**2, nGv*2), buffer=buf)
            tmp = ndarray((nocc,nao,nGv*2), buffer=buf1)
            dm_oo.reshape(naux, nocc*nocc).T.dot(auxG_conj, out=dm_ooG)
            dm_ooG = dm_ooG.reshape(nocc,nocc,nGv*2)
            contract('jiG,qi->jqG', dm_ooG, dm_factor_r, out=tmp)
            beta = 0
            if j_factor != 0:
                vG = auxvec.dot(auxG_conj)
                cp.multiply(dm_sorted[:,:,None], vG, out=dm_vG)
                beta = j_factor
            contract('jqG,pj->pqG', tmp, dm_factor_l, -.5*k_factor, beta, out=dm_vG)
            GvT = cp.asarray(Gv[p0:p1].T.ravel())
            err = kern(
                ctypes.cast(ejk_lr.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_vG.data.ptr, ctypes.c_void_p),
                ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(nbatches_shl_pair),
                ctypes.c_int(nGv),
                ctypes.c_int(shm_size),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ft_opt.permutation_symmetry))
            if err != 0:
                raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')

        pqG = ijG = tmp = dm_auxG = ip_auxG = None
        auxG = permuted_auxG = auxG_conj = dm_ooG = dm_vG = None
        buf = buf1 = buf2 = None
        ft_opt = eval_ft = None

        dims = aux_loc[1:] - aux_loc[:-1]
        atm_id_for_aux = np.repeat(auxcell._bas[:,ATOM_OF], dims)
        partial_daux = partial_daux.T[aux_sorting].get()
        ejk_aux = groupby(atm_id_for_aux, partial_daux, op='sum')
        if len(ejk_aux) < cell.natm:
            ejk[np.unique(atm_id_for_aux)] += ejk_aux
        else:
            ejk += ejk_aux
        ejk += ejk_lr.get() * 2
        log.timer_debug1('LR coulomb', *t0)

    dm_aux = None

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(-1, 54)
    lmax = cell.uniq_l_ctr[:,0].max()
    laux = auxcell.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, 1000000)[0]
    ao_pair_loc = get_ao_pair_loc(cell.uniq_l_ctr[:,0],
                                  int3c2e_opt.bas_ij_cache, cart=True)

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxcell.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxcell.uniq_l_ctr, batch_size)

    ksh_idx = _aggregate_bas_idx(
        l_ctr_aux_offsets, uniq_l_ctr_aux, bvk_ncells, auxcell.nbas)[1]
    ksh_idx += int3c2e_opt.bvkcell.nbas
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu, dtype=np.int32)

    diffuse_exps = cp.asarray(int3c2e_opt.diffuse_exps)
    diffuse_coefs = cp.asarray(int3c2e_opt.diffuse_coefs)
    atom_aux_exps = cp.asarray(diffuse_exps_by_atom(auxcell), dtype=np.float32)
    log_cutoff = math.log(int3c2e_opt.cutoff)

    ejk_sr = cp.zeros((cell.natm, 3))
    ejk_aux_sr = cp.zeros((cell.natm, 3))
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((blksize, nao, nao))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf1)
            tmp = ndarray((nocc,nao,dk), buffer=buf2)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm_sorted[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
            if hermi == 1:
                cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            else:
                dm_tensor1 = ndarray((nao,nao,dk), buffer=buf2)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk_sr.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux_sr.data.ptr, ctypes.c_void_p),
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
    if hermi == 1:
        ejk_sr *= 2.
        ejk_aux_sr *= 2.
    ejk_sr += ejk_aux_sr
    ejk += ejk_sr.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    if (exxdiv == 'ewald' and
        (cell.dimension == 3 or
         (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
        s0 = int1e.int1e_ovlp(cell)
        s1 = int1e.int1e_ipovlp(cell)
        k_dm = contract('pq,qr->pr', dm, s0)
        k_dm = contract('pr,rs->ps', k_dm, dm)
        # The cell object reorders the AOs. s1 and k_dm are stored in the order
        # of the original cell. It's necessary to pass the original cell to
        # contract_h1e_dm
        ejk_ewald = contract_h1e_dm(cell.cell, s1, k_dm, hermi=1)
        # the madelung function by default read the value of cell.omega.
        # cell.omega is not 0, which can lead to incorrect correction for
        # full-range ewald probe charge correction.
        if with_long_range:
            weighted_coulG_at_G0 = madelung(cell, omega=0)
        else:
            weighted_coulG_at_G0 = madelung(cell, omega=-omega)
        # Note the additional minus sign for nabla_A ovlp = -nabla ovlp
        ejk_ewald *= .5 * k_factor * weighted_coulG_at_G0
        ejk += ejk_ewald
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, with_long_range=True,
                       verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    dm = cell.apply_C_mat_CT(dm)
    if hermi != 1:
        dm = transpose_sum(dm, inplace=True)
        dm[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dm, hermi=1)
    t0 = log.timer_debug1('contract dm', *t0)

    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    omega = abs(int3c2e_opt.omega)
    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell)
    j2c = int2c2e_opt.int2c2e(sort_output=False)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        log.debug('mesh for LR coulG %s', mesh)
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        eval_ft = ft_opt.ft_evaluator(
            compressing=True, cart=True, original_ao_order=False)[0]
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws)
        ngrids = Gv.shape[0]
        Gv = asarray(Gv)

        pair_addresses, diag_idx = ft_opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)
        dm_tril = dm.ravel()[pair_addresses]
        dm_tril[diag_idx] *= .5
        dm_tril *= 2

        mem_avail = get_avail_mem(exclude_memory_pool=True)
        nao_pair = len(dm_tril)
        Gblksize = int(mem_avail//((nao_pair+naux*2)*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)

        auxvec_LR = cp.zeros(naux)
        rhoG = cp.empty(ngrids, dtype=np.complex128)
        buf  = cp.empty(max(nao_pair,naux)*Gblksize, dtype=np.complex128)
        buf1 = cp.empty((naux,Gblksize), dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            pqG = eval_ft(Gv[p0:p1], out=buf)
            rhoGz = cp.einsum('pG,p->G', pqG.view(np.float64), dm_tril)
            rhoG[p0:p1] = rhoGz.view(np.complex128)

            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            auxGw = ndarray((naux, nGv), dtype=np.complex128, buffer=buf1)
            cp.multiply(auxG, coulG_LR[p0:p1], out=auxGw)
            auxGw = auxGw.view(np.float64)
            contract('iG,jG->ij', auxG.view(np.float64), auxGw, beta=1, out=j2c)
            auxvec_LR += auxGw.dot(rhoGz)
        auxvec += auxcell.apply_CT_dot(auxvec_LR)
        buf = buf1 = pqG = rhoGz = auxvec_LR = auxG = auxGw = None
        eval_ft = None
        t0 = log.timer_debug1('lr_int2c2e via aft', *t0)

    j2c = auxcell.apply_CT_mat_C(j2c)

    if auxcell.cell.cart:
        raise NotImplementedError
    else:
        auxvec = cp.linalg.solve(j2c, auxvec)
    auxvec = auxcell.C_dot_mat(auxvec)
    j2c = None

    # (d/dX P|Q) contributions
    dm_aux = auxvec[:,None] * auxvec
    ej = -int2c2e_opt.energy_ip1_per_atom(dm_aux)
    dm_aux = None
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)

    if with_long_range:
        Gblksize = int(mem_avail//(naux*2*16))//32*32
        Gblksize = min(Gblksize, ngrids)
        partial_daux = cp.zeros((3, naux))
        vG = cp.empty(ngrids, dtype=np.complex128)
        buf = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            auxG = ft_ao.ft_ao(auxcell, Gv[p0:p1], out=buf).T
            rho_auxG = auxvec.dot(auxG.view(np.float64))
            vG[p0:p1] = rho_auxG.view(np.complex128)
            vG[p0:p1] *= coulG_LR[p0:p1]

            # (ii|r)^{[0]} * metric * (r|G)^{[1]} (jj|G)^{[0]}
            # = auxvec * (r|G)^{[1]} (jj|G)^{[0]}
            # IFT(nabla_A aux) = IFT(-nabla aux) = (iG IFT(aux)) = (iG conj(FT(aux)))
            ip_vG = rhoG[p0:p1] * coulG_LR[p0:p1] * 1j * Gv[p0:p1].T
            # (ii|r)^{[0]} * metric * -J2c^{[1]} * metric * (jj|r)^{[0]}
            # = auxvec * J2c^{[1]} * auxvec
            ip_vG -= vG[p0:p1] * 1j * Gv[p0:p1].T
            partial_daux += cp.einsum('xg,ag->xa', ip_vG.view(np.float64),
                                      auxG.view(np.float64))
        partial_daux *= auxvec
        buf = auxG = None

        dims = aux_loc[1:] - aux_loc[:-1]
        atm_id_for_aux = np.repeat(auxcell._bas[:,ATOM_OF], dims)
        partial_daux = partial_daux.T.real.get()
        ej_aux = groupby(atm_id_for_aux, partial_daux, op='sum')
        if len(ej_aux) < cell.natm:
            ej[np.unique(atm_id_for_aux)] += ej_aux
        else:
            ej += ej_aux
        t0 = log.timer_debug1('lr_int2c2e_ip1 via aft', *t0)

        ej_lr = cp.zeros((cell.natm, 3))
        vG = vG.conj()
        GvT = cp.asarray(Gv.T.ravel())
        bas_ij_idx, bas_ij_img_idx, shl_pair_offsets = aft_jk._generate_shl_pairs(ft_opt)
        nbatches_shl_pair = len(shl_pair_offsets) - 1
        aft_envs = ft_opt.aft_envs
        shm_size = aft_jk._estimate_max_shm_size(cell, (1, 0))
        err = libpbc.PBC_ft_aopair_ej_ip1(
            ctypes.cast(ej_lr.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.cast(vG.data.ptr, ctypes.c_void_p),
            ctypes.cast(GvT.data.ptr, ctypes.c_void_p),
            ctypes.byref(aft_envs),
            ctypes.c_int(nbatches_shl_pair),
            ctypes.c_int(ngrids),
            ctypes.c_int(shm_size),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_img_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ft_opt.permutation_symmetry))
        if err != 0:
            raise RuntimeError('PBC_ft_aopair_ej_ip1 failed')
        ej += ej_lr.get() * 2
        t0 = log.timer_debug1('lr_int3c2e_ip1 via aft', *t0)
        ft_opt = None

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

    ej_sr = cp.zeros((cell.natm, 3))
    ej_aux_sr = cp.zeros((cell.natm, 3))
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    err = kern(
        ctypes.cast(ej_sr.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux_sr.data.ptr, ctypes.c_void_p),
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
    ej_sr += ej_aux_sr
    ej += ej_sr.get() * 2
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ej

def int3c2e_scheme(omega=0, gout_width=None, shm_size=SHM_SIZE):
    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    order = li + lj + lk + 1
    nroots = (order//2 + 1)
    if omega < 0:
        nroots *= 2
    g_size = (li+2)*(lj+1)*(lk+2)
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
