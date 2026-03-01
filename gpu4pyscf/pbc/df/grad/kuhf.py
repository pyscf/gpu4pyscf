#!/usr/bin/env python
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
from pyscf.pbc.tools.k2gamma import double_translation_indices
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, unpack_tril
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.pbc.df.int3c2e import (
    libpbc, diffuse_exps_by_atom, _aggregate_bas_idx, POOL_SIZE)
from gpu4pyscf.pbc.tools.pbc import get_coulG, _Gv_wrap_around
from gpu4pyscf.pbc.df import ft_ao, aft_jk
from gpu4pyscf.pbc.df.grad.krhf import (
    _split_l_ctr_pattern, get_ao_pair_loc, int3c2e_scheme, factorize_dm,
    _j_energy_per_atom)
from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_LR
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt, _estimate_sr_2c2e_rcut
from gpu4pyscf.pbc.grad import kuhf as kuhf_grad
from gpu4pyscf.pbc.grad.krhf import contract_h1e_dm
from gpu4pyscf.gto.mole import groupby
from gpu4pyscf.pbc.lib.kpts_helper import (
    fft_matrix, kk_adapted_iter, conj_images_in_bvk_cell)

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, dm, kpts=None, hermi=0, j_factor=1., k_factor=1.,
                        exxdiv=None, with_long_range=True, verbose=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.
    '''
    if kpts is None:
        kpts = np.zeros((1, 3))
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        return _j_energy_per_atom(int3c2e_opt, dm[0]+dm[1], hermi,
                                  with_long_range, verbose) * j_factor

    assert hermi == 1 or hermi == 2
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    assert dm.ndim == 4
    dm_factor_l, dm_factor_r = factorize_dm(dm, hermi)
    # transform to the AO order in sorted_cell
    dm_factor_l = cell.apply_C_dot(dm_factor_l, axis=2)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l.conj()
    else:
        dm_factor_r = cell.apply_C_dot(dm_factor_r, axis=2)
    nkpts, nao, nocc = dm_factor_l.shape[1:]
    naux = auxcell.nao

    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)
    i_addr, j_addr = divmod(pair_addresses, bvk_ncells*nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxcell.ao_loc
    naux = int(aux_loc[-1])

    assert nkpts == len(kpts)
    expLk = cp.exp(1j*cp.asarray(int3c2e_opt.bvkmesh_Ls.dot(kpts.T)))
    expLk_conj = expLk.conj()
    expLk_conjz = expLk_conj.view(np.float64).reshape(bvk_ncells,nkpts,2)

    mem_free = cp.cuda.runtime.memGetInfo()[0]
    buffer_size = mem_free // 4
    batch_size = max(1, min(naux, buffer_size // (nao_pair*8*bvk_ncells)))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, buffer_size // ((nao*bvk_ncells)**2*8)))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    # k=ijk_conserv[i,j] provides: -i + j - k = 2n\pi
    # therefore, i=ijk_conserv[k,j]
    ijk_conserv = double_translation_indices(int3c2e_opt.bvk_kmesh)
    #for ki in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ki,kj] += j3c_tmp[ijk_conserv[ki,kj],ki]
    #        => order_KI = argsort([ki,ijk_conserv[ki,kj]])
    order_KI = cp.empty(nkpts**2, dtype=int)
    order_KI[(ijk_conserv * nkpts + np.arange(nkpts)).ravel()] = cp.arange(nkpts**2)
    #for kk in range(nkpts):
    #    for kj in range(nkpts):
    #        out[ijk_conserv[kk,kj],kj] += j3c_tmp[kk,kj]
    #        => order_KJ = [ijk_conserv[kk,kj],kj]
    order_KJ = cp.asarray((ijk_conserv * nkpts + np.arange(nkpts)).ravel())

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao*bvk_ncells*nao,blksize,nkpts), dtype=np.complex128)
    buf = cp.empty((bvk_ncells*batch_size, nao_pair))
    buf1 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    buf2 = cp.empty(((nao*bvk_ncells)**2*blksize), dtype=np.complex128)
    j3c_oo = cp.empty((2, naux, nkpts, nkpts, nocc, nocc), dtype=np.complex128)
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        compressed = contract('trL,LKz->trKz', compressed, expLk_conjz)
        compressed = compressed.view(np.complex128)[:,:,:,0]
        # *.5 because diagonal blocks are accessed twice
        compressed[diag_idx] *= .5
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            # TODO: decompress the j3c tensor using rsdf_builder._unpack_cderi_v2
            j3c = j3c_full[:,:dk]
            j3c[pair_addresses] = compressed[:,k0:k1]
            j3c = j3c.reshape(nao, bvk_ncells, nao, dk, nkpts)

            j3c_ij = ndarray((nkpts*nkpts, nao*nao*dk), dtype=np.complex128, buffer=buf1)
            j3c_tmp = ndarray((nkpts,nkpts, nao,nao,dk), dtype=np.complex128, buffer=buf2)
            j3c_tmp = contract('jLikK,LI->KIijk', j3c, expLk_conj, out=j3c_tmp)
            j3c_ij[order_KI] = j3c_tmp.reshape(nkpts**2,-1)
            j3c_tmp = contract('iLjkK,LJ->KJijk', j3c, expLk, out=j3c_tmp)
            j3c_ij[order_KJ] += j3c_tmp.reshape(nkpts**2,-1)
            j3c_ij = j3c_ij.reshape(nkpts, nkpts, nao, nao, dk)

            tmp = ndarray((nkpts, nkpts, nocc, nao, dk), dtype=np.complex128, buffer=buf2)
            contract('IJpqr,Ipi->IJiqr', j3c_ij, dm_factor_r[0], out=tmp)
            contract('IJiqr,Jqj->rIJij', tmp, dm_factor_l[0], out=j3c_oo[0,aux0:aux1])
            contract('IJpqr,Ipi->IJiqr', j3c_ij, dm_factor_r[1], out=tmp)
            contract('IJiqr,Jqj->rIJij', tmp, dm_factor_l[1], out=j3c_oo[1,aux0:aux1])
    j3c_full = buf = buf1 = buf2 = eval_j3c = None
    compressed = j3c = j3c_tmp = j3c_ij = tmp = None
    t0 = log.timer_debug1('contract dm', *t0)

    kpt_iters = list(kk_adapted_iter(int3c2e_opt.bvk_kmesh))
    uniq_kpts = kpts[[x[0] for x in kpt_iters]]
    nkpts_uniq = len(uniq_kpts)
    omega = abs(int3c2e_opt.omega)
    precision = auxcell.precision * 1e-6
    log.debug('Set 2c2e integrals precision %g', precision)
    auxcell.rcut = _estimate_sr_2c2e_rcut(auxcell, omega, precision)
    int2c2e_opt = Int2c2eOpt(auxcell, int3c2e_opt.bvk_kmesh)
    j2c = int2c2e_opt.int2c2e(uniq_kpts, sort_output=False)
    if j2c.dtype == np.float64:
        j2c = j2c.astype(np.complex128)

    if with_long_range:
        mesh = int3c2e_opt.mesh
        log.debug('mesh for LR coulG %s', mesh)
        ft_opt = ft_ao.FTOpt.from_intopt(int3c2e_opt)
        assert ft_opt.permutation_symmetry
        ft_kern = ft_opt.gen_ft_kernel(transform_ao=False)
        Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)
        ngrids = Gv.shape[0]

        # The _weighted_coulG_LR() for multiple k-points.
        # To ensure the symmetry between conjugated k-points, it is important to
        # wrap around the high-freq Gv.
        assert Gv[0].dot(Gv[0]) == 0
        Gk = (asarray(Gv) + asarray(uniq_kpts)[:,None]).reshape(-1, 3)
        Gk = _Gv_wrap_around(auxcell, Gk, cp.zeros(3), mesh)
        coulG_LR = get_coulG(auxcell, Gv=Gk, omega=omega).reshape(-1, ngrids)
        coulG_LR *= kws
        coulG_LR[0,0] -= np.pi / omega**2 / auxcell.vol
        Gk = Gk.reshape(nkpts_uniq, ngrids, 3)

        mem_avail = cp.cuda.runtime.memGetInfo()[0]
        Gblksize = int(mem_avail//((nao*2+nocc)*nao*16*nkpts))//32*32
        Gblksize = min(Gblksize, ngrids)
        assert Gblksize > 0
        log.debug1('%.3f GB free memory. blksize=%d for LR part',
                   mem_avail*1e-9, Gblksize)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            auxG = ft_ao.ft_ao(auxcell, Gk[:,p0:p1].reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)
            auxGw = auxG.conj()
            auxGw *= coulG_LR[:,p0:p1]
            contract('iKG,jKG->Kij', auxGw, auxG, beta=1, out=j2c)

            permuted_auxGw = auxG
            permuted_auxGw[aux_sorting] = auxGw
            # conj((r|G)^{[0]}) (ij|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                Gpq = ft_kern(Gv[p0:p1], kpts[kp], kpts, kj_idx)
                pqG, Gpq = Gpq.transpose(0,2,3,1), None
                tmp = contract('kpqG,skpi->skiqG', pqG, dm_factor_r)
                ijG = contract('skiqG,skqj->skijG', tmp, dm_factor_l[:,kj_idx])
                j3c_oo[:,:,ki_idx,kj_idx] += contract(
                    'rG,skijG->srkij', permuted_auxGw[:,j2c_idx], ijG)
                if kp != kp_conj:
                    tmp = contract('kqpG,skpi->skiqG', pqG.conj(), dm_factor_r[:,kj_idx])
                    ijG = contract('skiqG,skqj->skijG', tmp, dm_factor_l)
                    j3c_oo[:,:,kj_idx,ki_idx] += contract(
                        'rG,skijG->srkij', permuted_auxGw[:,j2c_idx].conj(), ijG)
                pqG = None
        tmp = ijG = auxG = auxGw = permuted_auxGw = None

    j2c = auxcell.apply_CT_mat_C(j2c)
    j2c_ip1 = auxcell.pbc_intor('int2c2e_ip1', kpts=uniq_kpts)

    aux_coeff = cp.asarray(auxcell.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    j_factor /= nkpts**2
    k_factor /= nkpts**2
    dm_oo = j3c_oo
    ejk = np.zeros((cell.natm, 3))
    buf = cp.empty((2, naux, nkpts, nocc, nocc), dtype=np.complex128)
    buf1 = cp.empty((2, naux, nkpts, nocc, nocc), dtype=np.complex128)
    dm_aux = cp.empty((nkpts_uniq, naux, naux), dtype=np.complex128)
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        metric = aux_coeff.dot(cp.linalg.solve(j2c[j2c_idx], aux_coeff.T))
        j3c_oo_k = j3c_oo[:,:,ki_idx,kj_idx]
        dm_oo_k = contract('uv,svnij->sunij', metric, j3c_oo_k, out=buf)
        dm_oo[:,:,ki_idx,kj_idx] = dm_oo_k
        if kp == 0:
            dm_oo_kconj = dm_oo_k
        elif kp == kp_conj:
            # for kp == kp_conj != 0, dm_oo_kconj and dm_oo_k correspond to
            # the same blocks in dm_oo, which has been updated previously
            dm_oo_kconj = dm_oo[:,:,kj_idx,ki_idx]
        else:
            j3c_oo_k = j3c_oo[:,:,kj_idx,ki_idx]
            dm_oo_kconj = contract('vu,svnij->sunij', metric, j3c_oo_k, out=buf1)
            dm_oo[:,:,kj_idx,ki_idx] = dm_oo_kconj

        beta = 0
        if j_factor != 0 and kp == 0:
            dm = contract('skpi,skqi->kpq', dm_factor_l, dm_factor_r)
            assert all(ki_idx == kj_idx)
            auxvec = cp.einsum('sunii->u', dm_oo_k)
            cp.multiply(auxvec[:,None], auxvec.conj(), out=dm_aux[j2c_idx])
            beta = j_factor

        dm_aux_k = contract('urkij,uskji->rs', dm_oo_k, dm_oo_kconj,
                            alpha=-k_factor, beta=beta, out=dm_aux[j2c_idx])
        j2c_k = asarray(j2c_ip1[j2c_idx])
        dm_aux_k = dm_aux_k[aux_sorting[:,None],aux_sorting]
        if kp == kp_conj:
            ejk += contract_h1e_dm(auxcell, j2c_k, dm_aux_k, hermi=0)
        else:
            ejk += 2 * contract_h1e_dm(auxcell, j2c_k, dm_aux_k, hermi=0)
        metric = j3c_oo_k = dm_oo_k = dm_oo_kconj = dm_aux_k = j2c_k = None
    ejk *= .5
    j2c = j2c_ip1 = j3c_oo = None
    aux_coeff = buf = buf1 = None
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
        buf2 = cp.empty(naux*Gblksize, dtype=np.complex128)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            nGv = p1 - p0
            # the auxliary dimension of dm_oo and dm_aux are regrouped and
            # permuted. Instead of sorting dm_oo (dm_oo[aux_sorting]) and
            # dm_aux, we reorder auxG here.
            auxG = ft_ao.ft_ao(auxcell, Gk[:,p0:p1].reshape(-1,3)).T
            auxG = auxG.reshape(naux, nkpts_uniq, nGv)
            permuted_auxG = cp.empty_like(auxG)
            permuted_auxG[aux_sorting] = auxG
            auxG, permuted_auxG = permuted_auxG, None

            # (ij|r)^{[0]} * metric * (r|G)^{[1]} (ji|G)^{[0]}
            for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
                Gpq = ft_kern(Gv[p0:p1], kpts[kp], kpts, kj_idx)
                pqG, Gpq = Gpq.transpose(0,2,3,1), None

                beta = 0
                dm_auxG = ndarray((naux,nGv), dtype=np.complex128, buffer=buf2)
                if j_factor != 0 and kp == 0:
                    rhoGz = cp.einsum('kpqG,kqp->G', pqG, dm)
                    cp.multiply(auxvec[:,None], rhoGz, out=dm_auxG)
                    beta = j_factor
                # einsum('pqG,pi,qj,rij,Gx,rG->rx', pqG, c, c, dm_oo, 1j*Gv, conj(auxG))
                tmp = contract('kpqG,skpi->skiqG', pqG, dm_factor_r)
                ijG = contract('skiqG,skqj->skijG', tmp, dm_factor_l[:,kj_idx])
                # (ji|r)^{[0]} * metric * (r|G)^{[1]} (G|ij)^{[0]}
                # contracting all [0] order terms -> dm_auxG
                dm_oo_k = dm_oo[:,:,kj_idx,ki_idx]
                contract('srkji,skijG->rG', dm_oo_k, ijG, -k_factor, beta, out=dm_auxG)

                # (ji|r)^{[0]} * metric * -J2c^{[1]} * metric * (ij|s)^{[0]}
                # = -(ji|r)^{[0]} * metric * (r|G)^{[1]} (G|s)^{[0]} * metric * (ij|s)^{[0]}
                contract('sr,sG->rG', dm_aux[j2c_idx], auxG[:,j2c_idx], -1, 1, out=dm_auxG)
                dm_auxG *= coulG_LR[j2c_idx, p0:p1]
                dm_auxG = dm_auxG.view(np.float64)

                # contract to (r|G)^{[1]}
                for i in range(3):
                    ip_auxG = auxG[:,j2c_idx] * (-1j*Gk[j2c_idx,p0:p1,i])
                    ip_auxG = ip_auxG.view(np.float64)
                    if kp != kp_conj:
                        partial_daux[i] += 2 * cp.einsum('ag,ag->a', ip_auxG, dm_auxG)
                    else:
                        partial_daux[i] += cp.einsum('ag,ag->a', ip_auxG, dm_auxG)

                pqG = None

                # (ji|r)^{[0]} * metric * (G|ij)^{[1]} (r|G)^{[0]}
                auxG_conj = auxG[:,j2c_idx].conj()
                auxG_conj *= coulG_LR[j2c_idx,p0:p1]

                # Note: PBC_ft_aopair_ek_ip1 kernel only processes the tril part.
                # dm_oo must be symmetric
                dm_ooG = contract('srkji,rG->skjiG', dm_oo_k, auxG_conj)
                tmp = contract('skjiG,skqi->skjqG', dm_ooG, dm_factor_r)
                beta = 0
                dm_vG = None
                if j_factor != 0 and kp == 0:
                    vG = auxvec.dot(auxG_conj)
                    dm_vG = dm[:,:,:,None] * vG
                    beta = j_factor
                dm_vG = contract('skjqG,skpj->kpqG', tmp, dm_factor_l[:,kj_idx],
                                 -k_factor, beta, out=dm_vG)
                dm_vG = contract('Lk,kpqG->LpqG', expLk, dm_vG)
                if kp != kp_conj:
                    dm_vG *= 2
                dm_vG = cp.asarray(dm_vG, order='C')

                GvT = cp.asarray(Gk[j2c_idx,p0:p1].T.ravel())
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
                    ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p))
                if err != 0:
                    raise RuntimeError('PBC_ft_aopair_ek_ip1 failed')
                dm_oo_k = dm_ooG = tmp = dm_vG = None

        pqG = ijG = tmp = dm_auxG = ip_auxG = None
        auxG = permuted_auxG = auxG_conj = None
        buf = buf1 = buf2 = None
        ft_opt = ft_kern = None

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
    aux_loc = auxcell.ao_loc

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
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, POOL_SIZE), dtype=np.uint32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libpbc.PBCsr_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size*bvk_ncells))
    buf1 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    buf2 = cp.empty((nkpts**2 * blksize*nao*nao), dtype=np.complex128)
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        naux_in_batch = aux_loc[ksh_offsets_cpu[kbatch+1]] - aux_ao_offset
        compressed = ndarray((nao_pair, naux_in_batch, bvk_ncells), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nkpts,nkpts,nao,nao,dk), dtype=np.complex128, buffer=buf2)
            tmp = ndarray((nkpts,nkpts,nocc,nao,dk), dtype=np.complex128, buffer=buf1)
            # Note the commutation of the indices due to the exchange term (ij|ji)
            contract('rJIji,Jqj->IJiqr', dm_oo[0,aux0:aux1], dm_factor_l[0], -k_factor, out=tmp)
            contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_r[0], out=dm_tensor)
            contract('rJIji,Jqj->IJiqr', dm_oo[1,aux0:aux1], dm_factor_l[1], -k_factor, out=tmp)
            contract('IJiqr,Ipi->IJpqr', tmp, dm_factor_r[1], beta=1, out=dm_tensor)
            dm_tensor = dm_tensor.reshape(nkpts**2,nao,nao,dk)
            dm_tensor = dm_tensor[order_KJ].reshape(nkpts,nkpts,nao,nao,dk)
            if j_factor != 0:
                dm_tensor[0] += j_factor * auxvec[aux0:aux1] * dm[:,:,:,None]
            tmp = ndarray((nkpts,nao,nao,dk,bvk_ncells), dtype=np.complex128, buffer=buf2)
            tmp1 = ndarray((nao,bvk_ncells,nao,dk,bvk_ncells), dtype=np.complex128, buffer=buf1)
            dm_tensor = contract('KJpqr,LK->JpqrL', dm_tensor, expLk_conj, out=tmp)
            dm_tensor = contract('JpqrL,NJ->pNqrL', dm_tensor, expLk, out=tmp1)
            dm_tensor = dm_tensor.reshape(-1,dk,bvk_ncells).real
            #:compressed[:,k0:k1] = dm_tensor[cgto_pair_addresses]
            cp.take(dm_tensor, pair_addresses, axis=0, out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk_sr.data.ptr, ctypes.c_void_p),
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
            ctypes.c_int(naux_in_batch * bvk_ncells),
            ctypes.cast(diffuse_exps.data.ptr, ctypes.c_void_p),
            ctypes.cast(diffuse_coefs.data.ptr, ctypes.c_void_p),
            ctypes.cast(atom_aux_exps.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff))
        if err != 0:
            raise RuntimeError('PBCsr_ejk_int3c2e_ip1 failed')
    ejk += ejk_sr.get() * 2
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    return ejk
