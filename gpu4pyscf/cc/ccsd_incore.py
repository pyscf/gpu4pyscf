# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''
Rewrite the pyscf/cc/ccsd.py using cupy, and GPU for ERIs.
This implementation requires that the GPU memory is large enough to hold at
least two t2 tensors.
'''

import time
import ctypes
import cupy
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf import __config__
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.lib import logger

FREE_CUPY_CACHE = True

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

libgint = load_library('libgint')
libgint.GINTfill_int2e.restype = ctypes.c_int

def update_amps(mycc, t1, t2, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    orbo = eris.mo_coeff[:,:nocc]
    orbv = eris.mo_coeff[:,nocc:]

    wpq, t1new, t2new, wVOov, wVooV = _direct_ovvv_vvvv(mycc, t1, t2)
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) at the end

    _einsum = cupy.einsum

    fov = fock[:nocc,nocc:].copy()
    t1new += fock[:nocc,nocc:]

    foo = fock[:nocc,:nocc] - np.diag(mo_e_o)
    foo += .5 * np.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = lib.einsum('pa,qp,qb->ab', orbv, wpq, orbv)
    t1new -= lib.einsum('ab,ib->ia', fvv, t1)

    fvv += fock[nocc:,nocc:] - np.diag(mo_e_v)
    fvv -= .5 * np.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    foo += lib.einsum('pi,qp,qj->ij', orbo, wpq, orbo)
    fov += lib.einsum('pi,qp,qa->ia', orbo, wpq, orbv)

    t1, t1_cpu = cupy.asarray(t1), t1
    t2, t2_cpu = cupy.asarray(t2), t2
    tau = _einsum('ia,jb->ijab', t1, t1)
    tau += t2
    woooo = _einsum('ijab,kabl->ijkl', tau, eris.ovvo)
    woooo += cupy.asarray(eris.oooo).transpose(0,2,1,3)
    tmp = _einsum('la,jaik->lkji', t1, eris.ovoo)
    woooo += tmp
    woooo += tmp.transpose(1,0,3,2)
    t2new += .5 * _einsum('ijkl,klab->ijab', woooo, tau).get()
    woooo = tau = None

    wVOov -= lib.einsum('jbik,ka->bjia', eris.ovoo, t1_cpu)
    t2new += wVOov.transpose(1,2,0,3)

    wVooV += lib.einsum('kbij,ka->bija', eris.ovoo, t1_cpu)
    wVooV -= eris.oovv.transpose(2,0,1,3)
    wVOov += wVooV*.5  #: bjia + bija*.5
    wVOov += eris.ovvo.transpose(2,3,0,1)

    t2new += (eris.ovvo*0.5).transpose(0,3,1,2)
    t1new += lib.einsum('pi,pq,qa->ia', orbo, wpq, orbv)

    tmp  = lib.einsum('ic,kjbc->ikjb', t1_cpu, eris.oovv)
    tmp += lib.einsum('jbck,ic->jkib', eris.ovvo, t1_cpu)
    t2new -= lib.einsum('ka,jkib->jiba', t1_cpu, tmp)
    tmp = None

    tau  = t2 * .5
    tau += _einsum('ia,jb->ijab', t1, t1)
    wVooV += _einsum('kbci,jkca->bija', eris.ovvo, tau).get()
    tau = None

    tmp = _einsum('jkca,ckib->jaib', t2, wVooV).get()
    t2new += tmp.transpose(2,0,1,3)
    tmp *= .5
    t2new += tmp.transpose(0,2,1,3)
    tmp = None

    tau  = np.einsum('ia,jb->iajb', t1_cpu*.5, t1_cpu)
    tau += t2_cpu.transpose(0,2,1,3)
    eris_ovOV = eris.ovvo.transpose(0,1,3,2) * 2
    eris_ovOV -= eris.ovvo.transpose(3,1,0,2)
    fvv -= lib.einsum('jcia,jcib->ab', tau, eris_ovOV)
    foo += lib.einsum('iakb,jakb->ij', eris_ovOV, tau)

    theta  = t2.transpose(0,2,1,3) * 2
    theta -= t2.transpose(1,2,0,3)
    tau = theta * .25
    tau -= _einsum('ia,jb->jaib', t1*.5, t1)
    wVOov += _einsum('kcia,kcjb->aijb', eris_ovOV, tau).get()
    eris_ovOV = tau = None

    t2new += _einsum('kcia,ckjb->ijab', theta, wVOov).get()
    theta = wVOov = wVooV = None

    t1new += np.einsum('jb,ijab->ia', fov, t2_cpu) * 2
    t1new -= np.einsum('jb,ijba->ia', fov, t2_cpu)
    ovoo = eris.ovoo * 2
    ovoo -= eris.ovoo.transpose(2,1,0,3)
    t1new -= lib.einsum('jbki,jkba->ia', ovoo, t2_cpu)
    ovoo = None

    ft_ij = foo + np.einsum('ja,ia->ij', .5*t1_cpu, fov)
    ft_ab = fvv - np.einsum('ia,ib->ab', .5*t1_cpu, fov)
    t2new += lib.einsum('ijac,bc->ijab', t2_cpu, ft_ab)
    t2new -= lib.einsum('ki,kjab->ijab', ft_ij, t2_cpu)

    eia = mo_e_o[:,None] - mo_e_v
    t1new += lib.einsum('ib,ab->ia', t1_cpu, fvv)
    t1new -= lib.einsum('ja,ji->ia', t1_cpu, foo)
    t1new /= eia

    t2new = t2new + t2new.transpose(1,0,3,2)
    t2new /= eia[:,None,:,None] + eia[:,None,:]

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new

# Corresponds to the _add_vvvv_tril function in pyscf.cc.ccsd
def _direct_ovvv_vvvv(mycc, t1, t2):
    nocc, nvir = t1.shape
    nocc2 = nocc*(nocc+1)//2
    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory*.9e6-t2.size*4*8)/8/nao_cart**2/3.5)**.5
    Ht2_mem = nocc2*nao_cart**2 * 8 * 2  # x2 and Ht2
    mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * .75)
    if mem_avail * .9 < Ht2_mem:
        raise RuntimeError(
            f'Not enough GPU memory. Available {mem_avail*1e-6} MB, required {Ht2_mem/.9e-6} MB')
    # Reserve some memory for ERIs?
    cupy.get_default_memory_pool().set_limit(mem_avail)

    blksize = max(BLKMIN, int(min((nao_cart+3)/4, blksize,
                                  ((mem_avail-Ht2_mem)*.5/8/nao_cart**2)**.5)))
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    vhfopt = gpu_hf._VHFOpt(mycc.mol, 'int2e')
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mol = vhfopt.mol

    _einsum = cupy.einsum

    mo = vhfopt.coeff.dot(cupy.asarray(mycc.mo_coeff))
    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])
    t1po = orbv.dot(cupy.asarray(t1).T)
    tau = make_tau_tril(t1, t2)
    x2 = _einsum('xab,pa->xpb', tau, orbv)
    x2 = _einsum('xpb,qb->xpq', x2, orbv)
    tau = None

    nao, nmo = mo.shape
    ao_loc = mol.ao_loc
    nao2 = nao * nao

    x2 = cupy.asarray(x2, order='C')
    Ht2ao = cupy.zeros_like(x2)
    _dgemm = cupy.cuda.cublas.dgemm
    handle = cupy.cuda.device.get_cublas_handle()
    N = cupy.cuda.cublas.CUBLAS_OP_N
    T = cupy.cuda.cublas.CUBLAS_OP_T
    one = np.ones(1)
    one_ptr = one.ctypes.data
    x2_ptr = np.int64(x2.data.ptr)
    Ht2ao_ptr = np.int64(Ht2ao.data.ptr)
    def contract_vvvv_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        eri = eri.reshape(-1,jc*nao)
        #:Ht2[:,j0:j1] += np.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm(handle, N, N, jc*nao, nocc2, ic*nao,
               one_ptr, eri.data.ptr, jc*nao, x2_ptr+i0*nao*8, nao2,
               one_ptr, Ht2ao_ptr+j0*nao*8, nao2)

        if i0 > j0:
            #:Ht2[:,i0:i1] += np.einsum('xef,abef->xab', x2[:,j0:j1], eri)
            _dgemm(handle, T, N, ic*nao, nocc2, jc*nao,
                   one_ptr, eri.data.ptr, jc*nao, x2_ptr+j0*nao*8, nao2,
                   one_ptr, Ht2ao_ptr+i0*nao*8, nao2)

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))

    if vhfopt.uniq_l_ctr[:,0].max() <= gpu_hf.LMAX_ON_GPU:
        # Computing ERIs on GPU
        idx, idy = cupy.tril_indices(nao)
        #eribuf = cupy.empty(blksize**2*nao**2)
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            #eri = cupy.ndarray((i1-i0, nao, j1-j0, nao), memptr=eribuf.data)
            #eri.fill(0.)
            eri = cupy.zeros([i1-i0,nao,j1-j0,nao])

            # strides to ensure data order consistent with eri(k1-k0,nao,l1-l0,nao)
            strides = [1, (j1-j0)*nao, (j1-j0)*nao**2, nao]
            ao_offsets = [0, 0, i0, j0]
            _fill_eri_block(eri, strides, ao_offsets, vhfopt, group_id)
            # Fill lower triangular part
            eri[:,idx,:,idy] = eri[:,idy,:,idx]
            return eri
    else:
        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        eribuf = np.empty((blksize,blksize,nao,nao))
        loadbuf = np.empty((blksize,blksize,nao,nao))
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            if ish0 != jsh0:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                aoblk = np.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
            else:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                eri = lib.unpack_tril(eri, axis=0)
                aoblk = np.ndarray((i1-i0,nao,i1-i0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, i0, i1),
                                       ctypes.c_int(nao))
            return cupy.asarray(aoblk)

    wVVoo = np.zeros((nao,nao,nocc,nocc))
    wVvoO = np.zeros((nao,nao,nocc,nocc))

    #mempool = cupy.get_default_memory_pool()
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > gpu_hf.LMAX_ON_GPU or lj > gpu_hf.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi+1]
        jsh1 = l_ctr_offsets[cpj+1]
        aoblk = fint(ish0, ish1, jsh0, jsh1, cp_ij_id)

        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        contract_vvvv_(aoblk, i0, i1, j0, j1)

        #:fvv += 2*np.einsum('kc,kcab->ab', t1, eris_ovvv)
        #:fvv -= np.einsum('kc,kbca->ab', t1, eris_ovvv)
        pppo = _einsum('prqs,si->prqi', aoblk, orbo)
        wVvoO[j0:j1] += _einsum('prqi,pj->qrij', pppo, t1po[i0:i1]).get()
        wVVoo[i0:i1,j0:j1] = _einsum('prqi,rj->pqij', pppo, t1po).get()
        pppo = None

        if ish0 != jsh0:
            wVVoo[j0:j1,i0:i1] = wVVoo[i0:i1,j0:j1].transpose(1,0,2,3)
            #mempool.free_all_blocks()
            tmp = _einsum('prqs,ri->piqs', aoblk, orbo)
            wVvoO[i0:i1] += _einsum('piqs,qj->psij', tmp, t1po[j0:j1]).get()

        aoblk = None
    eribuf = loadbuf = x2 = None

    #:t1new += 2*lib.einsum('edac,ikcd->ikea', eris_ovvv, t2)
    #:t1new +=  -lib.einsum('edac,ikdc->ikea', eris_ovvv, t2)
    Ht2full = _unpack_t2_tril(Ht2ao, nocc, nao)
    t1tmp  = _einsum('ijpq,qj->ip', Ht2full, orbo) * 2
    t1tmp -= _einsum('ijqp,qj->ip', Ht2full, orbo)
    t1new = t1tmp.dot(orbv).get()

    # vvvv-t2 contractions back to MO repr.
    Ht2tril = _einsum('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril = _einsum('xaq,qb->xab', Ht2tril, orbv)

    # part of ovvv-t2 contractions back to MO repr.
    #: tmp = np.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
    #: t2new -= tmp + tmp.transpose(1,0,3,2)
    t1pv = orbo.dot(cupy.asarray(t1))
    tmp = _einsum('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril -= _einsum('xaq,qb->xab', tmp, t1pv)

    tmp = _einsum('xpq,pa->xaq', Ht2ao, t1pv)
    Ht2tril -= _einsum('xaq,qb->xab', tmp, orbv)#_einsum('xpq,pa,qb->xab', Ht2ao, t1pv, orbv)

    t2new = _unpack_t2_tril(Ht2tril, nocc, nvir).get()
    Ht2ao = Ht2full = None

    c = vhfopt.coeff.get()
    wpq = 2 * lib.einsum('pqkk,pi,qj->ij', wVVoo, c, c)
    wpq -= lib.einsum('pqkk,pi,qj->ji', wVvoO, c, c)

    tmp = _einsum('pqji,qb->pbji', cupy.asarray(wVvoO), orbv)
    wVOov = _einsum('pbji,pa->bjia', tmp, orbv).get()
    #wVOov = _einsum('pqji,qb,pa->bjia', cupy.asarray(wVvoO), orbv, orbv).get()

    tmp = _einsum('pqji,pa->aqji', cupy.asarray(wVVoo), -orbv)
    wVooV = _einsum('aqji,qb->bjia', tmp, orbv).get()
    #wVooV = _einsum('pqji,pa,qb->bjia', cupy.asarray(wVVoo),-orbv, orbv).get()
    wVVoo = None

    if FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    return wpq, t1new, t2new, wVOov, wVooV

def make_tau_tril(t1, t2):
    nocc, nvir = t1.shape
    t1 = cupy.asarray(t1)
    tau = cupy.einsum('ia,jb->ijab', t1, t1)
    tau += cupy.asarray(t2)
    return tau[cupy.tril_indices(nocc)]

def _unpack_t2_tril(t2tril, nocc, nvir):
    t2 = cupy.empty((nocc,nocc,nvir,nvir))
    idx,idy = cupy.tril_indices(nocc)
    t2[idy,idx] = t2tril.transpose(0,2,1)
    t2[idx,idy] = t2tril
    return t2

def _fill_eri_block(eri, strides, ao_offsets, vhfopt, group_id):
    log_qs = vhfopt.log_qs
    cp_kl_id = group_id
    log_q_kl = log_qs[cp_kl_id]
    if log_q_kl.size == 0:
        return eri

    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))
    cpk = cp_idx[cp_kl_id]
    cpl = cp_jdx[cp_kl_id]
    lk = vhfopt.uniq_l_ctr[cpk,0]
    ll = vhfopt.uniq_l_ctr[cpl,0]
    if lk > gpu_hf.LMAX_ON_GPU or ll > gpu_hf.LMAX_ON_GPU:
        raise NotImplementedError

    stream = cupy.cuda.get_current_stream()
    log_cutoff = np.log(vhfopt.direct_scf_tol)
    omega = 0.

    l_symb = lib.param.ANGULAR
    nao = vhfopt.coeff.shape[0]
    bins_locs_kl = vhfopt.bins[cp_kl_id]
    bins_floor_kl = vhfopt.bins_floor[cp_kl_id]
    nbins_kl = len(bins_locs_kl) - 1

    fn = libgint.GINTfill_int2e
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > gpu_hf.LMAX_ON_GPU or lj > gpu_hf.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        t0 = time.perf_counter()
        bins_locs_ij = vhfopt.bins[cp_ij_id]
        bins_floor_ij = vhfopt.bins_floor[cp_ij_id]
        nbins_ij = len(bins_locs_ij) - 1

        err = fn(ctypes.cast(stream.ptr, ctypes.c_void_p), vhfopt.bpcache,
                 ctypes.cast(eri.data.ptr, ctypes.c_void_p), ctypes.c_int(nao),
                 (ctypes.c_int*4)(*strides), (ctypes.c_int*4)(*ao_offsets),
                 bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                 bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                 bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
                 bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(nbins_ij), ctypes.c_int(nbins_kl),
                 ctypes.c_int(cp_ij_id), ctypes.c_int(cp_kl_id),
                 ctypes.c_double(log_cutoff),
                 ctypes.c_double(omega))
        if err != 0:
            detail = f'CUDA Error for ({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
            raise RuntimeError(detail)
        logger.debug1(vhfopt.mol, '(%s%s|%s%s) on GPU %.3fs',
                      l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                      time.perf_counter() - t0)
    return eri

def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    # Cupy memory buffer may be created in previous SCF calculations.
    if FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()

    mol = mycc.mol
    mo_coeff = cupy.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc

    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory*.9e6-nocc**2*nao_cart**2*2*8)/8/nao_cart**2/2.5)**.5
    mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * .75)
    cupy.get_default_memory_pool().set_limit(mem_avail)
    blksize = max(BLKMIN, int(min((nao_cart+3)/4, blksize,
                                  (mem_avail*.5/8/nao_cart**2)**.5)))
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    vhfopt = gpu_hf._VHFOpt(mycc.mol, 'int2e')
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mol = vhfopt.mol
    mo = vhfopt.coeff.dot(mo_coeff)
    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])
    ao_loc = mol.ao_loc
    nao = mo.shape[0]

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))

    ppOO = np.empty((nao,nao,nocc,nocc))
    pPoO = np.zeros((nao,nao,nocc,nocc))
    eribuf = cupy.empty(blksize**2*nao**2)
    #mempool = cupy.get_default_memory_pool()
    idx, idy = cupy.tril_indices(nao)

    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > gpu_hf.LMAX_ON_GPU or lj > gpu_hf.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi+1]
        jsh1 = l_ctr_offsets[cpj+1]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        eri = cupy.ndarray((nao, i1-i0, j1-j0, nao), memptr=eribuf.data)
        eri.fill(0.)
        # strides to ensure data order consistent with eri(nao,k1-k0,l1-l0,nao)
        strides = [1, (i1-i0)*(j1-j0)*nao, (j1-j0)*nao, nao]
        ao_offsets = [0, 0, i0, j0]
        _fill_eri_block(eri, strides, ao_offsets, vhfopt, cp_ij_id)
        # Fill lower triangular part
        eri[idx,:,:,idy] = eri[idy,:,:,idx]

        pijo = cupy.dot(eri.reshape(-1,nao), orbo)
        ijoo = cupy.dot(pijo.reshape(nao,-1).T, orbo)
        ppOO[i0:i1,j0:j1] = ijoo.get().reshape(i1-i0,j1-j0,nocc,nocc)
        ijoo = None

        jopi = cupy.asarray(pijo.reshape(nao*(i1-i0),(j1-j0)*nocc).T, order='C')
        jopo = cupy.dot(jopi.reshape(-1,i1-i0), orbo[i0:i1])
        pPoO[j0:j1] += jopo.get().reshape(j1-j0,nocc,nao,nocc).transpose(0,2,1,3)
        pijo = jopo = None

        if ish0 != jsh0:
            ppOO[j0:j1,i0:i1] = ppOO[i0:i1,j0:j1].transpose(1,0,2,3)
            opio = cupy.dot(jopi.reshape(j1-j0,-1).T, orbo[j0:j1])
            pPoO[i0:i1] += opio.get().reshape(nocc,nao,i1-i0,nocc).transpose(2,1,0,3)
            jopi = opio = None

    ppOO = cupy.asarray(ppOO)
    pooo = cupy.dot(ppOO.reshape(nao,-1).T, orbo)
    oooo = cupy.dot(pooo.reshape(nao,-1).T, orbo).reshape(nocc,nocc,nocc,nocc)
    ooov = cupy.dot(pooo.reshape(nao,-1).T, orbv).reshape(nocc,nocc,nocc,nvir)
    eris.oooo = oooo.get()
    eris.ovoo = lib.transpose(ooov.get().reshape(nocc*nocc,nocc*nvir)).reshape(nocc,nvir,nocc,nocc)
    pooo = oooo = ooov = None

    poov = cupy.dot(ppOO.reshape(nao,-1).T, orbv)
    oovv = cupy.dot(poov.reshape(nao,-1).T, orbv).reshape(nocc,nocc,nvir,nvir)
    eris.oovv = oovv.get()
    ppOO = poov = oovv = None

    pPoO = cupy.asarray(pPoO)
    poov = cupy.dot(pPoO.reshape(nao,-1).T, orbv)
    voov = cupy.dot(orbv.T, poov.reshape(nao,-1))
    eris.ovvo = lib.transpose(voov.get().reshape(nvir*nocc,nocc*nvir)).reshape(nocc,nvir,nvir,nocc)
    eris.ovov = eris.ovvo.transpose(0,1,3,2)
    pPoO = poov = voov = None
    log.timer('CCSD integral transformation', *cput0)

    if FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    return eris

class CCSDBase(lib.StreamObject):
    # attributes
    _keys                  = ccsd.CCSDBase._keys
    max_cycle              = ccsd.CCSDBase.max_cycle
    conv_tol               = ccsd.CCSDBase.conv_tol
    iterative_damping      = ccsd.CCSDBase.iterative_damping
    conv_tol_normt         = ccsd.CCSDBase.conv_tol_normt

    diis                   = ccsd.CCSDBase.diis
    diis_space             = ccsd.CCSDBase.diis_space
    diis_file              = None
    diis_start_cycle       = ccsd.CCSDBase.diis_start_cycle
    diis_start_energy_diff = ccsd.CCSDBase.diis_start_energy_diff

    direct                 = ccsd.CCSDBase.direct
    async_io               = None
    incore_complete        = ccsd.CCSDBase.incore_complete
    cc2                    = ccsd.CCSDBase.cc2
    callback               = None

    # functions
    __init__           = ccsd.CCSDBase.__init__
    ecc                = ccsd.CCSDBase.ecc
    e_tot              = ccsd.CCSDBase.e_tot
    nocc               = ccsd.CCSDBase.nocc
    nmo                = ccsd.CCSDBase.nmo
    reset              = ccsd.CCSDBase.reset
    get_nocc           = ccsd.CCSDBase.get_nocc
    get_nmo            = ccsd.CCSDBase.get_nmo
    get_frozen_mask    = ccsd.CCSDBase.get_frozen_mask
    get_e_hf           = ccsd.CCSDBase.get_e_hf
    set_frozen         = ccsd.CCSDBase.set_frozen
    dump_flags         = ccsd.CCSDBase.dump_flags
    get_init_guess     = ccsd.CCSDBase.get_init_guess
    init_amps          = ccsd.CCSDBase.init_amps
    energy             = ccsd.CCSDBase.energy
    _add_vvvv          = ccsd.CCSDBase._add_vvvv
    update_amps        = update_amps
    kernel             = ccsd.CCSDBase.kernel
    _finalize          = ccsd.CCSDBase._finalize
    as_scanner         = ccsd.CCSDBase.as_scanner
    restore_from_diis_ = ccsd.CCSDBase.restore_from_diis_

    solve_lambda         = NotImplemented
    ccsd_t               = NotImplemented
    ipccsd               = NotImplemented
    eaccsd               = NotImplemented
    eeccsd               = NotImplemented
    eomee_ccsd_singlet   = NotImplemented
    eomee_ccsd_triplet   = NotImplemented
    eomsf_ccsd           = NotImplemented
    eomip_method         = NotImplemented
    eomea_method         = NotImplemented
    eomee_method         = NotImplemented
    make_rdm1            = NotImplemented
    make_rdm2            = NotImplemented
    ao2mo                = _make_eris_incore
    run_diis             = ccsd.CCSDBase.run_diis
    amplitudes_to_vector = ccsd.CCSDBase.amplitudes_to_vector
    vector_to_amplitudes = ccsd.CCSDBase.vector_to_amplitudes
    dump_chk             = None
    density_fit          = NotImplemented
    nuc_grad_method      = NotImplemented

    # to_cpu can be reused only when __init__ still takes mf
    def to_cpu(self):
        mf = self._scf.to_cpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

CCSDBase.ccsd = ccsd.CCSDBase.ccsd

class CCSD(CCSDBase):
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, mf, *args, **kwargs):
        if hasattr(mf, 'to_cpu'):
            mf = mf.to_cpu()
        if hasattr(mf, 'with_df') and mf.with_df:
            lib.logger.warn(mf.mol, 'DF-CCSD not available. Run the standard CCSD.')
        ccsd.CCSD.__init__(self, mf, *args, **kwargs)

