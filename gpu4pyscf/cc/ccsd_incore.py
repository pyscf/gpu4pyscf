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


'''
Rewrite the pyscf/cc/ccsd.py using cupy, and GPU for ERIs.
This implementation requires that the GPU memory is large enough to hold at
least two t2 tensors.
'''

import math
import ctypes
import cupy
import cupy as cp
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf import __config__
from gpu4pyscf.lib.cupy_helper import (
    load_library, get_avail_mem, fill_symmetric, contract)
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import int4c2e

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

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
    mol = int4c2e.SortedMole.from_mol(mycc.mol, decontract=False)
    vhfopt = int4c2e._VHFOpt(mol).build()

    nocc, nvir = t1.shape
    nocc2 = nocc*(nocc+1)//2

    uniq_l = mol.uniq_l_ctr[:,0]
    l_ctr_bas_loc = np.append(0, np.cumsum(mol.l_ctr_counts))
    bas_pair_cache = {k: [cp.asarray(x) for x in v]
                      for k, v in vhfopt.bas_pair_cache.items()}

    mo = mol.apply_C_dot(mycc.mo_coeff)
    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])
    t1po = orbv.dot(cupy.asarray(t1).T)
    tau = make_tau_tril(t1, t2)
    x2 = contract('xab,pa->xpb', tau, orbv)
    x2 = contract('xpb,qb->xpq', x2, orbv)
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

    if uniq_l.max() <= int4c2e.LMAX:
        # Computing ERIs on GPU
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        carts = [cp.arange(n) for n in nf]

        ao_loc_gpu = cp.asarray(ao_loc)
        bas_ij_idx = []
        pair_loc = []
        ao_pair_addresses = []
        ao_pair_offsets = {}
        p0 = p1 = 0
        for i, j in bas_pair_cache:
            bas_ij = bas_pair_cache[i, j][0]
            bas_ij_idx.append(bas_ij)
            pair_loc.append(cp.arange(len(bas_ij)+0, dtype=np.int32) * (nf[i] * nf[j]))
            ish, jsh = divmod(bas_ij, mol.nbas)
            iaddr = ao_loc_gpu[ish,None] + carts[i]
            jaddr = ao_loc_gpu[jsh,None] + carts[j]
            ao_pair_addresses.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
            p0, p1 = p1, p1 + len(ao_pair_addresses[-1])
            ao_pair_offsets[i, j] = (p0, p1)
        bas_ij_idx = cp.hstack(bas_ij_idx, dtype=np.int32)
        ao_pair_addresses = cp.hstack(ao_pair_addresses, dtype=np.int32)
        iaddr, jaddr = divmod(ao_pair_addresses, nao)
        nao_pairs = len(ao_pair_addresses)
        log_cutoff = math.log(vhfopt.direct_scf_tol)

        workers = int4c2e.gpu_specs['multiProcessorCount']
        # An additional integer to count for the proccessed pair_ijs
        pool = cp.empty(workers*int4c2e.QUEUE_DEPTH+1, dtype=np.int32)
        rys_envs = vhfopt.rys_envs
        kern = int4c2e.libvhf_rys.RYS_fill_int4c2e
        bas_pair_keys = list(bas_pair_cache.keys())

        def fint(ish0, ish1, jsh0, jsh1, kl_id):
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            di = i1 - i0
            dj = j1 - j0
            eri = cupy.zeros([di,nao,dj,nao])

            k, l = bas_pair_keys[kl_id]
            pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k, l]
            npairs_kl = pair_kl_mapping.size
            if npairs_kl == 0:
                return eri
            nao_kl_pairs = npairs_kl * nf[k] * nf[l]
            eri_tmp = cp.empty((nao_pairs, nao_kl_pairs))
            ij0 = ij1 = 0
            for ij_id, (i, j) in enumerate(bas_pair_cache):
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
                npairs_ij = pair_ij_mapping.size
                if npairs_ij == 0:
                    continue
                ij0, ij1 = ij1, ij1 + npairs_ij * nf[i] * nf[j]
                err = kern(
                    ctypes.cast(eri_tmp[ij0:ij1].data.ptr, ctypes.c_void_p),
                    ctypes.c_int(nao_kl_pairs),
                    ctypes.c_double(0.),
                    ctypes.byref(rys_envs), (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(int4c2e.shm_size),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_loc[ij_id].data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_loc[kl_id].data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    mol._bas.ctypes)
                if err != 0:
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    raise RuntimeError(f'fill_int4c2e kernel for {llll} failed')
            eri_tmp = fill_symmetric(eri_tmp, ao_pair_addresses, nao)

            p0, p1 = ao_pair_offsets[k, l]
            i = iaddr[p0:p1] - i0
            j = jaddr[p0:p1] - j0
            if ish0 == jsh0:
                eri_tmp = fill_symmetric(eri_tmp.reshape(nao**2,-1).T, i*dj+j, di)
                eri[:] = eri_tmp.reshape(di,di,nao,nao).transpose(0,2,1,3)
            else:
                eri_tmp = eri_tmp.transpose(2,0,1)
                eri[j,:,i] = eri_tmp
                eri[i,:,j] = eri_tmp
            return eri
    else:
        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            if ish0 != jsh0:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
                aoblk = np.empty((i1-i0,nao,j1-j0,nao))
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
            else:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
                eri = lib.unpack_tril(eri, axis=0)
                aoblk = np.empty((i1-i0,nao,i1-i0,nao))
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, i0, i1),
                                       ctypes.c_int(nao))
            return cupy.asarray(aoblk)

    wVVoo = np.zeros((nao,nao,nocc,nocc))
    wVvoO = np.zeros((nao,nao,nocc,nocc))

    for ij_id, (i, j) in enumerate(bas_pair_cache):
        ish0, ish1, jsh0, jsh1 = l_ctr_bas_loc[[i, i+1, j, j+1]]
        aoblk = fint(ish0, ish1, jsh0, jsh1, ij_id)

        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        contract_vvvv_(aoblk, i0, i1, j0, j1)

        #:fvv += 2*np.einsum('kc,kcab->ab', t1, eris_ovvv)
        #:fvv -= np.einsum('kc,kbca->ab', t1, eris_ovvv)
        pppo = contract('prqs,si->prqi', aoblk, orbo)
        wVvoO[j0:j1] += contract('prqi,pj->qrij', pppo, t1po[i0:i1]).get()
        wVVoo[i0:i1,j0:j1] = contract('prqi,rj->pqij', pppo, t1po).get()
        pppo = None

        if ish0 != jsh0:
            wVVoo[j0:j1,i0:i1] = wVVoo[i0:i1,j0:j1].transpose(1,0,2,3)
            tmp = contract('prqs,ri->piqs', aoblk, orbo)
            wVvoO[i0:i1] += contract('piqs,qj->psij', tmp, t1po[j0:j1]).get()

        aoblk = None
    x2 = None

    #:t1new += 2*lib.einsum('edac,ikcd->ikea', eris_ovvv, t2)
    #:t1new +=  -lib.einsum('edac,ikdc->ikea', eris_ovvv, t2)
    Ht2full = _unpack_t2_tril(Ht2ao, nocc, nao)
    t1tmp  = contract('ijpq,qj->ip', Ht2full, orbo) * 2
    t1tmp -= contract('ijqp,qj->ip', Ht2full, orbo)
    t1new = t1tmp.dot(orbv).get()

    # vvvv-t2 contractions back to MO repr.
    Ht2tril = contract('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril = contract('xaq,qb->xab', Ht2tril, orbv)

    # part of ovvv-t2 contractions back to MO repr.
    #: tmp = np.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
    #: t2new -= tmp + tmp.transpose(1,0,3,2)
    t1pv = orbo.dot(cupy.asarray(t1))
    tmp = contract('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril -= contract('xaq,qb->xab', tmp, t1pv)

    tmp = contract('xpq,pa->xaq', Ht2ao, t1pv)
    Ht2tril -= contract('xaq,qb->xab', tmp, orbv)#contract('xpq,pa,qb->xab', Ht2ao, t1pv, orbv)

    t2new = _unpack_t2_tril(Ht2tril, nocc, nvir).get()
    Ht2ao = Ht2full = None

    c = vhfopt.coeff.get()
    wpq = 2 * lib.einsum('pqkk,pi,qj->ij', wVVoo, c, c)
    wpq -= lib.einsum('pqkk,pi,qj->ji', wVvoO, c, c)

    tmp = contract('pqji,qb->pbji', cupy.asarray(wVvoO), orbv)
    wVOov = contract('pbji,pa->bjia', tmp, orbv).get()
    #wVOov = contract('pqji,qb,pa->bjia', cupy.asarray(wVvoO), orbv, orbv).get()

    tmp = contract('pqji,pa->aqji', cupy.asarray(wVVoo), -orbv)
    wVooV = contract('aqji,qb->bjia', tmp, orbv).get()
    #wVooV = contract('pqji,pa,qb->bjia', cupy.asarray(wVVoo),-orbv, orbv).get()
    wVVoo = None
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

def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    sorted_mol = int4c2e.SortedMole.from_mol(mol, decontract=False)
    vhfopt = int4c2e._VHFOpt(sorted_mol).build()

    mo_coeff = cupy.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc

    nao_cart = mo_coeff.shape[0]
    mem_avail = get_avail_mem()
    blksize = max(BLKMIN, int(min((nao_cart+3)/4, (mem_avail*.5/8/nao_cart**2)**.5)))
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    mo = sorted_mol.apply_C_dot(mo_coeff)
    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])

    eri = int4c2e._ao2mo_general(vhfopt, [orbo,orbo,mo,mo])
    eris.oooo = eri[:,:,:nocc,:nocc].get()
    eris.ovoo = eri[:,:,:nocc,nocc:].transpose(2,3,0,1).get()
    eris.oovv = eri[:,:,nocc:,nocc:].get()

    eris.ovov = int4c2e._ao2mo_general(vhfopt, [orbo,orbv,orbo,orbv]).get()
    eris.ovvo = eris.ovov.transpose(0,1,3,2)
    log.timer('CCSD integral transformation', *cput0)
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

