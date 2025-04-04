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

import ctypes
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cupy
from gpu4pyscf.df import int3c2e
from gpu4pyscf.scf.int4c2e import libgint
from gpu4pyscf.hessian.jk import _ao2mo
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract, cart2sph, reduce_to_device
from gpu4pyscf.__config__ import _streams, num_devices

NROOT_ON_GPU = 7

def _jk_task_with_mo1(dfobj, dms, mo_coeff, mo1s, occ_coeffs,
                      with_j=True, with_k=True, hermi=0, device_id=0):
    ''' Calculate J and K matrices with mo response
        For CP-HF
    '''
    assert hermi == 1
    with cupy.cuda.Device(device_id), _streams[device_id]:
        assert isinstance(dfobj.verbose, int)
        log = logger.new_logger(dfobj.mol, dfobj.verbose)
        t0 = log.init_timer()
        dms = cupy.asarray(dms)
        n_dm = dms.shape[0]
        mo1s = [cupy.asarray(mo1) for mo1 in mo1s]
        occ_coeffs = [cupy.asarray(occ_coeff) for occ_coeff in occ_coeffs]
        mo_coeff = [cupy.asarray(mo) for mo in mo_coeff]
        nao = dms.shape[-1]
        intopt = dfobj.intopt
        rows = intopt.cderi_row
        cols = intopt.cderi_col
        dms_shape = dms.shape
        if with_j:
            dm_sparse = dms[:,rows,cols]
            if hermi == 0:
                dm_sparse += dms[:,cols,rows]
            else:
                dm_sparse *= 2
            dm_sparse[:, intopt.cderi_diag] *= .5
        dms = None
        
        if with_k:
            vks = [cupy.zeros_like(mo1) for mo1 in mo1s]

        if with_j:
            vj_sparse = cupy.zeros_like(dm_sparse)

        nocc = max([mo1.shape[2] for mo1 in mo1s])
        blksize = dfobj.get_blksize(extra=2*nao*nocc)
        for cderi, cderi_sparse in dfobj.loop(blksize=blksize, unpack=with_k):
            if with_j:
                rhoj = dm_sparse.dot(cderi_sparse)
                vj_sparse += cupy.dot(rhoj, cderi_sparse.T)
                rhoj = None
            cderi_sparse = None
            if with_k:
                for occ_coeff, mo1, vk in zip(occ_coeffs, mo1s, vks):
                    nocc = occ_coeff.shape[1]
                    rhok = contract('Lij,jo->Loi', cderi, occ_coeff)
                    rhok_oo = contract('Loi,ip->Lop', rhok, occ_coeff).reshape([-1,nocc])
                    rhok = rhok.reshape([-1,nao])
                    for i in range(mo1.shape[0]):
                        rhok1 = contract('Lij,jo->Loi', cderi, mo1[i])
                        rhok1 = rhok1.reshape([-1,nao])
                        vk[i] += cupy.dot(rhok1.T, rhok_oo)

                        rhok1 = rhok1.reshape([-1,nocc,nao])
                        rhok1 = contract('Loi,ip->Lop', rhok1, occ_coeff)
                        rhok1 = rhok1.reshape([-1,nocc])
                        vk[i] += cupy.dot(rhok.T, rhok1)
                mo1 = rhok1 = rhok = rhok_oo = None
            cderi = None
        mo1s = None
        if with_j:
            vj = cupy.zeros(dms_shape)
            vj[:,rows,cols] = vj_sparse
            vj[:,cols,rows] = vj_sparse

        vj_mo = vk_mo = None
        if len(occ_coeffs) == 1:
            # Restricted case
            mo = mo_coeff[0]
            if with_j:
                vj_mo = _ao2mo(vj, occ_coeffs[0], mo).reshape(n_dm,-1)
                vj = None
            mo *= 2.0     # Due to double occupancy
            if with_k:
                vk_mo = contract('nio,ip->npo', vks[0], mo).reshape(n_dm,-1)
        elif len(occ_coeffs) == 2:
            # Unrestricted case
            n_dm_2 = n_dm // 2
            mocca, moccb = occ_coeffs
            moa, mob = mo_coeff
            nmoa, nmob = moa.shape[1], mob.shape[1]
            nocca, noccb = mocca.shape[1], moccb.shape[1]

            if with_j:
                vjab = vj[:n_dm_2] + vj[n_dm_2:]
                vj = None
                vj_mo = cupy.empty([n_dm_2,nmoa*nocca+nmob*noccb])
                vj_mo[:,:nmoa*nocca] = _ao2mo(vjab, mocca, moa).reshape(n_dm_2,-1)
                vj_mo[:,nmoa*nocca:] = _ao2mo(vjab, moccb, mob).reshape(n_dm_2,-1)
                vjab = None

            if with_k:
                vka, vkb = vks
                vk_mo = cupy.empty([n_dm_2,nmoa*nocca+nmob*noccb])
                vk_mo[:,:nmoa*nocca] = contract('nio,ip->npo', vka, moa).reshape(n_dm_2,-1)
                vk_mo[:,nmoa*nocca:] = contract('nio,ip->npo', vkb, mob).reshape(n_dm_2,-1)

        t0 = log.timer_debug1(f'vj and vk on Device {device_id}', *t0)
    return vj_mo, vk_mo

def get_jk(dfobj, dms_tag, mo_coeff, mocc, hermi=0,
           with_j=True, with_k=True, direct_scf_tol=1e-14, omega=None):
    ''' Compute J/K in MO with density fitting
    '''

    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    if not isinstance(dms_tag, cupy.ndarray):
        dms_tag = cupy.asarray(dms_tag)

    assert(with_j or with_k)
    if dms_tag is None: logger.error("dm is not given")
    nao = dms_tag.shape[-1]
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(direct_scf_tol=direct_scf_tol, omega=omega)
        t1 = log.timer_debug1('init jk', *t0)

    assert nao == dfobj.nao
    intopt = dfobj.intopt

    nao = dms_tag.shape[-1]
    dms = dms_tag.reshape([-1,nao,nao])
    intopt = dfobj.intopt
    dms = intopt.sort_orbitals(dms, axis=[1,2])

    occ_coeffs = dms_tag.occ_coeff
    mo1s = dms_tag.mo1

    if not isinstance(occ_coeffs, (tuple, list)):
        occ_coeffs = [occ_coeffs]
        mo1s = [mo1s]
        mo_coeff = [mo_coeff]
    else:
        assert isinstance(mo1s, (tuple, list))
        mo_coeff = [mo_coeff[0], mo_coeff[1]]

    occ_coeffs = [intopt.sort_orbitals(occ_coeff, axis=[0]) for occ_coeff in occ_coeffs]
    mo1s = [intopt.sort_orbitals(mo1, axis=[1]) for mo1 in mo1s]
    mo_coeff = [intopt.sort_orbitals(mo, axis=[0]) for mo in mo_coeff]

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _jk_task_with_mo1,
                dfobj, dms, mo_coeff, mo1s, occ_coeffs,
                hermi=hermi, device_id=device_id,
                with_j=with_j, with_k=with_k)
            futures.append(future)

    vj = vk = None
    if with_j:
        vj = [future.result()[0] for future in futures]
        vj = reduce_to_device(vj, inplace=True)

    if with_k:
        vk = [future.result()[1] for future in futures]
        vk = reduce_to_device(vk, inplace=True)
    t1 = log.timer_debug1('vj and vk', *t1)
    return vj, vk


def _get_int3c2e_ipip_slice(ip_type, intopt, cp_ij_id, aux_id, omega=None, stream=None):

    if omega is None: omega = 0.0
    if stream is None: stream = cupy.cuda.get_current_stream()

    fn = getattr(libgint, 'GINTfill_int3c2e_' + ip_type)
    nao = intopt._sorted_mol.nao
    naux = intopt._sorted_auxmol.nao
    norb = nao + naux + 1
    comp = 9
    order = 2
    nbins = 1

    cp_kl_id = aux_id + len(intopt.log_qs)
    lk = intopt.aux_angular[aux_id]

    cpi = intopt.cp_idx[cp_ij_id]
    cpj = intopt.cp_jdx[cp_ij_id]
    li = intopt.angular[cpi]
    lj = intopt.angular[cpj]

    i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
    j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
    k0, k1 = intopt.cart_aux_loc[aux_id], intopt.cart_aux_loc[aux_id+1]
    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0

    log_q_ij = intopt.log_qs[cp_ij_id]
    log_q_kl = intopt.aux_log_qs[aux_id]

    bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
    bins_locs_kl = np.array([0, len(log_q_kl)], dtype=np.int32)

    ao_offsets = np.array([i0,j0,nao+1+k0,nao], dtype=np.int32)
    strides = np.array([1, ni, ni*nj, ni*nj*nk], dtype=np.int32)

    # Use GPU kernels for low-angular momentum
    if (li + lj + lk + order)//2 + 1 < NROOT_ON_GPU:
        int3c_blk = cupy.zeros([comp, nk, nj, ni], order='C', dtype=np.float64)
        err = fn(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            intopt.bpcache,
            ctypes.cast(int3c_blk.data.ptr, ctypes.c_void_p),
            ctypes.c_int(norb),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins),
            ctypes.c_int(cp_ij_id),
            ctypes.c_int(cp_kl_id),
            ctypes.c_double(omega))
        if err != 0:
            raise RuntimeError(f'GINT_fill_int3c2e general failed, err={err}')
    else:
        from pyscf.gto.moleintor import getints, make_cintopt
        pmol = intopt._tot_mol
        intor = pmol._add_suffix('int3c2e_' + ip_type)
        opt = make_cintopt(pmol._atm, pmol._bas, pmol._env, intor)

        # TODO: sph2cart in CPU?
        ishl0, ishl1 = intopt.l_ctr_offsets[cpi], intopt.l_ctr_offsets[cpi+1]
        jshl0, jshl1 = intopt.l_ctr_offsets[cpj], intopt.l_ctr_offsets[cpj+1]
        kshl0, kshl1 = intopt.l_ctr_offsets[aux_id+1+intopt.nctr], intopt.l_ctr_offsets[aux_id+1+intopt.nctr+1]
        shls_slice = np.array([ishl0, ishl1, jshl0, jshl1, kshl0, kshl1], dtype=np.int64)
        int3c_cpu = getints(intor, pmol._atm, pmol._bas, pmol._env, shls_slice, cintopt=opt).transpose([0,3,2,1])
        int3c_blk = cupy.asarray(int3c_cpu)

    if not intopt.auxmol.cart:
        int3c_blk = cart2sph(int3c_blk, axis=1, ang=lk)
    if not intopt.mol.cart:
        int3c_blk = cart2sph(int3c_blk, axis=2, ang=lj)
        int3c_blk = cart2sph(int3c_blk, axis=3, ang=li)

    return int3c_blk


def _int3c2e_ipip_tasks(intopt, task_list, rhoj, rhok, dm0, orbo,
                        device_id=0, with_j=True, with_k=True, omega=None,
                        auxbasis_response=1):
    natm = intopt.mol.natm
    nao = dm0.shape[0]
    assert with_j or with_k
    ao_loc = intopt.ao_loc
    aux_ao_loc = intopt.aux_ao_loc
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(intopt.mol, intopt.mol.verbose)
        t0 = log.init_timer()
        orbo = cupy.asarray(orbo)
        dm0 = cupy.asarray(dm0)
        nao = dm0.shape[0]
        if with_j:
            naux = rhoj.shape[0]
            rhoj = cupy.asarray(rhoj)
            hj_ipip1 = cupy.zeros([9,nao])
            hj_ipip2 = cupy.zeros([9,naux])
            hj_ip1ip2 = cupy.zeros([9,nao,naux])
            hj_ipvip1 = cupy.zeros([9,nao,nao])
        if with_k:
            naux = rhok.shape[0]
            rhok = cupy.asarray(rhok)
            hk_ipip1 = cupy.zeros([9,nao])
            hk_ipip2 = cupy.zeros([9,naux])
            hk_ip1ip2 = cupy.zeros([9,nao,naux])
            hk_ipvip1 = cupy.zeros([9,nao,nao])

        cupy.get_default_memory_pool().free_all_blocks()
        for aux_id, cp_ij_id in task_list:
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            i0, i1 = ao_loc[cpi], ao_loc[cpi+1]
            j0, j1 = ao_loc[cpj], ao_loc[cpj+1]
            k0, k1 = aux_ao_loc[aux_id], aux_ao_loc[aux_id+1]

            if with_k:
                rhok_tmp = contract('por,ir->poi', rhok[k0:k1], orbo[i0:i1])
                rhok_tmp = contract('poi,jo->pji', rhok_tmp, orbo[j0:j1])
            
            # (20|0), (0|0)(0|00)
            int3c_blk = _get_int3c2e_ipip_slice('ipip1', intopt, cp_ij_id, aux_id, omega=omega)
            if with_j:
                tmp = contract('xpji,p->xji', int3c_blk, rhoj[k0:k1])
                hj_ipip1[:,i0:i1] += contract('xji,ij->xi', tmp, dm0[i0:i1,j0:j1])
            if with_k:
                hk_ipip1[:,i0:i1] += contract('xpji,pji->xi', int3c_blk, rhok_tmp)
            int3c_blk = tmp = None

            # (11|0), (0|0)(0|00) without response of RI basis
            int3c_blk = _get_int3c2e_ipip_slice('ipvip1', intopt, cp_ij_id, aux_id, omega=omega)
            if with_j:
                tmp = contract('xpji,p->xji', int3c_blk, rhoj[k0:k1])
                hj_ipvip1[:,i0:i1,j0:j1] += contract('xji,ij->xij', tmp, dm0[i0:i1,j0:j1])
            if with_k:
                hk_ipvip1[:,i0:i1,j0:j1] += contract('xpji,pji->xij', int3c_blk, rhok_tmp)
            int3c_blk = tmp = None

            if auxbasis_response < 1:
                continue

            # (10|1), (0|0)(0|00)
            int3c_blk = _get_int3c2e_ipip_slice('ip1ip2', intopt, cp_ij_id, aux_id, omega=omega)
            if with_j:
                tmp = contract('xpji,ij->xpi', int3c_blk, dm0[i0:i1,j0:j1])
                hj_ip1ip2[:,i0:i1,k0:k1] += contract('xpi,p->xip', tmp, rhoj[k0:k1])
            if with_k:
                hk_ip1ip2[:,i0:i1,k0:k1] += contract('xpji,pji->xip', int3c_blk, rhok_tmp)
            int3c_blk = tmp = None

            if auxbasis_response < 2:
                continue

            # (00|2), (0|0)(0|00)
            int3c_blk = _get_int3c2e_ipip_slice('ipip2', intopt, cp_ij_id, aux_id, omega=omega)
            if with_j:
                tmp = contract('xpji,ij->xp', int3c_blk, dm0[i0:i1,j0:j1])
                hj_ipip2[:,k0:k1] += contract('xp,p->xp', tmp, rhoj[k0:k1])
            if with_k:
                hk_ipip2[:,k0:k1] += contract('xpji,pji->xp', int3c_blk, rhok_tmp)
            int3c_blk = tmp = None
        auxslices = intopt.auxmol.aoslice_by_atom()
        aoslices = intopt.mol.aoslice_by_atom()
        ao2atom = int3c2e.get_ao2atom(intopt, aoslices)
        aux2atom = int3c2e.get_aux2atom(intopt, auxslices)

        hj = None
        if with_j:
            hj_ipvip1 = hj_ipvip1.reshape([3,3,nao,nao])
            tmp = contract('ia,xyij->ajxy', ao2atom, hj_ipvip1)
            hj = 2.0 * contract('jb,ajxy->abxy', ao2atom, tmp)

            hj_ipip1 = hj_ipip1.reshape([3,3,nao])
            tmp = contract('ia,xyi->axy', ao2atom, hj_ipip1)
            hj[range(natm), range(natm)] += 2.0 * tmp

        hk = None
        if with_k:
            hk_ipvip1 = hk_ipvip1.reshape([3,3,nao,nao])
            tmp = contract('ia,xyij->ajxy', ao2atom, hk_ipvip1)
            hk = contract('jb,ajxy->abxy', ao2atom, tmp)

            hk_ipip1 = hk_ipip1.reshape([3,3,nao])
            tmp = contract('ia,xyi->axy', ao2atom, hk_ipip1)
            hk[range(natm), range(natm)] += tmp

        if auxbasis_response > 0:
            if with_j:
                hj_ip1ip2 = hj_ip1ip2.reshape([3,3,nao,naux])
                tmp = contract('ia,xyij->ajxy', ao2atom, hj_ip1ip2)
                tmp = contract('jb,ajxy->abxy',aux2atom, tmp)
                tmp = tmp + tmp.transpose([1,0,3,2])
                hj += tmp
                if auxbasis_response > 1:
                    hj += tmp
            if with_k:
                hk_ip1ip2 = hk_ip1ip2.reshape([3,3,nao,naux])
                tmp = contract('ia,xyij->ajxy', ao2atom, hk_ip1ip2)
                tmp = contract('jb,ajxy->abxy', aux2atom, tmp)
                tmp = 0.5 * (tmp + tmp.transpose([1,0,3,2]))
                hk += tmp
                if auxbasis_response > 1:
                    hk += tmp

        if auxbasis_response > 1:
            if with_j:
                hj_ipip2 = hj_ipip2.reshape([3,3,naux])
                tmp = contract('ia,xyi->axy', aux2atom, hj_ipip2)
                hj[range(natm), range(natm)] += tmp
            if with_k:
                hk_ipip2 = hk_ipip2.reshape([3,3,naux])
                tmp = contract('ia,xyi->axy', aux2atom, hk_ipip2)
                hk[range(natm), range(natm)] += .5 * tmp
        t0 = log.timer_debug1(f'int3c2e_ipip on Device {device_id}', *t0)
    return hj, hk

def get_int3c2e_hjk(intopt, rhoj, rhok, dm0_tag, with_j=True, with_k=True,
                    omega=None, auxbasis_response=1):
    orbo = cupy.asarray(dm0_tag.occ_coeff, order='C')
    futures = []
    ncp_k = len(intopt.aux_log_qs)
    ncp_ij = len(intopt.log_qs)
    tasks = np.array(list(itertools.product(range(ncp_k), range(ncp_ij))))
    task_list = []
    for device_id in range(num_devices):
        task_list.append(tasks[device_id::num_devices])

    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _int3c2e_ipip_tasks, intopt, task_list[device_id],
                rhoj, rhok, dm0_tag, orbo, with_j=with_j, with_k=with_k,
                device_id=device_id, omega=omega,
                auxbasis_response=auxbasis_response)
            futures.append(future)

    hj_total = []
    hk_total = []
    for future in futures:
        hj, hk = future.result()
        hj_total.append(hj)
        hk_total.append(hk)

    hj = hk = None
    if with_j:
        hj = reduce_to_device(hj_total, inplace=True)
    if with_k:
        hk = reduce_to_device(hk_total, inplace=True)
    return hj, hk
