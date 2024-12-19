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
Compute J/K matrices for Hessian
'''
import ctypes
import math
import numpy as np
import cupy as cp
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pyscf import lib
from pyscf.scf import _vhf
from pyscf import __config__

from gpu4pyscf.scf.jk import (_make_tril_tile_mappings, quartets_scheme, QUEUE_DEPTH, 
                              _VHFOpt, LMAX, init_constant, libvhf_rys)
from gpu4pyscf.lib.cupy_helper import (condense, sandwich_dot, transpose_sum,
                                       reduce_to_device, contract)

from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.__config__ import _streams, _num_devices
from gpu4pyscf.lib import logger


def _ao2mo(v_ao, mocc, mo_coeff):
    v_ao = contract('nij,jo->nio', v_ao, mocc)
    return contract('nio,ip->npo', v_ao, mo_coeff)

def _jk_task(mol, dms, mo_coeff, mocc, vhfopt, task_list, hermi=0,
             device_id=0, with_j=True, with_k=True, verbose=0):
    nao, _ = vhfopt.coeff.shape
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    kern = libvhf_rys.RYS_build_jk
    
    timing_counter = Counter()
    kern_counts = 0
    with cp.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        dms = cp.asarray(dms)

        n_dm = dms.shape[0]
        tile_q_ptr = ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p)
        q_ptr = ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p)
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)
        
        vj = vk = None
        vj_ptr = vk_ptr = lib.c_null_ptr()
        assert with_j or with_k
        if with_k:
            vk = cp.zeros(dms.shape)
            vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
        if with_j:
            vj = cp.zeros(dms.shape)
            vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)
        
        ao_loc = mol.ao_loc
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_max_dm = dm_cond.max()
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        tile_mappings = _make_tril_tile_mappings(l_ctr_bas_loc, vhfopt.tile_q_cond,
                                                 log_cutoff-log_max_dm)
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
        info = cp.empty(2, dtype=np.uint32)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j in task_list:
            ij_shls = (l_ctr_bas_loc[i], l_ctr_bas_loc[i+1],
                       l_ctr_bas_loc[j], l_ctr_bas_loc[j+1])
            tile_ij_mapping = tile_mappings[i,j]
            for k in range(i+1):
                for l in range(k+1):
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    kl_shls = (l_ctr_bas_loc[k], l_ctr_bas_loc[k+1],
                                l_ctr_bas_loc[l], l_ctr_bas_loc[l+1])
                    tile_kl_mapping = tile_mappings[k,l]
                    scheme = quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
                    err = kern(
                        vj_ptr, vk_ptr, ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
                        (ctypes.c_int*8)(*ij_shls, *kl_shls),
                        ctypes.c_int(tile_ij_mapping.size),
                        ctypes.c_int(tile_kl_mapping.size),
                        ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                        tile_q_ptr, q_ptr, s_ptr,
                        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                        ctypes.cast(info.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(workers),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    if err != 0:
                        raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                    if log.verbose >= logger.DEBUG1:
                        msg = f'processing {llll}, tasks = {info[1].get()} on Device {device_id}'
                        t1, t1p = log.timer_debug1(msg, *t1), t1
                        timing_counter[llll] += t1[1] - t1p[1]
                        kern_counts += 1
        if with_j:
            vj *= 2.0
            vj = transpose_sum(vj)
        if with_k:
            vk = transpose_sum(vk)

        if isinstance(mocc, tuple):
            # Unrestricted case
            mocca, moccb = mocc
            moa, mob = mo_coeff
            nmoa, nmob = moa.shape[1], mob.shape[1]
            nocca, noccb = mocca.shape[1], moccb.shape[1]
            n_dm_2 = n_dm//2
            if with_j:
                vjab = vj[:n_dm_2] + vj[n_dm_2:]
                vj = cp.empty([n_dm_2,nmoa*nocca+nmob*noccb])
                vj[:,:nmoa*nocca] = _ao2mo(vjab, mocca, moa).reshape(n_dm_2,-1)
                vj[:,nmoa*nocca:] = _ao2mo(vjab, moccb, mob).reshape(n_dm_2,-1)
            if with_k:
                vka, vkb = vk[:n_dm_2], vk[n_dm_2:]
                vk = cp.empty([n_dm_2,nmoa*nocca+nmob*noccb])
                vk[:,:nmoa*nocca] = _ao2mo(vka, mocca, moa).reshape(n_dm_2,-1)
                vk[:,nmoa*nocca:] = _ao2mo(vkb, moccb, mob).reshape(n_dm_2,-1)
        else:
            if with_j:
                vj = _ao2mo(vj, mocc, mo_coeff).reshape(n_dm,-1)
            if with_k:
                vk = _ao2mo(vk, mocc, mo_coeff).reshape(n_dm,-1)
        
    return vj, vk, kern_counts, timing_counter

def get_jk(mol, dm, mo_coeff, mocc, hermi=0, vhfopt=None, 
           with_j=True, with_k=True, verbose=None):
    '''Compute J, K matrices in MO
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)

    # Transform MO coeffcients and DM into sorted, cartesian AO basis
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')
    coeff = vhfopt.coeff
    if isinstance(mocc, tuple):
        mocc = (coeff.dot(mocc[0]), coeff.dot(mocc[1]))
        mo_coeff = (coeff.dot(mo_coeff[0]), coeff.dot(mo_coeff[1]))
    else:
        mocc = coeff.dot(mocc)
        mo_coeff = coeff.dot(mo_coeff)
    n_dm = dms.shape[0]

    assert with_j or with_k

    init_constant(mol)

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = np.count_nonzero(uniq_l <= LMAX)

    tasks = [(i,j) for i in range(n_groups) for j in range(i+1)]
    tasks = np.array(tasks)
    task_list = []
    for device_id in range(_num_devices):
        task_list.append(tasks[device_id::_num_devices])

    cp.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _jk_task,
                mol, dms, mo_coeff, mocc, vhfopt, task_list[device_id], hermi=hermi,
                with_j=with_j, with_k=with_k, verbose=verbose, 
                device_id=device_id)
            futures.append(future)

    kern_counts = 0
    timing_collection = Counter()
    vj_dist = []
    vk_dist = []
    for future in futures:
        vj, vk, counts, counter = future.result()
        kern_counts += counts
        timing_collection += counter
        vj_dist.append(vj)
        vk_dist.append(vk)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)
    
    for s in _streams:
        s.synchronize()
    cp.cuda.get_current_stream().synchronize()
    vj = vk = None
    if with_k:
        vk = reduce_to_device(vk_dist, inplace=True)

    if with_j:
        vj = reduce_to_device(vj_dist, inplace=True)

    h_shls = vhfopt.h_shls
    assert len(h_shls) == 0
    if h_shls:
        cput1 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)
        log.debug3('Integrals for %s functions on CPU', l_symb[LMAX+1])
        scripts = []
        if with_j:
            scripts.append('ji->s2kl')
        if with_k:
            if hermi == 1:
                scripts.append('jk->s2il')
            else:
                scripts.append('jk->s1il')
        shls_excludes = [0, h_shls[0]] * 4
        if hermi == 1:
            dms = dms.get()
        else:
            dms = dms[:n_dm//2].get()
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms, 1, mol._atm, mol._bas, mol._env,
                                 shls_excludes=shls_excludes)
        if with_j and with_k:
            vj1 = vs_h[0]
            vk1 = vs_h[1]
        elif with_j:
            vj1 = vs_h[0]
        else:
            vk1 = vs_h[0]
        coeff = vhfopt.coeff
        idx, idy = np.tril_indices(nao, -1)
        if isinstance(mocc, tuple):
            mocca, moccb = mocc
            moa, mob = mo_coeff
            nmoa = moa.shape[1]
            nocca = mocca.shape[1]
            n_dm_2 = n_dm//2
            if with_j:
                vjab = vj1[:n_dm_2] + vj1[n_dm_2:]
                vj[:,:nmoa*nocca] += _ao2mo(vjab, mocca, moa).reshape(n_dm_2,-1)
                vj[:,nmoa*nocca:] += _ao2mo(vjab, moccb, mob).reshape(n_dm_2,-1)
            if with_k:
                vka, vkb = vk[:n_dm_2], vk[n_dm_2:]
                vk[:,:nmoa*nocca] += _ao2mo(vka, mocca, moa).reshape(n_dm_2,-1)
                vk[:,nmoa*nocca:] += _ao2mo(vkb, moccb, mob).reshape(n_dm_2,-1)
        else:
            if with_j:
                vj1[:,idy,idx] = vj1[:,idx,idy]
                vj += _ao2mo(cp.asarray(vj1), mocc, mo_coeff)
            if with_k:
                if hermi:
                    vk1[:,idy,idx] = vk1[:,idx,idy]
                vk += _ao2mo(cp.asarray(vk1), mocc, mo_coeff)
        log.timer_debug1('get_jk pass 2 for h functions on cpu', *cput1)
    log.timer('vj and vk', *cput0)
    return vj, vk
