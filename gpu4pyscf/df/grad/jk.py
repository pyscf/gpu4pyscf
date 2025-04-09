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

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cupy
from gpu4pyscf.df.int3c2e import get_int3c2e_ip_jk, VHFOpt, _split_tasks
from gpu4pyscf.lib.cupy_helper import contract, concatenate, reduce_to_device
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, num_devices

def _jk_task(with_df, dm, orbo, with_j=True, with_k=True, device_id=0):
    '''  # (L|ij) -> rhoj: (L), rhok: (L|oo)
    '''
    rhoj = rhok = None
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(with_df.mol, with_df.verbose)
        assert isinstance(with_df.verbose, int)
        t0 = log.init_timer()
        dm = cupy.asarray(dm)
        orbo = cupy.asarray(orbo)
        naux_slice = with_df._cderi[device_id].shape[0]
        nocc = orbo.shape[-1]
        rows = with_df.intopt.cderi_row
        cols = with_df.intopt.cderi_col
        dm_sparse = dm[rows, cols]
        dm_sparse[with_df.intopt.cderi_diag] *= .5

        blksize = with_df.get_blksize()
        if with_j:
            rhoj = cupy.empty([naux_slice])
        if with_k:
            rhok = cupy.empty([naux_slice, nocc, nocc], order='C')
        p0 = p1 = 0

        for cderi, cderi_sparse in with_df.loop(blksize=blksize):
            p1 = p0 + cderi.shape[0]
            if with_j:
                rhoj[p0:p1] = 2.0*dm_sparse.dot(cderi_sparse)
            if with_k:
                tmp = contract('Lij,jk->Lki', cderi, orbo)
                contract('Lki,il->Lkl', tmp, orbo, out=rhok[p0:p1])
            p0 = p1
            cupy.cuda.get_current_stream().synchronize()
        t0 = log.timer_debug1(f'rhoj and rhok on Device {device_id}', *t0)
    return rhoj, rhok

def get_rhojk(with_df, dm, orbo, with_j=True, with_k=True):
    ''' Calculate rhoj and rhok on Multi-GPU system
    '''
    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _jk_task, with_df, dm, orbo,
                with_j=with_j, with_k=with_k, device_id=device_id)
            futures.append(future)

    rhoj_total = []
    rhok_total = []
    for future in futures:
        rhoj, rhok = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)

    rhoj = rhok = None
    if with_j:
        rhoj = concatenate(rhoj_total)
    if with_k:
        rhok = concatenate(rhok_total)

    return rhoj, rhok

def _jk_ip_task(intopt, rhoj_cart, dm_cart, rhok_cart, orbo_cart, task_list,
                with_j=True, with_k=True, device_id=0, omega=None):
    mol = intopt.mol
    natm = mol.natm
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, mol.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        orbo_cart = cupy.asarray(orbo_cart)
        cart_aux_loc = intopt.cart_aux_loc
        ej = ek = ejaux = ekaux = None
        if with_j:
            rhoj_cart = cupy.asarray(rhoj_cart)
            dm_cart = cupy.asarray(dm_cart)
            ej = cupy.zeros((natm,3), order='C')
            ejaux = cupy.zeros((natm,3))
        if with_k:
            rhok_cart = cupy.asarray(rhok_cart)
            ek = cupy.zeros((natm,3), order='C')
            ekaux = cupy.zeros((natm,3))
        
        for cp_kl_id in task_list:
            k0, k1 = cart_aux_loc[cp_kl_id], cart_aux_loc[cp_kl_id+1]
            rhoj_tmp = rhok_tmp = None
            if with_j:
                rhoj_tmp = rhoj_cart[k0:k1]
            if with_k:
                rhok_tmp = contract('por,ir->pio', rhok_cart[k0:k1], orbo_cart)
                rhok_tmp = contract('pio,jo->pji', rhok_tmp, orbo_cart)
            '''
            if(rhoj_tmp.flags['C_CONTIGUOUS'] == False):
                rhoj_tmp = rhoj_tmp.astype(cupy.float64, order='C')

            if(rhok_tmp.flags['C_CONTIGUOUS'] == False):
                rhok_tmp = rhok_tmp.astype(cupy.float64, order='C')
            '''
            '''
            # outcore implementation
            buf = int3c2e.get_int3c2e_ip_slice(intopt, cp_kl_id, 1)
            size = 3*(k1-k0)*nao_cart*nao_cart
            int3c_ip = buf[:size].reshape([3,k1-k0,nao_cart,nao_cart], order='C')
            rhoj_tmp0 = contract('xpji,ij->xip', int3c_ip, dm_cart)
            vj_outcore = contract('xip,p->xi', rhoj_tmp0, rhoj_cart[k0:k1])
            vk_outcore = contract('pji,xpji->xi', rhok_tmp, int3c_ip)

            buf = int3c2e.get_int3c2e_ip_slice(intopt, cp_kl_id, 2)
            int3c_ip = buf[:size].reshape([3,k1-k0,nao_cart,nao_cart], order='C')
            rhoj_tmp0 = contract('xpji,ji->xp', int3c_ip, dm_cart)
            vjaux_outcore = contract('xp,p->xp', rhoj_tmp0, rhoj_cart[k0:k1])
            vkaux_outcore = contract('xpji,pji->xp', int3c_ip, rhok_tmp)
            '''
            ej_tmp, ek_tmp = get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip1', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
            if with_j: ej += ej_tmp
            if with_k: ek += ek_tmp
            ej_tmp, ek_tmp = get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip2', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
            if with_j: ejaux += ej_tmp
            if with_k: ekaux += ek_tmp

            rhoj_tmp = rhok_tmp = ej_tmp = ek_tmp = None
            t0 = log.timer_debug1(f'calculate {cp_kl_id:3d} / {len(intopt.aux_log_qs):3d}, {k1-k0:3d} slices', *t0)
    return ej, ek, ejaux, ekaux

def get_grad_vjk(with_df, mol, auxmol, rhoj_cart, dm_cart, rhok_cart, orbo_cart, 
                 with_j=True, with_k=True, omega=None):
    '''
    Calculate vj    = (i'j|L)(L|kl)(ij)(kl), vk    = (i'j|L)(L|kl)(ik)(jl)
              vjaux = (ij|L')(L|kl)(ij)(kl), vkaux = (ij|L')(L|kl)(ik)(jl)
    '''
    nao_cart = dm_cart.shape[0]
    block_size = with_df.get_blksize(nao=nao_cart)

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size_aux=block_size, verbose=0)#, group_size=block_size)

    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _jk_ip_task, intopt, rhoj_cart, dm_cart, rhok_cart, orbo_cart, task_list[device_id],
                with_j=with_j, with_k=with_k, device_id=device_id, omega=omega)
            futures.append(future)

    rhoj_total = []
    rhok_total = []
    vjaux_total = []
    vkaux_total = []
    for future in futures:
        rhoj, rhok, vjaux, vkaux = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)
        vjaux_total.append(vjaux)
        vkaux_total.append(vkaux)

    rhoj = rhok = vjaux = vkaux = None
    if with_j:
        rhoj = reduce_to_device(rhoj_total)
        vjaux = reduce_to_device(vjaux_total)
    if with_k:
        rhok = reduce_to_device(rhok_total)
        vkaux = reduce_to_device(vkaux_total)
    return rhoj, rhok, vjaux, vkaux


def _jk_task_td(with_df, dm, orbol, orbor, with_j=True, with_k=True, device_id=0):
    '''  
    (L|ij) -> rhoj: (L), rhok: (L|lr), for dm0 from scf, rhok is (L|oo) 
    '''
    rhoj = rhok = None
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(with_df.mol, with_df.verbose)
        assert isinstance(with_df.verbose, int)
        t0 = log.init_timer()
        dm = cupy.asarray(dm)
        naux_slice = with_df._cderi[device_id].shape[0]
        nl = orbol.shape[-1]
        nr = orbor.shape[-1]

        rows = with_df.intopt.cderi_row
        cols = with_df.intopt.cderi_col
        dm_sparse = dm[rows, cols]
        dm_sparse[with_df.intopt.cderi_diag] *= .5

        blksize = with_df.get_blksize()
        if with_j:
            rhoj = cupy.empty([naux_slice])
        if with_k:
            rhok = cupy.empty([naux_slice, nl, nr], order='C')
        p0 = p1 = 0

        for cderi, cderi_sparse in with_df.loop(blksize=blksize):
            p1 = p0 + cderi.shape[0]
            if with_j:
                rhoj[p0:p1] = 2.0*dm_sparse.dot(cderi_sparse)
            if with_k:
                rhok[p0:p1] = 0
                tmp = contract('Lij,jk->Lki', cderi, orbol)
                contract('Lki,il->Lkl', tmp, orbor, out=rhok[p0:p1])
            p0 = p1
            cupy.cuda.get_current_stream().synchronize()
        t0 = log.timer_debug1(f'rhoj and rhok on Device {device_id}', *t0)
    return rhoj, rhok


def get_rhojk_td(with_df, dm, orbol, orbor, with_j=True, with_k=True):
    ''' Calculate rhoj and rhok on Multi-GPU system
    '''
    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _jk_task_td, with_df, dm, orbol, orbor,
                with_j=with_j, with_k=with_k, device_id=device_id)
            futures.append(future)

    rhoj_total = []
    rhok_total = []
    for future in futures:
        rhoj, rhok = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)

    rhoj = rhok = None
    if with_j:
        rhoj = concatenate(rhoj_total)
    if with_k:
        rhok = concatenate(rhok_total)

    return rhoj, rhok


def _jk_ip_task_td(intopt, rhoj_cart, dm_cart, rhok_cart, orbol_cart, orbor_cart, task_list,
                with_j=True, with_k=True, device_id=0, omega=None):
    mol = intopt.mol
    natm = mol.natm
    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, mol.verbose)
        t0 = (logger.process_clock(), logger.perf_counter())

        cart_aux_loc = intopt.cart_aux_loc
        ej = ek = ejaux = ekaux = None
        if with_j:
            rhoj_cart = cupy.asarray(rhoj_cart)
            dm_cart = cupy.asarray(dm_cart)
            ej = cupy.zeros((natm,3), order='C')
            ejaux = cupy.zeros((natm,3))
        if with_k:
            rhok_cart = cupy.asarray(rhok_cart)
            ek = cupy.zeros((natm,3), order='C')
            ekaux = cupy.zeros((natm,3))
        
        for cp_kl_id in task_list:
            k0, k1 = cart_aux_loc[cp_kl_id], cart_aux_loc[cp_kl_id+1]
            rhoj_tmp = rhok_tmp = None
            if with_j:
                rhoj_tmp = rhoj_cart[k0:k1]
            if with_k:
                rhok_tmp = contract('por,ir->pio', rhok_cart[k0:k1], orbol_cart)
                rhok_tmp = contract('pio,jo->pji', rhok_tmp, orbor_cart)

            ej_tmp, ek_tmp = get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip1', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
            if with_j: ej += ej_tmp
            if with_k: ek += ek_tmp
            ej_tmp, ek_tmp = get_int3c2e_ip_jk(intopt, cp_kl_id, 'ip2', rhoj_tmp, rhok_tmp, dm_cart, omega=omega)
            if with_j: ejaux += ej_tmp
            if with_k: ekaux += ek_tmp

            rhoj_tmp = rhok_tmp = ej_tmp = ek_tmp = None
            t0 = log.timer_debug1(f'calculate {cp_kl_id:3d} / {len(intopt.aux_log_qs):3d}, {k1-k0:3d} slices', *t0)
    return ej, ek, ejaux, ekaux


def get_grad_vjk_td(with_df, mol, auxmol, rhoj_cart, dm_cart, rhok_cart, orbol_cart, orbor_cart, 
                 with_j=True, with_k=True, omega=None):
    '''
    Calculate vj    = (i'j|L)(L|kl)(ij)(kl), vk    = (i'j|L)(L|kl)(ik)(jl)
              vjaux = (ij|L')(L|kl)(ij)(kl), vkaux = (ij|L')(L|kl)(ik)(jl)
    '''
    nao_cart = dm_cart.shape[0]
    block_size = with_df.get_blksize(nao=nao_cart)

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size_aux=block_size, verbose=0)#, group_size=block_size)

    aux_ao_loc = np.array(intopt.aux_ao_loc)
    loads = aux_ao_loc[1:] - aux_ao_loc[:-1]
    task_list = _split_tasks(loads, num_devices)

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _jk_ip_task_td, intopt, rhoj_cart, dm_cart, rhok_cart, orbol_cart, orbor_cart, task_list[device_id],
                with_j=with_j, with_k=with_k, device_id=device_id, omega=omega)
            futures.append(future)

    rhoj_total = []
    rhok_total = []
    ejaux_total = []
    ekaux_total = []
    for future in futures:
        rhoj, rhok, ejaux, ekaux = future.result()
        rhoj_total.append(rhoj)
        rhok_total.append(rhok)
        ejaux_total.append(ejaux)
        ekaux_total.append(ekaux)

    rhoj = rhok = ejaux = ekaux = None
    if with_j:
        rhoj = reduce_to_device(rhoj_total)
        ejaux = reduce_to_device(ejaux_total)
    if with_k:
        rhok = reduce_to_device(rhok_total)
        ekaux = reduce_to_device(ekaux_total)
    return rhoj, rhok, ejaux, ekaux
