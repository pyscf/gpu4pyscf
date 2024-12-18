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
Non-relativistic RHF analytical Hessian
'''

import math
import ctypes
import numpy
import cupy
import cupy as cp
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pyscf.hessian import rhf as rhf_hess_cpu
from pyscf import lib
from pyscf.gto import ATOM_OF
# import _response_functions to load gen_response methods in SCF class
from gpu4pyscf.scf import _response_functions  # noqa
from gpu4pyscf.scf import cphf
from gpu4pyscf.lib.cupy_helper import (reduce_to_device,
    contract, tag_array, sandwich_dot, transpose_sum, get_avail_mem, condense,
    krylov)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.__config__ import _streams, _num_devices
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, libvhf_rys, _VHFOpt, init_constant,
    _make_tril_tile_mappings, _nearest_power2)
from gpu4pyscf.grad import rhf as rhf_grad

libvhf_rys.RYS_per_atom_jk_ip2_type12.restype = ctypes.c_int
libvhf_rys.RYS_per_atom_jk_ip2_type3.restype = ctypes.c_int
libvhf_rys.RYS_build_jk_ip1.restype = ctypes.c_int

GB = 1024*1024*1024
ALIGNED = 4

def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1mo=None,
              atmlst=None, max_memory=4000, verbose=None):
    ''' Different from PySF, using h1mo instead of h1ao for saving memory
    '''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = log.init_timer()
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is not None:
        assert len(atmlst) == mol.natm

    assert mo_coeff.dtype == cp.float64
    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)
    t1 = log.timer_debug1('hess elec', *t1)
    if h1mo is None:
        h1mo = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        if h1mo.size * 8 * 5 > get_avail_mem():
            # Reduce GPU memory footprint
            h1mo = h1mo.get()
        t1 = log.timer_debug1('making H1', *t1)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    mo1 = cupy.asarray(mo1)
    # *2 for double occupancy, *2 for +c.c.
    de2 += contract('kxpi,lypi->klxy', cupy.asarray(h1mo), mo1) * 4
    mo1 = contract('kxai,pa->kxpi', mo1, mo_coeff)
    mo_e1 = cupy.asarray(mo_e1)

    nao = mo_coeff.shape[0]
    mocc = mo_coeff[:,mo_occ>0]
    mocc_e = mocc * mo_energy[mo_occ>0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    s1a = cupy.asarray(s1a)

    aoslices = mol.aoslice_by_atom()
    for i0, (p0, p1) in enumerate(aoslices[:,2:]):
        s1ao = cupy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        tmp = contract('xpq,pi->xiq', s1ao, mocc)
        s1oo = contract('xiq,qj->xij', tmp, mocc)
        de2[i0] -= contract('xij,kyij->kxy', s1oo, mo_e1) * 2

        s1mo = contract('xpq,qi->xpi', s1ao, mocc_e)
        de2[i0] -= contract('xpi,kypi->kxy', s1mo, mo1) * 4

    de2 = de2 + de2.transpose(1,0,3,2)
    de2 *= .5
    log.timer('RHF hessian', *time0)
    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ejk = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                atmlst, max_memory, verbose)
    return e1 + ejk

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None,
                      j_factor=1., k_factor=1.):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    assert atmlst is None
    atmlst = range(mol.natm)

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = mocc.dot(mocc.T) * 2
    vhfopt = mf._opt_gpu.get(None, None)
    ejk = _partial_ejk_ip2(mol, dm0, vhfopt, j_factor, k_factor, verbose=log)
    t1 = log.timer_debug1('hessian of 2e part', *t1)

    # Energy weighted density matrix
    dme0 = (mocc * mo_energy[mo_occ>0]).dot(mocc.T) * 2
    de_hcore = _e_hcore_generator(hessobj, dm0)
    s1aa, s1ab, s1a = get_ovlp(mol)

    aoslices = mol.aoslice_by_atom()
    e1 = cupy.zeros((mol.natm,mol.natm,3,3))
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        e1[i0,i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            # *2 for +c.c.
            e1[i0,j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2
            e1[i0,j0] += de_hcore(ia, ja)

        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T

    log.timer('RHF partial hessian', *time0)
    return e1, ejk

def _ejk_ip2_task(mol, dms, vhfopt, task_list, j_factor=1.0, k_factor=1.0,
                  device_id=0, verbose=0):
    n_dm = dms.shape[0]
    assert n_dm <= 2
    assert isinstance(verbose, int)
    nao, _ = vhfopt.coeff.shape
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]

    kern1 = libvhf_rys.RYS_per_atom_jk_ip2_type12
    kern2 = libvhf_rys.RYS_per_atom_jk_ip2_type3

    timing_counter = Counter()
    kern_counts = 0
    with cp.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        dms = cp.asarray(dms)

        tile_q_ptr = ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p)
        q_ptr = ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p)
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)

        natm = mol.natm
        ejk = cp.zeros((natm, natm, 3, 3))

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
                    scheme = _ip2_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
                    err1 = kern1(
                        ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                        ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
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
                    err2 = kern2(
                        ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                        ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
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
                    if err1 != 0 or err2 != 0:
                        raise RuntimeError(f'RYS_per_atom_jk_ip2 kernel for {llll} failed')
                    if log.verbose >= logger.DEBUG1:
                        msg = f'processing {llll}, tasks = {info[1].get()} on Device {device_id}'
                        t1, t1p = log.timer_debug1(msg, *t1), t1
                        timing_counter[llll] += t1[1] - t1p[1]
                        kern_counts += 1

        ejk = ejk + ejk.transpose(1,0,3,2)
    return ejk, kern_counts, timing_counter

def _partial_ejk_ip2(mol, dm, vhfopt=None, j_factor=1., k_factor=1., verbose=None):
    '''Compute the energy per atom for
        j_factor * J_derivatives - k_factor * K_derivatives
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')

    init_constant(mol)

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    assert uniq_l.max() <= LMAX

    n_groups = len(uniq_l_ctr)
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
                _ejk_ip2_task,
                mol, dms, vhfopt, task_list[device_id],
                j_factor=j_factor, k_factor=k_factor, verbose=log.verbose,
                device_id=device_id)
            futures.append(future)

    kern_counts = 0
    timing_collection = Counter()
    ejk_dist = []
    for future in futures:
        ejk, counts, counter = future.result()
        kern_counts += counts
        timing_collection += counter
        ejk_dist.append(ejk)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    ejk = reduce_to_device(ejk_dist, inplace=True)

    timing_collection = {}
    kern_counts = 0

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    log.timer_debug1('ejk_ip2', *cput0)
    return ejk

def _ip2_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    g_size = (li+2)*(lj+2)*(lk+2)*(ll+2)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 2) // 2 + 1
    unit = nroots*2 + g_size*3 + ij_prims*4
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''Compute the first order Fock matrix (the force term of the CPHF
    equation). Returns of this function are different to the PySCF CPU version.
    This function returns matrices in the MO-occupied_orb basis, while the CPU
    version returns matrices in MO basis.
    '''
    assert atmlst is None
    mol = hessobj.mol
    natm = mol.natm
    nao = mo_coeff.shape[0]
    mo_coeff = cp.asarray(mo_coeff)
    mocc = cp.asarray(mo_coeff[:,mo_occ>0])
    dm0 = mocc.dot(mocc.T) * 2
    h1mo = rhf_grad.get_grad_hcore(hessobj.base.Gradients())

    avail_mem = get_avail_mem()
    slice_size = int(avail_mem*0.6) // (8*3*nao*nao)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vj, vk = _get_jk(mol, dm0, atoms_slice=atoms_slice, verbose=verbose)
        #:vhf = vj - vk * .5
        vhf = vk
        vhf *= -.5
        vhf += vj
        atom0, atom1 = atoms_slice
        for i, ia in enumerate(range(atom0, atom1)):
            for ix in range(3):
                h1mo[ia,ix] += mo_coeff.T.dot(vhf[i,ix].dot(mocc))
        vj = vk = vhf = None
    return h1mo


def _build_jk_ip1_task(mol, dms, vhfopt, task_list, atoms_slice,
                       device_id=0, with_j=True, with_k=True, verbose=0):
    assert isinstance(verbose, int)
    nao, _ = vhfopt.coeff.shape
    natm = mol.natm
    nbas = mol.nbas
    n_dm = dms.shape[0]
    if atoms_slice is None:
        atoms_slice = 0, natm
    atom0, atom1 = atoms_slice

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = len(uniq_l_ctr)
    kern = libvhf_rys.RYS_build_jk_ip1

    timing_counter = Counter()
    kern_counts = 0
    with cp.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        dms = cp.asarray(dms)

        vj = vk = None
        vj_ptr = vk_ptr = lib.c_null_ptr()
        assert with_j or with_k
        if with_k:
            vk = cp.zeros(((atom1-atom0)*3, nao, nao))
            vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
        if with_j:
            vj = cp.zeros(((atom1-atom0)*3, nao, nao))
            vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)

        ao_loc = mol.ao_loc
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_max_dm = dm_cond.max()
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        tril_tile_mappings = _make_tril_tile_mappings(
            l_ctr_bas_loc, vhfopt.tile_q_cond, log_cutoff-log_max_dm, 1)
        workers = gpu_specs['multiProcessorCount']
        QUEUE_DEPTH = 65536
        pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
        info = cp.empty(2, dtype=np.uint32)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j in task_list:
            ij_shls = (l_ctr_bas_loc[i], l_ctr_bas_loc[i+1],
                       l_ctr_bas_loc[j], l_ctr_bas_loc[j+1])
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            ij_shls = (ish0, ish1, jsh0, jsh1)

            sub_tile_q = vhfopt.tile_q_cond[ish0:ish1,jsh0:jsh1]
            mask = sub_tile_q > log_cutoff - log_max_dm
            mask[mol._bas[ish0:ish1,ATOM_OF] <  atom0] = False
            mask[mol._bas[ish0:ish1,ATOM_OF] >= atom1] = False
            t_ij = (cp.arange(ish0, ish1, dtype=np.int32)[:,None] * nbas +
                    cp.arange(jsh0, jsh1, dtype=np.int32))
            idx = cp.argsort(sub_tile_q[mask])[::-1]
            tile_ij_mapping = t_ij[mask][idx]
            for k in range(n_groups):
                for l in range(k+1):
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    kl_shls = (l_ctr_bas_loc[k], l_ctr_bas_loc[k+1],
                               l_ctr_bas_loc[l], l_ctr_bas_loc[l+1])
                    tile_kl_mapping = tril_tile_mappings[k,l]
                    scheme = _ip1_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
                    err = kern(
                        vj_ptr, vk_ptr, ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao), ctypes.c_int(atom0),
                        vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
                        (ctypes.c_int*8)(*ij_shls, *kl_shls),
                        ctypes.c_int(tile_ij_mapping.size),
                        ctypes.c_int(tile_kl_mapping.size),
                        ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p),
                        lib.c_null_ptr(),
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
    return vj, vk, kern_counts, timing_counter

def _get_jk(mol, dm, with_j=True, with_k=True, atoms_slice=None, verbose=None):
    r'''
    For each atom, compute
    J = ((\nabla_X i) j| kl) (D_lk + D_ji)
    K = ((\nabla_X i) j| kl) (D_jk + D_li)
    '''
    assert mol.omega >= 0
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    vhfopt = _VHFOpt(mol)
    # tile must set to 1. This tile size is assumed in the GPU kernel code
    vhfopt.tile = 1
    vhfopt.build()

    mol = vhfopt.mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')
    assert n_dm == 1

    natm = mol.natm
    if atoms_slice is None:
        atoms_slice = 0, natm
    atom0, atom1 = atoms_slice

    init_constant(mol)

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    assert uniq_l.max() <= LMAX

    nbas = mol.nbas
    assert vhfopt.tile_q_cond.shape == (nbas, nbas)

    n_groups = len(uniq_l_ctr)
    tasks = [(i,j) for i in range(n_groups) for j in range(n_groups)]
    tasks = np.array(tasks)
    task_list = []
    for device_id in range(_num_devices):
        task_list.append(tasks[device_id::_num_devices])

    cp.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _build_jk_ip1_task,
                mol, dms, vhfopt, task_list[device_id], atoms_slice,
                with_j=with_j, with_k=with_k, verbose=log.verbose,
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

    if with_k:
        vk = reduce_to_device(vk_dist, inplace=True)
        vk = sandwich_dot(vk, vhfopt.coeff)
        vk = transpose_sum(vk)
        vk = vk.reshape(atom1-atom0, 3, nao_orig, nao_orig)
    if with_j:
        vj = reduce_to_device(vj_dist, inplace=True)
        vj = sandwich_dot(vj, vhfopt.coeff)
        vj = transpose_sum(vj)
        vj *= 2.
        vj = vj.reshape(atom1-atom0, 3, nao_orig, nao_orig)
    log.timer('vj and vk gradients', *cput0)
    return vj, vk

def _ip1_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    gout_size = nfi * nfj * nfk * nfl
    g_size = (li+2)*(lj+1)*(lk+1)*(ll+1)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 1) // 2 + 1

    unit = nroots*2 + g_size*3
    shm_size -= ij_prims*12 * 8
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    gout_width = 18
    while gout_stride < 16 and gout_size / (gout_stride*gout_width) > 1:
        n //= 2
        gout_stride *= 2
    return n, gout_stride

def get_hcore(mol):
    '''Part of the second derivatives of core Hamiltonian'''
    h1aa = mol.intor('int1e_ipipkin', comp=9)
    h1ab = mol.intor('int1e_ipkinip', comp=9)
    if mol._pseudo:
        NotImplementedError('Nuclear hessian for GTH PP')
    else:
        h1aa+= mol.intor('int1e_ipipnuc', comp=9)
        h1ab+= mol.intor('int1e_ipnucip', comp=9)
    if mol.has_ecp():
        h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
    nao = h1aa.shape[-1]
    return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)

def get_ovlp(mol):
    s1a =-mol.intor('int1e_ipovlp', comp=3)
    nao = s1a.shape[-1]
    s1aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    s1ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    return cp.asarray(s1aa), cp.asarray(s1ab), cp.asarray(s1a)

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=50, level_shift=0):
    '''Solve the CPHF equation for the first orbitals.
    Note: These orbitals are represented in MO basis. This is different to the
    solve_mo1 function in the PySCF CPU version, which transforms the mo1 to AO
    basis. Additionally, the return format is also different to the CPU version.
    This function returns orbitals in a single tensor while the CPU version
    returns a list of mo1 orbitals.

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    t0 = log.init_timer()

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = 1 / (e_a[:,None] + level_shift - e_i)
    nvir, nocc = e_ai.shape

    mocc = mo_coeff[:,occidx]
    nao, nmo = mo_coeff.shape
    natm = mol.natm

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)

    def fvind_vo(mo1):
        mo1 = mo1.reshape(-1,nmo, nocc)
        v = fx(mo1).reshape(-1,nmo, nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v[:,viridx,:] *= e_ai
        v[:,occidx,:] = 0
        return v.reshape(-1,nmo*nocc)

    ipovlp = -mol.intor('int1e_ipovlp', comp=3)
    ipovlp = cp.asarray(ipovlp)
    cp.get_default_memory_pool().free_all_blocks()

    avail_mem = get_avail_mem()
    # *4 for input dm, vj, vk, and vxc
    blksize = int(min(avail_mem*.3 / (8*3*nao*nao*4),
                      avail_mem*.6 / (8*nmo*nocc*natm*3*5)))
    if blksize < ALIGNED**2:
        raise RuntimeError('GPU memory insufficient')

    blksize = (blksize // ALIGNED**2) * ALIGNED**2
    log.debug(f'GPU memory {avail_mem/GB:.1f} GB available')
    log.debug(f'{blksize} atoms in each block CPHF equation')

    mo1s = np.zeros(h1mo.shape)
    e1s = np.zeros((natm, 3, nocc, nocc))
    aoslices = mol.aoslice_by_atom()
    for i0, i1 in lib.prange(0, natm, blksize):
        log.info('Solving CPHF equation for atoms [%d:%d]', i0, i1)

        h1mo_blk = h1mo[i0:i1]
        if not isinstance(h1mo, cp.ndarray):
            h1mo_blk = cp.asarray(h1mo_blk)
        s1mo_blk = cp.empty_like(h1mo_blk)
        for k, (p0, p1) in enumerate(aoslices[i0:i1,2:]):
            s1ao = cp.zeros((3,nao,nao))
            s1ao[:,p0:p1] += ipovlp[:,p0:p1]
            s1ao[:,:,p0:p1] += ipovlp[:,p0:p1].transpose(0,2,1)
            tmp = contract('xij,jo->xio', s1ao, mocc)
            s1mo_blk[k] = contract('xio,ip->xpo', tmp, mo_coeff)

        mo1 = hs = h1mo_blk - s1mo_blk * e_i
        mo_e1 = hs[:,:,occidx]
        mo1[:,:,viridx] *= -e_ai
        mo1[:,:,occidx] = -s1mo_blk[:,:,occidx] * .5
        hs = s1mo_blk = h1mo_blk = None

        tol = mf.conv_tol_cpscf * (i1 - i0)
        raw_mo1 = krylov(fvind_vo, mo1.reshape(-1,nmo*nocc),
                         tol=tol, max_cycle=max_cycle, verbose=log)
        raw_mo1 = raw_mo1.reshape(i1-i0,3,nmo,nocc)
        raw_mo1[:,:,occidx] = mo1[:,:,occidx]

        v1 = fx(raw_mo1).reshape(i1-i0,3,nmo,nocc)
        mo1[:,:,viridx] -= v1[:,:,viridx] * e_ai
        mo_e1 += v1[:,:,occidx]
        mo_e1 += mo1[:,:,occidx] * (e_i[:,None] - e_i)

        mo1s[i0:i1] = mo1.get()
        e1s[i0:i1] = mo_e1.get()
        mo1 = raw_mo1 = mo_e1 = v1 = None
    log.timer('CPHF solver', *t0)
    return mo1s, e1s

def gen_vind(mf, mo_coeff, mo_occ):
    # Move data to GPU
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    mocc_2 = mocc * 2
    grids = getattr(mf, 'cphf_grids', None)
    if grids is not None:
        logger.info(mf, 'Secondary grids defined for CPHF in Hessian')
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1, grids=grids)

    def fx(mo1):
        mo1 = cupy.asarray(mo1)
        mo1 = mo1.reshape(-1,nmo,nocc)
        mo1_mo = contract('npo,ip->nio', mo1, mo_coeff)
        #dm1 = contract('nio,jo->nij', mo1_mo, mocc_2)
        #dm1 = dm1 + dm1.transpose(0,2,1)
        dm1 = mo1_mo.dot(mocc_2.T)
        transpose_sum(dm1)
        dm1 = tag_array(dm1, mo1=mo1_mo, occ_coeff=mocc, mo_occ=mo_occ)
        v1 = vresp(dm1)
        tmp = contract('nij,jo->nio', v1, mocc)
        v1vo = contract('nio,ip->npo', tmp, mo_coeff)
        return v1vo
    return fx

def hess_nuc_elec(mol, dm):
    '''
    calculate hessian contribution due to (nuc, elec) pair
    '''

    '''
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    hcore = numpy.zeros([3,3,natm,natm])
    # CPU version
    for ia in range(mol.natm):
        ish0, ish1, i0, i1 = aoslices[ia]
        zi = mol.atom_charge(ia)
        with mol.with_rinv_at_nucleus(ia):
            rinv2aa = mol.intor('int1e_ipiprinv', comp=9).reshape([3,3,nao,nao])
            rinv2ab = mol.intor('int1e_iprinvip', comp=9).reshape([3,3,nao,nao])
            rinv2aa *= zi
            rinv2ab *= zi

            hcore[:,:,ia,ia] -= numpy.einsum('xypq,pq->xy', rinv2aa+rinv2ab, dm)

            haa = numpy.einsum('xypq,pq->xyp', rinv2aa, dm)
            hab = numpy.einsum('xypq,pq->xyp', rinv2ab, dm)

            haa = [haa[:,:,p0:p1].sum(axis=2) for p0,p1 in aoslices[:,2:]]
            hab = [hab[:,:,p0:p1].sum(axis=2) for p0,p1 in aoslices[:,2:]]

            haa = numpy.stack(haa, axis=2)
            hab = numpy.stack(hab, axis=2)

            hcore[:,:,ia] += haa
            hcore[:,:,ia] += hab.transpose([1,0,2])

            hcore[:,:,:,ia] += haa.transpose([1,0,2])
            hcore[:,:,:,ia] += hab

    hcore = cupy.asarray(hcore)
    '''
    from gpu4pyscf.df import int3c2e
    hcore = int3c2e.get_hess_nuc_elec(mol, dm)
    return hcore * 2.0


def kernel(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    if mo_energy is None: mo_energy = hessobj.base.mo_energy
    if mo_coeff is None: mo_coeff = hessobj.base.mo_coeff
    if mo_occ is None: mo_occ = hessobj.base.mo_occ
    if atmlst is None:
        atmlst = hessobj.atmlst
    else:
        hessobj.atmlst = atmlst

    if hessobj.verbose >= logger.INFO:
        hessobj.dump_flags()

    de = hessobj.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst=atmlst)
    hessobj.de = de.get() + hessobj.hess_nuc(hessobj.mol, atmlst=atmlst)
    mf = hessobj.base
    if mf.do_disp():
        h_disp = hessobj.get_dispersion()
        hessobj.hess_disp = h_disp
        hessobj.hess_mf = hessobj.de
        atmlst = range(hessobj.mol.natm)
        for k, katm in enumerate(atmlst):
            for l, latm in enumerate(atmlst):
                hessobj.de[k,l] += h_disp[k,l]
    logger.timer(hessobj, 'SCF hessian', *cput0)

    return hessobj.de

def _e_hcore_generator(hessobj, dm):
    with_x2c = getattr(hessobj.base, 'with_x2c', None)
    if with_x2c:
        hcore_deriv = with_x2c.hcore_deriv_generator(deriv=2)
        dm = dm.get()
        return lambda i, j: cp.asarray(np.einsum('xypq,pq->xy', hcore_deriv(i, j), dm))

    assert dm.dtype == np.float64
    mol = hessobj.mol
    log = logger.new_logger(mol)
    t1 = log.init_timer()
    de_nuc_elec = hess_nuc_elec(mol, dm)
    t1 = log.timer_debug1('hess_nuc_elec', *t1)
    dm = dm.get()
    with_ecp = mol.has_ecp()
    if with_ecp:
        ecp_atoms = set(mol._ecpbas[:,ATOM_OF])
    else:
        ecp_atoms = ()
    aoslices = mol.aoslice_by_atom()
    nbas = mol.nbas
    nao = mol.nao_nr()

    # Move data to GPU, get_hcore is slow on CPU
    h1aa, h1ab = hessobj.get_hcore(mol)
    h1aa = cupy.asarray(h1aa)
    h1ab = cupy.asarray(h1ab)

    hcore = cupy.empty((3,3,nao,nao))
    t1 = log.timer_debug1('get_hcore', *t1)
    def get_hcore(iatm, jatm):
        nonlocal hcore
        ish0, ish1, i0, i1 = aoslices[iatm]
        jsh0, jsh1, j0, j1 = aoslices[jatm]
        rinv2aa = rinv2ab = None
        if iatm == jatm:
            with mol.with_rinv_at_nucleus(iatm):
                # The remaining integrals like int1e_ipiprinv are computed in
                # hess_nuc_elec(mol, dm)
                if with_ecp and iatm in ecp_atoms:
                    rinv2aa = -mol.intor('ECPscalar_ipiprinv', comp=9)
                    rinv2ab = -mol.intor('ECPscalar_iprinvip', comp=9)
                    rinv2aa = cupy.asarray(rinv2aa)
                    rinv2ab = cupy.asarray(rinv2ab)
                    rinv2aa = rinv2aa.reshape(3,3,nao,nao)
                    rinv2ab = rinv2ab.reshape(3,3,nao,nao)
            hcore[:] = 0.
            hcore[:,:,i0:i1] += h1aa[:,:,i0:i1]
            hcore[:,:,i0:i1,i0:i1] += h1ab[:,:,i0:i1,i0:i1]
            if rinv2aa is not None or rinv2ab is not None:
                hcore -= rinv2aa + rinv2ab
                hcore[:,:,i0:i1] += rinv2aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2ab[:,:,i0:i1]
                hcore[:,:,:,i0:i1] += rinv2aa[:,:,i0:i1].transpose(0,1,3,2)
                hcore[:,:,:,i0:i1] += rinv2ab[:,:,:,i0:i1]
        else:
            hcore[:] = 0.
            hcore[:,:,i0:i1,j0:j1] += h1ab[:,:,i0:i1,j0:j1]
            with mol.with_rinv_at_nucleus(iatm):
                if with_ecp and iatm in ecp_atoms:
                    shls_slice = (jsh0, jsh1, 0, nbas)
                    rinv2aa = -mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = -mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa = cupy.asarray(rinv2aa)
                    rinv2ab = cupy.asarray(rinv2ab)
                    hcore[:,:,j0:j1] += rinv2aa.reshape(3,3,j1-j0,nao)
                    hcore[:,:,j0:j1] += rinv2ab.reshape(3,3,j1-j0,nao).transpose(1,0,2,3)
            with mol.with_rinv_at_nucleus(jatm):
                if with_ecp and jatm in ecp_atoms:
                    shls_slice = (ish0, ish1, 0, nbas)
                    rinv2aa = -mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = -mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa = cupy.asarray(rinv2aa)
                    rinv2ab = cupy.asarray(rinv2ab)
                    hcore[:,:,i0:i1] += rinv2aa.reshape(3,3,i1-i0,nao)
                    hcore[:,:,i0:i1] += rinv2ab.reshape(3,3,i1-i0,nao)
        de = cupy.einsum('xypq,pq->xy', hcore, dm)
        de += cupy.einsum('xyqp,pq->xy', hcore, dm)
        return cp.asarray(de + de_nuc_elec[:,:,iatm,jatm])
    return get_hcore

def hcore_generator(hessobj, mol=None):
    raise NotImplementedError

class HessianBase(lib.StreamObject):
    # attributes
    max_cycle   = rhf_hess_cpu.HessianBase.max_cycle
    level_shift = rhf_hess_cpu.HessianBase.level_shift
    _keys       = rhf_hess_cpu.HessianBase._keys

    # methods
    hess_elec       = rhf_hess_cpu.HessianBase.hess_elec
    make_h1         = rhf_hess_cpu.HessianBase.make_h1
    hcore_generator = hcore_generator  # the functionality is different from cpu version
    hess_nuc        = rhf_hess_cpu.HessianBase.hess_nuc
    kernel = hess = kernel

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1mo,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1mo,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        if hasattr(self.base, 'converged') and not self.base.converged:
            log.warn('Ground state %s not converged',
                     self.base.__class__.__name__)
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        log.info('Max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def to_cpu(self):
        mf = self.base.to_cpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

class Hessian(HessianBase):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.base = scf_method
        self.max_memory = self.mol.max_memory
        self.atmlst = None
        self.de = numpy.zeros((0,0,3,3))  # (A,B,dR_A,dR_B)
        self._keys = set(self.__dict__.keys())

    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1
    gen_hop = NotImplemented

# Inject to RHF class
from gpu4pyscf import scf
scf.hf.RHF.Hessian = lib.class_as_method(Hessian)
