# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto
from pyscf.gto import ATOM_OF
from gpu4pyscf.gto.ecp import get_ecp_ip, get_ecp_ipip
from gpu4pyscf.scf import cphf, j_engine
from gpu4pyscf.lib.cupy_helper import (reduce_to_device,
    contract, tag_array, transpose_sum, get_avail_mem, condense,
    krylov)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib import utils
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, GROUP_SIZE, libvhf_rys, _VHFOpt,
    init_constant, _make_tril_tile_mappings, _make_tril_pair_mappings,
    _nearest_power2)
from gpu4pyscf.grad import rhf as rhf_grad

libvhf_rys.RYS_per_atom_jk_ip2_type12.restype = ctypes.c_int
libvhf_rys.RYS_per_atom_jk_ip2_type3.restype = ctypes.c_int
libvhf_rys.RYS_build_jk_ip1.restype = ctypes.c_int

GB = 1024*1024*1024
ALIGNED = 4
DD_CACHE_MAX = rhf_grad.DD_CACHE_MAX

libvhf_rys.RYS_build_vjk_ip1_init(ctypes.c_int(SHM_SIZE))
libvhf_rys.RYS_build_ejk_ip2_init(ctypes.c_int(SHM_SIZE))

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
        fx = hessobj.gen_vind(mo_coeff, mo_occ)
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo,
                                       fx, atmlst, max_memory, log)
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
    if atmlst is None:
        atmlst = range(mol.natm)

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = mocc.dot(mocc.T) * 2
    vhfopt = mf._opt_gpu.get(mol.omega)
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

def _partial_ejk_ip2(mol, dm, vhfopt=None, j_factor=1., k_factor=1., verbose=None):
    '''Compute the energy per atom for
        j_factor * J_derivatives - k_factor * K_derivatives
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol, tile=1).build()
    assert vhfopt.tile == 1

    mol = vhfopt.sorted_mol
    nao_orig = vhfopt.mol.nao

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)

    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    n_dm, nao = dms.shape[:2]
    assert n_dm <= 2

    ao_loc = mol.ao_loc
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert uniq_l.max() <= LMAX

    n_groups = len(uniq_l_ctr)
    tasks = ((i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1))

    def proc():
        device_id = cp.cuda.device.get_device_id()
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()

        timing_counter = Counter()
        kern_counts = 0
        kern1 = libvhf_rys.RYS_per_atom_jk_ip2_type12
        kern2 = libvhf_rys.RYS_per_atom_jk_ip2_type3

        _dms = cp.asarray(dms, order='C')
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)

        natm = mol.natm
        ejk = cp.zeros((natm, natm, 3, 3))

        dm_cond = cp.log(condense('absmax', _dms, ao_loc) + 1e-300).astype(np.float32)
        q_cond = cp.asarray(vhfopt.q_cond)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        pair_mappings = _make_tril_pair_mappings(
            l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
        rys_envs = vhfopt.rys_envs
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)
        dd_pool = cp.empty((workers, DD_CACHE_MAX), dtype=np.float64)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j, k, l in tasks:
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            pair_ij_mapping = pair_mappings[i,j]
            pair_kl_mapping = pair_mappings[k,l]
            npairs_ij = pair_ij_mapping.size
            npairs_kl = pair_kl_mapping.size
            if npairs_ij == 0 or npairs_kl == 0:
                continue
            scheme1 = _ip2_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            scheme3 = _ip2_type3_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                _npairs_kl = pair_kl1 - pair_kl0
                err1 = kern1(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                    ctypes.cast(_dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    rys_envs, (ctypes.c_int*2)(*scheme1),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    s_ptr,
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)

                err2 = kern2(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                    ctypes.cast(_dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    rys_envs, (ctypes.c_int*2)(*scheme3),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    s_ptr,
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)

                if err1 != 0 or err2 != 0:
                    raise RuntimeError(f'RYS_per_atom_jk_ip2 kernel for {llll} failed')
            if log.verbose >= logger.DEBUG1:
                ntasks = npairs_ij * npairs_kl
                msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                t1, t1p = log.timer_debug1(msg, *t1), t1
                timing_counter[llll] += t1[1] - t1p[1]
                kern_counts += 1

        ejk = ejk + ejk.transpose(1,0,3,2)
        return ejk, kern_counts, timing_counter

    results = multi_gpu.run(proc, non_blocking=True)

    kern_counts = 0
    timing_collection = Counter()
    ejk_dist = []
    for ejk, counts, counter in results:
        kern_counts += counts
        timing_collection += counter
        ejk_dist.append(ejk)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    ejk = reduce_to_device(ejk_dist, inplace=True)

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
    unit = nroots*2 + g_size*3 + 8
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = (shm_size - ij_prims*8) // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride

def _ip2_type3_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    g_size = (li+2)*(lj+1)*(lk+2)*(ll+1)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 2) // 2 + 1
    unit = nroots*2 + g_size*3 + 8
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = (shm_size - ij_prims*8) // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''Compute the first order Fock matrix (the force term of the CPHF
    equation). Returns of this function are different to the PySCF CPU version.
    This function returns matrices in the MO-occupied_orb basis, while the CPU
    version returns matrices in MO basis.
    '''
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst == range(natm)
    mo_coeff = cp.asarray(mo_coeff)
    mocc = cp.asarray(mo_coeff[:,mo_occ>0])
    dm0 = mocc.dot(mocc.T) * 2
    h1mo = rhf_grad.get_grad_hcore(hessobj.base.Gradients())

    # Estimate the size of intermediate variables
    # dm, vj, and vk in [natm,3,nao_cart,nao_cart]
    nao_cart = mol.nao_cart()
    avail_mem = get_avail_mem()
    slice_size = int(avail_mem*0.5) // (8*3*nao_cart*nao_cart*3)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vj, vk = _get_jk_ip1(mol, dm0, atoms_slice=atoms_slice, verbose=verbose)
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

def _get_jk_ip1(mol, dm, with_j=True, with_k=True, atoms_slice=None, verbose=None):
    r'''
    For each atom, compute
    J = ((\nabla_X i) j| kl) (D_lk + D_ji)
    K = ((\nabla_X i) j| kl) (D_jk + D_li)
    '''
    assert mol.omega >= 0
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    vhfopt = _VHFOpt(mol, tile=1).build()

    mol = vhfopt.sorted_mol
    nao_orig = vhfopt.mol.nao

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    n_dm, nao = dms.shape[:2]
    assert n_dm == 1

    natm = mol.natm
    if atoms_slice is None:
        atoms_slice = 0, natm
    atom0, atom1 = atoms_slice

    ao_loc = mol.ao_loc
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert uniq_l.max() <= LMAX

    n_groups = len(uniq_l_ctr)
    tasks = ((i, j, k, l)
             for i in range(n_groups)
             for j in range(n_groups)
             for k in range(n_groups)
             for l in range(k+1))

    def proc():
        device_id = cp.cuda.device.get_device_id()
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        timing_counter = Counter()
        kern_counts = 0
        kern = libvhf_rys.RYS_build_jk_ip1

        _dms = cp.asarray(dms)

        vj = vk = None
        vj_ptr = vk_ptr = lib.c_null_ptr()
        assert with_j or with_k
        if with_k:
            vk = cp.zeros(((atom1-atom0)*3, nao, nao))
            vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
        if with_j:
            vj = cp.zeros(((atom1-atom0)*3, nao, nao))
            vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)

        dm_cond = cp.log(condense('absmax', _dms, ao_loc) + 1e-300).astype(np.float32)
        q_cond = cp.asarray(vhfopt.q_cond)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        cutoff = log_cutoff - log_max_dm
        pair_kl_mappings = _make_tril_pair_mappings(
            l_ctr_bas_loc, q_cond, cutoff, tile=6)
        pair_ij_mappings = _make_pair_mappings_for_atoms_slice(
            mol, l_ctr_bas_loc, q_cond, cutoff, atoms_slice, tile=6)

        rys_envs = vhfopt.rys_envs
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j, k, l in tasks:
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            pair_ij_mapping = pair_ij_mappings[i,j]
            pair_kl_mapping = pair_kl_mappings[k,l]
            npairs_ij = pair_ij_mapping.size
            npairs_kl = pair_kl_mapping.size
            if npairs_ij == 0 or npairs_kl == 0:
                continue

            scheme = _ip1_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                _npairs_kl = pair_kl1 - pair_kl0
                err = kern(
                    vj_ptr, vk_ptr, ctypes.cast(_dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao), ctypes.c_int(atom0),
                    rys_envs, (ctypes.c_int*2)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(_npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(_pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
                    lib.c_null_ptr(),
                    ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                if err != 0:
                    raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
            if log.verbose >= logger.DEBUG1:
                ntasks = npairs_ij * npairs_kl
                msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                t1, t1p = log.timer_debug1(msg, *t1), t1
                timing_counter[llll] += t1[1] - t1p[1]
                kern_counts += 1
        return vj, vk, kern_counts, timing_counter

    results = multi_gpu.run(proc, non_blocking=True)

    kern_counts = 0
    timing_collection = Counter()
    vj_dist = []
    vk_dist = []
    for vj, vk, counts, counter in results:
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
        vk = vhfopt.apply_coeff_CT_mat_C(vk)
        vk = transpose_sum(vk)
        vk = vk.reshape(atom1-atom0, 3, nao_orig, nao_orig)
    if with_j:
        vj = reduce_to_device(vj_dist, inplace=True)
        vj = vhfopt.apply_coeff_CT_mat_C(vj)
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

    unit = nroots*2 + g_size*3 + 9
    counts = (shm_size - ij_prims * 8) // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    gout_width = 27
    while gout_stride < 16 and gout_size / (gout_stride*gout_width) > 1:
        n //= 2
        gout_stride *= 2
    return n, gout_stride

def _make_pair_mappings_for_atoms_slice(mol, l_ctr_bas_loc, q_cond, cutoff,
                                        atoms_slice, tile=4):
    nbas = q_cond.shape[0]
    atom0, atom1 = atoms_slice
    mask = q_cond > cutoff
    mask[(mol._bas[:,ATOM_OF] <  atom0) |
         (mol._bas[:,ATOM_OF] >= atom1)] = False
    mask = mask.ravel()
    n_groups = len(l_ctr_bas_loc) - 1
    pair_mappings = {}
    tile = 6
    for i in range(n_groups):
        for j in range(n_groups):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            nish = ish1 - ish0
            njsh = jsh1 - jsh0
            ntiles_i = (nish+tile-1) // tile
            ntiles_j = (njsh+tile-1) // tile
            pair_ij = (cp.arange(ish0, ish0+ntiles_i*tile, dtype=np.int32)[:,None] * nbas +
                       cp.arange(jsh0, jsh0+ntiles_j*tile, dtype=np.int32))
            pair_ij = pair_ij.reshape(ntiles_i,tile,ntiles_j,tile).transpose(0,2,1,3)
            ish = cp.arange(ish0, ish0+ntiles_i*tile, dtype=np.int32).reshape(ntiles_i,tile)
            jsh = cp.arange(jsh0, jsh0+ntiles_j*tile, dtype=np.int32).reshape(ntiles_j,tile)
            ish = ish[:,None,:,None]
            jsh = jsh[None,:,None,:]
            pair_ij = pair_ij[(ish < ish1) & (jsh < jsh1)]
            pair_ij = pair_ij[mask[pair_ij]]
            pair_mappings[i,j] = cp.asarray(pair_ij, dtype=np.int32)
    return pair_mappings

def get_hcore(mol):
    '''Part of the second derivatives of core Hamiltonian'''
    h1aa = mol.intor('int1e_ipipkin', comp=9)
    h1ab = mol.intor('int1e_ipkinip', comp=9)
    if mol._pseudo:
        NotImplementedError('Nuclear hessian for GTH PP')
    else:
        h1aa+= mol.intor('int1e_ipipnuc', comp=9)
        h1ab+= mol.intor('int1e_ipnucip', comp=9)
    h1aa = cupy.asarray(h1aa)
    h1ab = cupy.asarray(h1ab)
    if mol.has_ecp():
        #h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        #h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
        h1aa += get_ecp_ipip(mol, 'ipipv').sum(axis=0)
        h1ab += get_ecp_ipip(mol, 'ipvip').sum(axis=0)
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

    if cupy.any(cp.isinf(e_ai)) or cupy.any(cp.isnan(e_ai)):
        raise ValueError(f"e_ai = {e_ai} contains inf or nan, likely because HOMO-LUMO gap is zero.")

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
    blksize = int(min(avail_mem*.3 / (8*3*nao*nocc*4), # in MO
                      avail_mem*.3 / (8*nao*nao*3*3))) # vj, vk, dm in AO
    if blksize < ALIGNED**2:
        raise RuntimeError('GPU memory insufficient for solving CPHF equations')

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

def gen_vind(hessobj, mo_coeff, mo_occ):
    mol = hessobj.mol
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    mocc_2 = mocc * 2

    def fx(mo1):
        mo1 = cupy.asarray(mo1)
        mo1 = mo1.reshape(-1,nmo,nocc)
        mo1_mo = contract('npo,ip->nio', mo1, mo_coeff)
        dm1 = mo1_mo.dot(mocc_2.T)
        dm1 = transpose_sum(dm1)
        dm1 = tag_array(dm1, mo1=mo1_mo, occ_coeff=mocc, mo_occ=mo_occ)
        return hessobj.get_veff_resp_mo(mol, dm1, mo_coeff, mo_occ, hermi=1)
    return fx

def hess_nuc_elec(mol, dm):
    '''
    H1e hessian nuclear repulsion contribution that originates from differentiating nuclear position
    (both first derivative d2I_dAdC and second derivative d2I_dC2)
    w/o ECP
    '''
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    from gpu4pyscf.gto import int3c1e
    from gpu4pyscf.gto.int3c1e_ipip import int1e_grids_ip1ip2, int1e_grids_ipip2
    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    d2e = cupy.zeros([3, 3, mol.natm, mol.natm])

    for j_atom in range(mol.natm):
        # TODO: It is computing one charge at a time, and it's likely slow
        g0,g1 = j_atom,j_atom+1
        d2I_dAdC = int1e_grids_ip1ip2(mol, coords[g0:g1, :], charges = charges[g0:g1], intopt = intopt_derivative)

        for i_atom in range(mol.natm):
            p0,p1 = aoslice[i_atom, 2:]
            d2e[:, :, i_atom, j_atom] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :])
            d2e[:, :, i_atom, j_atom] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(0,1,3,2))

            d2e[:, :, j_atom, i_atom] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,2,3))
            d2e[:, :, j_atom, i_atom] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,3,2))
    d2I_dAdC = None

    d2I_dC2 = int1e_grids_ipip2(mol, coords, dm = dm, intopt = intopt_derivative)
    for i_atom in range(mol.natm):
        d2e[:, :, i_atom, i_atom] += d2I_dC2[:, :, i_atom] * charges[i_atom]
    d2I_dC2 = None

    return -d2e

def hess_nuc_elec_ecp(mol, dm):
    '''
    Calculate hessian contribution due to (nuc, elec) pair with ECP
    '''
    nao = mol.nao
    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    ecp_atoms = sorted(set(mol._ecpbas[:,ATOM_OF]))
    n_ecp_atm = len(ecp_atoms)
    de_ecp = cupy.zeros([3,3,natm,natm])
    rinv2aa = -get_ecp_ipip(mol, ip_type='ipipv').reshape(n_ecp_atm,3,3,nao,nao)
    for idx, atm_id in enumerate(ecp_atoms):
        de = contract('xypq,pq->xyp', rinv2aa[idx], dm)
        de = cupy.asarray([cupy.sum(de[:,:,p0:p1], axis=2) for p0,p1 in aoslices[:,2:]])
        de_ecp[:,:,atm_id] += de.transpose([1,2,0])
        de_ecp[:,:,:,atm_id] += de.transpose([2,1,0])
        
        # 2nd derivative on ECP basis
        de = contract('xypq,pq->xy', rinv2aa[idx], dm)
        de_ecp[:,:,atm_id,atm_id] -= de
    
    rinv2ab = -get_ecp_ipip(mol, ip_type='ipvip').reshape(n_ecp_atm,3,3,nao,nao)
    for idx, atm_id in enumerate(ecp_atoms):
        de = contract('xypq,pq->xyp', rinv2ab[idx], dm).transpose(1,0,2)
        de = cupy.asarray([cupy.sum(de[:,:,p0:p1], axis=2) for p0,p1 in aoslices[:,2:]])
        de_ecp[:,:,atm_id] += de.transpose([1,2,0])
        de_ecp[:,:,:,atm_id] += de.transpose([2,1,0])

        # 2nd derivative on ECP basis
        de = contract('xypq,pq->xy', rinv2ab[idx], dm)
        de_ecp[:,:,atm_id,atm_id] -= de

    return de_ecp

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
    aoslices = mol.aoslice_by_atom()
    if with_ecp:
        de_ecp = hess_nuc_elec_ecp(mol, dm)

    # Move data to GPU, get_hcore is slow on CPU
    h1aa, h1ab = hessobj.get_hcore(mol)
    h1aa = cupy.asarray(h1aa)
    h1ab = cupy.asarray(h1ab)

    t1 = log.timer_debug1('get_hcore', *t1)
    def get_hcore(iatm, jatm):
        i0, i1 = aoslices[iatm][2:]
        j0, j1 = aoslices[jatm][2:]
        if iatm == jatm:
            de = contract('xypq,pq->xy', h1aa[:,:,i0:i1], dm[i0:i1])
            de+= contract('xypq,pq->xy', h1ab[:,:,i0:i1,i0:i1], dm[i0:i1,i0:i1])
        else:
            de = contract('xypq,pq->xy', h1ab[:,:,i0:i1,j0:j1],dm[i0:i1,j0:j1])
        if with_ecp:
            de += de_ecp[:,:,iatm,jatm]
        # 2.0* due to the symmetry
        return cp.asarray(2.0*de + de_nuc_elec[:,:,iatm,jatm])
    return get_hcore

def hcore_generator(hessobj, mol=None):
    raise NotImplementedError

def _ao2mo(v_ao, mocc, mo_coeff):
    v_ao = contract('nij,jo->nio', v_ao, mocc)
    return contract('nio,ip->npo', v_ao, mo_coeff)

def _get_jk_mo(hessobj, mol, dms, mo_coeff, mo_occ,
            hermi=1, with_j=True, with_k=True, omega=None):
    ''' Compute J/K matrices in MO for multiple DMs
    '''
    assert hermi == 1
    mf = hessobj.base
    if omega is None:
        omega = mol.omega
    vj = vk = None
    nao = dms.shape[-1]
    dms = dms.reshape(-1,nao,nao)
    n_dm = len(dms)
    # When hessian obj is converted from CPU instance, _opt_jengine and _opt_gpu
    # might not be initialized
    if with_j:
        if omega not in mf._opt_jengine:
            mf._opt_jengine[omega] = j_engine._VHFOpt(mol, mf.direct_scf_tol).build()
        jopt = mf._opt_jengine[omega]
        _dms = jopt.apply_coeff_C_mat_CT(dms)
        vj = jopt.get_j(_dms, mf.verbose)
        _mo_coeff = jopt.apply_coeff_C_mat(mo_coeff)
        _mocc = _mo_coeff[:,mo_occ>0.5]
        vj = _ao2mo(vj, _mocc, _mo_coeff).reshape(n_dm,-1)
    if with_k:
        if omega not in mf._opt_gpu:
            with mol.with_range_coulomb(omega):
                mf._opt_gpu[omega] = _VHFOpt(mol, mf.direct_scf_tol, tile=1).build()
        kopt = mf._opt_gpu[omega]
        _dms = kopt.apply_coeff_C_mat_CT(dms)
        vk = kopt.get_k(_dms, hermi, mf.verbose)
        _mo_coeff = kopt.apply_coeff_C_mat(mo_coeff)
        _mocc = _mo_coeff[:,mo_occ>0.5]
        vk = _ao2mo(vk, _mocc, _mo_coeff).reshape(n_dm,-1)
    return vj, vk

def _get_veff_resp_mo(hessobj, mol, dms, mo_coeff, mo_occ, hermi=1, omega=None):
    vj, vk = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ,
                     hermi=hermi, with_j=True, with_k=True, omega=omega)
    return vj - 0.5 * vk

class HessianBase(lib.StreamObject):

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    # attributes
    max_cycle   = rhf_hess_cpu.HessianBase.max_cycle
    level_shift = rhf_hess_cpu.HessianBase.level_shift
    _keys       = rhf_hess_cpu.HessianBase._keys

    # methods
    hess_elec       = rhf_hess_cpu.HessianBase.hess_elec
    make_h1         = rhf_hess_cpu.HessianBase.make_h1
    hcore_generator = hcore_generator  # the functionality is different from cpu version
    hess_nuc        = rhf_hess_cpu.HessianBase.hess_nuc
    gen_vind        = NotImplemented
    get_jk          = NotImplemented
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

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(mol)
        return self

class Hessian(HessianBase):
    '''Non-relativistic restricted Hartree-Fock hessian'''

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
    gen_vind = gen_vind
    get_jk_mo = _get_jk_mo
    get_veff_resp_mo = _get_veff_resp_mo

# Inject to RHF class
from gpu4pyscf import scf
scf.hf.RHF.Hessian = lib.class_as_method(Hessian)
