#!/usr/bin/env python
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
Non-relativistic RHF analytical Hessian
'''

import math
import ctypes
import numpy
import cupy
import cupy as cp
import numpy as np
from pyscf.hessian import rhf as rhf_hess_cpu
from pyscf import lib
from pyscf.gto import ATOM_OF
# import _response_functions to load gen_response methods in SCF class
from gpu4pyscf.scf import _response_functions  # noqa
from gpu4pyscf.gto.mole import sort_atoms
from gpu4pyscf.scf import cphf
from gpu4pyscf.lib.cupy_helper import (
    contract, tag_array, sandwich_dot, transpose_sum, get_avail_mem, condense)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, libvhf_rys, _VHFOpt, init_constant,
    _make_tril_tile_mappings, _nearest_power2)
from gpu4pyscf.grad import rhf as rhf_grad

libvhf_rys.RYS_per_atom_jk_ip2.restype = ctypes.c_int
libvhf_rys.RYS_build_jk_ip1.restype = ctypes.c_int

GB = 1024*1024*1024
ALIGNED = 4

def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1mo=None,
              atmlst=None, max_memory=4000, verbose=None):
    ''' Different from PySF, using h1mo instead of h1ao for saving memory
    '''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)
    t1 = log.timer_debug1('hess elec', *t1)
    if h1mo is None:
        h1mo = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        t1 = log.timer_debug1('making H1', *t1)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)
    
    nao = mo_coeff.shape[0]
    mocc = cupy.array(mo_coeff[:,mo_occ>0])
    mo_energy = cupy.array(mo_energy)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    s1a = cupy.asarray(s1a)
    aoslices = mol.aoslice_by_atom()
    if atmlst is None:
        atmlst = range(mol.natm)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = cupy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        tmp = contract('xpq,pi->xiq', s1ao, mocc)
        s1oo = contract('xiq,qj->xij', tmp, mocc)

        s1mo = contract('xij,ip->xpj', s1ao, mo_coeff)

        for j0 in range(i0+1):
            ja = atmlst[j0]
            q0, q1 = aoslices[ja][2:]
# *2 for double occupancy, *2 for +c.c.
            de2[i0,j0] += contract('xpi,ypi->xy', h1mo[ia], mo1[ja]) * 4
            dm1 = contract('ypi,qi->ypq', mo1[ja], mocc*mo_energy[mo_occ>0])
            de2[i0,j0] -= contract('xpq,ypq->xy', s1mo, dm1) * 4
            de2[i0,j0] -= contract('xpq,ypq->xy', s1oo, mo_e1[ja]) * 2
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('RHF hessian', *time0)

    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True):
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
    ej, ek = _partial_ejk_ip2(mol, dm0, vhfopt, with_k, verbose=log)
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
    return e1, ej, ek

def _partial_ejk_ip2(mol, dm, vhfopt=None, with_k=True, verbose=None):
    assert mol.omega >= 0
    log = logger.new_logger(mol, verbose)
    cput0 = t1 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')
    assert n_dm <= 2

    natm = mol.natm
    ej = cp.zeros((natm, natm, 3, 3))
    ek = cp.zeros((natm, natm, 3, 3))
    vj_ptr = ctypes.cast(ej.data.ptr, ctypes.c_void_p)
    if with_k:
        vk_ptr = ctypes.cast(ek.data.ptr, ctypes.c_void_p)
    else:
        vk_ptr = lib.c_null_ptr()

    init_constant(mol)
    ao_loc = mol.ao_loc
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_max_dm = dm_cond.max()
    log_cutoff = math.log(vhfopt.direct_scf_tol)

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    assert uniq_l.max() <= LMAX
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = len(uniq_l_ctr)
    tile_mappings = _make_tril_tile_mappings(
        l_ctr_bas_loc, vhfopt.tile_q_cond, log_cutoff-log_max_dm)
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
    info = cp.empty(2, dtype=np.uint32)
    t1 = log.timer_debug1('dm_cond', *t1)

    timing_collection = {}
    kern_counts = 0
    kern = libvhf_rys.RYS_per_atom_jk_ip2

    for i in range(n_groups):
        for j in range(i+1):
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
                    err = kern(
                        vj_ptr, vk_ptr, ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
                        (ctypes.c_int*8)(*ij_shls, *kl_shls),
                        ctypes.c_int(tile_ij_mapping.size),
                        ctypes.c_int(tile_kl_mapping.size),
                        ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                        ctypes.cast(info.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(workers),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    if err != 0:
                        raise RuntimeError(f'RYS_per_atom_jk_ip2 kernel for {llll} failed')
                    if log.verbose >= logger.DEBUG1:
                        t1, t1p = log.timer_debug1(f'processing {llll}, tasks = {info[1]}', *t1), t1
                        if llll not in timing_collection:
                            timing_collection[llll] = 0
                        timing_collection[llll] += t1[1] - t1p[1]
                        kern_counts += 1

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    # *8 for the symmetry (i,j) = (j,i), (k,l) = (l,k) and (ij,kl) = (kl,ij)
    # The additional factor 1/2 is from the two-electron Coulomb operator
    ej *= 4
    if n_dm == 2:
        # corresponding to the symmetry (i,j) = (j,i) and (k,l) = (l,k) for UHF
        # density matrices. Including the additional factor 1/2 from operator,
        # ek * 2 is required. For RHF, dm=2*dm_a, a factor of 4 has been
        # included, which is cancelled by the contribution from dm_b (a
        # factor of 2), the symmetry between i,j and k,l (a factor of 4), and
        # the Coulomb operator (1/2). ek does not need to be scaled in RHF.
        ek *= 2
    log.timer_debug1('ejk_ip2', *cput0)
    return ej, ek

def _ip2_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    g_size = (li+2)*(lj+2)*(lk+2)*(ll+2)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 2) // 2 + 1

    unit = nroots*2 + g_size*3 + ij_prims*4
    counts = shm_size // (unit*8)
    n = THREADS // 16
    while n >= counts:
        n >>= 1
    gout_stride = THREADS // n
    return n, gout_stride

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
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

    vj = vk = None
    vj_ptr = vk_ptr = lib.c_null_ptr()
    assert with_j or with_k
    if with_k:
        vk = cp.zeros(((atom1-atom0)*3, nao, nao))
        vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
    if with_j:
        vj = cp.zeros(((atom1-atom0)*3, nao, nao))
        vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)

    init_constant(mol)
    ao_loc = mol.ao_loc
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_max_dm = dm_cond.max()
    log_cutoff = math.log(vhfopt.direct_scf_tol)

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    assert uniq_l.max() <= LMAX
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = len(uniq_l_ctr)
    tril_tile_mappings = _make_tril_tile_mappings(
        l_ctr_bas_loc, vhfopt.tile_q_cond, log_cutoff-log_max_dm, 1)
    workers = gpu_specs['multiProcessorCount']
    QUEUE_DEPTH = 65536 # See rys_contract_jk_ip1 kernel
    pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
    info = cp.empty(2, dtype=np.uint32)
    t1 = log.timer_debug1('q_cond and dm_cond', *cput0)

    timing_collection = {}
    kern_counts = 0
    kern = libvhf_rys.RYS_build_jk_ip1

    nbas = mol.nbas
    assert vhfopt.tile_q_cond.shape == (nbas, nbas)
    for i in range(n_groups):
        for j in range(n_groups):
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
                        t1, t1p = log.timer_debug1(f'processing {llll}, tasks = {info[1]}', *t1), t1
                        if llll not in timing_collection:
                            timing_collection[llll] = 0
                        timing_collection[llll] += t1[1] - t1p[1]
                        kern_counts += 1

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    if with_k:
        vk = sandwich_dot(vk, vhfopt.coeff)
        vk = transpose_sum(vk)
        vk = vk.reshape(atom1-atom0, 3, nao_orig, nao_orig)
    if with_j:
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
    '''Solve the first order equation
    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    nao = mo_coeff.shape[0]
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    s1a = cupy.asarray(s1a)

    def _ao2mo(mat):
        tmp = contract('xij,jo->xio', mat, mocc)
        return contract('xik,ip->xpk', tmp, mo_coeff)
    cupy.get_default_memory_pool().free_all_blocks()

    avail_mem = get_avail_mem()
    blksize = int(avail_mem*0.4) // (8*3*nao*nao*4) // ALIGNED * ALIGNED
    blksize = min(32, blksize)
    log.debug(f'GPU memory {avail_mem/GB:.1f} GB available')
    log.debug(f'{blksize} atoms in each block CPHF equation')

    # sort atoms to improve the convergence
    sorted_idx = sort_atoms(mol)
    atom_groups = []
    for p0,p1 in lib.prange(0,mol.natm,blksize):
        blk = sorted_idx[p0:p1]
        atom_groups.append(blk)

    mo1s = [None] * mol.natm
    e1s = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()

    for group in atom_groups:
        s1vo = []
        h1vo = []
        for ia in group:
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = cupy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1vo.append(_ao2mo(s1ao))
            h1vo.append(h1mo[ia])

        log.info(f'Solving CPHF equation for atoms {len(group)}/{mol.natm}')
        h1vo = cupy.vstack(h1vo)
        s1vo = cupy.vstack(s1vo)
        tol = mf.conv_tol_cpscf
        mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                             level_shift=level_shift, tol=tol, verbose=verbose)

        mo1 = mo1.reshape(-1,3,nao,nocc)
        e1 = e1.reshape(-1,3,nocc,nocc)

        for k, ia in enumerate(group):
            mo1s[ia] = mo1[k]
            e1s[ia] = e1[k].reshape(3,nocc,nocc)
        mo1 = e1 = None
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
