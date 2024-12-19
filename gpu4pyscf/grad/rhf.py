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

import time
import ctypes
import math
import numpy as np
import cupy as cp
import cupy
import numpy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pyscf import lib, gto
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.lib import utils
from gpu4pyscf.scf.hf import KohnShamDFT
from gpu4pyscf.lib.cupy_helper import tag_array, contract, condense, sandwich_dot, reduce_to_device
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.__config__ import _streams, _num_devices
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, libvhf_rys, _VHFOpt, init_constant,
    _make_tril_tile_mappings, _nearest_power2)

libvhf_rys.RYS_per_atom_jk_ip1.restype = ctypes.c_int

__all__ = [
    'SCF_GradScanner',
    'Gradients',
    'Grad'
]

def _ejk_ip1_task(mol, dms, vhfopt, task_list, j_factor=1.0, k_factor=1.0,
                 device_id=0, verbose=0):
    n_dm = dms.shape[0]
    assert n_dm <= 2
    nao, _ = vhfopt.coeff.shape
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    kern = libvhf_rys.RYS_per_atom_jk_ip1

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

        ejk = cp.zeros((mol.natm, 3))

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
                    scheme = _ejk_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
                    err = kern(
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
                    if err != 0:
                        raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
                    if log.verbose >= logger.DEBUG1:
                        msg = f'processing {llll}, tasks = {info[1].get()} on Device {device_id}'
                        t1, t1p = log.timer_debug1(msg, *t1), t1
                        timing_counter[llll] += t1[1] - t1p[1]
                        kern_counts += 1
    return ejk, kern_counts, timing_counter

def _jk_energy_per_atom(mol, dm, vhfopt=None,
                        j_factor=1., k_factor=1., verbose=None):
    ''' Computes the first-order derivatives of the energy per atom for
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
                _ejk_ip1_task,
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

    log.timer_debug1('grad jk energy', *cput0)
    return ejk

def _ejk_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    g_size = (li+2)*(lj+2)*(lk+2)*(ll+2)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 1) // 2 + 1
    unit = nroots*2 + g_size*3 + ij_prims*4
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride

def get_dh1e_ecp(mol, dm):
    natom = mol.natm
    dh1e_ecp = cupy.zeros([natom,3])
    with_ecp = mol.has_ecp()
    if not with_ecp:
        return dh1e_ecp
    ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
    for ia in ecp_atoms:
        with mol.with_rinv_at_nucleus(ia):
            ecp = mol.intor('ECPscalar_iprinv', comp=3)
            dh1e_ecp[ia] = contract('xij,ij->x', cupy.asarray(ecp), dm)
    return 2.0 * dh1e_ecp

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of RHF/RKS gradients
    Args:
        mf_grad : grad.rhf.Gradients or grad.rks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    t0 = log.init_timer()

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    # (\nabla i | hcore | j) - (\nabla i | j)
    h1 = cupy.asarray(mf_grad.get_hcore(mol))
    s1 = cupy.asarray(mf_grad.get_ovlp(mol))

    # (i | \nabla hcore | j)
    t3 = log.init_timer()
    dh1e = int3c2e.get_dh1e(mol, dm0)

    if mol.has_ecp():
        dh1e += get_dh1e_ecp(mol, dm0)
    t3 = log.timer_debug1('gradients of h1e', *t3)

    dvhf = mf_grad.get_veff(mol, dm0)
    log.timer_debug1('gradients of veff', *t3)
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')

    dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    extra_force = cupy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())

    log.timer_debug1('gradients of 2e part', *t3)

    dh = contract('xij,ij->xi', h1, dm0)
    ds = contract('xij,ij->xi', s1, dme0)
    delec = 2.0*(dh - ds)

    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * dvhf + dh1e + delec + extra_force

    # for backforward compatiability
    if(hasattr(mf, 'disp') and mf.disp is not None):
        g_disp = mf_grad.get_dispersion()
        mf_grad.grad_disp = g_disp
        mf_grad.grad_mf = de

    log.timer_debug1('gradients of electronic part', *t0)
    return de.get()

def get_grad_hcore(mf_grad, mo_coeff=None, mo_occ=None):
    '''
    derivatives of Hcore in MO bases
    '''
    mf = mf_grad.base
    mol = mf.mol
    natm = mol.natm
    nao = mol.nao
    if mo_coeff is None: mo_coeff = cupy.asarray(mf.mo_coeff)
    if mo_occ is None: mo_occ = mf.mo_occ

    orbo = mo_coeff[:,mo_occ>0]
    nocc = orbo.shape[1]

    # derivative w.r.t nuclie position
    dh1e = cupy.zeros([natm,3,nao,nocc])
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
    orbo_sorted = intopt.sort_orbitals(orbo, axis=[0])
    mo_coeff_sorted = intopt.sort_orbitals(mo_coeff, axis=[0])
    for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e[k0:k1,:,j0:j1,:] += contract('xkji,io->kxjo', int3c_blk, orbo_sorted[i0:i1])
        dh1e[k0:k1,:,i0:i1,:] += contract('xkji,jo->kxio', int3c_blk, orbo_sorted[j0:j1])
    dh1e = contract('kxjo,k->kxjo', dh1e, -charges)
    dh1e = contract('kxjo,jp->kxpo', dh1e, mo_coeff_sorted)

    # derivative w.r.t. atomic orbitals
    h1 = mf_grad.get_hcore(mol)
    aoslices = mol.aoslice_by_atom()
    with_ecp = mol.has_ecp()
    if with_ecp:
        ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
    else:
        ecp_atoms = ()
    for atm_id in range(natm):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        h1ao = numpy.zeros([3,nao,nao])
        with mol.with_rinv_at_nucleus(atm_id):
            if with_ecp and atm_id in ecp_atoms:
                h1ao += mol.intor('ECPscalar_iprinv', comp=3)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        h1ao += h1ao.transpose([0,2,1])
        h1ao = cupy.asarray(h1ao)
        h1mo = contract('xij,jo->xio', h1ao, orbo)
        dh1e[atm_id] += contract('xio,ip->xpo', h1mo, mo_coeff)
    return dh1e

def as_scanner(mf_grad):
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad
    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)
    name = mf_grad.__class__.__name__ + SCF_GradScanner.__name_mixin__
    return lib.set_class(SCF_GradScanner(mf_grad),
                         (SCF_GradScanner, mf_grad.__class__), name)

class SCF_GradScanner(lib.GradScanner):
    def __init__(self, g):
        lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self.base
        e_tot = mf_scanner(mol)

        if isinstance(mf_scanner, KohnShamDFT):
            if getattr(self, 'grids', None):
                self.grids.reset(mol)
            if getattr(self, 'nlcgrids', None):
                self.nlcgrids.reset(mol)

        de = self.kernel(**kwargs)
        return e_tot, de

class GradientsBase(lib.StreamObject):
    '''
    Basic nuclear gradient functions for non-relativistic methods
    '''

    _keys = {'mol', 'base', 'unit', 'atmlst', 'de'}
    __init__    = rhf_grad_cpu.GradientsBase.__init__

    dump_flags  = rhf_grad_cpu.GradientsBase.dump_flags

    reset       = rhf_grad_cpu.GradientsBase.reset
    get_hcore   = rhf_grad_cpu.GradientsBase.get_hcore
    get_ovlp    = rhf_grad_cpu.GradientsBase.get_ovlp
    get_jk      = NotImplemented
    get_j       = NotImplemented
    get_k       = NotImplemented
    get_veff    = NotImplemented
    make_rdm1e  = rhf_grad_cpu.GradientsBase.make_rdm1e
    grad_nuc    = rhf_grad_cpu.GradientsBase.grad_nuc
    grad_elec   = NotImplemented
    optimizer   = rhf_grad_cpu.GradientsBase.optimizer
    extra_force = rhf_grad_cpu.GradientsBase.extra_force
    kernel      = rhf_grad_cpu.GradientsBase.kernel
    grad        = rhf_grad_cpu.GradientsBase.grad
    _finalize   = rhf_grad_cpu.GradientsBase._finalize
    _write      = rhf_grad_cpu.GradientsBase._write
    as_scanner  = as_scanner
    _tag_rdm1   = rhf_grad_cpu.GradientsBase._tag_rdm1


class Gradients(GradientsBase):

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    make_rdm1e = rhf_grad_cpu.Gradients.make_rdm1e
    grad_elec = grad_elec

    def get_veff(self, mol=None, dm=None, verbose=None):
        '''
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        vhfopt = self.base._opt_gpu.get(None, None)
        return _jk_energy_per_atom(mol, dm, vhfopt, verbose=verbose)

Grad = Gradients

from gpu4pyscf import scf
scf.hf.RHF.Gradients = lib.class_as_method(Gradients)
