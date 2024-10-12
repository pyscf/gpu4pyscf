# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

import time
import ctypes
import math
import numpy as np
import cupy as cp
import cupy
import numpy
from pyscf import lib, gto
from pyscf.gto import ATOM_OF
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.scf.hf import KohnShamDFT
from gpu4pyscf.lib.cupy_helper import tag_array, contract, take_last2d, condense
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.lib import logger
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, libvhf_rys, _VHFOpt, init_constant,
    _make_tril_tile_mappings, _nearest_power2)

libvhf_rys.RYS_per_atom_jk_ip1.restype = ctypes.c_int
libvhf_rys.RYS_build_jk_ip1.restype = ctypes.c_int

__all__ = [
    'get_jk',
    'SCF_GradScanner',
    'Gradients',
    'Grad'
]

def get_jk(mol, dm, with_j=True, with_k=True, atoms_slice=None, verbose=None):
    '''J = ((-nabla i) j| kl) D_lk
    K = ((-nabla i) j| kl) D_jk
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = cp.asarray(dms, order='C')

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
        vk = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vk, vhfopt.coeff)
        vk = vk + vk.transpose(0,2,1)
        vk = vk.reshape(atom1-atom0, 3, nao_orig, nao_orig)
    if with_j:
        vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vj, vhfopt.coeff)
        vj = vj + vj.transpose(0,2,1)
        vj *= 2.
        vj = vj.reshape(atom1-atom0, 3, nao_orig, nao_orig)
    log.timer('vj and vk gradients', *cput0)
    return vj, vk

def _jk_energy_per_atom(mol, dm, vhfopt=None, with_j=True, with_k=True, verbose=None):
    log = logger.new_logger(mol, verbose)
    cput0 = t1 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = cp.asarray(dms, order='C')
    assert n_dm <= 2

    vj = vk = None
    vj_ptr = vk_ptr = lib.c_null_ptr()

    assert with_j or with_k
    if with_k:
        vk = cp.zeros((mol.natm, 3))
        vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
    if with_j:
        vj = cp.zeros((mol.natm, 3))
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
    tile_mappings = _make_tril_tile_mappings(l_ctr_bas_loc, vhfopt.tile_q_cond,
                                             log_cutoff-log_max_dm)
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
    info = cp.empty(2, dtype=np.uint32)
    t1 = log.timer_debug1('dm_cond', *t1)

    timing_collection = {}
    kern_counts = 0
    kern = libvhf_rys.RYS_per_atom_jk_ip1

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
                    scheme = _ejk_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
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
                        raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
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

    if with_j:
        vj *= 2.
    log.timer_debug1('grad jk energy', *cput0)
    return vj, vk

def _ejk_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    g_size = (li+2)*(lj+2)*(lk+2)*(ll+2)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 1) // 2 + 1

    unit = nroots*2 + g_size*3 + ij_prims*4
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride

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

    # CPU tasks are executed on background
    def calculate_h1e(h1_gpu, s1_gpu):
        # (\nabla i | hcore | j) - (\nabla i | j)
        h1_cpu = mf_grad.get_hcore(mol)
        s1_cpu = mf_grad.get_ovlp(mol)
        h1_gpu[:] = cupy.asarray(h1_cpu)
        s1_gpu[:] = cupy.asarray(s1_cpu)
        return

    h1 = cupy.empty([3, dm0.shape[0], dm0.shape[1]])
    s1 = cupy.empty([3, dm0.shape[0], dm0.shape[1]])
    with lib.call_in_background(calculate_h1e) as calculate_hs:
        calculate_hs(h1, s1)
        # (i | \nabla hcore | j)
        dh1e = int3c2e.get_dh1e(mol, dm0)
        if mol.has_ecp():
            dh1e += get_dh1e_ecp(mol, dm0)
        t1 = log.timer_debug1('gradients of h1e', *t0)

        vhfopt = mf._opt_gpu.get(None, None)
        ej, ek = _jk_energy_per_atom(mol, dm0, vhfopt, verbose=log)
        veff = ej - ek * .5

        dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
        extra_force = cupy.zeros((len(atmlst),3))
        for k, ia in enumerate(atmlst):
            extra_force[k] += mf_grad.extra_force(ia, locals())

        log.timer_debug1('gradients of 2e part', *t1)

    dh = contract('xij,ij->xi', h1, dm0)
    ds = contract('xij,ij->xi', s1, dme0)
    delec = 2.0*(dh - ds)

    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = 2.0 * veff + dh1e + delec + extra_force

    # for backforward compatiability
    if(hasattr(mf, 'disp') and mf.disp is not None):
        g_disp = mf_grad.get_dispersion()
        mf_grad.grad_disp = g_disp
        mf_grad.grad_mf = de

    log.timer_debug1('gradients of electronic part', *t0)
    return de.get()

def get_grad_hcore(mf_grad, mo_coeff=None, mo_occ=None):
    '''
    derivative of hcore in MO
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
    dh1e = cupy.zeros([3,natm,nao,nocc])
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
    orbo_sorted = orbo[intopt.ao_idx]
    mo_coeff_sorted = mo_coeff[intopt.ao_idx]
    for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e[:,k0:k1,j0:j1,:] += contract('xkji,io->xkjo', int3c_blk, orbo_sorted[i0:i1])
        dh1e[:,k0:k1,i0:i1,:] += contract('xkji,jo->xkio', int3c_blk, orbo_sorted[j0:j1])
    dh1e = contract('xkjo,k->xkjo', dh1e, -charges)
    dh1e = contract('xkjo,jp->xkpo', dh1e, mo_coeff_sorted)

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
        dh1e[:,atm_id] += contract('xio,ip->xpo', h1mo, mo_coeff)
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
    get_jk      = get_jk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        with mol.with_range_coulomb(omega):
            vj, _ = self.get_jk(mol, dm, with_k=False, omega=omega)
        return vj

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        with mol.with_range_coulomb(omega):
            _, vk = self.get_jk(mol, dm, with_j=False, omega=omega)
        return vk

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

    # to_cpu can be reused only when __init__ still takes mf
    def to_cpu(self):
        mf = self.base.to_cpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

class Gradients(GradientsBase):
    from gpu4pyscf.lib.utils import to_gpu, device

    make_rdm1e = rhf_grad_cpu.Gradients.make_rdm1e
    grad_elec = grad_elec

Grad = Gradients

from gpu4pyscf import scf
scf.hf.RHF.Gradients = lib.class_as_method(Gradients)
