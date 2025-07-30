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
from pyscf.grad.dispersion import get_dispersion
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.lib import utils
from gpu4pyscf.scf.hf import KohnShamDFT
from gpu4pyscf.lib.cupy_helper import (
    tag_array, contract, condense, reduce_to_device, transpose_sum, ensure_numpy)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.scf import jk
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, libvhf_rys, _VHFOpt,
    _make_tril_pair_mappings, _nearest_power2)

__all__ = [
    'SCF_GradScanner',
    'Gradients',
    'Grad'
]

libvhf_rys.RYS_per_atom_jk_ip1.restype = ctypes.c_int
# The max. size of nf*nsq_per_block for each block.
# If shared memory is 48KB, this is enough to cache up to g-type functions,
# corresponding to 15^4 with nsq_per_block=2. All other cases require smaller
# cache for the product of density matrices. Although nsq_per_block would be
# larger, the overall cache requirements is smaller. The following code gives
# the size estimation for each angular momentum pattern (see also
# _ejk_quartets_scheme)
# for li, lj, lk, ll in itertools.product(*[range(LMAX+1)]*4):
#     nf = (li+1)*(li+2) * (lj+1)*(lj+2) * (lk+1)*(lk+2) * (ll+1)*(ll+2) // 16
#     g_size = (li+2)*(lj+1)*(lk+2)*(ll+1)
#     dd_cache_size = nf * min(THREADS, _nearest_power2(SHM_SIZE//(g_size*3*8)))
DD_CACHE_MAX = 101250 * (SHM_SIZE//48000)

libvhf_rys.RYS_build_vjk_ip1_init(ctypes.c_int(SHM_SIZE))

def _jk_energy_per_atom(mol, dm, vhfopt=None,
                        j_factor=1., k_factor=1., verbose=None):
    ''' Computes the first-order derivatives of the energy per atom for
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
        kern = libvhf_rys.RYS_per_atom_jk_ip1

        _dms = cp.asarray(dms, order='C')
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)

        ejk = cp.zeros((mol.natm, 3))

        dm_cond = cp.log(condense('absmax', _dms, ao_loc) + 1e-300).astype(np.float32)
        q_cond = cp.asarray(vhfopt.q_cond)
        log_max_dm = float(dm_cond.max())
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        pair_mappings = _make_tril_pair_mappings(
            l_ctr_bas_loc, q_cond, log_cutoff-log_max_dm, tile=6)
        rys_envs = vhfopt.rys_envs
        workers = gpu_specs['multiProcessorCount']
        # An additional integer to count for the proccessed pair_ijs 
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
            scheme = _ejk_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            for pair_kl0, pair_kl1 in lib.prange(0, npairs_kl, QUEUE_DEPTH):
                _pair_kl_mapping = pair_kl_mapping[pair_kl0:]
                _npairs_kl = pair_kl1 - pair_kl0
                err = kern(
                    ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                    ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                    ctypes.cast(_dms.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(nao),
                    rys_envs, (ctypes.c_int*2)(*scheme),
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
                if err != 0:
                    raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
            if log.verbose >= logger.DEBUG1:
                ntasks = npairs_ij * npairs_kl
                msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                t1, t1p = log.timer_debug1(msg, *t1), t1
                timing_counter[llll] += t1[1] - t1p[1]
                kern_counts += 1
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

    log.timer_debug1('grad jk energy', *cput0)
    return ejk

def _ejk_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    g_size = (li+2)*(lj+1)*(lk+2)*(ll+1)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = (order + 1) // 2 + 1
    unit = nroots*2 + g_size*3 + 6
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = (shm_size - ij_prims*8) // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    return n, gout_stride

def get_dh1e_ecp(mol, dm):
    '''
    Nuclear gradients of core Hamiltonian due to ECP
    '''
    with_ecp = mol.has_ecp()
    if not with_ecp:
        raise RuntimeWarning("ECP not found")
    
    h1_ecp = get_ecp_ip(mol)
    dh1e_ecp = contract('nxij,ij->nx', h1_ecp, dm)
    return 2.0 * dh1e_ecp

def get_hcore(mf, mol, exclude_ecp=False):
    '''
    Nuclear gradients of core Hamiltonian
    '''
    h = mol.intor('int1e_ipkin', comp=3)
    if mol._pseudo:
        NotImplementedError('Nuclear gradients for GTH PP')
    else:
        h += mol.intor('int1e_ipnuc', comp=3)
    h = cupy.asarray(h)
    if not exclude_ecp and mol.has_ecp():
        h += get_ecp_ip(mol).sum(axis=0)
    return -h

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
    t0 = t3 = log.init_timer()

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    # (\nabla i | hcore | j) - (\nabla i | j)
    h1 = cupy.asarray(mf_grad.get_hcore(mol, exclude_ecp=True))
    s1 = cupy.asarray(mf_grad.get_ovlp(mol))

    # (i | \nabla hcore | j)
    dh1e = int3c2e.get_dh1e(mol, dm0)

    # Calculate ECP contributions in (i | \nabla hcore | j) and 
    # (\nabla i | hcore | j) simultaneously
    if mol.has_ecp():
        # TODO: slice ecp_atoms
        ecp_atoms = sorted(set(mol._ecpbas[:,gto.ATOM_OF]))
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        h1 -= h1_ecp.sum(axis=0)

        dh1e[ecp_atoms] += 2.0 * contract('nxij,ij->nx', h1_ecp, dm0)
    t3 = log.timer_debug1('gradients of h1e', *t3)

    dvhf = mf_grad.get_veff(mol, dm0)
    log.timer_debug1('gradients of veff', *t3)
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')

    dm0 = tag_array(dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)
    extra_force = np.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += ensure_numpy(mf_grad.extra_force(ia, locals()))

    log.timer_debug1('gradients of 2e part', *t3)

    dh = contract('xij,ij->xi', h1, dm0)
    ds = contract('xij,ij->xi', s1, dme0)
    delec = 2.0*(dh - ds)

    delec = cupy.asarray([cupy.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:,2:]])
    de = ensure_numpy(2.0 * dvhf + dh1e + delec)
    de += extra_force
    log.timer_debug1('gradients of electronic part', *t0)
    return de

def get_grad_hcore(mf_grad, mo_coeff=None, mo_occ=None):
    '''
    Calculate derivatives of Hcore in MO bases
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
    h1 = cupy.asarray(mf_grad.get_hcore(mol))
    aoslices = mol.aoslice_by_atom()
    
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        h1mo = contract('xij,jo->xio', h1[:,p0:p1], orbo)
        dh1e[atm_id] += contract('xio,ip->xpo', h1mo, mo_coeff[p0:p1])
        h1mo = contract('xij,jp->xpi', h1[:,p0:p1], mo_coeff)
        dh1e[atm_id] += contract('xpi,io->xpo', h1mo, orbo[p0:p1])

    # Contributions due to ECP
    if mol.has_ecp():
        ecp_atoms = sorted(set(mol._ecpbas[:,gto.ATOM_OF]))
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        h1_ecp = h1_ecp + h1_ecp.transpose([0,1,3,2])
        h1mo = contract('nxij,jo->nxio', h1_ecp, orbo)
        dh1e[ecp_atoms] += contract('nxio,ip->nxpo', h1mo, mo_coeff)

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
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self.base
        e_tot = mf_scanner(mol)

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
    get_hcore   = get_hcore
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

    get_dispersion = get_dispersion

    @property
    def grad_disp(self):
        logger.warn(self, 'Attributes grad_disp and grad_mf are deprecated. '
                    'They will be removed in the future')
        g_disp = 0
        mf = self.base
        if hasattr(mf, 'disp') and mf.disp is not None:
            g_disp = self.get_dispersion()
        return g_disp

    @property
    def grad_mf(self):
        return self.de - self.grad_disp


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
        vhfopt = self.base._opt_gpu.get(mol.omega)
        return _jk_energy_per_atom(mol, dm, vhfopt, verbose=verbose)

Grad = Gradients
