# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.grad.dispersion import get_dispersion
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import (
    tag_array, contract, condense, transpose_sum, get_avail_mem, ndarray)
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.df import int3c2e      #TODO: move int3c2e to out of df
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.scf.jk import (
    LMAX, QUEUE_DEPTH, SHM_SIZE, THREADS, libvhf_rys, _VHFOpt,
    _nearest_power2, _TimingCollector)
from gpu4pyscf.gto.mole import groupby, extract_pgto_params, SortedMole
from gpu4pyscf.df.int3c2e_bdiv import (
    Int3c2eOpt, _int3c2e_ip1_evaluator, int3c2e_scheme, int3c2e_scheme_ip1)

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
DD_CACHE_MAX = 101250 * (SHM_SIZE//(45*1024))

def _jk_energy_per_atom(vhfopt, dm, j_factor=1., k_factor=1., verbose=None):
    '''
    Computes the first-order derivatives of the energy per atom for
    j_factor * J_derivatives - k_factor * K_derivatives
    '''
    log = logger.new_logger(vhfopt.mol, verbose)
    cput0 = log.init_timer()
    mol = vhfopt.sorted_mol

    dm = cp.asarray(dm, order='C')
    nao_orig = dm.shape[-1]
    dms = dm.reshape(-1,nao_orig,nao_orig)

    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    n_dm, nao = dms.shape[:2]
    assert n_dm <= 2

    ao_loc = mol.ao_loc
    uniq_l_ctr = mol.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = np.append(0, np.cumsum(mol.l_ctr_counts))
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    assert uniq_l.max() <= LMAX

    log_cutoff = math.log(vhfopt.direct_scf_tol)
    dm_penalty = 0
    diffuse_exps, diffuse_ctr_coef = extract_pgto_params(mol, 'diffuse')

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

        timing_collection = _TimingCollector(log.timer_debug1)
        kern_counts = 0
        kern = libvhf_rys.RYS_per_atom_jk_ip1

        _dms = cp.asarray(dms, order='C')
        ejk = cp.zeros((mol.natm, 3))

        dm_cond = cp.log(condense('absmax', _dms, ao_loc) + 1e-300).astype(np.float32)
        _diffuse_exps = cp.asarray(diffuse_exps, dtype=np.float32)
        bas_pair_cache = {k: [cp.asarray(x) for x in v]
                          for k, v in vhfopt.bas_pair_cache.items()}

        rys_envs = vhfopt.rys_envs
        workers = gpu_specs['multiProcessorCount']
        # An additional integer to count for the proccessed pair_ijs
        pool = cp.empty(workers*QUEUE_DEPTH+1, dtype=np.int32)
        dd_pool = cp.empty((workers, DD_CACHE_MAX), dtype=np.float64)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j, k, l in tasks:
            shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
            pair_ij_mapping, q_cond_ij, s_cond_ij = bas_pair_cache[i,j]
            pair_kl_mapping, q_cond_kl, s_cond_kl = bas_pair_cache[k,l]
            npairs_ij = pair_ij_mapping.size
            npairs_kl = pair_kl_mapping.size
            if npairs_ij == 0 or npairs_kl == 0:
                continue
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            scheme = _ejk_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            err = kern(
                ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                ctypes.c_double(j_factor), ctypes.c_double(k_factor),
                ctypes.cast(_dms.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_dm), ctypes.c_int(nao),
                rys_envs, (ctypes.c_int*2)(*scheme),
                (ctypes.c_int*8)(*shls_slice),
                ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
                ctypes.cast(_diffuse_exps.data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.c_float(dm_penalty),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
                mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
            if err != 0:
                raise RuntimeError(f'RYS_per_atom_jk_ip1 kernel for {llll} failed')
            kern_counts += 1
            if log.verbose >= logger.DEBUG1:
                ntasks = npairs_ij * npairs_kl
                msg = f'processing {llll} on Device {device_id} tasks ~= {ntasks}'
                t1 = timing_collection.collect(llll, t1, msg)
        return ejk, kern_counts, timing_collection

    results = multi_gpu.run(proc, non_blocking=True)
    ejk = multi_gpu.array_reduce([x[0] for x in results], inplace=True)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', sum(x[1] for x in results))
        _TimingCollector.summary(log.debug1, (x[2] for x in results))

    log.timer_debug1('grad jk energy', *cput0)
    return ejk.get()

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
    with_ecp = len(mol._ecpbas) > 0
    if not with_ecp:
        raise RuntimeWarning("ECP not found")

    h1_ecp = get_ecp_ip(mol)
    dh1e_ecp = contract('nxij,ij->nx', h1_ecp, dm)
    return 2.0 * dh1e_ecp

def get_hcore(mf, mol, exclude_ecp=False):
    '''
    Nuclear gradients of core Hamiltonian
    '''
    from gpu4pyscf.pbc.gto.int1e import int1e_ipkin
    if mol._pseudo:
        NotImplementedError('Nuclear gradients for GTH PP')

    if getattr(mf, 'with_x2c', None):
        raise NotImplementedError('X2C gradients')

    sorted_mol = SortedMole.from_mol(mol, decontract=True)
    h = int1e_ipkin(sorted_mol)
    h += int1e_ipnuc(sorted_mol)
    if not exclude_ecp and len(mol._ecpbas) > 0:
        h += get_ecp_ip(mol).sum(axis=0)
    return -h

def int1e_ipnuc(mol):
    '''
    cp.array(mol.intor('int1e_ipnuc'))
    '''
    coords = mol.atom_coords()
    charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)

    sorted_mol = SortedMole.from_mol(mol, decontract=True)
    opt = Int3c2eOpt(sorted_mol, fakemol).build(tril=False)
    batch_size = 32
    omega = 0.
    eval_ip1, _, aux_offsets = _int3c2e_ip1_evaluator(
        opt, int3c2e_scheme_ip1(omega, 27), batch_size, 'fill_int3c2e_ip1', omega)
    ipnuc = 0
    for batch_id, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        ipnuc += cp.einsum('xpk,k->xp', eval_ip1(batch_id), -charges[p0:p1])

    pair_addresses = opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]
    nao = sorted_mol.nao
    out = cp.zeros([3, nao*nao])
    out[:,pair_addresses] = ipnuc
    out = sorted_mol.apply_CT_mat_C(out.reshape(3, nao, nao))
    return out

def _grad_nuc_without_ecp(mol, dm0):
    from gpu4pyscf.pbc.gto.int1e import int1e_ipkin
    coords = mol.atom_coords()
    charges = cp.asarray(-mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)

    sorted_mol = SortedMole.from_mol(mol, decontract=True)
    int3c2e_opt = Int3c2eOpt(sorted_mol, fakemol).build()
    dm = sorted_mol.apply_C_mat_CT(dm0)

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=False, gout_width=54, deriv=(1,0,0))
    lmax = sorted_mol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[0,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = sorted_mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    auxmol = int3c2e_opt.auxmol
    ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+sorted_mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    de = cp.zeros((mol.natm, 3))
    # de_aux * 2 is identical to the results of int3c2e.get_dh1e
    de_aux = cp.zeros((mol.natm, 3))
    err = kern(
        ctypes.cast(de.data.ptr, ctypes.c_void_p),
        ctypes.cast(de_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.cast(charges.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_gpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(mol.natm), ctypes.c_int(mol.natm))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    de += de_aux
    de *= 2
    de = de.get()

    t = int1e_ipkin(sorted_mol)
    de -= contract_h1e_dm(mol, t, dm0, hermi=1)
    return de

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

    e1_grad = mf_grad._hcore_energy(dm0, dme0)
    t3 = log.timer_debug1('gradients of h1e', *t3)

    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    e2_grad = mf_grad.energy_ee(mol, dm0)
    log.timer_debug1('gradients of 2e part', *t3)

    de = e1_grad + e2_grad
    de += cupy.asnumpy(mf_grad.extra_force())
    log.timer_debug1('gradients of electronic part', *t0)
    return de

def _hcore_energy(mf_grad, dm0, dme0):
    mol = mf_grad.mol
    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    # dh = (\nabla i | hcore | j) + (i | \nabla Vnuc | j)
    dh = _grad_nuc_without_ecp(mol, dm0)

    # Calculate ECP contributions in (i | \nabla hcore | j) and
    # (\nabla i | hcore | j) simultaneously
    if len(mol._ecpbas) > 0:
        # TODO: slice ecp_atoms
        ecp_atoms = np.unique(mol._ecpbas[:,gto.ATOM_OF])
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        dh -= contract_h1e_dm(mol, h1_ecp.sum(axis=0), dm0, hermi=1)
        dh[ecp_atoms] += 2.0 * contract('nxij,ij->nx', h1_ecp, dm0).get()

    s1 = cupy.asarray(mf_grad.get_ovlp(mol))
    dh -= contract_h1e_dm(mol, s1, dme0, hermi=1)
    return dh

def get_grad_hcore(mf_grad, mo_coeff=None, mo_occ=None):
    '''
    Calculate derivatives of Hcore in MO bases
    '''
    mf = mf_grad.base
    mol = mf.mol
    natm = mol.natm
    if mo_coeff is None: mo_coeff = cupy.asarray(mf.mo_coeff)
    if mo_occ is None: mo_occ = mf.mo_occ

    if getattr(mf, 'with_x2c', None):
        raise NotImplementedError('X2C gradients')

    orbo = mo_coeff[:,mo_occ>0]
    nmo = mo_coeff.shape[1]
    nocc = orbo.shape[1]
    dh1e = cupy.empty([natm,3,nmo,nocc])

    # derivative w.r.t nuclie position
    coords = mol.atom_coords()
    charges = cupy.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    sorted_mol = SortedMole.from_mol(mol, decontract=True)
    mo_sorted = sorted_mol.apply_C_dot(mo_coeff, axis=0)
    orbo_sorted = mo_sorted[:,mo_occ>0]

    opt = Int3c2eOpt(sorted_mol, fakemol).build(tril=False)
    batch_size = min(32, natm)
    omega = 0.
    eval_ip1, _, aux_offsets = _int3c2e_ip1_evaluator(
        opt, int3c2e_scheme_ip1(omega, 27), batch_size, 'fill_int3c2e_ip1', omega)
    pair_addresses = opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]
    nao1 = sorted_mol.nao
    work = cp.zeros([batch_size,3,nao1*nao1])
    for batch_id, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        tmp = eval_ip1(batch_id)
        tmp *= -charges[p0:p1]
        h1 = work[:p1-p0]
        h1[:,:,pair_addresses] = tmp.transpose(2,0,1)
        h1 = transpose_sum(h1.reshape((p1-p0)*3, nao1, nao1))
        tmp = contract('kxpq,qj->kxpj', h1.reshape(p1-p0,3,nao1,nao1), orbo_sorted)
        contract('kxpj,pi->kxij', tmp, mo_sorted, out=dh1e[p0:p1])
    work = h1 = tmp = None

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
    if len(mol._ecpbas) > 0:
        ecp_atoms = np.unique(mol._ecpbas[:,gto.ATOM_OF])
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        h1_ecp = h1_ecp + h1_ecp.transpose([0,1,3,2])
        h1mo = contract('nxij,jo->nxio', h1_ecp, orbo)
        dh1e[ecp_atoms] += contract('nxio,ip->nxpo', h1mo, mo_coeff)

    if mol._pseudo:
        raise NotImplementedError("Pseudopotential gradient not supported for molecular system yet")

    return dh1e

def contract_h1e_dm(mol, h1e, dm, hermi=0):
    '''Evaluate
    einsum('xij,ji->x', h1e[:,AO_idx_for_atom], (dm+dm.T)[:,AO_idx_for_atom])
    for all atoms. hermi=1 indicates that dm is a hermitian matrix.
    '''
    assert h1e.ndim == dm.ndim + 1
    ao_loc = mol.ao_loc
    dims = ao_loc[1:] - ao_loc[:-1]
    atm_id_for_ao = np.repeat(mol._bas[:,gto.ATOM_OF], dims)

    if dm.ndim == 2: # RHF
        de_partial = cp.einsum('xij,ji->ix', h1e, dm).real
        if hermi != 1:
            de_partial += cp.einsum('xij,ij->ix', h1e, dm).real
    else: # UHF
        de_partial = cp.einsum('sxij,sji->ix', h1e, dm).real
        if hermi != 1:
            de_partial += cp.einsum('sxij,sij->ix', h1e, dm).real

    de_partial = de_partial.get()
    de = groupby(atm_id_for_ao, de_partial, op='sum')
    if hermi == 1:
        de *= 2

    if len(de) < mol.natm:
        # Handle the case where basis sets are not specified for certain atoms
        de, de_tmp = np.zeros((mol.natm, 3)), de
        de[np.unique(atm_id_for_ao)] = de_tmp
    return de

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
    get_jk      = NotImplemented
    get_j       = NotImplemented
    get_k       = NotImplemented
    get_veff    = NotImplemented
    make_rdm1e  = NotImplemented
    grad_nuc    = rhf_grad_cpu.GradientsBase.grad_nuc
    grad_elec   = NotImplemented
    optimizer   = rhf_grad_cpu.GradientsBase.optimizer
    grad        = rhf_grad_cpu.GradientsBase.grad
    _finalize   = rhf_grad_cpu.GradientsBase._finalize
    _write      = rhf_grad_cpu.GradientsBase._write
    as_scanner  = as_scanner

    get_hcore   = get_hcore
    get_dispersion = get_dispersion
    _hcore_energy = _hcore_energy

    def get_ovlp(self, mol):
        from gpu4pyscf.pbc.gto.int1e import int1e_ipovlp
        return -int1e_ipovlp(mol)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        if mo_energy is None:
            if self.base.mo_energy is None:
                self.base.run()
            mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_energy, mo_coeff, mo_occ)
        self.de = de + self.grad_nuc()
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de)
        if self.base.do_disp():
            self.de += self.get_dispersion()
        log.timer('SCF gradients', *t0)
        self._finalize()
        return self.de

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, omega=0,
                           hermi=0, verbose=None):
        '''
        Computes the first-order derivatives of the energy per atom for
        j_factor * J_derivatives - k_factor * K_derivatives
        '''
        raise NotImplementedError

    def extra_force(self, atom_id=None):
        '''Hook for additional contributions to the analytical gradients.

        `atom_id` is the index of the atom for which to compute the force.
        If not provided, the contribution for all atoms is returned.
        '''
        return 0


class Gradients(GradientsBase):

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    make_rdm1e = rhf_grad_cpu.Gradients.make_rdm1e
    grad_elec = grad_elec

    def get_veff(self, mol=None, dm=None, verbose=None):
        '''
        Computes the first-order derivatives of the energy contributions from
        Veff per atom, corresponding to contracting dm with Veff:
        [np.einsum('xpq,pq->x', veff[:,AO_idx_for_atom], dm[AO_idx_for_atom]) for all atoms]
        This contraction is equal to 1/2 of the nuclear derivatives of the
        two-electron potential.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        '''
        raise DeprecationWarning

    def energy_ee(self, mol, dm):
        return self.jk_energy_per_atom(dm)

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, omega=0,
                           hermi=0, verbose=None):
        '''
        Computes the first-order derivatives of the energy per atom for
        j_factor * J_derivatives - k_factor * K_derivatives
        '''
        if dm is None: dm = self.base.make_rdm1()
        mf = self.base
        vhfopt = mf._opt_gpu.get(omega)
        if vhfopt is None:
            # For LDA and GGA, only mf._opt_jengine is initialized
            mol = mf.mol
            with mol.with_range_coulomb(omega):
                vhfopt = mf._opt_gpu[omega] = _VHFOpt(
                    mol, mf.direct_scf_tol).build()
        return _jk_energy_per_atom(vhfopt, dm, j_factor, k_factor, verbose)

Grad = Gradients
