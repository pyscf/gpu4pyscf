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
The RHF implementation in this module prioritizes a reduced GPU memory footprint
at the cost of efficiency.
'''

import numpy as np
import cupy as cp
from pyscf.scf import hf as hf_cpu
from pyscf.scf import chkfile
from gpu4pyscf.lib.cupy_helper import asarray, pack_tril, unpack_tril, eigh
from gpu4pyscf import lib
from gpu4pyscf.scf import diis, jk, j_engine, hf
from gpu4pyscf.lib import logger

__all__ = [
    'RHF',
]

def kernel(mf, dm0=None, conv_tol=1e-10, conv_tol_grad=None,
          dump_chk=True, callback=None, conv_check=True, **kwargs):
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mol, verbose)
    cput0 = cput1 = log.init_timer()

    mf.dump_flags()
    mf.build(mf.mol)
    if log.verbose >= logger.DEBUG1:
        mem_avail = log.print_mem_info()
        log.debug1('available GPU memory for SCF: %.3f GB', mem_avail/1e9)

    conv_tol = mf.conv_tol
    if(conv_tol_grad is None):
        conv_tol_grad = conv_tol**.5
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    if dm0 is None:
        if mf.mo_coeff is not None and mf.mo_occ is not None:
            # Initial guess from existing wavefunction
            dm0 = mf.make_rdm1()
        else:
            dm0 = mf.get_init_guess(mol, mf.init_guess)
            if hasattr(dm0, 'mo_coeff') and dm0.mo_coeff.size > dm0.size:
                # Discard the mo_coeff attribute, force the vxc evaluator to
                # eval_rho using dm directly 
                dm0 = asarray(dm0, order='C')
            else:
                dm0 = mf.make_wfn(dm0.mo_coeff, dm0.mo_occ)
        cput1 = log.timer_debug1('generating initial guess', *cput1)

    h1e = mf.get_hcore(mol) # On CPU
    cput1 = log.timer_debug1('hcore', *cput1)
    dm, dm0 = dm0, None # on GPU
    vhf = mf.get_veff(mol, dm) # On CPU
    cp.get_default_memory_pool().free_all_blocks()
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        mf.e_tot = e_tot
        if mf.mo_coeff is None:
            s1e = mf.get_ovlp(mol)
            fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
            mf.mo_energy, mf.mo_coeff = mf.eig(fock, s1e)
            mf.mo_occ = mf.get_occ(mf.mo_energy, mf.mo_coeff)
            mf.converged = scf_conv
        return e_tot

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        # The Corth in DIIS calls the eigh function that does not overwrite
        # the input matrices. The input can be overwritten so as to reduce GPU
        # memory footprint.
        s1e = asarray(mf.get_ovlp(mol))
        c = eigh(unpack_tril(asarray(h1e)), s1e, overwrite=True)[1]
        mf_diis.Corth = c.get()
        s1e = c = None
    else:
        mf_diis = None

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        log.warn('Low-mem SCF does not support dumping chkfile')
    cp.get_default_memory_pool().free_all_blocks()
    if log.verbose >= logger.DEBUG1:
        mem_avail = log.print_mem_info()
        log.debug1('available GPU memory after SCF initialization: %.3f GB', mem_avail/1e9)
    t1 = log.timer_debug1('SCF initialization', *cput1)
    natm = mol.natm

    for cycle in range(mf.max_cycle):
        t0 = t1
        mo_coeff = mo_occ = mo_energy = fock = None
        last_hf_e = e_tot

        s1e = asarray(mf.get_ovlp(mol))
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis) # on GPU
        t1 = log.timer_debug1('DIIS', *t1)
        cp.get_default_memory_pool().free_all_blocks()
        mo_energy, mo_coeff = mf.eig(fock, s1e) # on GPU
        fock = s1e = None
        t1 = log.timer_debug1('eig', *t1)

        mo_occ = mf.get_occ(mo_energy, mo_coeff) # on GPU
        # Update mo_coeff and mo_occ, allowing get_veff to generate DMs on the fly
        dm, dm_last = mf.make_wfn(mo_coeff, mo_occ), dm # on GPU
        vhf = mf.get_veff(mol, dm, dm_last, vhf) # on CPU
        cp.get_default_memory_pool().free_all_blocks()
        if log.verbose >= logger.DEBUG1:
            mem_avail = log.print_mem_info()
            log.debug1('available GPU memory: %.3f GB', mem_avail/1e9)

        fock = mf.get_fock(h1e, None, vhf)  # = h1e + vhf, no DIIS
        norm_gorb = cp.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        fock = None
        e_tot = mf.energy_tot(dm, h1e, vhf)
        t1 = log.timer_debug1('SCF iteration', *t0)
        log.info('cycle= %d E= %.15g  delta_E= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e)

        e_diff = abs(e_tot-last_hf_e)
        if e_diff < conv_tol and norm_gorb/natm**.5 < conv_tol_grad:
            scf_conv = True
            break
    else:
        log.warn("SCF failed to converge")

    if scf_conv and abs(mf.level_shift) > 0:
        # An extra diagonalization, to remove level shift
        s1e = asarray(mf.get_ovlp(mol))
        fock = mf.get_fock(h1e, s1e, vhf)

        cp.get_default_memory_pool().free_all_blocks()
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        fock = s1e = None
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_wfn(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        cp.get_default_memory_pool().free_all_blocks()

        fock = mf.get_fock(h1e, None, vhf)
        norm_gorb = cp.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        fock = None
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        else:
            log.warn("Level-shifted SCF extra cycle failed to converge")
            scf_conv = False
        log.info('Extra cycle= %d E= %.15g  delta_E= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e)

    mf.converged = scf_conv
    mf.e_tot = e_tot
    mf.mo_energy = mo_energy
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    log.timer('SCF', *cput0)
    mf._finalize()
    return e_tot

class WaveFunction:
    def __init__(self, mo_coeff, mo_occ):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

    def make_rdm1(self):
        raise NotImplementedError

class RHFWfn(WaveFunction):
    def make_rdm1(self):
        mo_coeff = cp.asarray(self.mo_coeff)
        mo_occ = cp.asarray(self.mo_occ)
        is_occ = mo_occ > 0
        mocc = mo_coeff[:, is_occ]
        mocc *= mo_occ[is_occ]**.5
        dm = mocc.dot(mocc.conj().T)
        return dm

class CDIIS(diis.CDIIS):
    def update(self, s, d, f, *args, **kwargs):
        out = super().update(s, d, f, *args, **kwargs)
        if isinstance(self.Corth, cp.ndarray):
            # Store Corth on host to reduce GPU memory pressure
            self.Corth = self.Corth.get()
        return out

class RHF(hf.RHF):
    '''The low-memory RHF class for large systems. Not fully compatible with the
    default RHF class in hf.py . Some methods return the lower-triangular part
    of the square matrix; some methods are simplified.
    '''

    DIIS = CDIIS

    kernel = scf = kernel
    density_fit              = NotImplemented
    as_scanner               = NotImplemented
    newton                   = NotImplemented
    x2c = x2c1e = sfx2c1e    = NotImplemented
    stability                = NotImplemented

    def check_sanity(self):
        mol = self.mol
        if mol.spin != 0:
            raise RuntimeError(
                f'Invalid number of electrons {mol.nelectron} for RHF method.')
        return self

    def get_hcore(self, mol=None):
        '''The lower triangular part of Hcore'''
        hcore = hf.get_hcore(mol)
        return pack_tril(hcore).get()

    def get_jk(self, mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
        raise NotImplementedError

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        raise NotImplementedError

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        raise NotImplementedError

    def get_veff(self, mol, dm_or_wfn, dm_last=None, vhf_last=None, hermi=1):
        '''Constructus the lower-triangular part of the Veff matrix.'''
        log = logger.new_logger(mol, self.verbose)
        cput0 = log.init_timer()

        omega = mol.omega
        if omega in self._opt_gpu:
            vhfopt = self._opt_gpu[omega]
        else:
            self._opt_gpu[omega] = vhfopt = jk._VHFOpt(mol, self.direct_scf_tol).build()
        if omega in self._opt_jengine:
            jopt = self._opt_jengine[omega]
        else:
            self._opt_jengine[omega] = jopt = j_engine._VHFOpt(mol, self.direct_scf_tol).build()

        #:vj, vk = vhfopt.get_jk(dm, hermi, True, True, log)
        dm = lambda: self._delta_rdm1(dm_or_wfn, dm_last, jopt)
        vj = jopt.get_j(dm, log)
        assert vj.ndim == 3
        vj = jopt.apply_coeff_CT_mat_C(vj)
        vhf, vj = vj, None

        dm = lambda: self._delta_rdm1(dm_or_wfn, dm_last, vhfopt)
        vk = vhfopt.get_jk(dm, hermi, False, True, log)[1]
        assert vk.ndim == 3
        vk = vhfopt.apply_coeff_CT_mat_C(vk)
        vk *= -.5

        vhf += vk
        vhf = pack_tril(vhf[0])
        if vhf_last is not None:
            vhf += asarray(vhf_last)
        log.timer('veff', *cput0)
        return vhf.get()

    def _delta_rdm1(self, dm_or_wfn, dm_last, vhfopt):
        '''Construct dm-dm_last suitable for the vhfopt.get_jk method'''
        if dm_or_wfn is None:
            dm = self.make_rdm1()
        elif isinstance(dm_or_wfn, WaveFunction):
            dm = dm_or_wfn.make_rdm1()
        else:
            dm = dm_or_wfn
            if dm_last is not None:
                dm = dm.copy()

        if isinstance(dm_last, WaveFunction):
            dm_last = dm_last.make_rdm1()
        if dm_last is not None:
            dm -= dm_last

        dm = vhfopt.apply_coeff_C_mat_CT(dm)
        return dm[None] # Add an additional axis, as required by the get_jk function

    def make_wfn(self, mo_coeff, mo_occ):
        return RHFWfn(mo_coeff, mo_occ)

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return self.make_wfn(mo_coeff, mo_occ).make_rdm1()

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm_or_wfn=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        '''Return Fock matrix in square storage'''
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm_or_wfn)
        if not isinstance(h1e, cp.ndarray): h1e = asarray(h1e)
        if not isinstance(vhf, cp.ndarray): vhf = asarray(vhf)
        # h1e and vhf must be both in tril storage or square-matrix storage
        assert h1e.shape[-1] == vhf.shape[-1]
        assert h1e.ndim == vhf.ndim == 1
        nao = int((h1e.size*2)**.5)
        f = cp.empty((nao, nao))
        f = unpack_tril(asarray(h1e + vhf), out=f)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if s1e is None: s1e = self.get_ovlp()
        if dm_or_wfn is None:
            dm = self.make_rdm1()
        elif isinstance(dm_or_wfn, WaveFunction):
            dm = dm_or_wfn.make_rdm1()
        else:
            dm = dm_or_wfn
        if not isinstance(s1e, cp.ndarray): s1e = asarray(s1e)
        if not isinstance(dm, cp.ndarray): dm = asarray(dm)
        # Ensure overlap in square-matrix format
        if s1e.shape[-1] != f.shape[-1]:
            s1e = unpack_tril(s1e)

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp

        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
            f = hf.damping(s1e, dm*.5, f, damp_factor)
        if diis is not None and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm, f)
            cp.get_default_memory_pool().free_all_blocks()

        if abs(level_shift_factor) > 1e-4:
            dm_vir, dm = dm, None
            #:f = hf.level_shift(s1e, dm*.5, f, level_shift_factor)
            dm_vir = s1e.dot(dm_vir)
            dm_vir = dm_vir.dot(s1e)
            dm_vir *= -.5
            dm_vir += s1e
            dm_vir *= level_shift_factor
            f += dm_vir
        return f

    def energy_elec(self, dm_or_wfn, h1e, vhf):
        '''
        electronic energy
        '''
        assert h1e.ndim == vhf.ndim == 1
        if isinstance(dm_or_wfn, WaveFunction):
            dm = dm_or_wfn.make_rdm1()
        else:
            dm = dm_or_wfn
        assert dm.dtype == np.float64
        dm_tril = pack_tril(dm)
        nao = dm.shape[0]
        i = cp.arange(nao)
        diag = i*(i+1)//2 + i
        dm_tril[diag] *= .5
        dm_tril = dm_tril.get()
        e1 = float(h1e.dot(dm_tril) * 2)
        e_coul = float(vhf.dot(dm_tril))
        vtmp = h1e * 2
        vtmp += vhf
        e_tot = float(vtmp.dot(dm_tril))
        self.scf_summary['e1'] = e1
        self.scf_summary['e2'] = e_coul
        logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
        return e_tot, e_coul

    def _eigh(self, h, s):
        # In DIIS, fock and overlap matrices are temporarily constructed and
        # discarded, they can be overwritten in the eigh solver.
        e, c = eigh(h, s, overwrite=True)
        # eigh allocates a large memory buffer "work". Immediately free the cupy
        # memory after the eigh function to avoid this buffer being trapped by
        # small-sized arrays.
        cp.get_default_memory_pool().free_all_blocks()
        return e, c

    def to_cpu(self):
        raise NotImplementedError
