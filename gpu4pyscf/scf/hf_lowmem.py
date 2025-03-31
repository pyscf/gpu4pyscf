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
from gpu4pyscf.lib.cupy_helper import (
    asarray, pack_tril, unpack_tril, ConditionalMemoryPool)
from gpu4pyscf.scf import diis, jk, hf
from gpu4pyscf.lib import logger

__all__ = [
    'RHF',
]

def kernel(mf, dm0=None, conv_tol=1e-10, conv_tol_grad=None,
          dump_chk=True, callback=None, conv_check=True, **kwargs):
    mol = mf.mol
    verbose = mf.verbose
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()

    mf.dump_flags()
    mf.build(mf.mol)

    conv_tol = mf.conv_tol
    if(conv_tol_grad is None):
        conv_tol_grad = conv_tol**.5
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    if dm0 is None:
        if mf.mo_coeff is not None and mf.mo_occ is not None:
            # Initial guess from existing wavefunction
            dm0 = mf.make_rdm1()
        else:
            dm0 = mf.get_init_guess(mol, mf.init_guess)

    dm, dm0 = asarray(dm0, order='C'), None
    h1e = cp.asarray(mf.get_hcore(mol))
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)
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

    mf_diis = mf.DIIS(mf, mf.diis_file)
    mf_diis.space = mf.diis_space
    mf_diis.rollback = mf.diis_space_rollback

    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        mo_coeff = mo_occ = mo_energy = fock = None
        last_hf_e = e_tot

        s1e = mf.get_ovlp(mol)
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        t1 = log.timer_debug1('DIIS', *t0)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        fock = s1e = None
        t1 = log.timer_debug1('eig', *t1)

        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        # Update mo_coeff and mo_occ, allowing get_veff to generate DMs on the fly
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
        vhf = mf.get_veff(mol, None, dm, vhf)

        fock = mf.get_fock(h1e, None, vhf)  # = h1e + vhf, no DIIS
        norm_gorb = cp.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        fock = None
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        e_tot = mf.energy_tot(dm, h1e, vhf)
        norm_ddm = cp.linalg.norm(dm-dm_last)
        dm_last = None
        t1 = log.timer_debug1('SCF iteration', *t0)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_ddm)

        if dump_chk:
            mf.dump_chk(locals())

        e_diff = abs(e_tot-last_hf_e)
        if(e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break
    else:
        logger.warn(mf, "SCF failed to converge")

    mf.converged = scf_conv
    mf.e_tot = e_tot
    mf.mo_energy = mo_energy
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    logger.timer(mf, 'SCF', *cput0)
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

    def kernel(self, *args, **kwargs):
        try:
            default_allocator = cp.cuda.memory.get_allocator()
            nao = self.mol.nao
            thresh = nao**2 // 2 * 8
            cp.cuda.memory.set_allocator(ConditionalMemoryPool(thresh).malloc)
            return kernel(self, *args, **kwargs)
        finally:
            cp.cuda.memory.set_allocator(default_allocator)
    scf = kernel

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
        hcore = hf_cpu.RHF.get_hcore(self, mol)
        nao = hcore.shape[0]
        idx = np.arange(nao)
        return cp.asarray(hcore[idx[:,None] >= idx])

    def get_jk(self, mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
        raise NotImplementedError

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        raise NotImplementedError

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        raise NotImplementedError

    def get_veff(self, mol, dm, dm_last=None, vhf_last=0, hermi=1):
        '''Constructus the lower-triangular part of the Fock matrix.'''
        log = logger.new_logger(mol, self.verbose)
        cput0 = log.init_timer()

        omega = mol.omega
        vhfopt = self._opt_gpu.get(omega)
        if vhfopt is None:
            vhfopt = self._opt_gpu[omega] = jk._VHFOpt(mol, self.direct_scf_tol).build()

        dm = self._delta_rdm1(dm, dm_last, vhfopt)
        vj, vk = vhfopt.get_jk(dm, hermi, True, True, log)
        assert vj.ndim == 3
        dm = None
        vk *= -.5
        vj += vk
        vj = vhfopt.apply_coeff_CT_mat_C(vj)
        vk = None
        if isinstance(vhf_last, cp.ndarray):
            vhf = asarray(vhf_last) # remove attributes if vhf_last is a tag_array
            vhf_last = None
            vhf += pack_tril(vj[0]) # Reuse the memory of previous Veff
        else:
            vhf = pack_tril(vj[0])
        log.timer('vj and vk', *cput0)
        return vhf

    def _delta_rdm1(self, dm, dm_last, vhfopt):
        '''Construct dm-dm_last suitable for the vhfopt.get_jk method'''
        if dm is None:
            if isinstance(dm, WaveFunction):
                mo_coeff = dm.mo_coeff
                mo_occ = dm.mo_occ
            else:
                mo_coeff = self.mo_coeff
                mo_occ = self.mo_occ
            mask = mo_occ > 0
            occ_coeff = mo_coeff[:,mask]
            occ_coeff *= mo_occ[mask]**.5

            if dm_last is None:
                occ_coeff = vhfopt.apply_coeff_C_mat(occ_coeff)
                dm = occ_coeff.dot(occ_coeff.T)
            else:
                dm = occ_coeff.dot(occ_coeff.T)
                dm -= dm_last
                dm = vhfopt.apply_coeff_C_mat_CT(dm)
        else:
            if dm_last is not None:
                dm = dm.copy()
                dm -= dm_last
            dm = vhfopt.apply_coeff_C_mat_CT(dm)
        return dm[None]

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return RHFWfn(mo_coeff, mo_occ).make_rdm1()

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        '''Return Fock matrix in square storage'''
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if not isinstance(h1e, cp.ndarray): h1e = cp.asarray(h1e)
        if not isinstance(vhf, cp.ndarray): vhf = cp.asarray(vhf)
        # h1e and vhf must be both in tril storage or square-matrix storage
        assert h1e.shape[-1] == vhf.shape[-1]
        assert h1e.ndim == vhf.ndim == 1
        f = unpack_tril(h1e + vhf)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if s1e is None: s1e = self.get_ovlp()
        if dm is None: dm = self.make_rdm1()
        if not isinstance(s1e, cp.ndarray): s1e = cp.asarray(s1e)
        if not isinstance(dm, cp.ndarray): dm = cp.asarray(dm)
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

        if abs(level_shift_factor) > 1e-4:
            #:f = hf.level_shift(s1e, dm*.5, f, level_shift_factor)
            dm_vir = s1e.dot(dm).dot(s1e)
            dm_vir *= -.5
            dm_vir += s1e
            dm_vir *= level_shift_factor
            f += dm_vir
        return f

    def energy_elec(self, dm, h1e, vhf):
        '''
        electronic energy
        '''
        assert dm.dtype == np.float64
        assert h1e.ndim == vhf.ndim == 1
        dm_tril = pack_tril(dm)
        nao = dm.shape[0]
        i = cp.arange(nao)
        diag = i*(i+1)//2 + i
        dm_tril[diag] *= .5
        e1 = float(h1e.dot(dm_tril) * 2)
        e_coul = float(vhf.dot(dm_tril))
        self.scf_summary['e1'] = e1
        self.scf_summary['e2'] = e_coul
        logger.debug(self, 'E1 = %s  E_coul = %s', e1, e_coul)
        return e1+e_coul, e_coul

    def to_cpu(self):
        raise NotImplementedError
