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
Unrestricted Hartree-Fock for periodic systems with k-point sampling
'''

__all__ = [
    'KUHF'
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.scf import kuhf as kuhf_cpu
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.scf import khf
from gpu4pyscf.pbc.scf import uhf as pbcuhf
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import (
    return_cupy_array, contract, tag_array, sandwich_dot)


def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    '''Alpha and beta spin one particle density matrices for all k-points.

    Returns:
        dm_kpts : (2, nkpts, nao, nao) ndarray
    '''
    mo_occ_kpts = cp.asarray(mo_occ_kpts)
    mo_coeff_kpts = cp.asarray(mo_coeff_kpts)
    assert mo_occ_kpts.dtype == np.float64
    c = mo_coeff_kpts * mo_occ_kpts[:,:,None,:]
    dm = contract('nkpi,nkqi->nkpq', mo_coeff_kpts, c.conj())
    return tag_array(dm, mo_coeff=mo_coeff_kpts, mo_occ=mo_occ_kpts)

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    f_kpts = h1e_kpts + vhf_kpts
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f_kpts

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s_kpts is None: s_kpts = mf.get_ovlp()
    if dm_kpts is None: dm_kpts = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4 and fock_last is not None:
        f_a = []
        f_b = []
        for k in range(len(s_kpts)):
            f_a.append(mol_hf.damping(f_kpts[0][k], fock_last[0][k], dampa))
            f_b.append(mol_hf.damping(f_kpts[1][k], fock_last[1][k], dampa))
        f_kpts = [f_a, f_b]
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts, f_prev=fock_last)
    if abs(level_shift_factor) > 1e-4:
        f_kpts =([mol_hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta)
                  for k, s in enumerate(s_kpts)],
                 [mol_hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb)
                  for k, s in enumerate(s_kpts)])
    return cp.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''A pair of Fermi level for spin-up and spin-down orbitals
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ
    assert isinstance(mo_energy_kpts, cp.ndarray) and mo_energy_kpts.ndim == 3
    assert isinstance(mo_occ_kpts, cp.ndarray) and mo_occ_kpts.ndim == 3

    nocca, noccb = mf.nelec
    fermi_a = cp.partition(mo_energy_kpts[0].ravel(), nocca-1)[nocca-1]
    fermi_b = cp.partition(mo_energy_kpts[1].ravel(), noccb-1)[noccb-1]

    if mf.verbose >= logger.DEBUG:
        for k, mo_e in enumerate(mo_energy_kpts[0]):
            mo_occ = mo_occ_kpts[0][k]
            if mo_occ[mo_e > fermi_a].sum() > 0.5:
                logger.warn(mf, 'Alpha occupied band above Fermi level: \n'
                            'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
        for k, mo_e in enumerate(mo_energy_kpts[1]):
            mo_occ = mo_occ_kpts[1][k]
            if mo_occ[mo_e > fermi_b].sum() > 0.5:
                logger.warn(mf, 'Beta occupied band above Fermi level: \n'
                            'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    return (fermi_a, fermi_b)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''

    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    assert isinstance(mo_energy_kpts, cp.ndarray)

    nocc_a, nocc_b = mf.nelec
    nmo = mo_energy_kpts.shape[-1]
    mo_energy_a = cp.sort(mo_energy_kpts[0].ravel())
    fermi_a = mo_energy_a[nocc_a-1]
    mo_occ_kpts = cp.zeros_like(mo_energy_kpts)
    mo_occ_kpts[0] = (mo_energy_kpts[0] <= fermi_a).astype(np.float64)
    if nocc_b > 0:
        mo_energy_b = cp.sort(mo_energy_kpts[1].ravel())
        fermi_b = mo_energy_b[nocc_b-1]
        mo_occ_kpts[1] = (mo_energy_kpts[1] <= fermi_b).astype(np.float64)

    if mf.verbose >= logger.DEBUG:
        if nocc_a < nmo:
            logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g',
                        fermi_a, mo_energy_a[nocc_a])
        else:
            logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', fermi_a)
        if 0 < nocc_b < nmo:
            logger.info(mf, 'beta HOMO = %.12g  LUMO = %.12g',
                        fermi_b, mo_energy_b[nocc_b])
        elif 0 < nocc_b:
            logger.info(mf, 'beta HOMO = %.12g  (no LUMO because of small basis) ', fermi_b)
    return mo_occ_kpts


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(h1e_kpts)
    e1 = 1./nkpts * cp.einsum('kij,kji', dm_kpts[0], h1e_kpts)
    e1+= 1./nkpts * cp.einsum('kij,kji', dm_kpts[1], h1e_kpts)
    e_coul = 1./nkpts * cp.einsum('kij,kji', dm_kpts[0], vhf_kpts[0]) * 0.5
    e_coul+= 1./nkpts * cp.einsum('kij,kji', dm_kpts[1], vhf_kpts[1]) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if abs(e_coul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)

    fock_a = sandwich_dot(fock[0], mo_coeff_kpts[0])
    fock_b = sandwich_dot(fock[1], mo_coeff_kpts[1])
    occidx = mo_occ_kpts == 1
    viridx = ~occidx
    mo_coeff = cp.empty_like(mo_coeff_kpts)
    mo_energy = cp.empty(mo_occ_kpts.shape, dtype=np.float64)

    for k, f in enumerate(fock_a):
        for idx in (occidx[0], viridx[0]):
            if cp.count_nonzero(idx) > 0:
                e, c = cp.linalg.eigh(f[idx[:,None],idx])
                mo_coeff[0,k,:,idx] = mo_coeff_kpts[0,k,:,idx].dot(c)
                mo_energy[0,k,idx] = e

    for k, f in enumerate(fock_b):
        for idx in (occidx[1], viridx[1]):
            if cp.count_nonzero(idx) > 0:
                e, c = cp.linalg.eigh(f[idx[:,None],idx])
                mo_coeff[1,k,:,idx] = mo_coeff_kpts[1,k,:,idx].dot(c)
                mo_energy[1,k,idx] = e
    return mo_energy, mo_coeff

class KUHF(khf.KSCF):
    '''UHF class with k-point sampling.
    '''
    conv_tol_grad = kuhf_cpu.KUHF.conv_tol_grad
    init_guess_breaksym = kuhf_cpu.KUHF.init_guess_breaksym

    _keys = kuhf_cpu.KUHF._keys

    def __init__(self, cell, kpts=None, exxdiv='ewald'):
        khf.KSCF.__init__(self, cell, kpts, exxdiv)
        self.nelec = None

    nelec = kuhf_cpu.KUHF.nelec
    dump_flags = kuhf_cpu.KUHF.dump_flags

    init_guess_by_1e     = pbcuhf.UHF.init_guess_by_1e
    init_guess_by_minao  = pbcuhf.UHF.init_guess_by_minao
    init_guess_by_atom   = pbcuhf.UHF.init_guess_by_atom
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    energy_elec = energy_elec
    get_rho = khf.get_rho
    analyze = NotImplemented
    canonicalize = canonicalize

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm_kpts = cp.asarray(mol_hf.SCF.get_init_guess(self, cell, key))
        assert dm_kpts.shape[0] == 2
        nkpts = len(self.kpts)
        if dm_kpts.ndim != 4:
            # dm[spin,nao,nao] at gamma point -> dm_kpts[spin,nkpts,nao,nao]
            dm_kpts = cp.repeat(dm_kpts[:,None,:,:], nkpts, axis=1)

        ne = cp.einsum('xkij,kji->x', dm_kpts, s1e).real
        nelec = cp.asarray(self.nelec)
        if any(abs(ne - nelec) > 0.01*nkpts):
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne.mean()/nkpts, nelec/nkpts)
            dm_kpts *= (nelec / ne).reshape(2,1,1,1)
        return dm_kpts

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        vhf = vj[0] + vj[1] - vk
        return vhf

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        def grad(mo, mo_occ, fock):
            occidx = mo_occ > 0
            viridx = ~occidx
            g = mo[:,viridx].conj().T.dot(fock.dot(mo[:,occidx]))
            return g.ravel()

        nkpts = len(mo_occ_kpts[0])
        grad_kpts = [grad(mo_coeff_kpts[0][k], mo_occ_kpts[0][k], fock[0][k])
                     for k in range(nkpts)]
        grad_kpts+= [grad(mo_coeff_kpts[1][k], mo_occ_kpts[1][k], fock[1][k])
                     for k in range(nkpts)]
        return cp.hstack(grad_kpts)

    def eig(self, h_kpts, s_kpts):
        e_a, c_a = khf.KSCF.eig(self, h_kpts[0], s_kpts)
        e_b, c_b = khf.KSCF.eig(self, h_kpts[1], s_kpts)
        return cp.asarray((e_a,e_b)), cp.asarray((c_a,c_b))

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None: mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None: mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = kpts_band.ndim == 1
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
        fock += self.get_hcore(cell, kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        e, c = self.eig(fock, s1e)
        if single_kpt_band:
            e = e[:,0]
            c = c[:,0]
        return e, c

    init_guess_by_chkfile = return_cupy_array(kuhf_cpu.KUHF.init_guess_by_chkfile)

    mulliken_meta = NotImplemented
    mulliken_meta_spin = NotImplemented
    mulliken_pop = NotImplemented
    dip_moment = NotImplemented
    spin_square = NotImplemented
    stability = NotImplemented
    to_ks = NotImplemented
    convert_from_ = NotImplemented

    density_fit = khf.KRHF.density_fit

    def to_cpu(self):
        mf = kuhf_cpu.KUHF(self.cell)
        utils.to_cpu(self, out=mf)
        return mf
