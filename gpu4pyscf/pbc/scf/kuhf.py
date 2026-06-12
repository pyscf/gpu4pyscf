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

import functools
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.data.nist import HARTREE2EV
from pyscf.pbc.scf import kuhf as kuhf_cpu
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.scf import khf
from gpu4pyscf.pbc.scf import uhf as pbcuhf
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import (
    return_cupy_array, contract, tag_array, sandwich_dot, asarray)


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

    if s_kpts is None: s_kpts = mf.get_ovlp()
    if dm_kpts is None: dm_kpts = mf.make_rdm1()

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if damp_factor is None:
        damp_factor = mf.damp
    if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
        if isinstance(damp_factor, (tuple, list, np.ndarray)):
            dampa, dampb = damp_factor
        else:
            dampa = dampb = damp_factor
        f_a = []
        f_b = []
        for k in range(len(s_kpts)):
            f_a.append(asarray(mol_hf.damping(f_kpts[0][k], fock_last[0][k], dampa)))
            f_b.append(asarray(mol_hf.damping(f_kpts[1][k], fock_last[1][k], dampb)))
        f_kpts = cp.asarray([f_a, f_b])
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts, f_prev=fock_last)

    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if level_shift_factor is not None:
        if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
            shifta, shiftb = level_shift_factor
        else:
            shifta = shiftb = level_shift_factor
        f_kpts =([asarray(mol_hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta))
                  for k, s in enumerate(s_kpts)],
                 [asarray(mol_hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb))
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
    fermi_a = float(fermi_a.get())
    fermi_b = float(fermi_b.get())
    return (fermi_a, fermi_b)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''

    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    assert isinstance(mo_energy_kpts, cp.ndarray)

    nocc_a, nocc_b = mf.nelec
    mo_energy_a = cp.sort(mo_energy_kpts[0].ravel())
    nmo = mo_energy_a.size
    if nocc_a > nmo or nocc_b > nmo:
        raise RuntimeError('Failed to assign mo_occ. '
                           f'Nocc ({nocc_a}, {nocc_b}) > Nmo ({nmo})')
    fermi_a = mo_energy_a[nocc_a-1]
    mo_occ_kpts = cp.zeros_like(mo_energy_kpts)
    mo_occ_kpts[0] = (mo_energy_kpts[0] <= fermi_a).astype(np.float64)
    if nocc_b > 0:
        mo_energy_b = cp.sort(mo_energy_kpts[1].ravel())
        fermi_b = mo_energy_b[nocc_b-1]
        mo_occ_kpts[1] = (mo_energy_kpts[1] <= fermi_b).astype(np.float64)

    if nocc_a < nmo and nocc_b < nmo:
        homo = homo_a = fermi_a
        homo_b = None
        if nocc_b > 0:
            homo = max(homo, fermi_b)
        lumo = lumo_b = mo_energy_b[nocc_b]
        lumo_a = None
        if nocc_a < nmo:
            lumo_a = mo_energy_a[nocc_a]
            lumo = min(lumo, lumo_a)
        gap = (lumo - homo) * HARTREE2EV
        mf.scf_summary['gap'] = gap
        if mf.verbose >= logger.INFO:
            if lumo_a is not None:
                logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', homo_a, lumo_a)
            else:
                logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', homo_a)
            if homo_b is not None:
                logger.info(mf, 'beta HOMO = %.12g  LUMO = %.12g', homo_b, lumo_b)
            else:
                logger.info(mf, 'beta               LUMO = %.12g', lumo_b)
            if homo+1e-3 > lumo:
                logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', homo, lumo)
            else:
                logger.info(mf, '  HOMO = %.12g  LUMO = %.12g  gap/eV = %.5f',
                            homo, lumo, gap)
    return mo_occ_kpts


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None or getattr(vhf_kpts, 'ecoul', None) is None:
        vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(h1e_kpts)
    e1 = 1./nkpts * cp.einsum('skij,kji->', dm_kpts, h1e_kpts).get()
    e2 = 1./nkpts * cp.einsum('skij,skji->', dm_kpts, vhf_kpts).get() * 0.5
    ecoul = vhf_kpts.ecoul
    exx = e2 - ecoul
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e2.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exx.real
    logger.debug(mf, 'E1 = %s  E2 = %s  Ecoul = %s  Exc = %s', e1, e2, ecoul, exx)
    if abs(e2.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e2.imag)
    return (e1+e2).real, e2.real

def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)
    ea, ca = khf.canonicalize(mf, mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
    eb, cb = khf.canonicalize(mf, mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
    mo_energy = cp.stack([ea, eb])
    mo_coeff = cp.stack([ca, cb])
    return mo_energy, mo_coeff

def _cast_mol_init_guess(fn):
    @functools.wraps(fn)
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = fn(mf, cell)
        assert dm.ndim == 3
        nkpts = len(kpts)
        if hasattr(dm, 'mo_coeff'):
            idx = np.where(cp.asnumpy(dm.mo_occ.sum(axis=0)) > 0)[0]
            mo_coeff = cp.repeat(asarray(dm.mo_coeff[:,None,:,idx]), nkpts, axis=1)
            mo_occ = cp.repeat(asarray(dm.mo_occ[:,None,idx]), nkpts, axis=1)
            dm = cp.repeat(asarray(dm[:,None]), nkpts, axis=1)
            dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        else:
            dm = cp.repeat(asarray(dm[:,None]), nkpts, axis=1)
        return dm
    return fn_init_guess

class KUHF(khf.KSCF):
    '''UHF class with k-point sampling.
    '''
    conv_tol_grad = kuhf_cpu.KUHF.conv_tol_grad
    init_guess_breaksym = kuhf_cpu.KUHF.init_guess_breaksym

    _keys = kuhf_cpu.KUHF._keys

    def __init__(self, cell, kpts=None, exxdiv='ewald'):
        khf.KSCF.__init__(self, cell, kpts, exxdiv)
        self.nelec = None

    def dump_flags(self, verbose=None):
        khf.KSCF.dump_flags(self, verbose)
        logger.info(self, 'number of electrons per cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    nelec = kuhf_cpu.KUHF.nelec

    init_guess_by_minao = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_atom)
    init_guess_by_huckel = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_huckel)
    init_guess_by_mod_huckel = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_mod_huckel)
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    energy_elec = energy_elec
    get_rho = khf.get_rho
    canonicalize = canonicalize

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        logger.info(self, 'Initial guess from hcore.')
        h = self.get_hcore(cell)
        s = self.get_ovlp(cell)
        e, c = self.eig((h, h), s)
        mo_occ = self.get_occ(e, c)
        nocc = int((mo_occ > 0).sum(axis=2).max())
        dm = self.make_rdm1(c[:,:,:,:nocc], mo_occ[:,:,:nocc])
        return dm

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm = cp.asarray(mol_hf.SCF.get_init_guess(self, cell, key))
        nkpts = len(self.kpts)
        assert dm.ndim == 4 and dm.shape[:2] == (2, nkpts)

        ne = cp.einsum('xkij,kji->x', dm, s1e).real.get()
        nelec = self.nelec
        if any(abs(ne - nelec) > 0.01*nkpts):
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons (%g, %g)',
                         ne.mean()/nkpts, nelec[0]/nkpts, nelec[1]/nkpts)
            ne[1] += 1e-300 # Number of beta electrons may be 0
            dm[0] *= nelec[0] / ne[0]
            dm[1] *= nelec[1] / ne[1]
            if hasattr(dm, 'mo_coeff'):
                dm.mo_occ[0] *= nelec[0] / ne[0]
                dm.mo_occ[1] *= nelec[1] / ne[1]
        return dm

    def get_veff(self, cell=None, dm_kpts=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None):
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        def trace(dm, vj):
            if kpts_band is not None:
                return None
            if vj.ndim == 2:
                return cp.einsum('nij,ji->', dm_kpts, vj).real.get() * .5
            return cp.einsum('nKij,Kji->', dm_kpts, vj).real.get() * .5

        if self.rsjk or isinstance(self.j_engine, (PBCJKMatrixOpt, PBCJMatrixOpt)):
            incremental_veff = dm_last is not None and hasattr(vhf_last, 'sr')
            ddm = dm_kpts
            if incremental_veff:
                ddm = dm_kpts - dm_last

            vj_sr = vk_sr = 0
            ecoul = ecoul_sr = None
            if isinstance(self.j_engine, (PBCJKMatrixOpt, PBCJMatrixOpt)):
                if self.j_engine.supmol is None:
                    self.j_engine.build(kpts)
                vj_sr = self.j_engine._get_j_sr(ddm.sum(axis=0), hermi, kpts, kpts_band)
                vj = self.j_engine._get_j_lr(dm_kpts.sum(axis=0), hermi, kpts, kpts_band)
                if incremental_veff:
                    if hasattr(vhf_last, 'ecoul_sr'):
                        ecoul_sr = trace(dm_last, vj_sr) * 2
                        ecoul_sr += trace(ddm, vj_sr)
                        ecoul_sr += vhf_last.ecoul_sr
                        ecoul = trace(dm_kpts, vj) + ecoul_sr
                else:
                    ecoul_sr = trace(dm_kpts, vj_sr)
                    ecoul = trace(dm_kpts, vj) + ecoul_sr
            else:
                vj = self.get_j(cell, dm_kpts.sum(axis=0), hermi, kpts, kpts_band)
                ecoul = trace(dm_kpts, vj)

            if self.rsjk:
                if self.rsjk.supmol is None:
                    self.rsjk.build(kpts)
                vk_sr = self.rsjk._get_k_sr(ddm, hermi, kpts, kpts_band, self.exxdiv)
                vk = self.rsjk._get_k_lr(dm_kpts, hermi, kpts, kpts_band, self.exxdiv)
            else:
                vk = self.get_k(cell, dm_kpts, hermi, kpts, kpts_band)

            vhf_sr = vj_sr - vk_sr
            if incremental_veff:
                vhf_sr += vhf_last.sr
            vhf = vj - vk + vhf_sr
            vhf = tag_array(vhf, sr=vhf_sr)
            if ecoul is not None:
                vhf.ecoul = ecoul
                if ecoul_sr is not None:
                    vhf.ecoul_sr = ecoul_sr
        else:
            vj, vk = self.with_df.get_jk(
                dm_kpts, hermi, kpts, kpts_band, with_j=True, with_k=True,
                exxdiv=self.exxdiv)
            vj = vj.sum(axis=0)
            ecoul = trace(dm_kpts, vj)
            vhf = tag_array(vj - vk, ecoul=ecoul)
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

    def eig(self, h_kpts, s_kpts, overwrite=False, x=None):
        e_a, c_a = khf.KSCF.eig(self, h_kpts[0], s_kpts, x=x)
        e_b, c_b = khf.KSCF.eig(self, h_kpts[1], s_kpts, overwrite, x)
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

    def Gradients(self):
        from gpu4pyscf.pbc.grad.kuhf import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = kuhf_cpu.KUHF(self.cell)
        with lib.temporary_env(self, _numint=None):
            utils.to_cpu(self, out=mf)
        return mf

    def analyze(self, verbose=None, **kwargs):
        '''Analyze the given SCF object:  print orbital energies, occupancies;
        print orbital coefficients; Mulliken population analysis; Dipole moment
        '''
        from pyscf.pbc.scf.kuhf import mulliken_meta
        if verbose is None:
            verbose = self.verbose
        log = logger.new_logger(self, verbose)
        mo_energy = self.mo_energy.get()
        mo_occ = self.mo_occ.get()
        cell = self.cell
        kpts = self.kpts
        if log.verbose >= logger.NOTE:
            self.dump_scf_summary(log)
            log.note('**** MO energy ****')
            log.note('                           alpha                               | beta')
            log.note('k-point                    nocc    HOMO/AU         LUMO/AU     | nocc    HOMO/AU         LUMO/AU')
            for k, kpt in enumerate(cell.get_scaled_kpts(kpts)):
                nocca = np.count_nonzero(mo_occ[0,k])
                noccb = np.count_nonzero(mo_occ[1,k])
                homoa = mo_energy[0,k,nocca-1]
                homob = mo_energy[1,k,noccb-1]
                lumoa = mo_energy[0,k,nocca  ]
                lumob = mo_energy[1,k,noccb  ]
                log.note('%2d (%6.3f %6.3f %6.3f) %2d   %15.9f %15.9f |%2d   %15.9f %15.9f',
                         k, kpt[0], kpt[1], kpt[2], nocca, homoa, lumoa, noccb, homob, lumob)

        log.note('**** Population analysis for atoms in the reference cell ****')
        s = self.get_ovlp(kpts=kpts).get()
        dm = self.make_rdm1().get()
        pop, chg = mulliken_meta(cell, dm, kpts=kpts, s=s, verbose=verbose)
        dip = None
        return (pop, chg), dip
