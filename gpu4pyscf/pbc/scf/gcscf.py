# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Constant-chemical-potential SCF with a compensating planar gate for the 1st case.

The gate and finite-width dipole energies are differentiated with respect to
both the density and their compensating charge. All charges and energies are
per primitive cell; cell.charge is only used by the initial density guess.
'''

from numbers import Real
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.scf import smearing
from gpu4pyscf.pbc.scf.slabgate import SlabGateCharge, _sheet_potential
from gpu4pyscf.pbc.scf.slabdipole import (
    SlabDipoleCorrection, _dipole_grids, _sawtooth, _slab_length,
)


def constant_mu(mf, mu, sigma):
    '''
    Enable fixed-mu SCF on a 3D RKS/UKS/KRKS/KUKS gate calculation.

    Apply slabgate.gate_charge first, with barrier=False. mu is the
    electron chemical potential in Hartree in the variational gate reference;
    sigma > 0 is the Fermi-Dirac width in Hartree. A single chemical potential 
    is used for both spins.

    Returns the decorated SCF object. The ordinary DIIS, damping and level-shift 
    settings are left unchanged. For damping-only calculations, set diis=False, 
    damp > 0 and diis_start_cycle > max_cycle to keep ordinary Fock damping active.

    e_tot remains the internal energy, e_free the Helmholtz free energy,
    and e_grand = e_free - mu0*gcscf_nelec the grand potential. The default
    convergence check uses the grand potential, density, electron number and
    fixed-mu occupation residuals. conv_tol_nelec defaults to 1e-6 electrons
    per cell. An instance check_convergence callback takes precedence.
    As for smearing, thermodynamic energies use the latest get_occ occupations;
    they require the corresponding density, and are not arbitrary-DM entropy
    evaluations. A zero-cycle run does not produce self-consistent free energies.

    It should be note that, the mu should be changed according to the neutral reference.
    mu_target = mu - e delta U, where U is the voltage, and is 
    mu_target = mu - delta U/27.2114, if U is in V.
    '''
    if not isinstance(mu, Real) or not np.isfinite(mu):
        raise ValueError('mu must be a finite real scalar in Hartree')
    if not isinstance(sigma, Real) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be a positive Fermi-Dirac width in Hartree')
    # TODO: this may change, due to more environment support for constant-mu SCF.
    if not isinstance(mf, SlabGateCharge):
        raise ValueError('Apply gate_charge before constant_mu')
    # TODO: support 2D cell.
    if mf.cell.dimension != 3 or not isinstance(mf, SlabDipoleCorrection):
        raise NotImplementedError('Constant-mu gate SCF requires a 3D cell with dipole correction')
    if mf.barrier:
        raise NotImplementedError('Constant-mu gate SCF does not support barrier')

    # if set mu0, constant mu routine is enabled.
    mf = smearing.smearing(mf, sigma=float(sigma), method='fermi',
                           mu0=float(mu), fix_spin=False)
    if not isinstance(mf, ConstantMuSCF):
        lib.set_class(mf, (ConstantMuSCF, mf.__class__))
    mf._gcscf_cache = None
    mf._gcscf_last_grand = None
    mf.gcscf_nelec = None
    mf.e_grand = None
    mf.entropy = mf.e_free = mf.e_zero = None
    logger.warn(mf, 'GCSCF has convergence issues: its SCF iteration has not been '
                'optimized and can be very difficult to converge. '
                'Use the FCP method instead.')
    return mf


class ConstantMuSCF:
    __name_mixin__ = 'ConstantMu'
    conv_tol_nelec = 1e-6
    gcscf_nelec = None
    e_grand = None
    _gcscf_cache = None
    _gcscf_last_grand = None

    _keys = {'conv_tol_nelec', 'gcscf_nelec', 'e_grand'}

    def reset(self, cell=None):
        self._gcscf_cache = None
        self._gcscf_last_grand = None
        self.gcscf_nelec = None
        self.e_grand = None
        self.entropy = self.e_free = self.e_zero = None
        return super().reset(cell)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'Constant-mu gate SCF: mu = %.12g Ha, sigma = %.6g Ha',
                    self.mu0, self.sigma)
        logger.info(self, 'Electron-number convergence threshold = %g', self.conv_tol_nelec)
        return self

    def get_hcore(self, cell=None, kpts=None, *, kpt=None):
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts if self.istype('KSCF') else (self.kpt if kpt is None else kpt)
        # The common SCF kernel caches hcore. All density-dependent gate terms
        # must instead enter get_veff; skip the fixed-charge gate hcore.
        return super(SlabGateCharge, self).get_hcore(cell, kpts)

    def energy_nuc(self):
        # Ionic-gate and sheet self energies are included in the dynamic energy.
        return super(SlabGateCharge, self).energy_nuc()

    def _gcscf_matrices(self, cell, kpts):
        from gpu4pyscf.pbc.dft import numint

        length = _slab_length(cell)
        if self.barrier:
            raise NotImplementedError('Constant-mu gate SCF does not support barrier')
        if self.cut is None:
            raise ValueError('Constant-mu SCF requires the fixed gate dipole geometry')
        width = self.dipole_width
        if not 0 <= width < length:
            raise ValueError('Dipole width must satisfy 0 <= width < cell length')
        zgate = self.zgate % length
        cut = self.cut % length
        kpts = np.asarray(kpts).reshape(-1, 3)
        grids = _dipole_grids(self, cell)
        mesh = getattr(grids, 'mesh', None)
        if mesh is None:
            mesh = cell.mesh
        key = (id(cell), tuple(kpts.ravel()), zgate, cut, width, tuple(mesh))
        if self._gcscf_cache is not None and self._gcscf_cache[0] == key:
            return self._gcscf_cache[1]

        area = cell.vol / length
        ni = numint.KNumInt()
        c1 = position = None
        for ao_kpts, weight, coords in ni.block_loop(cell, grids, kpts=kpts):
            if c1 is None:
                shape = (len(kpts), cell.nao, cell.nao)
                c1 = cp.zeros(shape, dtype=ao_kpts.dtype)
                position = cp.zeros_like(c1)
            z = coords[:, 2]
            vc = -_sheet_potential(z, zgate, length, area, 1.)
            f = _sawtooth(z, cut, length, width)
            for k, ao in enumerate(ao_kpts):
                c1[k] += ao.conj().T @ ((weight*vc)[:, None] * ao)
                position[k] += ao.conj().T @ ((weight*f)[:, None] * ao)
        if c1 is None:
            raise ValueError('Empty integration grid for constant-mu gate SCF')

        points = kpts if self.istype('KSCF') else kpts[0]
        overlap = cp.asarray(self.get_ovlp(cell, points)).reshape(c1.shape)
        charges = cell.atom_charges()
        z = cell.atom_coords()[:, 2]
        a_c = float(np.dot(charges, _sheet_potential(z, zgate, length, area, 1.).get()))
        b_c = np.pi*length/(3*area)
        p_ion = float(np.dot(charges, _sawtooth(z, cut, length, width).get()))
        f_c = float(_sawtooth(zgate, cut, length, width))
        data = (overlap, c1, position, a_c, b_c, p_ion, f_c)
        self._gcscf_cache = key, data
        self.dipole_cut = cut
        self.dipole_z0 = zgate
        self.dipole_offset = f_c
        return data

    def _gcscf_veff(self, dm, cell=None, kpts=None):
        if cell is None:
            cell = self.cell
        is_kpts = self.istype('KSCF')
        if kpts is None:
            kpts = self.kpts if is_kpts else self.kpt
        kpts = np.asarray(kpts).reshape(-1, 3)
        overlap, c1, position, a_c, b_c, p_ion, f_c = self._gcscf_matrices(cell, kpts)
        dm_charge = cp.asarray(dm)
        if self.istype('UHF') or self.istype('KUHF'):
            dm_charge = dm_charge[0] + dm_charge[1]
        dm_charge = dm_charge.reshape(c1.shape)
        nkpts = len(kpts)
        nelec = float(cp.einsum('kij,kji->', dm_charge, overlap).real) / nkpts
        qc = nelec - cell.atom_charges().sum()
        h_c = a_c + float(cp.einsum('kij,kji->', dm_charge, c1).real) / nkpts
        dipole = (p_ion + qc*f_c
                  - float(cp.einsum('kij,kji->', dm_charge, position).real) / nkpts)
        field = 4*np.pi*dipole/cell.vol
        shift = h_c + qc*b_c + field*f_c
        egate = qc*h_c + .5*qc**2*b_c
        edip = .5*field*dipole
        # first two terms are identical of constant electron gate.
        potential = qc*c1 - field*position + shift*overlap
        if not is_kpts:
            potential = potential[0]
        return tag_array(potential, nelec=nelec, gate_charge=float(qc),
                         egate=float(egate), edip=float(edip), dipole=float(dipole),
                         gcscf_shift=float(shift))

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None, kpt=None):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpts is None:
            kpts = self.kpts if self.istype('KSCF') else (self.kpt if kpt is None else kpt)
        # Skip the old dipole potential, which uses the fixed cell.charge.
        veff = super(SlabDipoleCorrection, self).get_veff(
            cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)
        extra = self._gcscf_veff(dm, cell, kpts)
        out = tag_array(veff + extra, **veff.__dict__)
        # DFT energy_elec uses ecoul + exc, not 1/2 Tr(D Vextra). Preserve the
        # bare JK tags for incremental builds, and add each environment energy once.
        out.exc = veff.exc + extra.egate + extra.edip
        for name in ('nelec', 'gate_charge', 'egate', 'edip', 'dipole', 'gcscf_shift'):
            setattr(out, name, getattr(extra, name))
        self.gcscf_nelec = extra.nelec
        return out

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if vhf is None or not hasattr(vhf, 'gate_charge'):
            vhf = self.get_veff(self.cell, dm)
        # Skip the static gate bookkeeping as well as its ionic-energy path.
        energy = super(SlabGateCharge, self).energy_elec(dm, h1e, vhf)
        self.gcscf_nelec = vhf.nelec
        self.scf_summary.update(nelec=vhf.nelec, gate_charge=vhf.gate_charge,
                                egate=vhf.egate, edip=vhf.edip)
        return energy

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = super().energy_tot(dm, h1e, vhf)
        entropy = 0. if self.entropy is None else self.entropy
        self.e_free = e_tot - self.sigma*entropy
        self.e_zero = e_tot - .5*self.sigma*entropy
        self._gcscf_last_grand = self.e_grand
        self.e_grand = self.e_free - self.mu0*self.gcscf_nelec
        self.scf_summary['grand_potential'] = self.e_grand
        logger.info(self, '    N = %.10g  Qgate = %.10g  Grand potential = %.15g',
                    self.gcscf_nelec, self.scf_summary['gate_charge'], self.e_grand)
        return e_tot

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if (not isinstance(self.mu0, Real) or not np.isfinite(self.mu0)
                or not isinstance(self.sigma, Real) or self.sigma <= 0 or not np.isfinite(self.sigma)
                or self.smearing_method != 'fermi' or self.fix_spin):
            raise ValueError('Constant-mu SCF requires scalar mu0 and positive Fermi smearing')
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_energy is None:
            raise ValueError('Orbital energies are not available')
        return super().get_occ(cp.asarray(mo_energy), mo_coeff)

    def check_convergence(self, envs):
        if envs['cycle'] == 0 or self._gcscf_last_grand is None:
            return False
        dm_diff = envs['dm'] - envs['dm_last']
        if self.istype('UHF') or self.istype('KUHF'):
            dm_diff = dm_diff[0] + dm_diff[1]
        overlap = envs['s1e']
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        shape = (nkpts, self.cell.nao, self.cell.nao)
        dnelec = abs(float(cp.einsum('kij,kji->', dm_diff.reshape(shape),
                                    overlap.reshape(shape)).real) / nkpts)
        # Orbital rotations alone miss the diagonal (occupation) stationarity,
        # particularly for one-AO systems or strongly damped Fock matrices.
        coeff = envs['mo_coeff']
        coeff_h = coeff.conj().swapaxes(-1, -2)
        eps = (coeff_h @ envs['fock'] @ coeff).diagonal(axis1=-2, axis2=-1).real
        norm = (coeff_h @ overlap @ coeff).diagonal(axis1=-2, axis2=-1).real
        degeneracy = 1 if self.istype('UHF') or self.istype('KUHF') else 2
        target_occ = degeneracy * cp.exp(-cp.logaddexp(0., (eps-self.mu0)/self.sigma))
        # Linear-dependency removal may pad different k-points with zero columns.
        target_occ = cp.where(norm > .5, target_occ, 0.)
        occ_error = float(cp.linalg.norm(target_occ-envs['mo_occ'])) / np.sqrt(nkpts)
        return bool(abs(self.e_grand-self._gcscf_last_grand) < envs['conv_tol']
                    and envs['norm_gorb'] < envs['conv_tol_grad']
                    and envs['norm_ddm'] < envs['conv_tol_grad']
                    and occ_error < envs['conv_tol_grad']
                    and dnelec < self.conv_tol_nelec)

    def Gradients(self):
        raise NotImplementedError('Constant-mu gate gradients')

    def undo_smearing(self):
        raise NotImplementedError('Create a new SCF object to remove constant-mu smearing')

    def to_cpu(self):
        raise NotImplementedError('Constant-mu gate SCF is a GPU addon')
