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
Fully converged constant potential with fixed-N KRKS and a planar gate.

All electron numbers and energies are per primitive cell. Energies, including
mu and sigma, are in Hartree. Only Fermi smearing and equal-weight k-points
are supported. The input Cell retains its reference charge; fcp_nelec controls
the actual electron number and compensating charge of each inner calculation.
'''

import numpy as np
import cupy as cp
from numpy.polynomial import Polynomial
from scipy.optimize import brentq
from scipy.special import expit, xlogy
from pyscf import lib
from pyscf.data.nist import HARTREE2EV
from pyscf.pbc.lib.kpts import KPoints
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.scf import smearing
from gpu4pyscf.pbc.scf.gcscf import ConstantMuSCF
from gpu4pyscf.pbc.scf.slabgate import SlabGateCharge, _sheet_potential
from gpu4pyscf.pbc.scf.slabdipole import _dipole_grids, _sawtooth, _slab_length


def _occupations(energies, nelec, sigma):
    '''
    Return one chemical potential, KRKS occupations and entropy per cell.
    '''
    if sigma is None or sigma <= 0:
        raise ValueError('FCP requires Fermi smearing with sigma > 0')
    energies = np.asarray(energies)
    nkpts, nmo = energies.shape
    if not 0 < nelec < 2*nmo:
        raise ValueError('Electron number must lie inside the orbital capacity')

    def residual(eta):
        return 2*expit((eta-energies)/sigma).sum()/nkpts - nelec

    eta = brentq(residual, energies.min()-50*sigma,
                 energies.max()+50*sigma, xtol=1e-14)
    occ = expit((eta-energies)/sigma)
    entropy = -2*np.sum(xlogy(occ, occ) + xlogy(1-occ, 1-occ))/nkpts
    return eta, 2*occ, float(entropy)


def _fit_potential(nelec, mu, tol):
    '''
    Fit in a scaled coordinate and increase the degree only when needed.
    '''
    nelec = np.asarray(nelec)
    mu = np.asarray(mu)
    max_degree = min(3, len(np.unique(nelec))-1)
    if max_degree < 1:
        raise ValueError('Fitting requires at least two different electron numbers')
    for degree in range(1, max_degree+1):
        # this follows the original FCP paper
        fit = Polynomial.fit(nelec, mu, degree)
        rms = np.sqrt(np.mean((fit(nelec)-mu)**2))
        if rms <= tol:
            break
    return fit


def _newton_step(fit, nelec, mu, lr, max_step):
    slope = float(fit.deriv()(nelec))
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError('FCP requires a positive chemical-potential slope')
    step = -lr*(float(fit(nelec))-mu)/slope
    return nelec + np.clip(step, -max_step, max_step), slope


class _FixedNKRKS:

    __name_mixin__ = 'FixedN'
    _keys = {'fcp_nelec', 'fcp_eta'}

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.smearing_method != 'fermi' or self.mu0 is not None:
            raise ValueError('FCP requires fixed-N Fermi smearing (mu0=None)')
        if mo_energy is None:
            mo_energy = self.mo_energy
            if mo_coeff is None:
                mo_coeff = self.mo_coeff

        eta, occ, entropy = _occupations(cp.asnumpy(mo_energy),
                                        self.fcp_nelec, self.sigma)
        self.fcp_eta = eta
        self.entropy = entropy
        logger.debug(self, '    FCP N = %.10f eta = %.12g entropy = %.12g',
                     self.fcp_nelec, eta, entropy)
        return cp.asarray(occ)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        self._fcp_previous_free = self.e_free
        return super().energy_tot(dm, h1e, vhf)

    def check_convergence(self, envs):
        if envs['cycle'] == 0 or self._fcp_previous_free is None:
            return False
        # Check occupations against the physical Fock, not the DIIS Fock.
        coeff = envs['mo_coeff']
        coeff_h = coeff.conj().swapaxes(-1, -2)
        eps = (coeff_h @ envs['fock'] @ coeff).diagonal(axis1=-2, axis2=-1).real
        eta, occ, _ = _occupations(cp.asnumpy(eps), self.fcp_nelec, self.sigma)
        occ_error = float(cp.linalg.norm(cp.asarray(occ)-envs['mo_occ'])) / np.sqrt(len(self.kpts))
        # A uniform Fock shift leaves fixed-N occupations unchanged but changes eta.
        eta_error = abs(eta-self.fcp_eta)
        return bool(abs(self.e_free-self._fcp_previous_free) < envs['conv_tol']
                    and envs['norm_gorb'] < envs['conv_tol_grad']
                    and envs['norm_ddm'] < envs['conv_tol_grad']
                    and occ_error < envs['conv_tol_grad']
                    and eta_error < envs['conv_tol_grad'])

    def undo_smearing(self):
        raise NotImplementedError('FCP requires positive Fermi smearing')


class FCP:
    '''Outer Newton-fitting loop around a gate-decorated KRKS calculation.

    Example::

        mf = gate_charge(dft.KRKS(cell, kpts), zgate=12.)
        calc = FCP(mf, mu=-0.2)
        calc.kernel()

    Fermi smearing is enabled by default (0.1 eV). sigma=None or zero is an
    error, not a request for zero-temperature filling. mu is the target in the
    variational gate reference; it is not an SHE potential or smearing.mu0.
    kernel returns the grand potential, with the final state in calc.mf.
    Set nelec_bounds from fixed-charge trial calculations for the chosen model.
    conv_tol and fit_tol are in Hartree; sample_step and max_step are electrons
    per cell. The inner SCF tolerances must resolve the target chemical potential.
    Trial steps, including presampling, respect max_step and nelec_bounds.
    A failed inner SCF or exhausted outer loop raises RuntimeError and leaves
    converged=False.

    Geometry, basis, gate and dipole parameters must remain fixed while reusing
    this object. Create a new FCP object after changing them.
    '''

    conv_tol = 1e-3/HARTREE2EV
    fit_tol = .1/HARTREE2EV
    max_cycle = 100
    max_step = .2
    sample_step = .05
    adaptive_lr = False
    nelec_bounds = None

    def __init__(self, mf, mu, sigma=.1/HARTREE2EV):
        if sigma is None or sigma <= 0:
            raise ValueError('FCP requires Fermi smearing with sigma > 0')
        from gpu4pyscf.pbc.dft.krks import KRKS

        if not isinstance(mf, KRKS):
            raise NotImplementedError('FCP requires KRKS with an equal-weight k-point array')
        # TODO: 2D not support yet.
        if not isinstance(mf, SlabGateCharge) or mf.cell.dimension != 3 or mf.barrier:
            raise NotImplementedError('FCP requires a 3D gate calculation without barrier')
        if isinstance(mf, ConstantMuSCF) or mf.level_shift:
            raise NotImplementedError('FCP cannot wrap constant-mu SCF (GCSCF) or level shifting')
        mf = smearing.smearing(mf, sigma=sigma, method='fermi', mu0=None)
        if not isinstance(mf, _FixedNKRKS):
            lib.set_class(mf, (_FixedNKRKS, mf.__class__))
        self.mf = mf
        self.mu = float(mu)
        self.nelec = float(mf.cell.atom_charges().sum())
        mf.fcp_nelec = self.nelec
        mf.fcp_eta = None
        mf._fcp_previous_free = None
        self.converged = False
        self.history = []
        self._matrices = None
        self.dm = None
        self.eta = self.shift = self.mu_eff = None
        self.e_tot = self.e_free = self.e_grand = None

    def _prepare(self):
        # TODO: this should be moved to gate_charge.
        from gpu4pyscf.pbc.dft import numint

        mf = self.mf
        cell = mf.cell
        length = _slab_length(cell)
        area = cell.vol/length
        zgate = mf.zgate % length
        cut = mf.cut % length
        width = mf.dipole_width
        # Use the same zero-offset Brumme reference as the work equations.
        mf.z0 = (cut+length/2) % length
        overlap = cp.asarray(mf.get_ovlp())
        c1 = cp.zeros(overlap.shape, dtype=complex)
        position = cp.zeros_like(c1)
        grids = _dipole_grids(mf, cell)
        ni = numint.KNumInt()
        for ao_kpts, weight, coords in ni.block_loop(cell, grids, kpts=mf.kpts):
            z = coords[:, 2]
            vc = -_sheet_potential(z, zgate, length, area, 1.)
            f = _sawtooth(z, cut, length, width)
            for k, ao in enumerate(ao_kpts):
                c1[k] += ao.conj().T @ ((weight*vc)[:, None]*ao)
                position[k] += ao.conj().T @ ((weight*f)[:, None]*ao)
        charges = cell.atom_charges()
        z = cell.atom_coords()[:, 2]
        a_c = float(np.dot(charges, _sheet_potential(z, zgate, length, area, 1.).get()))
        b_c = np.pi*length/(3*area)
        p_ion = float(np.dot(charges, _sawtooth(z, cut, length, width).get()))
        f_c = float(_sawtooth(zgate, cut, length, width))
        self._matrices = overlap, c1, position, a_c, b_c, p_ion, f_c

    def chemical_potential_shift(self, dm):
        '''
        Gate and dipole charge derivative Lambda, in Hartree.
        '''
        if self._matrices is None:
            self._prepare()
        overlap, c1, position, a_c, b_c, p_ion, f_c = self._matrices
        nkpts = len(self.mf.kpts)
        qc = self.mf.fcp_nelec-self.mf.cell.atom_charges().sum()
        h_c = a_c + float(cp.einsum('kij,kji->', dm, c1).real)/nkpts
        dipole = p_ion + qc*f_c - float(cp.einsum('kij,kji->', dm, position).real)/nkpts
        return h_c + qc*b_c + 4*np.pi*dipole/self.mf.cell.vol*f_c

    def evaluate(self, nelec, dm0=None):
        '''
        Run one complete fixed-N SCF and return eta + Lambda.
        Pass the previous density directly to the ordinary SCF kernel.
        '''
        self.converged = False
        mf = self.mf
        if dm0 is None:
            dm0 = self.dm
        # Do not expose the previous solution as the result of a failed trial.
        self.dm = None
        self.eta = self.shift = self.mu_eff = None
        self.e_tot = self.e_free = self.e_grand = None
        mf.converged = False
        if mf.max_cycle <= 0:
            raise ValueError('FCP requires a complete SCF, not a zero-cycle calculation')
        if self.nelec_bounds is not None:
            lower, upper = self.nelec_bounds
            if not lower <= nelec <= upper:
                raise ValueError('Trial electron number is outside nelec_bounds')
        mf.fcp_nelec = float(nelec)
        if self._matrices is None:
            self._prepare()
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError('Fixed-electron-number SCF did not converge')
        self.nelec = float(nelec)
        self.dm = mf.make_rdm1()
        self.eta = mf.fcp_eta
        self.shift = self.chemical_potential_shift(self.dm) # electron changed part
        self.mu_eff = self.eta + self.shift
        self.e_tot = mf.e_tot
        self.e_free = mf.e_free
        self.e_grand = self.e_free-self.mu*self.nelec
        return self.mu_eff

    def kernel(self, nelec=None, dm0=None):
        '''
        Converge the actual chemical potential.
        '''
        if nelec is None:
            nelec = self.nelec
        initial = float(nelec)
        self.converged = False
        self.history = []
        self._matrices = None
        if self.nelec_bounds is not None:
            lower, upper = self.nelec_bounds
            if not lower < upper or not lower <= initial <= upper:
                raise ValueError('Initial electron number must lie inside nelec_bounds')
        # The second sampling move crosses both sides of the initial point.
        step = min(self.sample_step, self.max_step/2)
        samples = [initial-step, initial+step]
        if self.nelec_bounds is not None:
            samples = np.clip(samples, *self.nelec_bounds)
        samples = [float(n) for n in samples if n != initial]
        lr, oscillations = 1., 1
        previous = None
        numbers, potentials = [], []
        for cycle in range(self.max_cycle):
            mu_eff = self.evaluate(nelec, dm0)
            dm0 = self.dm
            residual = mu_eff-self.mu
            numbers.append(self.nelec)
            potentials.append(mu_eff)
            self.history.append((self.nelec, mu_eff, self.e_grand))
            logger.info(self.mf, 'FCP %d N = %.10f mu = %.10f residual = %.3g Ha',
                        cycle+1, self.nelec, mu_eff, residual)
            if abs(residual) < self.conv_tol:
                self.converged = True
                return self.e_grand
            if cycle+1 == self.max_cycle:
                break
            # handle oscillations
            if self.adaptive_lr and previous is not None and previous*residual < 0:
                oscillations += 1
                lr -= .5**oscillations
            previous = residual
            if samples:
                # for starting steps
                nelec = samples.pop(0)
            else:
                # next step nelec from newton update
                fit = _fit_potential(numbers, potentials, self.fit_tol)
                nelec, slope = _newton_step(fit, self.nelec, self.mu, lr, self.max_step)
                if self.nelec_bounds is not None:
                    nelec = float(np.clip(nelec, *self.nelec_bounds))
                    if nelec == self.nelec:
                        raise RuntimeError('FCP target cannot be reached within nelec_bounds')
                logger.debug(self.mf, 'FCP fit degree = %d slope = %.8g lr = %.6g',
                             fit.degree(), slope, lr)
        raise RuntimeError('FCP did not converge within max_cycle')
