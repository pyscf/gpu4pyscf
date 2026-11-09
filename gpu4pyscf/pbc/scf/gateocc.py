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
Real electron counts for explicitly charged RKS/KRKS gate calculations.

Changes from the normal SCF paths are noted inline. The standard smearing
kernels, root solver, entropy and initial-guess generators are called directly.

TODO: this may be applied to normal SCF calculations as well.
'''

import numpy as np


def _check_nelectron(cell):
    if cell._nelectron is None:
        raise ValueError('Set cell.nelectron explicitly for fractional gate occupations')
    nelectron = float(cell.nelectron)
    charge = float(cell.charge)
    if not np.isfinite(nelectron) or nelectron < 0 or not np.isfinite(charge):
        raise ValueError('Gate electron number must be finite and nonnegative')
    if abs(nelectron + charge - cell.atom_charges().sum()) > 1e-9:
        raise ValueError('cell.nelectron and cell.charge must match per primitive cell')
    if cell.spin != 0:
        raise NotImplementedError('Fractional gate occupations require spin=0 RKS/KRKS')
    return nelectron


def _zero_temperature_occ(mo_energy, nelectron):
    '''
    Global filling adapted from pbc.scf.khf.get_occ.
    '''
    energies = np.asarray(mo_energy)
    if not 0 <= nelectron <= 2 * energies.size:
        raise ValueError('Electron number exceeds the available orbital capacity')
    occ = np.zeros_like(energies, dtype=float)
    if nelectron == 0:
        return occ
    # Unlike khf.get_occ's N//2, include the partially occupied frontier state.
    fermi = np.sort(energies.ravel())[int(np.ceil(nelectron / 2)) - 1]
    # Replace the normal energy <= fermi -> 2 rule by sharing the remainder.
    below = energies < fermi - 1e-9
    frontier = abs(energies - fermi) <= 1e-9
    occ[below] = 2
    occ[frontier] = (nelectron - 2 * below.sum()) / frontier.sum()
    return occ


class GateFractionalOccupations:
    '''
    Opt-in, per-cell electron counting; ordinary SCF objects are unchanged.
    '''
    __name_mixin__ = 'Fractional'

    def _gate_nelectron(self, cell=None):
        nelectron = _check_nelectron(self.cell if cell is None else cell)
        if getattr(self, 'mu0', None) is not None:
            raise ValueError('Gate occupations fix the electron number; mu0 must be None')
        if self.level_shift:
            raise NotImplementedError('Level shifting with fractional gate occupations')
        return nelectron

    def build(self, cell=None):
        self._gate_nelectron(cell)
        return super().build(cell)

    def reset(self, cell=None):
        self._gate_nelectron(cell)
        return super().reset(cell)

    def _get_gate_occ(self, mo_energy=None, mo_coeff=None):
        '''
        Restricted-spin adaptation of _SmearingSCF/_SmearingKSCF.get_occ.
        '''
        import cupy as cp
        from gpu4pyscf.lib import logger
        from gpu4pyscf.scf import smearing

        nelectron = self._gate_nelectron()
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        if mo_energy is None:
            mo_energy = self.mo_energy
        energies = cp.asnumpy(mo_energy)
        # Do not use Cell.tot_electrons(nkpts): it rounds the explicit count.
        target = nelectron * nkpts
        if not np.all(np.isfinite(energies)):
            raise ValueError('Gate orbital energies must be finite')
        if not 0 <= target <= 2 * energies.size:
            raise ValueError('Electron number exceeds the available orbital capacity')
        sigma = getattr(self, 'sigma', None)
        method = getattr(self, 'smearing_method', None)
        if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
            raise ValueError('Smearing sigma must be finite and nonnegative')
        if sigma and method:
            if method.lower() == 'fermi':
                f_occ = smearing._fermi_smearing_occ
            elif method.lower() == 'gaussian':
                f_occ = smearing._gaussian_smearing_occ
            else:
                raise ValueError('Gate smearing supports fermi or gaussian')
            if target == 0 or target == 2 * energies.size:
                occ = np.full_like(energies, 0. if target == 0 else 2., dtype=float)
                self.entropy = 0.
            else:
                # Reuse the solver, replacing KRKS's (N+1)//2 by real N/2.
                mu, occ = smearing._smearing_optimize(
                    f_occ, energies.ravel(), target / 2, sigma)
                self.entropy = 2 * self._get_entropy(energies.ravel(), occ, mu) / nkpts
                occ = (2 * occ).reshape(energies.shape)
                logger.info(self, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                            sigma, mu, self.entropy)
        else:
            # Normal smearing falls back to integer filling at sigma=0.
            occ = _zero_temperature_occ(energies, target)
            if hasattr(self, 'entropy'):
                self.entropy = 0.
                self.e_free = self.e_zero = None
        if abs(occ.sum() / nkpts - nelectron) > 1e-9:
            raise RuntimeError('Gate occupations do not conserve the electron number')
        return cp.asarray(occ)

    get_occ = _get_gate_occ

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        import cupy as cp
        from gpu4pyscf.scf import hf

        if cell is None:
            cell = self.cell
        target = self._gate_nelectron(cell)
        # Adapt pbc.scf.hf.RHF / khf.KRHF.get_init_guess: reuse the common
        # dispatcher, but normalize to the real target rather than tot_electrons.
        dm = hf.RHF.get_init_guess(self, cell, key)
        if s1e is None:
            s1e = self.get_ovlp(cell)
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        count = float(cp.einsum('...ij,...ji->...', dm, s1e).real.sum()) / nkpts
        if count == 0:
            if target != 0:
                raise ValueError('Cannot normalize an empty gate initial density')
            return dm
        # Unlike the normal 0.01-electron threshold, always match the real target.
        scale = target / count
        dm *= scale
        if hasattr(dm, 'mo_occ'):
            dm.mo_occ *= scale
        return dm

    def _get_gate_grad(self, mo_coeff, mo_occ, fock=None):
        import cupy as cp

        self._gate_nelectron()
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore() + self.get_veff(self.cell, dm)
        f_mo = mo_coeff.swapaxes(-1, -2).conj() @ fock @ mo_coeff
        # Normal hf/khf.get_grad only pairs occupied and empty orbitals.
        # Occupation differences also include full-to-partial rotations.
        gradient = f_mo * (mo_occ[..., None, :] - mo_occ[..., :, None])
        i, j = cp.tril_indices(mo_occ.shape[-1], -1)
        return gradient[..., i, j].ravel()

    get_grad = _get_gate_grad

    def get_fermi(self, mo_energy=None, mo_occ=None):
        import cupy as cp

        target = self._gate_nelectron()
        if target == 0:
            return -np.inf
        if mo_energy is None:
            mo_energy = self.mo_energy
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        # Unlike normal get_fermi's integer truncation, include the partial state.
        # This remains a zero-temperature reference, not the smearing chemical potential.
        index = int(np.ceil(target * nkpts / 2)) - 1
        return float(cp.sort(cp.asarray(mo_energy).ravel())[index])

    def newton(self):
        raise NotImplementedError('Second-order SCF with fractional gate occupations')

    def Gradients(self):
        raise NotImplementedError('Nuclear gradients with fractional gate occupations')
