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

'''CPU atomic UHF calculations for GPU molecular SAD guesses.'''

import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf.data import elements
from pyscf.scf import atom_hf, uhf


class _FractionalUHF(uhf.UHF):
    def get_occ(self, mo_energy=None, mo_coeff=None):
        return self._atomic_occ.copy()

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        # The ordinary occupied/virtual gradient assumes integer occupations.
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore() + self.get_veff(self.mol, dm)
        return np.concatenate([
            ((occ[None, :] - occ[:, None]) * (c.conj().T @ f @ c)).ravel()
            for c, occ, f in zip(mo_coeff, mo_occ, fock)])

    def _finalize(self):
        # UHF's determinant <S^2> diagnostic does not apply to fractional occs.
        return atom_hf.AtomSphAverageRHF._finalize(self)


class _SphericalUHF(_FractionalUHF):
    atomic_configuration = elements.CONFIGURATION

    def eig(self, fock, s, **kwargs):
        # Average and diagonalize each l block separately for each spin.
        results = [atom_hf.AtomSphAverageRHF.eig(self, f, s) for f in fock]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])

    def get_occ(self, mo_energy=None, mo_coeff=None):
        occ = atom_hf.AtomSphAverageRHF.get_occ(self)
        return np.array((np.minimum(occ, 1), np.maximum(occ - 1, 0)))


def _atomic_mol(mol, ia, spherical):
    '''Extract the built basis, including decontraction and atom-specific ECPs.'''
    atom = mol.copy(deep=False)
    atom.atom = atom._atom = mol._atom[ia:ia+1]
    atom._atm = mol._atm[ia:ia+1].copy()
    atom._bas = mol._bas[mol._bas[:, gto.ATOM_OF] == ia].copy()
    atom._ecpbas = mol._ecpbas[mol._ecpbas[:, gto.ATOM_OF] == ia].copy()
    atom._bas[:, gto.ATOM_OF] = 0
    atom._ecpbas[:, gto.ATOM_OF] = 0
    atom.symmetry = False
    atom.symm_orb = None
    atom.groupname = 'C1'
    atom.charge = 0
    atom.enuc = 0
    atom._nelectron = None
    atom._nao = None
    if spherical:
        atom.cart = False
    z = gto.charge(mol.atom_symbol(ia))
    # Hund's rule for each partially filled subshell of the ground configuration.
    configuration = elements.CONFIGURATION[z]
    atom.spin = sum(min(n % (4*l+2), (4*l+2) - n % (4*l+2))
                    for l, n in enumerate(configuration))
    return atom


def _valence_occupations(atom):
    '''Psi4-style fractional occupations spanning the next noble-gas closure.

    This follows SAD_FRAC_OCC=true, SAD_SPIN_AVERAGE=false in Psi4's internal
    atomic UHF solver. Spins come from PySCF's ground configurations, rather
    than Psi4's separate empirical spin table. ECP cores are removed first.
    https://github.com/psi4/psi4/blob/258591c1c483c2231f228187ba4bcb30ba1178f9/psi4/src/psi4/libscf_solver/sad.cc#L428
    '''
    ne = atom.nelectron
    closures = np.array([0, 2, 10, 18, 36, 54, 86, 118])
    upper = np.searchsorted(closures, ne)
    if upper == len(closures):
        raise NotImplementedError('Atomic fractional occupations beyond 118 electrons')
    if closures[upper] == ne:
        ncore, nactive = 0, ne // 2
    else:
        ncore = closures[upper-1] // 2
        nactive = (closures[upper] - closures[upper-1]) // 2
    nactive = min(nactive, atom.nao - ncore)
    if nactive <= 0:
        raise ValueError('Insufficient atomic basis for fractional UHF occupations')
    occ = np.zeros((2, atom.nao))
    occ[:, :ncore] = 1
    for spin, n in enumerate(atom.nelec):
        frac = (n - ncore) / nactive
        if not 0 <= frac <= 1:
            raise ValueError('Insufficient atomic valence space for high-spin UHF')
        occ[spin, ncore:ncore+nactive] = frac
    return occ


def _atomic_uhf(atom, spherical):
    mf = _SphericalUHF(atom) if spherical else _FractionalUHF(atom)
    if not spherical:
        mf._atomic_occ = _valence_occupations(atom)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.chkfile = None
    # Start with spherical core orbitals and the actual fractional occupations.
    h = mf.get_hcore()
    e, c = mf.eig((h, h), mf.get_ovlp())
    dm = mf.make_rdm1(c, mf.get_occ(e, c))
    mf.kernel(dm0=dm)
    if not mf.converged:
        raise RuntimeError('Atomic UHF SAD calculation did not converge for '
                           f'{atom.atom_symbol(0)}')
    return mf


def atomic_uhf_orbitals(mol, spherical=True):
    '''Return block-diagonal atomic UHF orbitals and occupations on the CPU.

    With spherical=True, each spin retains equal occupations within magnetic
    multiplets throughout SCF. Otherwise use the Psi4-style fractional valence
    occupations; the caller can spin-average the resulting density. Both modes
    use exact HF integrals, including for molecular DFT. Ghost blocks are zero.

    Returns:
        NumPy orbital coefficients (2, nao, nmo) and occupations (2, nmo).
        Only orbitals occupied in at least one spin channel are retained.
    '''
    if mol._pseudo:
        raise NotImplementedError('Atomic UHF SAD does not support GTH pseudopotentials')
    coeffs = [[], []]
    occupations = [[], []]
    cache = {}
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        if gto.is_ghost_atom(mol.atom_symbol(ia)) or p0 == p1:
            c = np.zeros((2, p1-p0, 0))
            occ = np.zeros((2, 0))
        else:
            atom = _atomic_mol(mol, ia, spherical)
            key = (mol.atom_symbol(ia), atom._bas[:, 1:].tobytes(),
                   atom._ecpbas[:, 1:].tobytes())
            if key not in cache:
                mf = _atomic_uhf(atom, spherical)
                c, occ = mf.mo_coeff, mf.mo_occ
                if mol.cart and spherical:
                    c = np.array([atom.cart2sph_coeff() @ cs for cs in c])
                occupied = np.any(occ > 0, axis=0)
                cache[key] = c[:, :, occupied], occ[:, occupied]
            c, occ = cache[key]
        for spin in range(2):
            coeffs[spin].append(c[spin])
            occupations[spin].append(occ[spin])
    return (np.array([scipy.linalg.block_diag(*cs) for cs in coeffs]),
            np.array([np.concatenate(ns) for ns in occupations]))
