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

import unittest
import numpy as np
import cupy as cp
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import gate_charge
from gpu4pyscf.pbc.scf.slabgate import _zero_temperature_occ
from gpu4pyscf.pbc.scf.slabdipole import _slab_charge


def make_cell(nelectron=7.7):
    cell = gto.M(
        a=np.diag([7., 8., 22.]), unit='Bohr',
        atom='O 3 4 10; H 3 2.57 11.1; H 3 5.43 11.1',
        basis='gth-szv', pseudo='gth-pade', mesh=[9, 9, 25],
        precision=1e-6, verbose=0)
    if nelectron is None:
        cell.charge = 2
    else:
        cell.charge = 8 - nelectron
        cell.nelectron = nelectron
    return cell


def make_mf(cell, name='RKS', nkpts=1):
    if name.startswith('K'):
        mf = getattr(dft, name)(cell, cell.make_kpts([nkpts, 1, 1]), xc='lda,vwn')
    else:
        mf = getattr(dft, name)(cell, xc='lda,vwn')
    mf.chkfile = None
    return mf


def energies(name, nkpts):
    energy = cp.linspace(-1., 1., 6)
    if name.startswith('K'):
        energy = energy[None, :] + cp.arange(nkpts)[:, None] * 1e-4
    return energy


class KnownValues(unittest.TestCase):
    def assert_density_count(self, dm, overlap, nkpts):
        count = float(cp.einsum('...ij,...ji->...', dm, overlap).real.sum()) / nkpts
        self.assertAlmostEqual(count, 7.7, delta=1e-9)

    def test_zero_temperature(self):
        cases = [('RKS', 1, 7), ('RKS', 1, 8), ('RKS', 1, 7.7),
                 ('KRKS', 1, 7.7), ('KRKS', 2, 7.),
                 ('KRKS', 2, 7.7), ('KRKS', 2, 8.3), ('KRKS', 20, 7.7)]
        for name, nkpts, target in cases:
            with self.subTest(name=name, nkpts=nkpts, target=target):
                mf = gate_charge(make_mf(make_cell(target), name, nkpts),
                                 fractional_electrons=True)
                energy = energies(name, nkpts)
                occ = mf.get_occ(energy)
                # Independent filling by cumulative capacity for distinct levels.
                expected = cp.clip(target * nkpts - 2 * cp.arange(energy.size), 0, 2)
                cp.testing.assert_allclose(occ.ravel()[cp.argsort(energy.ravel())],
                                           expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            _zero_temperature_occ([0., 1., 2., 3., 3.], 7.7), [2, 2, 2, .85, .85])

    def test_smearing_orders(self):
        for name, nkpts, method in [('RKS', 1, 'fermi'), ('KRKS', 2, 'fermi'),
                                    ('KRKS', 2, 'gaussian')]:
            for gate_first in (False, True):
                with self.subTest(name=name, method=method, gate_first=gate_first):
                    mf = make_mf(make_cell(), name, nkpts)
                    if gate_first:
                        mf = gate_charge(mf, fractional_electrons=True)
                        mf = mf.smearing(.03, method=method)
                    else:
                        mf = gate_charge(mf.smearing(.03, method=method),
                                         fractional_electrons=True)
                    energy = energies(name, nkpts)
                    self.assertAlmostEqual(float(mf.get_occ(energy).sum()) / nkpts,
                                           7.7, delta=1e-10)
                    self.assertGreater(mf.entropy, 0.)
                    mf.sigma = 0.
                    self.assertAlmostEqual(float(mf.get_occ(energy).sum()) / nkpts,
                                           7.7, delta=1e-10)
                    self.assertEqual(mf.entropy, 0.)

    def test_validation_and_legacy_charge(self):
        old_cell = make_cell(None)
        old_mf = gate_charge(make_mf(old_cell, 'KRKS', 2))
        self.assertFalse(hasattr(old_mf, '_gate_nelectron'))
        self.assertEqual(_slab_charge(old_mf, old_cell), 1.)
        mf = gate_charge(make_mf(make_cell(), 'KRKS', 2), fractional_electrons=True)
        self.assertAlmostEqual(_slab_charge(mf, mf.cell), .3)
        mf.cell.charge = -.3
        with self.assertRaisesRegex(ValueError, 'match'):
            mf.get_occ(energies('KRKS', 2))
        cell = make_cell(None)
        cell.charge = .3
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            gate_charge(make_mf(cell), fractional_electrons=True)

        mf = make_mf(make_cell())
        mf.cell.spin = 1
        with self.assertRaisesRegex(NotImplementedError, 'spin=0'):
            gate_charge(mf, fractional_electrons=True)
        with self.assertRaisesRegex(NotImplementedError, 'RKS/KRKS'):
            gate_charge(make_mf(make_cell(), 'UKS'), fractional_electrons=True)
        mf = gate_charge(make_mf(make_cell()), fractional_electrons=True).smearing(.03)
        mf.mu0 = 0.
        with self.assertRaisesRegex(ValueError, 'mu0'):
            mf.get_occ(energies('RKS', 1))

    def test_explicit_switch(self):
        # Explicit integer N alone keeps the original occupation methods.
        mf = make_mf(make_cell(8))
        original_get_occ = mf.get_occ.__func__
        gate_charge(mf)
        self.assertFalse(hasattr(mf, '_gate_nelectron'))
        self.assertIs(mf.get_occ.__func__, original_get_occ)
        for charge, nelectron in [(.3, None), (.3, 7.7), (0, 7.7), (1, 7)]:
            with self.subTest(charge=charge, nelectron=nelectron):
                cell = make_cell(nelectron)
                cell.charge = charge
                with self.assertRaisesRegex(ValueError, 'fractional_electrons=True'):
                    gate_charge(make_mf(cell), fractional_electrons=False)
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            gate_charge(make_mf(make_cell(None)), fractional_electrons=True)

        # Legacy explicit N must match charge/Nk, not charge per cell.
        cell = make_cell(None)
        cell.nelectron = 7
        mf = gate_charge(make_mf(cell, 'KRKS', 2))
        self.assertFalse(hasattr(mf, '_gate_nelectron'))
        self.assertEqual(_slab_charge(mf, cell), 1.)
        with self.assertRaisesRegex(ValueError, 'match charge/nkpts'):
            gate_charge(make_mf(make_cell(6), 'KRKS', 2))

        mf = gate_charge(make_mf(make_cell()), fractional_electrons=True)
        self.assertIs(gate_charge(mf, fractional_electrons=True), mf)
        with self.assertRaisesRegex(ValueError, 'Cannot disable'):
            gate_charge(mf, fractional_electrons=False)
        self.assertAlmostEqual(mf._gate_nelectron(), 7.7)

    def test_fractional_orbital_gradient(self):
        for name, nkpts in [('RKS', 1), ('KRKS', 2)]:
            with self.subTest(name=name):
                # Smearing outside gate must also use the fractional gradient.
                mf = gate_charge(make_mf(make_cell(), name, nkpts),
                                 fractional_electrons=True).smearing(.03)
                coeff, occ = cp.eye(4), cp.asarray([2., 2., 2., 1.7])
                fock = cp.ones((4, 4))
                if name == 'KRKS':
                    coeff = cp.repeat(coeff[None], nkpts, axis=0)
                    occ = cp.repeat(occ[None], nkpts, axis=0)
                    fock = cp.repeat(fock[None], nkpts, axis=0)
                expected = cp.tile(cp.asarray([0., 0., 0., .3, .3, .3]), nkpts)
                for sigma in (.03, 0.):
                    mf.sigma = sigma
                    cp.testing.assert_allclose(mf.get_grad(coeff, occ, fock), expected,
                                               atol=1e-12, rtol=0)

    def test_scf_run(self):
        for name, nkpts, sigma in [('RKS', 1, 0.), ('KRKS', 2, .03)]:
            with self.subTest(name=name):
                mf = gate_charge(make_mf(make_cell(), name, nkpts), zgate=17.,
                                 fractional_electrons=True)
                if sigma:
                    mf = mf.smearing(sigma)
                overlap = mf.get_ovlp()
                dm = mf.get_init_guess(s1e=overlap)
                self.assert_density_count(dm, overlap, nkpts)
                # Check the SCF path and electron count, not energy convergence.
                mf.max_cycle = 2
                mf.diis = False
                self.assertTrue(np.isfinite(mf.kernel(dm0=dm)))
                self.assertAlmostEqual(float(mf.mo_occ.sum()) / nkpts, 7.7, delta=1e-10)
                self.assert_density_count(mf.make_rdm1(), overlap, nkpts)
                self.assertTrue(np.isfinite(mf.scf_summary['egate']))


if __name__ == '__main__':
    print("Full tests for pbc gate fractional occupations...")
    unittest.main()
