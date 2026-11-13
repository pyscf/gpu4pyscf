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
from unittest import mock
import numpy as np
import cupy as cp
from gpu4pyscf.pbc.scf import slabgate
from gpu4pyscf.pbc.scf.addons import gate_charge, slab_dipole_correction
from gpu4pyscf.pbc.grad.slabgate import gate_grad
from gpu4pyscf.pbc.grad.slabdipole import dipole_grad
from gpu4pyscf.pbc.scf.tests.test_pbc_slabgate import make_cell, make_mf
from gpu4pyscf.pbc.grad.tests.test_pbc_grad_slabdipole import (
    displaced_cell, finite_difference, DISP)


def gate_energy(mf, dm):
    vc, vb = mf._gate_matrices(mf.cell, mf.kpts)
    if mf.istype('UHF') or mf.istype('KUHF'):
        dm = dm[0] + dm[1]
    dm = dm.reshape(vc.shape)
    electronic = float(cp.einsum('kij,kji->', dm, vc+vb).real) / len(vc)
    return electronic + sum(mf._gate_ionic_energy(mf.cell))


class KnownValues(unittest.TestCase):
    @mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', 2./22.)
    def check_explicit(self, name='RKS', gate_side=-1):
        cell = make_cell()

        def build(c):
            mf = make_mf(c, name)
            # Enclose the gate and return region, while deliberately placing
            # an ion on the slab-facing ramp to test the ionic barrier force.
            if gate_side == -1:
                zgate, start, end = 5., 2.9, 12.4
            else:
                zgate, start, end = 17., 9.6, 19.1
            return gate_charge(mf, zgate=zgate, barrier=True, barrier_start=start,
                               barrier_end=end, barrier_width=1.,
                               dipole_gap=.1, gate_side=gate_side)

        mf = build(cell)
        dm = mf.get_init_guess()
        grad = mf.Gradients()
        grad.grid_response = False

        def energy(c):
            return gate_energy(build(c), dm)

        numeric_gate = finite_difference(cell, energy)
        np.testing.assert_allclose(gate_grad(grad, dm), numeric_gate, atol=3e-7, rtol=0)

        def dipole_energy(c):
            return build(c).get_veff(dm=dm).edip

        numeric_dipole = finite_difference(cell, dipole_energy)
        np.testing.assert_allclose(dipole_grad(grad, dm), numeric_dipole, atol=3e-7, rtol=0)

    def test_explicit_finite_difference(self):
        self.check_explicit('RKS')
        self.check_explicit('KUKS', gate_side=1)

    def test_ions_in_dipole_return_region(self):
        cell = make_cell(charge=0)

        def build(c):
            return slab_dipole_correction(make_mf(c), cut=10.4, z0=21.4, width=2.)

        mf = build(cell)
        dm = mf.get_init_guess()

        def energy(c):
            return build(c).get_veff(dm=dm).edip

        numeric = finite_difference(cell, energy)
        np.testing.assert_allclose(dipole_grad(mf.Gradients(), dm), numeric, atol=3e-7, rtol=0)

    @mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', 2./22.)
    def check_scf(self, name='RKS', multigrid=False, gate_side=-1):
        cell = make_cell()
        cell.atom = 'C 2.1 2.8 10.; H 2.8 3.3 11.7'
        cell.charge = 1
        cell.build()
        if name.startswith('K'):
            cell.charge = 2  # One hole per primitive cell for two k-points.
        if gate_side == -1:
            zgate, start, end = 5., 2.9, 7.
        else:
            zgate, start, end = 17., 15., 19.1
        mf = gate_charge(make_mf(cell, name, multigrid), zgate=zgate,
                         barrier=True, barrier_start=start, barrier_end=end,
                         barrier_width=.7, dipole_gap=.1,
                         gate_side=gate_side)
        mf.kernel()
        self.assertTrue(mf.converged)
        grad = mf.Gradients()
        analytic = grad.kernel()
        self.assertTrue(np.isfinite(analytic).all())
        cut = mf.dipole_cut
        scanner = grad.as_scanner()
        for axis in range(3):
            ep, _ = scanner(displaced_cell(cell, 1, axis, DISP))
            self.assertTrue(scanner.base.converged)
            em, _ = scanner(displaced_cell(cell, 1, axis, -DISP))
            self.assertTrue(scanner.base.converged)
            self.assertAlmostEqual(analytic[1, axis], (ep-em)/(2*DISP),
                                   delta=1e-5)
        self.assertEqual(scanner.base.dipole_cut, cut)
        self.assertEqual(scanner.base.gate_side, gate_side)

    def test_scf_finite_difference(self):
        self.check_scf('RKS')
        self.check_scf('KUKS', gate_side=1)

    def test_multigrid_scf_finite_difference(self):
        self.check_scf('RKS', multigrid=True, gate_side=1)

    @mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', 2./22.)
    def test_fractional_scf_finite_difference(self):
        cell = make_cell(charge=0)
        cell.charge = .3
        cell.nelectron = 7.7
        for name, nkpts, sigma in [('RKS', 1, 0.), ('KRKS', 2, .03)]:
            with self.subTest(name=name, sigma=sigma):
                def build(c):
                    mf = gate_charge(
                        make_mf(c, name), fractional_electrons=True, zgate=17.,
                        barrier=True, barrier_start=14.9, barrier_end=19.,
                        barrier_width=.7, dipole_gap=.1)
                    return mf.smearing(sigma, method='fermi') if sigma else mf

                mf = build(cell)
                mf.kernel()
                self.assertTrue(mf.converged)
                self.assertAlmostEqual(float(mf.mo_occ.sum()) / nkpts, 7.7, delta=1e-9)
                analytic = mf.nuc_grad_method().kernel()
                self.assertTrue(np.isfinite(analytic).all())
                dm0 = mf.make_rdm1()
                energies = []
                # Move H along z, keeping N, the cell, grid and gate fixed.
                for step in (-DISP, DISP):
                    displaced = build(displaced_cell(cell, 1, 2, step))
                    displaced.kernel(dm0=dm0.copy())
                    self.assertTrue(displaced.converged)
                    self.assertAlmostEqual(float(displaced.mo_occ.sum()) / nkpts,
                                           7.7, delta=1e-9)
                    # Fermi-smearing forces are derivatives of the free energy.
                    energies.append(displaced.e_free if sigma else displaced.e_tot)
                numeric = (energies[1] - energies[0]) / (2 * DISP)
                self.assertAlmostEqual(analytic[1, 2], numeric, delta=1e-5)


if __name__ == '__main__':
    unittest.main()
