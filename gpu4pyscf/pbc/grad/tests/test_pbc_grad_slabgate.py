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
    def check_explicit(self, name='RKS', multigrid=False, becke=False):
        cell = make_cell()
        def build(c):
            mf = make_mf(c, name, multigrid, becke)
            # Intentionally place ions on the barrier ramps to test their force.
            gate_charge(mf, zgate=17., barrier=True, barrier_start=9.6,
                        barrier_end=12.4, barrier_width=1.)
            return slab_dipole_correction(mf, cut=20., z0=9., width=2.)
        mf = build(cell)
        dm = mf.get_init_guess()
        rng = np.random.default_rng(5)
        ddm = rng.random(dm.shape)
        if name.startswith('K') and not multigrid:
            ddm = ddm + 1j*rng.random(dm.shape)
        ddm = cp.asarray(ddm)
        dm = dm + .01*(ddm + ddm.swapaxes(-1, -2).conj())
        if name in ('UKS', 'KUKS'):
            dm[0] *= 1.1
            dm[1] *= .9
        grad = mf.nuc_grad_method()
        if becke:
            mf.grids.build()
            fixed = mf.grids
            def build(c):
                m = make_mf(c, name, becke=True)
                m.grids.coords, m.grids.weights = fixed.coords, fixed.weights
                gate_charge(m, zgate=17., barrier=True, barrier_start=9.6,
                            barrier_end=12.4, barrier_width=1.)
                return slab_dipole_correction(m, cut=20., z0=9., width=2.)
        grad.grid_response = False
        numeric_gate = finite_difference(cell, lambda c: gate_energy(build(c), dm))
        np.testing.assert_allclose(gate_grad(grad, dm), numeric_gate, atol=3e-7, rtol=0)
        numeric_dipole = finite_difference(cell, lambda c: build(c).get_veff(dm=dm).edip)
        np.testing.assert_allclose(dipole_grad(grad, dm), numeric_dipole, atol=3e-7, rtol=0)
        if becke:
            grad.grid_response = True
            with self.assertRaises(NotImplementedError):
                gate_grad(grad, dm)
            with self.assertRaises(NotImplementedError):
                dipole_grad(grad, dm)

    def test_explicit_finite_difference(self):
        for name in ('RKS', 'UKS', 'KRKS', 'KUKS'):
            with self.subTest(name=name):
                self.check_explicit(name)

    def test_multigrid_and_frozen_becke(self):
        self.check_explicit('KUKS', multigrid=True)
        self.check_explicit('RKS', becke=True)

    def test_ions_in_dipole_return_region(self):
        cell = make_cell(charge=0)
        def build(c):
            return slab_dipole_correction(make_mf(c), cut=10.4, z0=21.4, width=2.)
        mf = build(cell)
        dm = mf.get_init_guess()
        numeric = finite_difference(cell, lambda c: build(c).get_veff(dm=dm).edip)
        np.testing.assert_allclose(dipole_grad(mf.Gradients(), dm), numeric, atol=3e-7, rtol=0)

    def check_scf(self, name='RKS', multigrid=False):
        cell = make_cell()
        # CH+ has four valence electrons and avoids the nearly degenerate
        # partially filled p orbitals of the FH2+ explicit-derivative fixture.
        cell.atom = 'C 2.1 2.8 10.; H 2.8 3.3 11.7'
        cell.charge = 1
        cell.build()
        if name.startswith('K'):
            cell.charge = 2  # One hole per primitive cell for two k-points.
        def build(c, z0=None):
            mf = gate_charge(make_mf(c, name, multigrid), zgate=17.,
                             barrier=True, barrier_start=15., barrier_end=19.,
                             barrier_width=.7)
            return slab_dipole_correction(mf, cut=20., width=2., z0=z0)
        mf = build(cell)
        mf.kernel()
        self.assertTrue(mf.converged)
        grad = mf.Gradients()
        analytic = grad.kernel()
        self.assertTrue(np.isfinite(analytic).all())
        self.assertTrue(np.isnan(grad.stress).all())
        dm = mf.make_rdm1()
        for axis in range(3):
            energies = []
            for step in (DISP, -DISP):
                displaced = build(displaced_cell(cell, 1, axis, step))
                energies.append(displaced.kernel(dm0=dm))
                self.assertTrue(displaced.converged)
            self.assertAlmostEqual(analytic[1, axis], (energies[0]-energies[1])/(2*DISP),
                                   delta=1e-5)
        shifted = build(cell, z0=19.7)  # Potential zero inside the return region.
        shifted.kernel(dm0=dm)
        self.assertTrue(shifted.converged)
        self.assertAlmostEqual(shifted.e_tot, mf.e_tot, delta=1e-9)
        np.testing.assert_allclose(shifted.Gradients().kernel(), analytic, atol=5e-7)
        self.assertAlmostEqual(shifted.dipole_z0, 19.7)

        scanner = mf.Gradients().as_scanner()
        ep, _ = scanner(displaced_cell(cell, 1, 2, DISP))
        em, _ = scanner(displaced_cell(cell, 1, 2, -DISP))
        self.assertEqual(scanner.base.dipole_cut, 20.)
        self.assertAlmostEqual(analytic[1, 2], (ep-em)/(2*DISP), delta=1e-5)

    def test_scf_finite_difference(self):
        self.check_scf('RKS')
        self.check_scf('KUKS')

    def test_multigrid_scf_finite_difference(self):
        self.check_scf('RKS', multigrid=True)


if __name__ == '__main__':
    unittest.main()
