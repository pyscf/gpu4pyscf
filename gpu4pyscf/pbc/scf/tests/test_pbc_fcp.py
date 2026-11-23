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
from gpu4pyscf.pbc.scf import fcp
from gpu4pyscf.pbc.scf.addons import gate_charge
from gpu4pyscf.pbc.scf.slabdipole import _slab_charge


def make_fcp():
    cell = gto.M(
        a=np.diag([7., 8., 18.]), unit='Bohr',
        atom='H 3 3 7.; H 3.2 3 8.4',
        basis='gth-szv', pseudo='gth-pade',
        mesh=[15, 15, 35], precision=1e-9, verbose=0)
    mf = dft.KRKS(cell, cell.make_kpts([2, 1, 1]), xc='lda,vwn')
    mf = gate_charge(mf, zgate=12., gate_side=1, barrier=False)
    mf.chkfile = None
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-6
    mf.max_cycle = 100
    return fcp.FCP(mf, mu=-.2, sigma=.05)


class KnownValues(unittest.TestCase):
    def test_global_fractional_occupations(self):
        energies = np.array([[-.5, .2], [-.3, .4]])
        eta, occ, entropy = fcp._occupations(energies, 1.7, .1)
        self.assertAlmostEqual(occ.sum()/2, 1.7, delta=1e-10)
        expected = 2/(1 + np.exp((energies-eta)/.1))
        np.testing.assert_allclose(occ, expected, atol=1e-12)
        self.assertGreater(entropy, 0.)

        energies = np.zeros((2, 1))
        eta, occ, entropy = fcp._occupations(energies, 1., .1)
        self.assertAlmostEqual(eta, 0., delta=1e-12)
        np.testing.assert_allclose(occ, np.ones((2, 1)), atol=1e-12)
        self.assertAlmostEqual(entropy, 2*np.log(2), delta=1e-12)

    def test_newton_fit(self):
        numbers = np.array([1.9, 2., 2.1])
        mu = .5*(numbers-2.)
        fit = fcp._fit_potential(numbers, mu, 1e-10)
        new_number, slope = fcp._newton_step(fit, 2., .05, 1., .2)
        self.assertEqual(fit.degree(), 1)
        self.assertAlmostEqual(slope, .5, delta=1e-12)
        self.assertAlmostEqual(new_number, 2.1, delta=1e-12)

        numbers = np.array([1., 1.5, 2., 2.5, 3.])
        fit = fcp._fit_potential(numbers, numbers**2, .01)
        self.assertEqual(fit.degree(), 2)
        self.assertAlmostEqual(fit(2.), 4., delta=1e-12)
        self.assertAlmostEqual(fit.deriv()(2.), 4., delta=1e-12)

        fit = fcp._fit_potential([1., 2., 3.], [1., 2., 3.], 1e-10)
        new_number, slope = fcp._newton_step(fit, 2., 10., 1., .2)
        self.assertAlmostEqual(new_number, 2.2, delta=1e-12)
        self.assertAlmostEqual(slope, 1., delta=1e-12)

    def test_fractional_count_and_gate(self):
        calc = make_fcp()
        calc.mf.fcp_nelec = 2.3
        energies = cp.array([[-.5, .2], [-.3, .4]])
        occ = calc.mf.get_occ(energies)
        self.assertAlmostEqual(float(occ.sum())/2, 2.3, delta=1e-10)
        self.assertAlmostEqual(_slab_charge(calc.mf, calc.mf.cell), -.3, delta=1e-12)
        self.assertEqual(calc.mf.cell.charge, 0)
        self.assertTrue(np.isfinite(calc.mf.fcp_eta))

    def test_environment_energy_and_shift(self):
        calc = make_fcp()
        mf = calc.mf
        dm0 = mf.get_init_guess()
        plain = dft.KRKS(mf.cell, mf.kpts, xc=mf.xc)
        for side in (-1, 1):
            gate_charge(mf, zgate=12., gate_side=side, barrier=False)
            for nelec in (2., 2.1):
                with self.subTest(side=side, nelec=nelec):
                    mf.fcp_nelec = nelec
                    dm = (nelec/2) * dm0
                    vhf = mf.get_veff(dm=dm)
                    energy = mf.energy_tot(dm, mf.get_hcore(), vhf)
                    shift = mf.scf_summary['gate_shift']
                    e0 = plain.energy_tot(dm)
                    self.assertAlmostEqual(energy-e0, mf.scf_summary['egate']+vhf.edip,
                                           delta=1e-10)

                    # Lambda is the explicit charge derivative at fixed density,
                    # also at qc=0, where the sheet energy itself vanishes.
                    step = 1e-4
                    mf.fcp_nelec = nelec+step
                    plus = mf.energy_tot(dm)
                    mf.fcp_nelec = nelec-step
                    minus = mf.energy_tot(dm)
                    self.assertAlmostEqual((plus-minus)/(2*step), shift, delta=1e-8)

    def test_constant_fock_shift_is_not_converged(self):
        calc = make_fcp()
        energies = cp.array([[-.5, .2], [-.3, .4]])
        occ = calc.mf.get_occ(energies)
        coeff = cp.array([np.eye(2), np.eye(2)])
        fock = cp.array([np.diag([0., .7]), np.diag([.2, .9])])
        calc.mf.e_free = calc.mf._fcp_previous_free = 0.
        envs = {'cycle': 1, 'mo_coeff': coeff, 'mo_occ': occ, 'fock': fock,
                'conv_tol': 1e-9, 'conv_tol_grad': 1e-6,
                'norm_gorb': 0., 'norm_ddm': 0.}
        self.assertFalse(calc.mf.check_convergence(envs))

    def test_free_energy_derivative(self):
        calc = make_fcp()
        mu = calc.evaluate(2.1)
        self.assertAlmostEqual(mu, calc.eta+calc.shift, delta=1e-12)
        self.assertAlmostEqual(calc.e_free,
                               calc.e_tot-calc.mf.sigma*calc.mf.entropy, delta=1e-12)
        self.assertAlmostEqual(calc.e_grand, calc.e_free-calc.mu*calc.nelec, delta=1e-12)
        nelec = float(cp.einsum('kij,kji->', calc.dm, calc.mf.get_ovlp()).real)/2
        self.assertAlmostEqual(nelec, 2.1, delta=1e-8)
        step = .001
        calc.evaluate(2.1+step)
        plus = calc.e_free
        calc.evaluate(2.1-step)
        minus = calc.e_free
        self.assertAlmostEqual((plus-minus)/(2*step), mu, delta=2e-5)

    def test_outer_loop(self):
        calc = make_fcp()
        calc.mu = calc.evaluate(2.1)
        calc.max_cycle = 20
        calc.conv_tol = 1e-5
        calc.fit_tol = 1e-6
        calc.sample_step = .005
        calc.nelec_bounds = (2.08, 2.3)
        energy = calc.kernel(nelec=2.08)
        self.assertTrue(calc.converged)
        self.assertLess(abs(calc.mu_eff-calc.mu), calc.conv_tol)
        self.assertAlmostEqual(calc.nelec, 2.1, delta=5e-4)
        self.assertTrue(all(2.08 <= n <= 2.3 for n, _, _ in calc.history))
        self.assertEqual(energy, calc.e_grand)
        nelec = float(cp.einsum('kij,kji->', calc.dm, calc.mf.get_ovlp()).real)/2
        self.assertAlmostEqual(nelec, calc.nelec, delta=1e-8)


if __name__ == '__main__':
    print('Full Tests for FCP...')
    unittest.main()
