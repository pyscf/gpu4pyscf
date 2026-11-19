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
from gpu4pyscf.pbc.scf import gcscf
from gpu4pyscf.pbc.scf.addons import gate_charge


def make_cell():
    return gto.M(
        a=np.diag([7., 8., 18.]), unit='Bohr',
        atom='H 3 3 7.; H 3.2 3 8.4', basis='gth-szv', pseudo='gth-pade',
        mesh=[21, 23, 55], precision=1e-9, verbose=0)


def constant_mu(mf):
    diis = mf.diis
    mf = gate_charge(mf, zgate=12., gate_side=1, barrier=False)
    mf = gcscf.constant_mu(mf, mu=-.5, sigma=.2)
    assert mf.diis is diis
    mf.diis = False
    mf.damp = .8
    mf.max_cycle = 150
    mf.diis_start_cycle = mf.max_cycle + 1
    mf.chkfile = None
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    return mf


class KnownValues(unittest.TestCase):
    def test_charge_per_cell(self):
        cell = make_cell()
        mf = constant_mu(dft.KUKS(cell, cell.make_kpts([2, 1, 1]), xc='lda,vwn'))
        dm = mf.get_init_guess()
        dm[0] *= .9
        dm[1] *= .7
        potential = mf._gcscf_veff(dm)
        nelec = float(cp.einsum('skij,kji->', dm, mf.get_ovlp()).real) / len(mf.kpts)
        self.assertAlmostEqual(potential.nelec, nelec, delta=1e-12)
        self.assertAlmostEqual(potential.gate_charge,
                               nelec-cell.atom_charges().sum(), delta=1e-12)
        self.assertGreater(abs(nelec-cell.nelectron), .1)
        self.assertEqual(cell.charge, 0)

    def test_energy_derivative(self):
        cell = make_cell()
        plain = dft.RKS(cell, xc='lda,vwn')
        mf = constant_mu(dft.RKS(cell, xc='lda,vwn'))
        dm = .85 * mf.get_init_guess()
        ddm = .1 * dm
        step = 1e-4
        potential = mf._gcscf_veff(dm)
        plus = mf._gcscf_veff(dm + step*ddm)
        minus = mf._gcscf_veff(dm - step*ddm)
        derivative = (plus.egate+plus.edip-minus.egate-minus.edip) / (2*step)
        expected = float(cp.einsum('ij,ji->', potential, ddm).real)
        self.assertAlmostEqual(derivative, expected, delta=1e-8)
        v0 = plain.get_veff(dm=dm)
        v = mf.get_veff(dm=dm)
        cp.testing.assert_allclose(v-v0, potential, atol=1e-10)
        e0 = plain.energy_tot(dm, plain.get_hcore(), v0)
        energy = mf.energy_tot(dm, mf.get_hcore(), v)
        self.assertAlmostEqual(energy-e0, potential.egate+potential.edip, delta=1e-10)

    def test_damped_scf(self):
        mf = constant_mu(dft.RKS(make_cell(), xc='lda,vwn'))
        mf.kernel()
        self.assertTrue(mf.converged)
        dm = mf.make_rdm1()
        nelec = float(cp.einsum('ij,ji->', dm, mf.get_ovlp()).real)
        self.assertAlmostEqual(mf.gcscf_nelec, nelec, delta=1e-8)
        self.assertGreater(abs(nelec-mf.cell.nelectron), .01)
        fock = mf.get_hcore() + mf.get_veff(dm=dm)
        energies, _ = mf.eig(fock, mf.get_ovlp())
        occupations = 2 / (1 + cp.exp((energies-mf.mu0)/mf.sigma))
        cp.testing.assert_allclose(mf.mo_occ, occupations, atol=2e-6, rtol=0)
        self.assertAlmostEqual(mf.scf_summary['gate_charge'], nelec-2, delta=1e-8)
        self.assertAlmostEqual(mf.e_free, mf.e_tot-mf.sigma*mf.entropy, delta=1e-8)
        self.assertAlmostEqual(mf.e_grand, mf.e_free-mf.mu0*nelec, delta=1e-8)


if __name__ == '__main__':
    print("Full Tests for Constant Mu...")
    unittest.main()
