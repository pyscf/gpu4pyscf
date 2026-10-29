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

import unittest
import numpy as np
import cupy as cp
import scipy.linalg
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
from gpu4pyscf.nac import tdrhf as tdrhf_nac

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""
atom1 = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "cc-pvdz"

def setUpModule():
    global mol, mol1
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)
    mol1 = pyscf.M(
        atom=atom1, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1


class KnownValues(unittest.TestCase):
    def test_nac_scanner_ge(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac1.kernel()

        nac_scanner = nac1.as_scanner()
        new_nac = nac_scanner(mol1)

        assert (new_nac[1]*nac1.de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac1.de) > 0.99

    def test_nac_scanner_ee(self):
        mf = scf.RHF(mol).to_gpu()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac1.kernel()

        nac_scanner = nac1.as_scanner()
        new_nac = nac_scanner(mol1)

        assert (new_nac[1]*nac1.de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac1.de) > 0.99

    def test_sign(self):
        mol = pyscf.M(atom='H 0 1 0; H 0 0 1; H .2 .5, .8; H 1. 0. .2', basis=[[0, [1,1]]])
        s = mol.intor('int1e_ovlp')
        h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        mo1 = scipy.linalg.eigh(h, s)[1]
        mo2 = mo1[:,[1,0,3,2]]
        mo2[:,2] *= -1
        nocc = 2
        nvir = mo1.shape[1] - nocc
        x0 = np.zeros((nocc, nvir))
        x0[0,1] = .5**.5
        x1 = np.zeros((nocc, nvir))
        x1[1,0] = -.5**.5
        assert abs(tdrhf_nac._wfn_overlap(mo1, mo2, x0, x1, s) - -1) < 1e-9


class SmallSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small_mol = pyscf.M(atom=atom, basis='6-31g', verbose=0)
        cls.mf = scf.RHF(cls.small_mol).to_gpu().set(conv_tol=1e-12).run()
        # Keep trial-vector filtering below the requested residual tolerance.
        cls.td = cls.mf.TDA().set(nstates=2, conv_tol=1e-10, lindep=1e-20).run()
        assert cls.mf.converged and all(cls.td.converged)

    @classmethod
    def tearDownClass(cls):
        del cls.td, cls.mf, cls.small_mol

    def test_matching_real_orbitals_after_permutation_and_phase_change(self):
        coeff = self.mf.mo_coeff
        overlap = cp.asarray(self.small_mol.intor('int1e_ovlp'))
        order = np.arange(coeff.shape[1])[::-1]
        phases = np.where(np.arange(coeff.shape[1]) % 2, -1., 1.)
        changed = coeff[:, order] * cp.asarray(phases)
        reordered, indices, signs = tdrhf_nac.match_and_reorder_mos(
            overlap, coeff, changed)
        np.testing.assert_array_equal(indices, np.argsort(order))
        np.testing.assert_array_equal(signs, phases[indices])
        np.testing.assert_allclose(cp.asnumpy(reordered) * signs,
                                   cp.asnumpy(coeff), atol=1e-12, rtol=0)

    def test_ground_cis_orthogonality_and_cis_phase(self):
        coeff = self.mf.mo_coeff
        overlap = self.small_mol.intor('int1e_ovlp')
        amplitudes = self.td.xy[0][0]
        for left, right in ((None, amplitudes), (amplitudes, None)):
            value = tdrhf_nac._wfn_overlap(
                coeff, coeff, left, right, overlap,
                num_to_consider=amplitudes.size)
            self.assertLess(abs(value), 1e-10)
        # With all determinants retained, a normalized state overlaps itself
        # by one; changing the electronic phase reverses the overlap sign.
        for phase in (1., -1.):
            value = tdrhf_nac._wfn_overlap(
                coeff, coeff, amplitudes, phase * amplitudes, overlap,
                num_to_consider=amplitudes.size)
            self.assertAlmostEqual(value, phase, delta=1e-10)

    def test_converged_highest_requested_root(self):
        for states in ((0, 2), (1, 2)):
            with self.subTest(states=states):
                scanner = self.td.nac_method().as_scanner(states=states)
                self.assertIs(scanner.as_scanner(), scanner)
                self.assertTrue(scanner.converged)

    def test_etf_translation(self):
        for states in ((0, 1), (1, 2)):
            with self.subTest(states=states):
                method = self.td.nac_method().set(cphf_conv_tol=1e-10)
                values = method.kernel(states=states)
                # Electron-translation factors remove rigid-translation coupling.
                np.testing.assert_allclose(values[2].sum(axis=0), 0, atol=1e-8, rtol=0)
                np.testing.assert_allclose(values[3].sum(axis=0), 0, atol=1e-8, rtol=0)

    def test_tuple_and_list_select_the_same_atom_rows(self):
        selected = []
        for atoms in ([2, 0], (2, 0)):
            method = self.td.nac_method().set(cphf_conv_tol=1e-10)
            selected.append(method.kernel(states=(0, 1), atmlst=atoms))
        for as_list, as_tuple in zip(*selected):
            self.assertEqual(as_tuple.shape, (2, 3))
            np.testing.assert_allclose(as_tuple, as_list, atol=1e-7, rtol=0)

    def test_geometry_array_and_explicit_scanner_states(self):
        scanner = self.td.nac_method().set(cphf_conv_tol=1e-10).as_scanner(states=(0, 1))
        # The molecule was built in Angstrom; scanner array inputs use that unit.
        geometry = self.small_mol.atom_coords(unit='Angstrom')
        geometry[1, 2] += 0.002
        result = scanner(geometry, states=(1, 2))
        self.assertTrue(scanner.converged)
        self.assertEqual(scanner.states, (1, 2))
        np.testing.assert_allclose(scanner.mol.atom_coords(unit='Angstrom'),
                                   geometry, atol=1e-12, rtol=0)
        cpu_mf = scf.RHF(scanner.mol).set(conv_tol=1e-12).run()
        cpu_td = cpu_mf.TDA().set(nstates=2, conv_tol=1e-10).run()
        self.assertTrue(cpu_mf.converged and all(cpu_td.converged))
        np.testing.assert_allclose(cp.asnumpy(result[0]), cpu_td.e_tot, atol=1e-8, rtol=0)
        direct = scanner.base.nac_method().set(cphf_conv_tol=1e-10)
        reference = direct.kernel(states=(1, 2))
        for actual, expected in zip(result[1:], reference):
            np.testing.assert_allclose(actual, scanner.sign * expected,
                                       atol=1e-7, rtol=0)


if __name__ == "__main__":
    print("Full Tests for TD-RHF nonadiabatic coupling vectors scanner.")
    unittest.main()
