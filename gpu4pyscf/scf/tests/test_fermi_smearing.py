# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from scipy.special import erfcinv
from packaging.version import Version
import pyscf
from pyscf.scf import addons as cpu_addons
from pyscf.scf import hf as cpu_hf
from gpu4pyscf.scf import hf, smearing


def setUpModule():
    global mol
    atom = """
    O       0.0000000000    -0.0000000000     0.1174000000
    H      -0.7570000000    -0.0000000000    -0.4696000000
    H       0.7570000000     0.0000000000    -0.4696000000
    """

    mol = pyscf.M(
        atom=atom,  # water molecule
        basis="6-31g",  # basis set
        verbose=7,
        output="/dev/null",
    )


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):

    def test_gradient(self):
        gpu_mf = hf.RHF(mol).smearing(sigma=0.1).run()
        gpu_gradient = gpu_mf.nuc_grad_method().kernel()
        cpu_mf = cpu_addons.smearing(cpu_hf.RHF(mol), sigma=0.1).run()
        cpu_gradient = cpu_mf.nuc_grad_method().kernel()
        assert np.allclose(gpu_mf.e_tot, cpu_mf.e_tot, atol=1e-9)
        assert np.allclose(gpu_gradient, cpu_gradient, atol=1e-7)

    def test_df_uhf_gradient(self):
        gpu_mf = mol.UHF().to_gpu().density_fit().smearing(sigma=0.1).run()
        gpu_gradient = gpu_mf.nuc_grad_method().kernel()
        cpu_mf = cpu_addons.smearing(mol.UHF().density_fit(), sigma=0.1).run()
        cpu_gradient = cpu_mf.nuc_grad_method().kernel()
        assert np.allclose(gpu_mf.e_tot, cpu_mf.e_tot, atol=1e-9)
        assert np.allclose(gpu_gradient, cpu_gradient, atol=1e-7)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.13'),
                     'Require new interface developed in pyscf-2.13')
    def test_to_gpu(self):
        from pyscf.scf.smearing import _SmearingSCF as _SmearingSCF_cpu
        mf = mol.RHF().smearing(sigma=0.1)
        gpu_mf = mf.to_gpu()
        assert isinstance(gpu_mf, smearing._SmearingSCF)
        assert gpu_mf.sigma == 0.1

        mf = gpu_mf.to_cpu()
        assert isinstance(mf, _SmearingSCF_cpu)
        assert mf.sigma == 0.1


SMALL_ENERGIES = np.array([[-.7, -.1, .05, .25, .8],
                           [-.62, -.06, .12, .3, .88]])


class UHFSmearing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        molecule = pyscf.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
        cls.small_uhf = molecule.UHF(), molecule.UHF().to_gpu()

    @classmethod
    def tearDownClass(cls):
        del cls.small_uhf

    def test_uhf_fermi_occupations_and_entropy(self):
        for fix_spin, mu in [(False, None), (False, .07), (True, None),
                  (True, .07), (True, (-.04, .12))]:
            with self.subTest(fix_spin=fix_spin, mu=mu):
                small_uhf = self.small_uhf
                cpu_base, gpu_base = small_uhf
                cpu = cpu_addons.smearing(cpu_base, .12, 'fermi', mu, fix_spin)
                gpu = smearing.smearing(gpu_base, .12, 'fermi', mu, fix_spin)
                expected = np.asarray(cpu.get_occ(SMALL_ENERGIES))
                actual = gpu.get_occ(cp.asarray(SMALL_ENERGIES)).get()
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
                assert np.all((actual >= 0) & (actual <= 1))
                if mu is None:
                    if fix_spin:
                        np.testing.assert_allclose(actual.sum(axis=1), gpu.nelec, atol=2e-13)
                    else:
                        np.testing.assert_allclose(actual.sum(), gpu.mol.nelectron, atol=2e-13)
                else:
                    # Fixed mu is grand-canonical, not an electron-number constraint.
                    assert abs(actual.sum()-gpu.mol.nelectron) > .1
                f = actual.ravel()
                f = f[(f > 0) & (f < 1)]
                entropy = -(f*np.log(f) + (1-f)*np.log1p(-f)).sum()
                np.testing.assert_allclose(gpu.entropy, entropy, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(gpu.entropy, cpu.entropy, atol=2e-13, rtol=2e-13)

    def test_uhf_gaussian_occupations_and_entropy(self):
        cpu_base, gpu_base = self.small_uhf
        cpu = cpu_addons.smearing(cpu_base, .12, 'gauss')
        gpu = smearing.smearing(gpu_base, .12, 'gauss')
        # Optimize a common chemical potential for Gaussian spin occupations.
        actual = gpu.get_occ(cp.asarray(SMALL_ENERGIES)).get()
        expected = np.asarray(cpu.get_occ(SMALL_ENERGIES))
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
        np.testing.assert_allclose(actual.sum(), gpu.mol.nelectron, atol=2e-13)
        f = actual.ravel()
        entropy = np.exp(-erfcinv(2*f)**2).sum()/(2*np.sqrt(np.pi))
        np.testing.assert_allclose(gpu.entropy, entropy, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(gpu.entropy, cpu.entropy, atol=2e-13, rtol=2e-13)

    def test_uhf_reconfigure_and_inplace_smearing(self):
        small_uhf = self.small_uhf
        cpu_base, gpu_base = small_uhf
        gpu = smearing.smearing(gpu_base, .12, 'fermi', .07, True)
        fermi = gpu.get_occ(cp.asarray(SMALL_ENERGIES)).get()
        assert smearing.smearing(gpu, .08, 'gauss', (-.04, .12), True) is gpu
        assert gpu.smearing_method == 'gauss'
        cpu = cpu_addons.smearing(cpu_base, .08, 'gauss', (-.04, .12), True)
        actual = gpu.get_occ(cp.asarray(SMALL_ENERGIES)).get()
        np.testing.assert_allclose(actual, np.asarray(cpu.get_occ(SMALL_ENERGIES)), atol=2e-13)
        assert np.max(abs(actual-fermi)) > .01
        inplace = gpu_base.copy()
        assert smearing.smearing_(inplace, .12, 'fermi') is inplace
        assert isinstance(inplace, smearing._SmearingSCF)
        undone = inplace.undo_smearing()
        assert not isinstance(undone, smearing._SmearingSCF)
        np.testing.assert_array_equal(undone.get_occ(cp.asarray(SMALL_ENERGIES)).get(),
                                      gpu_base.get_occ(cp.asarray(SMALL_ENERGIES)).get())

    def test_uhf_smearing_energy_and_gradient(self):
        small_uhf = self.small_uhf
        cpu_base, gpu_base = small_uhf
        cpu = cpu_addons.smearing(cpu_base, .12, 'fermi', fix_spin=True)
        gpu = smearing.smearing(gpu_base, .12, 'fermi', fix_spin=True)
        occupation = np.asarray(cpu.get_occ(SMALL_ENERGIES))
        gpu_occ = gpu.get_occ(cp.asarray(SMALL_ENERGIES))
        eigenvalues, vectors = np.linalg.eigh(cpu.get_ovlp())
        coefficients = np.array([vectors/np.sqrt(eigenvalues)]*2)
        dm = cpu.make_rdm1(coefficients, occupation)
        hcore = cpu.get_hcore()
        potential = cpu.get_veff(dm=dm)
        expected = cpu.energy_tot(dm, hcore, potential)
        actual = gpu.energy_tot(cp.asarray(dm), cp.asarray(hcore), cp.asarray(potential))
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(gpu.e_free, actual-.12*gpu.entropy, atol=1e-13)
        np.testing.assert_allclose(gpu.e_zero, (actual+gpu.e_free)/2, atol=1e-13)
        grad = gpu.get_grad(cp.asarray(coefficients), gpu_occ)
        np.testing.assert_allclose(grad.get(), cpu.get_grad(coefficients, occupation),
                                   atol=2e-10, rtol=2e-10)
        gpu.sigma = 0
        zero_occ = gpu.get_occ(cp.asarray(SMALL_ENERGIES))
        np.testing.assert_array_equal(zero_occ.get(),
                                      gpu_base.get_occ(cp.asarray(SMALL_ENERGIES)).get())
        fock = cp.asarray(hcore + potential)
        gradient = gpu.get_grad(cp.asarray(coefficients), zero_occ, fock)
        assert gradient.size > 0
        np.testing.assert_allclose(gradient.get(),
            gpu_base.get_grad(cp.asarray(coefficients), zero_occ, fock).get(), atol=1e-13)


if __name__ == "__main__":
    print("Basic Tests for GPU Fermi Smearing")
    unittest.main()
