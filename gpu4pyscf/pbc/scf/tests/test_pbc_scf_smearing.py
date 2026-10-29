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
from pyscf.pbc.scf import addons as cpu_addons
from packaging.version import Version
import pyscf
from pyscf import lib
from gpu4pyscf.pbc.scf import smearing


def setUpModule():
    global cell
    cell = pyscf.M(
        atom = 'He 0 0 1; He 1 0 1',
        basis = [[0, [1., 1.]], [0, [0.5, 1]]],
        a = np.eye(3) * 3,
        verbose = 7,
        output = '/dev/null')

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_krhf_fermi_and_occupations(self):
        crystal = pyscf.M(
            atom='He .2 .3 .4', unit='Bohr', a=np.diag([5., 5.2, 5.4]),
            basis=[[0, [.8, 1.]], [1, [.4, 1.]]], pseudo='gth-pade',
            mesh=[25, 27, 29], precision=1e-10, verbose=0)
        cpu = crystal.KRHF(kpts=crystal.make_kpts([2, 1, 1]))
        mf = cpu.to_gpu()
        mf.run(conv_tol=1e-11)
        cpu.run(conv_tol=1e-11)
        self.assertTrue(mf.converged and cpu.converged)
        np.testing.assert_allclose(mf.e_tot, cpu.e_tot, atol=1e-9, rtol=0)
        reference = cpu.get_occ(mf.mo_energy.get())
        np.testing.assert_array_equal(mf.get_occ().get(), np.asarray(reference))
        with lib.temporary_env(mf, verbose=lib.logger.DEBUG):
            fermi = mf.get_fermi()
            explicit = mf.get_fermi(mf.mo_energy, mf.mo_occ)
        ref = cpu.get_fermi(mf.mo_energy.get(), mf.mo_occ.get())
        np.testing.assert_allclose(fermi, ref, atol=1e-12, rtol=0)
        np.testing.assert_allclose(explicit, ref, atol=1e-12, rtol=0)

    def test_kuhf_fermi_and_occupations(self):
        crystal = pyscf.M(
            atom='He .2 .3 .4', unit='Bohr', a=np.diag([5., 5.2, 5.4]),
            basis=[[0, [.8, 1.]], [1, [.4, 1.]]], pseudo='gth-pade',
            mesh=[25, 27, 29], precision=1e-10, verbose=0)
        cpu = crystal.KUHF(kpts=crystal.make_kpts([2, 1, 1]))
        mf = cpu.to_gpu()
        mf.run(conv_tol=1e-11)
        cpu.run(conv_tol=1e-11)
        self.assertTrue(mf.converged and cpu.converged)
        np.testing.assert_allclose(mf.e_tot, cpu.e_tot, atol=1e-9, rtol=0)
        reference = cpu.get_occ(mf.mo_energy.get())
        np.testing.assert_array_equal(mf.get_occ().get(), np.asarray(reference))
        with lib.temporary_env(mf, verbose=lib.logger.DEBUG):
            fermi = mf.get_fermi()
            explicit = mf.get_fermi(mf.mo_energy, mf.mo_occ)
        ref = cpu.get_fermi(mf.mo_energy.get(), mf.mo_occ.get())
        np.testing.assert_allclose(fermi, ref, atol=1e-12, rtol=0)
        np.testing.assert_allclose(explicit, ref, atol=1e-12, rtol=0)

    def test_krhf_smearing(self):
        nao = cell.nao
        mf = cell.KRHF(kpts=cell.make_kpts([2,1,1])).to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        nkpts = len(mf.kpts)
        mo_energy_kpts = cp.array([cp.arange(nao)*.2+np.cos(i+.5)*.1 for i in range(nkpts)])
        mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1656394960533021/2, 9)

        mf.smearing_method = 'gauss'
        mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 0.94924016074521311/2, 9)

        mf.kernel(dm0=np.array([np.eye(nao)]*nkpts))
        self.assertAlmostEqual(mf.entropy, 0, 15)

    def test_kuhf_smearing(self):
        nao = cell.nao
        mf = cell.KUHF(kpts=cell.make_kpts([2,1,1])).to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        nkpts = len(mf.kpts)
        mo_energy_kpts = cp.array([cp.arange(nao)*.2+np.cos(i+.5)*.1 for i in range(nkpts)])
        mo_energy_kpts = cp.array([mo_energy_kpts, mo_energy_kpts+cp.cos(mo_energy_kpts)*.02])
        mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1803390081500869/2, 9)

        mf.smearing_method = 'gauss'
        mf.mu0 = 0.3
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 0.5066105772152231, 9)

        mf = mf.to_cpu()
        ref = np.array(mf.get_occ(mo_energy_kpts.get()))
        self.assertAlmostEqual(abs(occ.get() - ref).max(), 0, 9)

    def test_rhf_smearing(self):
        nao = cell.nao
        mf = cell.RHF().to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        mo_energy = cp.arange(nao)*.2+cp.cos(.5)*.1
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.0922723199786408, 9)

        mf.smearing_method = 'gauss'
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 0.4152467504725415, 9)

        mf.kernel()
        self.assertAlmostEqual(mf.entropy, 0, 15)

    def test_uhf_smearing(self):
        nao = cell.nao
        mf = cell.UHF().to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        mo_energy = cp.arange(nao)*.2+cp.cos(.5)*.1
        mo_energy = cp.array([mo_energy, mo_energy+cp.cos(mo_energy)*.02])
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.1007387905421022, 9)

        mf.smearing_method = 'gauss'
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 0.42189309944541731, 9)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.13'),
                     'Require new interface developed in pyscf-2.13')
    def test_to_gpu(self):
        from pyscf.pbc.scf import smearing as cpu_smearing
        mf = cell.KRHF().smearing(sigma=0.1)
        gpu_mf = mf.to_gpu()
        assert isinstance(gpu_mf, smearing._SmearingKSCF)
        assert gpu_mf.sigma == 0.1

        mf = gpu_mf.to_cpu()
        assert isinstance(mf, cpu_smearing._SmearingKSCF)
        assert mf.sigma == 0.1

        mf = cell.RHF().smearing(sigma=0.1)
        gpu_mf = mf.to_gpu()
        assert isinstance(gpu_mf, smearing.mol_smearing._SmearingSCF)
        assert gpu_mf.sigma == 0.1


SMALL_K_ENERGIES = np.array([[[-.7, -.1, .05, .25, .8], [-.67, -.08, .08, .29, .82]],
                             [[-.62, -.06, .12, .3, .88], [-.59, -.02, .14, .34, .93]]])


class KSCFSmearing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        crystal = pyscf.M(atom='Li 0 0 0', basis='sto-3g', spin=1, a=np.eye(3)*8,
                          unit='Bohr', mesh=[7]*3, verbose=0)
        kpts = crystal.make_kpts([2, 1, 1])
        cpu = crystal.KUHF(kpts=kpts, exxdiv=None)
        # KUHF.nelec counts both sampled cells. Set the supported public spin
        # populations explicitly: two copies of Li's (2 alpha, 1 beta) electrons.
        cpu.nelec = (4, 2)
        cls.small_kuhf = cpu, cpu.to_gpu()

    @classmethod
    def tearDownClass(cls):
        del cls.small_kuhf

    def test_kuhf_spectral_occupations_and_entropy(self):
        for fix_spin, mu in [(False, None), (False, .07), (True, None),
                             (True, .07), (True, (-.04, .12))]:
            with self.subTest(fix_spin=fix_spin, mu=mu):
                small_kuhf = self.small_kuhf
                cpu_base, gpu_base = small_kuhf
                cpu = cpu_addons.smearing(cpu_base, .12, 'fermi', mu, fix_spin)
                gpu = smearing.smearing(gpu_base, .12, 'fermi', mu, fix_spin)
                actual = gpu.get_occ(cp.asarray(SMALL_K_ENERGIES)).get()
                expected = np.asarray(cpu.get_occ(SMALL_K_ENERGIES))
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
                assert actual.shape == (2, 2, 5)
                assert np.all((actual >= 0) & (actual <= 1))
                if mu is None:
                    if fix_spin:
                        np.testing.assert_allclose(actual.sum(axis=(1, 2)), gpu.nelec, atol=2e-13)
                    else:
                        np.testing.assert_allclose(actual.sum()/2, gpu.cell.nelectron, atol=2e-13)
                else:
                    # Occupations at fixed chemical potential need not conserve N.
                    assert abs(actual.sum()/2-gpu.cell.nelectron) > .1
                f = actual.ravel()
                f = f[(f > 0) & (f < 1)]
                entropy = -(f*np.log(f) + (1-f)*np.log1p(-f)).sum()/2
                np.testing.assert_allclose(gpu.entropy, entropy, atol=3e-13, rtol=3e-13)
                np.testing.assert_allclose(gpu.entropy, cpu.entropy, atol=2e-13, rtol=2e-13)

    def test_krhf_gaussian_occupations_and_entropy(self):
        cpu_base = cell.KRHF(kpts=cell.make_kpts([2, 1, 1]))
        gpu_base = cpu_base.to_gpu()
        energies = SMALL_K_ENERGIES[0]
        cpu = cpu_addons.smearing(cpu_base, .12, 'gauss')
        gpu = smearing.smearing(gpu_base, .12, 'gauss')
        actual = gpu.get_occ(cp.asarray(energies)).get()
        expected = np.asarray(cpu.get_occ(energies))
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
        self.assertEqual(actual.shape, (2, 5))
        self.assertTrue(np.all((actual >= 0) & (actual <= 2)))
        np.testing.assert_allclose(actual.sum()/2, cell.nelectron, atol=2e-13)
        f = actual.ravel()/2
        entropy = np.exp(-erfcinv(2*f)**2).sum()/(2*np.sqrt(np.pi))
        np.testing.assert_allclose(gpu.entropy, entropy, atol=3e-13, rtol=3e-13)
        np.testing.assert_allclose(gpu.entropy, cpu.entropy, atol=2e-13, rtol=2e-13)

    def test_kuhf_smearing_energy_and_gradient_identities(self):
        small_kuhf = self.small_kuhf
        cpu_base, gpu_base = small_kuhf
        cpu = cpu_addons.smearing(cpu_base, .12, 'fermi', fix_spin=True)
        gpu = smearing.smearing(gpu_base, .12, 'fermi', fix_spin=True)
        occupation = np.asarray(cpu.get_occ(SMALL_K_ENERGIES))
        gpu_occ = gpu.get_occ(cp.asarray(SMALL_K_ENERGIES))
        eigenvalues, vectors = np.linalg.eigh(cpu.get_ovlp())
        coefficients = np.array([vectors/np.sqrt(eigenvalues)[:, None, :]]*2)
        dm = cpu.make_rdm1(coefficients, occupation)
        hcore = cpu.get_hcore()
        potential = cpu.get_veff(dm_kpts=dm)
        expected = cpu.energy_tot(dm, hcore, potential)
        actual = gpu.energy_tot(cp.asarray(dm), cp.asarray(hcore), cp.asarray(potential))
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(gpu.e_free, actual-.12*gpu.entropy, atol=1e-13)
        np.testing.assert_allclose(gpu.e_zero, (actual+gpu.e_free)/2, atol=1e-13)
        fock = hcore + potential
        grad = gpu.get_grad(cp.asarray(coefficients), gpu_occ, cp.asarray(fock))
        np.testing.assert_allclose(grad.get(), cpu.get_grad(coefficients, occupation, fock),
                                   atol=2e-11, rtol=2e-11)
        # The omitted-Fock interface must rebuild from these orbitals/occupations,
        # not from stale mean-field state. Compare with an explicitly formed Fock.
        gpu_fock = gpu.get_hcore() + gpu.get_veff(gpu.mol, cp.asarray(dm))
        np.testing.assert_allclose(gpu.get_grad(cp.asarray(coefficients), gpu_occ).get(),
            gpu.get_grad(cp.asarray(coefficients), gpu_occ, gpu_fock).get(),
            atol=2e-11, rtol=2e-11)
        gpu.sigma = 0
        zero_occ = gpu.get_occ(cp.asarray(SMALL_K_ENERGIES))
        np.testing.assert_array_equal(zero_occ.get(),
                                      gpu_base.get_occ(cp.asarray(SMALL_K_ENERGIES)).get())
        np.testing.assert_allclose(gpu.get_grad(cp.asarray(coefficients), zero_occ, cp.asarray(fock)).get(),
            gpu_base.get_grad(cp.asarray(coefficients), zero_occ, cp.asarray(fock)).get(), atol=1e-13)


if __name__ == "__main__":
    print("Basic Tests for GPU PBC-SCF Smearing")
    unittest.main()
