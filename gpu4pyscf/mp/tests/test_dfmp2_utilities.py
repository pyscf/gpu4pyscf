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


"""
Unittest for DF-MP2 utility functions.

This will only test intermediate results, not the final MP2 energy and the full process correctness.

Some additional assumptions:
- no cartesian
- restricted reference
"""

import unittest
import cupy as cp
import numpy as np
import pyscf
import gpu4pyscf

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import pyscf.mp.dfmp2

from gpu4pyscf.mp import dfmp2_addons, dfmp2_drivers


class Intermediates_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global mol, aux, mf, mp, intopt
        token = """
        O    0.    0.    0.
        H    0.94  0.    0.
        H   -0.24  0.    0.91
        """
        mol = pyscf.gto.Mole(atom=token, basis='def2-TZVPP', max_memory=32000, output='/dev/null', cart=False).build()
        aux = pyscf.gto.Mole(atom=token, basis='def2-TZVPP-ri', max_memory=32000, output='/dev/null', cart=False).build()
        mol.output = aux.output = '/dev/null'
        mol.incore_anyway = True
        mf = pyscf.scf.RHF(mol).density_fit().run()
        mp = pyscf.mp.dfmp2.DFMP2(mf)
        mp.with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri')
        mp.run()
        intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux).build()

    @classmethod
    def tearDownClass(cls):
        global mol, aux, mf, mp, intopt
        mol.stdout.close()
        aux.stdout.close()
        del mol, aux, mf, mp, intopt

    def test_get_j2c_decomp_cpu(self):
        j2c = aux.intor('int2c2e')

        # test usual case of cholesky decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'cd', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'tag'})
        self.assertTrue(j2c_decomp['tag'] == 'cd')
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(np.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(np.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition with lower threshold
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-3)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertFalse(np.allclose(j2c, j2c_rebuild))
        self.assertTrue(np.allclose(j2c, j2c_rebuild, atol=1e-4, rtol=1e-3))

    def test_get_j2c_decomp_gpu(self):
        j2c = cp.asarray(aux.intor('int2c2e'))

        # test usual case of cholesky decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'cd', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'tag'})
        self.assertTrue(j2c_decomp['tag'] == 'cd')
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(cp.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-15)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertTrue(cp.allclose(j2c, j2c_rebuild))

        # test eigenvalue decomposition with lower threshold
        j2c_decomp = dfmp2_addons.get_j2c_decomp(mol, j2c, 'eig', thresh_lindep=1e-3)
        self.assertTrue(j2c_decomp.keys() == {'j2c_l', 'j2c_l_inv', 'tag'})
        j2c_l = j2c_decomp['j2c_l']
        j2c_rebuild = j2c_l @ j2c_l.T
        self.assertFalse(cp.allclose(j2c, j2c_rebuild))
        self.assertTrue(cp.allclose(j2c, j2c_rebuild, atol=1e-4, rtol=1e-3))

    def test_j2c(self):
        j2c_cpu = aux.intor('int2c2e')
        j2c_gpu = dfmp2_addons.get_j2c_bdiv(intopt)
        self.assertTrue(cp.allclose(j2c_gpu, j2c_cpu))

    def test_j3c_ovl_bdiv(self):
        nocc = mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        naux_cart = aux.nao_cart()
        occ_coeff = mf.mo_coeff[:, :nocc]
        vir_coeff = mf.mo_coeff[:, nocc:]

        j3c_cpu = pyscf.df.incore.aux_e2(mol, aux)
        j3c_ovl_cpu = pyscf.lib.einsum('uvP, ui, va -> iaP', j3c_cpu, occ_coeff, vir_coeff, optimize=True)

        # j3c_ovl on GPU
        j3c_ovl_cart = cp.empty([nocc, nvir, naux_cart])
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(cp.allclose(j3c_ovl_set[0], j3c_ovl_cpu))

        # j3c_ovl on CPU
        j3c_ovl_cart = np.empty([nocc, nvir, naux_cart])
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(np.allclose(j3c_ovl_set[0], j3c_ovl_cpu))

        # j3c_ovl on GPU, FP32
        j3c_ovl_cart = cp.empty([nocc, nvir, naux_cart], dtype=cp.float32)
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(j3c_ovl_set[0].dtype == cp.float32)
        self.assertTrue(cp.allclose(j3c_ovl_set[0], j3c_ovl_cpu, atol=1e-6))

        # j3c_ovl on CPU, FP32
        j3c_ovl_cart = np.empty([nocc, nvir, naux_cart], dtype=np.float32)
        j3c_ovl_set = dfmp2_addons.get_j3c_ovl_gpu_bdiv(mol, intopt, [occ_coeff], [vir_coeff], [j3c_ovl_cart])
        self.assertTrue(j3c_ovl_set[0].dtype == np.float32)
        self.assertTrue(np.allclose(j3c_ovl_set[0], j3c_ovl_cpu, atol=1e-6))

    def test_dfmp2_kernel_one_gpu(self):
        nocc = mol.nelectron // 2
        occ_coeff = mf.mo_coeff[:, :nocc]
        vir_coeff = mf.mo_coeff[:, nocc:]
        occ_energy = mf.mo_energy[:nocc]
        vir_energy = mf.mo_energy[nocc:]

        result = dfmp2_drivers.dfmp2_kernel_one_gpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='bdiv')
        self.assertAlmostEqual(result['e_corr_os'], mp.e_corr_os, 7)
        self.assertAlmostEqual(result['e_corr_os'], -0.2132034596360335, 7)
        self.assertAlmostEqual(result['e_corr_ss'], mp.e_corr_ss, 7)
        self.assertAlmostEqual(result['e_corr_ss'], -0.06542902835968734, 7)


class DFMP2Utilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)

    @classmethod
    def tearDownClass(cls):
        del cls.mol

    def test_balanced_split(self):
        self.assertEqual(dfmp2_addons.balanced_split(9, 3), [3, 3, 3])
        self.assertEqual(dfmp2_addons.balanced_split(10, 3), [4, 3, 3])
        self.assertEqual(dfmp2_addons.balanced_split(5, 10), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_three_center_whitening(self):
        cases = [(np.float32, False, 'cd'), (np.float32, True, 'eig'),
                 (np.float64, True, 'cd'), (np.float64, False, 'eig')]
        for dtype, host, algorithm in cases:
            with self.subTest(dtype=dtype, host=host, algorithm=algorithm):
                metric = np.array([[3., .2, .1], [.2, 2., -.1], [.1, -.1, 1.5]])
                rng = np.random.default_rng(91)
                original = [rng.normal(size=shape).astype(dtype)
                            for shape in ((2, 4, 3), (1, 2, 3))]
                tensors = [x.copy() if host else cp.asarray(x) for x in original]
                decomposition = dfmp2_addons.get_j2c_decomp(self.mol, metric, algorithm)
                result = dfmp2_addons.decompose_j3c_gpu(self.mol, decomposition, tensors)
                assert result is tensors
                if algorithm == 'cd':
                    transform = np.linalg.inv(np.linalg.cholesky(metric)).T
                else:
                    eigenvalues, vectors = np.linalg.eigh(metric)
                    transform = (vectors / np.sqrt(eigenvalues)) @ vectors.T
                tolerance = 20 * np.finfo(dtype).eps
                for before, after in zip(original, result):
                    assert after.shape == before.shape
                    assert after.dtype == dtype
                    assert isinstance(after, np.ndarray if host else cp.ndarray)
                    actual = after if host else after.get()
                    flat = before.reshape(-1, 3)
                    expected = (flat @ transform).reshape(before.shape)
                    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
                    # Whitened pair products reproduce contraction with the inverse metric.
                    np.testing.assert_allclose(actual.reshape(-1, 3) @ actual.reshape(-1, 3).T,
                                               flat @ np.linalg.solve(metric, flat.T),
                                               atol=tolerance, rtol=tolerance)

    def test_metric_decomposition_with_linear_dependence(self):
        for on_gpu in [False, True]:
            with self.subTest(on_gpu=on_gpu):
                metric = np.diag([4., -1e-5, 1.])
                data = cp.asarray(metric) if on_gpu else metric
                result = dfmp2_addons.get_j2c_decomp(self.mol, data, 'cd', thresh_lindep=1e-8)
                assert result['tag'] == 'eig'
                factor = result['j2c_l']
                inverse = result['j2c_l_inv']
                if on_gpu:
                    factor, inverse = factor.get(), inverse.get()
                np.testing.assert_allclose(factor @ factor.T, np.diag([4., 0., 1.]), atol=1e-14)
                np.testing.assert_allclose(inverse @ factor, np.diag([1., 0., 1.]), atol=1e-14)

    def test_host_device_pair_partition_matches_dense_contraction(self):
        for modes in [(True, True, True), (False, False, False),
                      (True, False, None), (None, None, None)]:
            with self.subTest(modes=modes):
                rng = np.random.default_rng(19)
                device_tensor = rng.normal(size=(3, 2, 4))
                host_tensors = [rng.normal(size=(n, 2, 4)) for n in (3, 2, 1)]
                occupied = np.array([-1.2, -.9, -.6])
                host_energies = [np.linspace(-1.1, -.5, n) for n in (3, 2, 1)]
                virtual = np.array([.3, .8])
                combined = np.concatenate(host_tensors)
                energies = np.concatenate(host_energies)
                integrals = np.einsum('iap,jbp->ijab', device_tensor, combined)
                denominator = (occupied[:, None, None, None] + energies[None, :, None, None]
                               - virtual[None, None, :, None] - virtual[None, None, None, :])
                expected = [(integrals**2 / denominator).sum(axis=(2, 3)),
                            (integrals * integrals.swapaxes(2, 3) / denominator).sum(axis=(2, 3))]
                selected = np.zeros((3, 6), dtype=bool)
                offset = 0
                for block, mode in zip(host_tensors, modes):
                    size = len(block)
                    if mode is not None:
                        same_half = (np.arange(3)[:, None] < 1) == (np.arange(size)[None, :] < size//2)
                        selected[:, offset:offset+size] = same_half if mode else ~same_half
                    offset += size
                actual = dfmp2_addons.get_dfmp2_energy_pair_inter(
                    self.mol, device_tensor, occupied, virtual, host_tensors, host_energies, modes)
                for result, reference in zip(actual, expected):
                    np.testing.assert_allclose(result, np.where(selected, reference, 0.), atol=1e-13, rtol=0)
                np.testing.assert_array_equal(np.concatenate(host_tensors), combined)

    def test_device_t2_amplitudes_and_pair_energies(self):
        for same_spin in [False, True]:
            with self.subTest(same_spin=same_spin):
                tensor = np.random.default_rng(21).normal(size=(3, 2, 4))
                occupied, virtual = np.array([-1.2, -.9, -.6]), np.array([.3, .8])
                integrals = np.einsum('iap,jbp->ijab', tensor, tensor)
                denominator = (occupied[:, None, None, None] + occupied[None, :, None, None]
                               - virtual[None, None, :, None] - virtual[None, None, None, :])
                kernel = integrals - integrals.swapaxes(2, 3) if same_spin else integrals
                expected_t2 = kernel / denominator
                t2 = cp.full((3, 3, 2, 2), cp.nan)
                result = dfmp2_addons.get_dfmp2_energy_pair_intra(
                    self.mol, cp.asarray(tensor), occupied, virtual, t2=t2, ss_only=same_spin)
                np.testing.assert_allclose(t2.get(), expected_t2, atol=1e-13, rtol=0)
                if same_spin:
                    np.testing.assert_allclose(result, (kernel * expected_t2).sum(axis=(2, 3)), atol=1e-13, rtol=0)
                else:
                    np.testing.assert_allclose(result[0], (integrals * expected_t2).sum(axis=(2, 3)), atol=1e-13, rtol=0)
                    np.testing.assert_allclose(result[1], (integrals * expected_t2.swapaxes(2, 3)).sum(axis=(2, 3)), atol=1e-13, rtol=0)
