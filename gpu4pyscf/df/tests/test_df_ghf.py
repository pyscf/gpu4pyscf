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
import cupy
import pyscf
from pyscf import scf as cpu_scf
from pyscf.df import df_jk as cpu_df_jk
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.df import df_jk as gpu_df_jk


atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''


bas='def2tzvpp'


def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, charge=1, spin=1, max_memory=32000,
                      output='/dev/null', verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    '''
    known values are obtained by PySCF
    '''
    def test_ghf(self):
        mf_gpu = gpu_scf.GHF(mol).density_fit(auxbasis='def2-tzvpp-jkfit')
        e_tot = mf_gpu.kernel()
        mf_cpu = cpu_scf.GHF(mol).density_fit(auxbasis='def2-tzvpp-jkfit')
        e_pyscf = mf_cpu.kernel()

        assert np.abs(e_tot - e_pyscf) < 1e-5

    def test_to_cpu(self):
        mf = gpu_scf.GHF(mol).density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert isinstance(mf, cpu_df_jk._DFHF)
        assert np.abs(e_gpu - e_cpu) < 1e-5

    @unittest.skip("skip test_to_gpu")
    def test_to_gpu(self):
        mf = cpu_scf.GHF(mol).density_fit()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert isinstance(mf, gpu_df_jk._DFHF)
        assert np.abs(e_gpu - e_cpu) < 1e-5


class SmallBatchedSpinDensity(unittest.TestCase):
    """The shared GHF adapter must retain each density's four spin blocks."""

    @classmethod
    def setUpClass(cls):
        from scipy.linalg import eigh
        cls.mol = pyscf.M(atom='H 0 0 0; H .2 .1 .8', basis='6-31g', verbose=0)
        cpu = cpu_scf.RHF(cls.mol)
        overlap = cpu.get_ovlp()
        _, coeff = eigh(cpu.get_hcore(), overlap)
        n = cls.mol.nao
        densities = []
        for theta, phi in ((.28, .63), (.8, .37)):
            occupied = np.zeros((2*n, 2))
            occupied[:n, 0] = np.cos(theta)*coeff[:, 0]
            occupied[n:, 0] = np.sin(theta)*coeff[:, 1]
            occupied[:n, 1] = -np.sin(phi)*coeff[:, 2]
            occupied[n:, 1] = np.cos(phi)*coeff[:, 3]
            dm = occupied @ occupied.T
            spin_overlap = np.kron(np.eye(2), overlap)
            np.testing.assert_allclose(dm @ spin_overlap @ dm, dm, atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.trace(dm @ spin_overlap), 2, atol=1e-12, rtol=0)
            assert np.linalg.norm(dm[:n, n:]) > .1
            densities.append(dm)
        cls.densities = np.stack(densities)
        assert np.linalg.norm(cls.densities[0]-cls.densities[1]) > .1

    def _check_batch(self, strategy):
        n = self.mol.nao
        for complex_dm in (False, True):
            with self.subTest(strategy=strategy, complex_dm=complex_dm):
                cpu = cpu_scf.GHF(self.mol)
                gpu = gpu_scf.GHF(self.mol)
                kwargs = dict(auxbasis='weigend', only_dfj=(strategy == 'dfj'))
                cpu = cpu.density_fit(**kwargs)
                gpu = gpu.density_fit(**kwargs)
                dm = self.densities.copy()
                if complex_dm:
                    phase = np.diag(np.r_[np.ones(n), np.full(n, np.exp(.37j))])
                    dm = phase @ dm @ phase.conj().T
                # Evaluate independent CPU references one density at a time;
                # a shared batch-assembly mistake cannot cancel in this oracle.
                pairs = [cpu.get_jk(dm=d, hermi=1) for d in dm]
                reference_j = np.stack([p[0] for p in pairs])
                reference_k = np.stack([p[1] for p in pairs])
                gpu_dm = cupy.asarray(dm)
                j, k = gpu.get_jk(dm=gpu_dm, hermi=1)
                self.assertEqual(j.shape, dm.shape)
                self.assertEqual(k.shape, dm.shape)
                np.testing.assert_allclose(j.get(), reference_j, atol=1e-10, rtol=0)
                np.testing.assert_allclose(k.get(), reference_k, atol=1e-10, rtol=0)
                reversed_j, reversed_k = gpu.get_jk(dm=gpu_dm[::-1].copy(), hermi=1)
                np.testing.assert_allclose(reversed_j.get(), reference_j[::-1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(reversed_k.get(), reference_k[::-1], atol=1e-10, rtol=0)
                j_only, unused = gpu.get_jk(dm=gpu_dm, hermi=1, with_k=False)
                assert unused is None
                unused, k_only = gpu.get_jk(dm=gpu_dm, hermi=1, with_j=False)
                assert unused is None
                np.testing.assert_allclose(j_only.get(), reference_j, atol=1e-10, rtol=0)
                np.testing.assert_allclose(k_only.get(), reference_k, atol=1e-10, rtol=0)
                np.testing.assert_array_equal(gpu_dm.get(), dm)

    def test_df_jk_batch(self):
        self._check_batch('df')

    def test_df_j_exact_k_batch(self):
        self._check_batch('dfj')


if __name__ == "__main__":
    print("Full Tests for Generalized Hartree-Fock")
    unittest.main()
