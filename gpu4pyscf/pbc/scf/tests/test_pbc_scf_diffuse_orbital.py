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
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt
import gpu4pyscf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = pbcgto.M(
            a = np.array([
                [2, 0, 0],
                [0, 1.5, 0],
                [0, 0, 5],
            ]),
            atom = """
                H 0 0 0
                H 1.1 0.1 0
            """,
            basis = """
                H  DZV-GTH-q1 DZV-GTH
                2
                1  0  0  4  2
                    8.3744350009  -0.0283380461   0.0000000000
                    1.8058681460  -0.1333810052   0.0000000000
                    0.4852528328  -0.3995676063   0.0000000000
                    0.1658236932  -0.5531027541   1.0000000000
                # 2  1  1  1  1
                #     0.7270000000   1.0000000000
                1 1 1 1 1
                    0.08  1.0
            """, # This is the gth-dzv basis
            verbose = 4,
            precision = 1e-7,
            output = '/dev/null',
        )
        cls.cell = cell

        cls.kpts = cell.make_kpts([3,2,1])

        cls.energy_threshold = 1e-10
        cls.gradient_threshold = 1e-7

        assert gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue is False
        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = True

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()

        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = False

    # Henry 20251121: All of the following tests are consistency tests. Not sure how to get external reference results.
    #                 There's a sanity check that the energy and gradient with diffuse p orbital are between
    #                 those from gth-dzv(s only) and gth-dzv(normal, with correct p) basis.
    #                 For smearing tests, another sanity check is that for sigma = 1e-4, the result is the same as no smearing.

    def test_rks(self):
        cell = self.cell
        mf = cell.KRKS(xc = "PBE").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ### gth-dzv with s orbital only
        # ref_energy = -1.643960408622607
        # ref_gradient = np.array([[-3.75331300e-02,  9.33043462e-03,  1.27319097e-12],
        #        [ 3.75331299e-02, -9.33043444e-03, -2.18457183e-12]])
        ### gth-dzv with correct p orbitals
        # ref_energy = -1.6440107444409877
        # ref_gradient = np.array([[-3.70278610e-02,  9.30086766e-03,  1.19546740e-12],
        #        [ 3.70278608e-02, -9.30086747e-03, -2.27375130e-12]])

        ref_energy = -1.6439608521190807
        ref_gradient = np.array([[-3.75319097e-02,  9.32974780e-03, -4.73203810e-11],
                                 [ 3.75319095e-02, -9.32974768e-03, -4.51863943e-11]])

        assert abs(test_energy - ref_energy) <= self.energy_threshold
        assert np.max(np.abs(test_gradient - ref_gradient)) <= self.gradient_threshold

    def test_krks(self):
        cell = self.cell
        raise

    def test_krks_aftdf(self):
        cell = self.cell
        raise

    def test_uks(self):
        cell = self.cell
        raise

    def test_kuks(self):
        cell = self.cell
        raise

    def test_rks_smearing(self):
        cell = self.cell
        raise

    def test_krks_smearing(self):
        cell = self.cell
        raise

    def test_krks_aftdf_smearing(self):
        cell = self.cell
        raise

    def test_uks_smearing(self):
        cell = self.cell
        raise

    def test_kuks_smearing(self):
        cell = self.cell
        raise

if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
