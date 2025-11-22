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
from gpu4pyscf.pbc.df import AFTDF
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

        assert gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue is False
        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = True

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()

        gpu4pyscf.scf.hf.remove_overlap_zero_eigenvalue = False

    # Henry 20251121: All of the following tests are consistency tests. Not sure how to get external reference results.
    #                 For non-smearing tests, a sanity check that the energy and gradient with diffuse p orbital are between
    #                 those from gth-dzv(s only) and gth-dzv(normal, with correct p) basis.
    #                 For smearing tests, the criteria above is not valid, since virtual orbital is also occupied. 
    #                 The only sanity check is that for sigma = 1e-4, the result is the same as no smearing,
    #                 and with a higher temperature, the energy is higher.

    def test_rks(self):
        cell = self.cell
        mf = cell.RKS(xc = "PBE").to_gpu()
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

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_krks(self):
        cell = self.cell
        kpts = self.kpts
        mf = cell.KRKS(kpts = kpts, xc = "r2SCAN").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ### gth-dzv with s orbital only
        # ref_energy = -1.030681108515315
        # ref_gradient = np.array([[ 2.74046900e-02, -2.58274368e-02, -9.72284259e-10],
        #        [-2.74047072e-02,  2.58584329e-02, -9.66630729e-10]])
        ### gth-dzv with correct p orbitals
        # ref_energy = -1.0326055423733533
        # ref_gradient = np.array([[ 2.83562648e-02, -2.29156096e-02, -8.75419169e-10],
        #        [-2.83562828e-02,  2.29441356e-02, -8.70495011e-10]])

        ref_energy = -1.0310588575348987
        ref_gradient = np.array([[ 2.77114603e-02, -2.47140931e-02,  3.94331325e-10],
                                 [-2.77114767e-02,  2.47502293e-02,  4.16056183e-10]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_krks_aftdf(self):
        cell = self.cell
        kpts = cell.make_kpts([3,1,1])
        mf = cell.KRKS(kpts = kpts, xc = "PBE0").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        mf.with_df = AFTDF(cell, kpts = kpts)
        test_energy = mf.kernel()

        # Gradient is not supported

        ### gth-dzv with s orbital only
        # ref_energy = -1.43916426819719
        ### gth-dzv with correct p orbitals
        # ref_energy = -1.4402294429423825

        ref_energy = -1.439218776262645

        assert abs(test_energy - ref_energy) <= 1e-10

    def test_uks(self):
        cell = self.cell
        mf = cell.UKS(xc = "LDA").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ### gth-dzv with s orbital only
        # ref_energy = -1.5347566337664909
        # ref_gradient = np.array([[-4.10216694e-02,  1.00157684e-02, -1.13644262e-12],
        #        [ 4.10216695e-02, -1.00157684e-02, -4.65232380e-12]])
        ### gth-dzv with correct p orbitals
        # ref_energy = -1.5348119419460333
        # ref_gradient = np.array([[-4.04657343e-02,  9.98731778e-03, -1.21802950e-12],
        #        [ 4.04657344e-02, -9.98731778e-03, -4.76048018e-12]])

        ref_energy = -1.5347576584734226
        ref_gradient = np.array([[-4.10163065e-02,  1.00148573e-02, -4.86782781e-11],
                                 [ 4.10163065e-02, -1.00148573e-02, -4.62222600e-11]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_kuks(self):
        cell = self.cell
        kpts = self.kpts
        mf = cell.KRKS(kpts = kpts, xc = "LDA").to_gpu()
        mf.conv_tol = 1e-10
        # mf._numint = MultiGridNumInt(cell)
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ### gth-dzv with s orbital only
        # ref_energy = -0.9082336217304214
        # ref_gradient = np.array([[ 1.84461136e-02, -2.32893781e-02, -1.73157268e-16],
        #        [-1.84461136e-02,  2.32893781e-02, -7.57429697e-17]])
        ### gth-dzv with correct p orbitals
        # ref_energy = -0.909636987193341
        # ref_gradient = np.array([[ 1.91407104e-02, -2.07970047e-02, -1.85706741e-16],
        #        [-1.91407104e-02,  2.07970047e-02, -9.92724335e-17]])

        ref_energy = -0.9086798110324885
        ref_gradient = np.array([[ 1.88442348e-02, -2.23928792e-02,  6.31616636e-14],
                                 [-1.88442348e-02,  2.23928792e-02, -6.31292040e-14]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_rks_smearing(self):
        cell = self.cell
        mf = cell.KRKS(xc = "PBE").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        mf = mf.smearing(5e-2, 'fermi')
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ref_energy = -1.6420959103726285
        ref_gradient = np.array([[-3.75888726e-02,  9.34455109e-03, -4.80129776e-11],
                                 [ 3.75888723e-02, -9.34455096e-03, -4.57646238e-11]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_krks_smearing(self):
        cell = self.cell
        kpts = self.kpts
        mf = cell.KRKS(kpts = kpts, xc = "r2SCAN").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        mf = mf.smearing(5e-2, 'fermi')
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ref_energy = -1.007566518169512
        ref_gradient = np.array([[-2.80927897e-04, -1.55260904e-02, -7.70325423e-11],
                                 [ 2.80905389e-04,  1.55173700e-02, -8.01210477e-11]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_uks_smearing(self):
        cell = self.cell
        mf = cell.UKS(xc = "LDA").to_gpu()
        mf.conv_tol = 1e-10
        mf._numint = MultiGridNumInt(cell)
        mf = mf.smearing(5e-2, 'fermi')
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ref_energy = -1.5323753648477274
        ref_gradient = np.array([[-4.10910618e-02,  1.00340747e-02, -4.86288417e-11],
                                 [ 4.10910618e-02, -1.00340747e-02, -4.60603924e-11]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

    def test_kuks_smearing(self):
        cell = self.cell
        kpts = self.kpts
        mf = cell.KRKS(kpts = kpts, xc = "LDA").to_gpu()
        mf.conv_tol = 1e-10
        # mf._numint = MultiGridNumInt(cell)
        mf = mf.smearing(5e-2, 'fermi')
        test_energy = mf.kernel()

        gobj = mf.Gradients()
        test_gradient = gobj.kernel()

        ref_energy = -0.8899476624776543
        ref_gradient = np.array([[-1.14818193e-02, -1.14416801e-02,  2.82385089e-14],
                                 [ 1.14818192e-02,  1.14416801e-02, -2.82459068e-14]])

        assert abs(test_energy - ref_energy) <= 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) <= 1e-8

if __name__ == '__main__':
    print("Full Tests for PBC with diffused orbitals")
    unittest.main()
