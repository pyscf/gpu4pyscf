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
import pytest
import numpy as np
import cupy as cp
import pyscf, gpu4pyscf
from gpu4pyscf.pbc.dft.krks import KRKS
from gpu4pyscf.pbc.dft.kuks import KUKS
from gpu4pyscf.pbc.scf.khf import KRHF
from gpu4pyscf.pbc.scf.kuhf import KUHF
from gpu4pyscf.scf.gdm import gdm
from gpu4pyscf.pbc.dft import multigrid_v3
from gpu4pyscf.pbc.dft.gen_grid import BeckeGrids

class KnownValues(unittest.TestCase):
    def test_pbc_gdm_krhf_gamma(self):
        cell = pyscf.M(
            a = """
                2.1 0 0
                0 100 0
                0 0 100
            """,
            atom = """
                H 0 0 0
                F 1 0 0.3
            """,
            basis = "6-31g",
            verbose = 0,
        )
        kpts = cell.make_kpts([1, 1, 1])

        mf = KRHF(cell, kpts = kpts).density_fit()
        mf.conv_tol = 1e-10
        ref_energy = mf.kernel()
        assert mf.converged
        ref_n_cycle = mf.cycles
        ref_gradient = mf.Gradients().kernel()
        del mf

        mf = KRHF(cell, kpts = kpts).density_fit()
        mf.conv_tol = 1e-10
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert test_n_cycle <= ref_n_cycle + 5

    @pytest.mark.slow
    def test_pbc_gdm_krhf(self):
        cell = pyscf.M(
            a = '''
                0.000000000, 3.370137329, 3.370137329
                3.370137329, 0.000000000, 3.370137329
                3.370137329, 3.370137329, 0.000000000
            ''',
            atom = """
                C 0 0 0
                C 1.685068664391 1.685068664391 1.785068664391
            """,
            unit = 'bohr',
            basis = "6-31g",
            mesh = [10086] * 3, # Avoid going into the FFTDF gradient path
            verbose = 0,
        )
        kpts = cell.make_kpts([3, 1, 1])

        # mf = KRHF(cell, kpts = kpts).density_fit()
        # mf.conv_tol = 1e-10
        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_n_cycle = mf.cycles
        # ref_gradient = mf.Gradients().kernel()
        # del mf
        ref_energy = -75.2289652802803
        ref_n_cycle = 7
        ref_gradient = np.array([
            [ 0.0166432606,    -0.0166432604,    -0.1134059674],
            [-0.0166432603,     0.0166432601,     0.1134059671],
        ])

        mf = KRHF(cell, kpts = kpts).density_fit()
        mf.conv_tol = 1e-10
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert test_n_cycle <= ref_n_cycle + 5

    def test_pbc_gdm_kuhf(self):
        cell = pyscf.M(
            a = '''
                -2.2657170000000000   -2.0833919999999999    0.0000000000000000
                0.0000000000000000    2.0833919999999999    2.2657159999999998
                -2.2657159999999998    4.1667850000000000   -2.2657159999999998
            ''',
            atom = """
                Fe 0.5000000000000000    0.5000000000000000    0.5001000000000000
                Fe 0.0000000000000000    0.0000000000000000    0.0000000000000000
                O  0.7500000000000000    0.2500000000000000    0.2500000000000000
                O  0.2500000000000000    0.7500000000000000    0.7500000000000000
            """,
            fractional = True,
            basis = "gth-szv-molopt-sr",
            pseudo = "gth-pbe",
            verbose = 0,
        )
        kpts = cell.make_kpts([3, 1, 1])

        # mf = KUHF(cell, kpts = kpts).density_fit()
        # mf.conv_tol = 1e-10
        # mf.with_df.linear_dep_threshold = 1e-9 # The default in pyscf==2.8.0 is 1e-9, in pyscf==2.14.0 is 1e-10
        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_n_cycle = mf.cycles
        # ref_gradient = mf.Gradients().kernel()
        # del mf
        ref_energy = -276.432107475977
        ref_n_cycle = 29
        # ref_gradient = np.array([
        #     [-0.0000601663,     0.0000899938,    -0.0000062124],
        #     [ 0.0000685774,    -0.0001356575,     0.0001127860],
        #     [-0.0000042301,     0.0000228900,    -0.0000532994],
        #     [-0.0000041829,     0.0000227730,    -0.0000532736],
        # ])

        mf = KUHF(cell, kpts = kpts).density_fit()
        mf.conv_tol = 1e-10
        mf.with_df.linear_dep_threshold = 1e-9 # The default in pyscf==2.8.0 is 1e-9, in pyscf==2.14.0 is 1e-10
        mf = gdm(mf)
        mf.check_occupation_per_n_iteration = 1
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        # test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-10
        # assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6 # Not checking gradient because it's too slow
        assert test_n_cycle <= ref_n_cycle + 10

    def test_pbc_gdm_krks_uniform_grid(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.5668,
            atom = '''
                C     0.      0.      0.
                C     0.8917  0.8917  0.9017
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = 'gth-szv',
            pseudo = 'gth-pbe',
            verbose = 0,
        )
        kpts = cell.make_kpts([1, 1, 3])

        # mf = KRKS(cell, xc = "PBE", kpts = kpts)
        # mf.conv_tol = 1e-10
        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_n_cycle = mf.cycles
        # ref_gradient = mf.Gradients().kernel()
        # del mf
        ref_energy = -45.0600590144487
        ref_n_cycle = 7
        ref_gradient = np.array([
            [-0.0012387104,    -0.0012387104,    -0.0020099571],
            [ 0.0000039304,     0.0000039304,     0.0095626070],
            [ 0.0012385344,     0.0012385344,    -0.0020098241],
            [-0.0000042025,    -0.0000042025,     0.0011324616],
            [-0.0012880526,     0.0012879820,    -0.0020506418],
            [-0.0000042043,     0.0000042033,    -0.0012879828],
            [ 0.0012879820,    -0.0012880526,    -0.0020506418],
            [ 0.0000042033,    -0.0000042043,    -0.0012879829],
        ])

        mf = KRKS(cell, xc = "PBE", kpts = kpts)
        mf.conv_tol = 1e-10
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert test_n_cycle <= ref_n_cycle + 5

    def test_pbc_gdm_kuks_multigrid(self):
        cell = pyscf.M(
            a = '''
                -2.1283550000000000   -2.1283550000000000    0.0000000000000000
                -2.1283550000000000    0.0000000000000000   -2.1283550000000000
                 0.0000000000000000   -2.1283550000000000   -2.1283550000000000
            ''',
            atom = """
                Mg 0.5000000000000000    0.5000000000000000    0.5001000000000000
                O  0.0000000000000000    0.0000000000000000    0.0000000000000000
            """,
            fractional = True,
            basis = "gth-dzvp-molopt-sr",
            pseudo = "gth-pbe",
            spin = 2,
            verbose = 0,
        )
        kpts = cell.make_kpts([2, 2, 2])
        xc = "LDA"

        # mf = KUKS(cell, xc = xc, kpts = kpts)
        # mf._numint = multigrid_v3.MultiGridNumInt(cell)
        # mf.conv_tol = 1e-10
        # mf.max_cycle = 100
        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_n_cycle = mf.cycles
        # ref_gradient = mf.Gradients().kernel()
        # del mf
        ref_energy = -78.0287048479913
        ref_n_cycle = 12
        ref_gradient = np.array([
            [ 0.0000000001,    -0.0000222442,    -0.0000222442],
            [-0.0000000000,     0.0000230580,     0.0000230580],
        ])

        mf = KUKS(cell, xc = xc, kpts = kpts)
        mf._numint = multigrid_v3.MultiGridNumInt(cell)
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert test_n_cycle <= ref_n_cycle + 5

    def test_pbc_gdm_kuks_beckegrid(self):
        cell = pyscf.M(
            a = '''
                100 0 0
                0 100 0
                0 0 100
            ''',
            atom = """
                C  -0.6695000000   0.0000000000   0.0000000000
                C   0.6695000000   0.0000000000   0.0000000000
                H  -1.2342172746   0.9287967484   0.0000000000
                H  -1.2342172746  -0.9287967484   0.0000000000
                H   1.2342172746   0.3176671970   0.8727834506
                H   1.2342172746  -0.3176671970  -0.8727834506
            """,
            basis = "def2-svp",
            spin = 2,
            verbose = 0,
        )
        kpts = cell.make_kpts([1, 1, 1])
        xc = "r2scan"

        # mf = KUKS(cell, xc = xc, kpts = kpts).density_fit(auxbasis = "def2-universal-jkfit")
        # mf.grids = BeckeGrids(cell)
        # mf.grids.atom_grid = (99, 590)
        # mf.conv_tol = 1e-10
        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_n_cycle = mf.cycles
        # ref_gradient = mf.Gradients().kernel()
        # del mf
        ref_energy = -78.3655656124592
        ref_n_cycle = 9
        ref_gradient = np.array([
            [ 0.0913143427,    -0.0000000000,     0.0000000000],
            [-0.0913192722,    -0.0000000000,     0.0000000000],
            [ 0.0092354581,    -0.0057841156,     0.0077438774],
            [ 0.0092354581,     0.0057841156,    -0.0077438774],
            [-0.0092352643,     0.0052979879,    -0.0080838157],
            [-0.0092352643,    -0.0052979879,     0.0080838157],
        ])

        mf = KUKS(cell, xc = xc, kpts = kpts).density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids = BeckeGrids(cell)
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-10
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert test_n_cycle <= ref_n_cycle + 5

    def test_pbc_gdm_linear_dependent_basis(self):
        cell = pyscf.M(
            a = '''
                0.000000000, 3.370137329, 3.370137329
                3.370137329, 0.000000000, 3.370137329
                3.370137329, 3.370137329, 0.000000000
            ''',
            atom = """
                C 0 0 0
                C 1.685068664391 1.685068664391 1.785068664391
            """,
            unit = 'bohr',
            basis = "gth-tzvp",
            pseudo = "gth-pbe",
            verbose = 0,
        )
        kpts = cell.make_kpts([2, 2, 2])
        xc = "pbe"

        # mf = KRKS(cell, xc = xc, kpts = kpts).density_fit(auxbasis = "def2-universal-jkfit")
        # mf._numint = multigrid_v3.MultiGridNumInt(cell)
        # mf.conv_tol = 1e-10
        # ref_energy = mf.kernel()
        # assert mf.converged
        # ref_n_cycle = mf.cycles
        # ref_gradient = mf.Gradients().kernel()
        # del mf
        ref_energy = -11.2616170936777
        ref_n_cycle = 7
        ref_gradient = np.array([
            [ 0.0000000000,     0.0000000000,    -0.0586528814],
            [-0.0000000001,    -0.0000000001,     0.0586528110],
        ])

        mf = KRKS(cell, xc = xc, kpts = kpts).density_fit(auxbasis = "def2-universal-jkfit")
        mf._numint = multigrid_v3.MultiGridNumInt(cell)
        mf.conv_tol = 1e-10
        mf = gdm(mf)
        test_energy = mf.kernel()
        mf = mf.undo_gdm()
        assert mf.converged
        test_n_cycle = mf.cycles
        test_gradient = mf.Gradients().kernel()

        assert np.abs(test_energy - ref_energy) < 1e-10
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert test_n_cycle <= ref_n_cycle + 5

if __name__ == "__main__":
    print("Full Tests for GDM")
    unittest.main()
