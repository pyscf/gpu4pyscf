# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf.pbc.dft import gen_grid as gen_grid_cpu
from gpu4pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks as rks_cpu
from gpu4pyscf.pbc.dft import rks
from pyscf.pbc.dft import krks as krks_cpu
from gpu4pyscf.pbc.dft import krks
from gpu4pyscf.pbc.dft.gen_grid import get_becke_weight_derivative

class KnownValues(unittest.TestCase):
    def test_argsort(self):
        cell = pyscf.M(atom='He 0 0 0', a=np.eye(3)*3)
        grids = gen_grid.UniformGrids(cell)
        grids.mesh = [19] * 3
        for tile in [3, 4, 6, 8]:
            idx = grids.argsort(tile=tile)
            self.assertEqual(len(np.unique(idx)), 19**3)

    def test_becke_grid_atom_grid(self):
        cell = pyscf.M(
            atom = """
                H 0 0 0
                F 1 0 0.1
            """,
            a = np.diag([2.5, 3, 4]),
            basis = "6-31g",
            # verbose = 4,
        )

        mf = rks_cpu.RKS(cell, xc = 'pbe0').density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid_cpu.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        ref_energy = mf.kernel()
        assert mf.converged

        ref_grid_coords = mf.grids.coords
        ref_grid_weights = mf.grids.weights

        mf = rks.RKS(cell, xc = 'pbe0').density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = (50,194)
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        test_energy = mf.kernel()
        assert mf.converged

        test_grid_coords = mf.grids.coords.get()
        test_grid_weights = mf.grids.weights.get()

        assert np.abs(test_energy - ref_energy) < 1e-6
        assert np.max(np.abs(test_grid_coords - ref_grid_coords)) < 1e-14
        assert np.max(np.abs(test_grid_weights - ref_grid_weights)) < 1e-12

    def test_becke_grid_level(self):
        cell = pyscf.M(
            atom = """
                H 0 0 0
                F 1 0 0.1
            """,
            a = np.diag([2.5, 3, 3]),
            basis = "6-31g",
            # verbose = 4,
        )

        kpts = cell.make_kpts([3,1,1])
        mf = krks_cpu.KRKS(cell, xc = 'pbe0', kpts = kpts).density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid_cpu.BeckeGrids(cell)
        mf.grids.level = 2
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        ref_energy = mf.kernel()
        assert mf.converged

        ref_grid_coords = mf.grids.coords
        ref_grid_weights = mf.grids.weights

        mf = krks.KRKS(cell, xc = 'pbe0', kpts = kpts).density_fit()
        mf.conv_tol = 1e-9
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.level = 2
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        test_energy = mf.kernel()
        assert mf.converged

        test_grid_coords = mf.grids.coords.get()
        test_grid_weights = mf.grids.weights.get()

        assert np.abs(test_energy - ref_energy) < 1e-6
        assert np.max(np.abs(test_grid_coords - ref_grid_coords)) < 1e-14
        assert np.max(np.abs(test_grid_weights - ref_grid_weights)) < 1e-12

    def test_becke_weight_derivative(self):
        cell = pyscf.M(
            a = np.eye(3) * 3.5668 * 1.01, # The additional factor of 1.01 guarantees no grid point is right at the -0.5 ~ 0.5 box cutoff
            atom = '''
                C     0.      0.      0.
                C     0.8917  0.8917  0.8917
                C     1.7834  1.7834  0.
                C     2.6751  2.6751  0.8917
                C     1.7834  0.      1.7834
                C     2.6751  0.8917  2.6751
                C     0.      1.7834  1.7834
                C     0.8917  2.6751  2.6751
            ''',
            basis = 'sto-6g',
        )
        grids = gen_grid.BeckeGrids(cell)
        grids.atom_grid = (10,14)
        grids.build()

        analytic_gradient = get_becke_weight_derivative(grids, cell.natm)

        dx = 1e-5
        numerical_gradient = cp.empty([cell.natm, 3, grids.coords.shape[0]])
        cell_copy = cell.copy()
        for i_atom in range(cell.natm):
            for i_xyz in range(3):
                xyz_p = cell.atom_coords()
                xyz_p[i_atom, i_xyz] += dx
                cell_copy.set_geom_(xyz_p, unit='Bohr')
                cell_copy.build()
                grids.reset(cell_copy)
                grids.build()
                w_p = grids.weights.copy()

                xyz_m = cell.atom_coords()
                xyz_m[i_atom, i_xyz] -= dx
                cell_copy.set_geom_(xyz_m, unit='Bohr')
                cell_copy.build()
                grids.reset(cell_copy)
                grids.build()
                w_m = grids.weights.copy()

                numerical_gradient[i_atom, i_xyz, :] = (w_p - w_m) / (2 * dx)

        assert cp.max(cp.abs(analytic_gradient - numerical_gradient)) < 2e-9

if __name__ == '__main__':
    print("Full Tests for pbc.dft.numint")
    unittest.main()
