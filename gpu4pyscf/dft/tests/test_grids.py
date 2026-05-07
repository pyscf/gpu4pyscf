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
import pyscf

import cupy
from pyscf import lib, scf
from pyscf import dft as cpu_dft
from pyscf.dft import Grids as Grids_cpu
from pyscf.dft.numint import NumInt as pyscf_numint
from pyscf.dft import gen_grid as gen_grid_cpu
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft as gpu_dft
from gpu4pyscf.dft import Grids as Grids_gpu
from gpu4pyscf.dft import gen_grid as gen_grid_gpu

def find_matching_index_between_two_grids(coords1, weights1, rho1, coords2, weights2, rho2):
    # If there's no rho, you can pass in rho1 = rho2 = 1.0

    if isinstance(coords1, cupy.ndarray): coords1 = coords1.get()
    if isinstance(weights1, cupy.ndarray): weights1 = weights1.get()
    if isinstance(rho1, cupy.ndarray): rho1 = rho1.get()
    if isinstance(coords2, cupy.ndarray): coords2 = coords2.get()
    if isinstance(weights2, cupy.ndarray): weights2 = weights2.get()
    if isinstance(rho2, cupy.ndarray): rho2 = rho2.get()

    nonzero1 = np.where(weights1 * rho1 > 1e-10)[0]
    nonzero2 = np.where(weights2 * rho2 > 1e-10)[0]
    assert len(nonzero1) == len(nonzero2), \
        f"The two sets of grids have different number of grids with non-zero rhos ({len(nonzero1)} vs {len(nonzero2)})."

    coords1 = coords1[nonzero1]
    coords2 = coords2[nonzero2]
    coords1 = np.round(coords1, decimals = 10)
    coords2 = np.round(coords2, decimals = 10)
    sort1 = np.lexsort(coords1.T)
    sort2 = np.lexsort(coords2.T)

    coords1 = coords1[sort1]
    coords2 = coords2[sort2]
    coords_diff = np.max(np.abs(coords1 - coords2)) # Already rounded to nearest 1e-10
    assert coords_diff == 0, f"The two sets of grids have different coordinates (max diff = {coords_diff})."

    map1 = nonzero1[sort1]
    map2 = nonzero2[sort2]
    weights_diff = np.max(np.abs(weights1[map1] - weights2[map2]))
    assert weights_diff < 1e-10, f"The two sets of grids have different weights (max diff = {weights_diff})."

    return map1, map2

def setUpModule():
    global mol, grids_cpu, grids_gpu
    mol = pyscf.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'ccpvqz',
        charge = 1,
        spin = 1,  # = 2S = spin_up - spin_down
        output = '/dev/null')
    grids_cpu = Grids_cpu(mol)
    grids_cpu.level = 3
    grids_cpu.alignment = 1
    grids_cpu.build(sort_grids=False)

    grids_gpu = Grids_gpu(mol)
    grids_gpu.level = 3
    grids_gpu.alignment = 1
    grids_gpu.build(sort_grids=False)

def tearDownModule():
    global mol, grids_cpu, grids_gpu
    mol.stdout.close()
    del mol, grids_cpu, grids_gpu

class KnownValues(unittest.TestCase):
    def test_grids(self):
        ngrids = grids_cpu.coords.shape[0]
        coords_cpu = grids_cpu.coords
        coords_gpu = grids_gpu.coords[:ngrids].get()
        weights_cpu = grids_cpu.weights
        weights_gpu = grids_gpu.weights[:ngrids].get()

        assert np.linalg.norm(coords_cpu - coords_gpu) < 1e-10
        assert np.linalg.norm(weights_cpu - weights_gpu) < 1e-10

    def test_sg1(self):
        from pyscf.dft.gen_grid import sg1_prune as cpu_prune
        from gpu4pyscf.dft.gen_grid import sg1_prune as gpu_prune

        gpu_grids = gpu_dft.gen_grid.gen_atomic_grids(mol, prune=gpu_prune)
        cpu_grids = cpu_dft.gen_grid.gen_atomic_grids(mol, prune=cpu_prune)
        for sym in gpu_grids:
            gpu_coords, gpu_weights = gpu_grids[sym]
            cpu_coords, cpu_weights = cpu_grids[sym]
            assert np.linalg.norm(gpu_coords.get() - cpu_coords) < 1e-6
            assert np.linalg.norm(gpu_weights.get() - cpu_weights) < 1e-6

    def test_nwchem(self):
        from pyscf.dft.gen_grid import nwchem_prune as cpu_prune
        from gpu4pyscf.dft.gen_grid import nwchem_prune as gpu_prune

        gpu_grids = gpu_dft.gen_grid.gen_atomic_grids(mol, prune=gpu_prune)
        cpu_grids = cpu_dft.gen_grid.gen_atomic_grids(mol, prune=cpu_prune)
        for sym in gpu_grids:
            gpu_coords, gpu_weights = gpu_grids[sym]
            cpu_coords, cpu_weights = cpu_grids[sym]
            assert np.linalg.norm(gpu_coords.get() - cpu_coords) < 1e-6
            assert np.linalg.norm(gpu_weights.get() - cpu_weights) < 1e-6

    def test_treutler(self):
        from pyscf.dft.gen_grid import treutler_prune as cpu_prune
        from gpu4pyscf.dft.gen_grid import treutler_prune as gpu_prune

        gpu_grids = gpu_dft.gen_grid.gen_atomic_grids(mol, prune=gpu_prune)
        cpu_grids = cpu_dft.gen_grid.gen_atomic_grids(mol, prune=cpu_prune)
        for sym in gpu_grids:
            gpu_coords, gpu_weights = gpu_grids[sym]
            cpu_coords, cpu_weights = cpu_grids[sym]
            assert np.linalg.norm(gpu_coords.get() - cpu_coords) < 1e-6
            assert np.linalg.norm(gpu_weights.get() - cpu_weights) < 1e-6

    def test_stratmann_scheme(self):
        grids_cpu = Grids_cpu(mol)
        grids_cpu.atom_grid = (50,194)
        grids_cpu.becke_scheme = gen_grid_cpu.stratmann
        grids_cpu.build()

        grids_gpu = Grids_gpu(mol)
        grids_gpu.atom_grid = (50,194)
        grids_gpu.becke_scheme = gen_grid_gpu.stratmann
        grids_gpu.build()

        idx1, idx2 = find_matching_index_between_two_grids(grids_cpu.coords, grids_cpu.weights, 1.0,
                                                           grids_gpu.coords, grids_gpu.weights, 1.0,)
        assert np.linalg.norm(grids_gpu.coords[idx2].get() - grids_cpu.coords[idx1]) < 1e-10
        assert np.linalg.norm(grids_gpu.weights[idx2].get() - grids_cpu.weights[idx1]) < 1e-10

        mf = mol.RKS(xc = "r2scan").density_fit(auxbasis = "cc-pvqz-jkfit").to_gpu()
        mf.grids.becke_scheme = gen_grid_gpu.stratmann
        mf.conv_tol = 1e-12
        test_energy = mf.kernel()
        assert mf.converged

        ref_energy = -75.96234634235809 # From pyscf

        assert np.abs(test_energy - ref_energy) < 1e-10

    def test_no_radii_adjustment(self):
        grids_cpu = Grids_cpu(mol)
        grids_cpu.atom_grid = (50,194)
        grids_cpu.radii_adjust = None
        grids_cpu.becke_scheme = gen_grid_cpu.stratmann
        grids_cpu.build()

        grids_gpu = Grids_gpu(mol)
        grids_gpu.atom_grid = (50,194)
        grids_gpu.radii_adjust = None
        grids_gpu.becke_scheme = gen_grid_gpu.stratmann
        grids_gpu.build()

        idx1, idx2 = find_matching_index_between_two_grids(grids_cpu.coords, grids_cpu.weights, 1.0,
                                                           grids_gpu.coords, grids_gpu.weights, 1.0,)
        assert np.linalg.norm(grids_gpu.coords[idx2].get() - grids_cpu.coords[idx1]) < 1e-10
        assert np.linalg.norm(grids_gpu.weights[idx2].get() - grids_cpu.weights[idx1]) < 1e-10

        mf = mol.RKS(xc = "r2scan").density_fit(auxbasis = "cc-pvqz-jkfit").to_gpu()
        mf.grids.radii_adjust = None
        mf.grids.becke_scheme = gen_grid_cpu.stratmann
        mf.conv_tol = 1e-12
        test_energy = mf.kernel()
        assert mf.converged

        ref_energy = -75.96234774753147 # From pyscf

        assert np.abs(test_energy - ref_energy) < 1e-10

    def test_default_grids(self):
        grids = Grids_gpu(mol)
        grids.atom_grid = {'default': (6, 50), 'S': (99, 590)}
        grids.prune = None
        grids.build()
        # grids.size != 900 due to alignment
        assert 900 <= grids.size <= 1024

if __name__ == "__main__":
    print("Full Tests for grids")
    unittest.main()
