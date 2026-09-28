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

    def test_GDFTscreen_index_kernel_bugfix(self):
        # To reproduce the bug, remove the first __syncthreads() in _screen_index() CUDA kernel.
        # See issue 919. Not able to reproduce with smaller molecule or basis.
        mol = pyscf.M(
            atom = """
                O      6.45893747000000     9.93116808000000    10.97643293000000
                O      9.42527815000000    10.14590587000000     7.81102800000000
                O      9.64696581000000     6.46209074000000     9.02490091000000
                O      5.23011095000000     8.51174284000000     7.49046864000000
                O      8.16099437000000     5.11473047000000     6.90504217000000
                O      5.15045626000000     7.79109026000000    12.30054617000000
                O      4.80053694000000    11.57217284000000     9.15394163000000
                O      6.88023836000000     9.31882215000000     5.04551697000000
                O     10.79761454000000     8.66076494000000    12.04761948000000
                O     11.61422235000000     7.83941995000000     7.46961035000000
                O      6.60868723000000     5.42187429000000    12.13381280000000
                O      8.66451903000000     6.93763948000000     4.99853052000000
                O      7.70942575000000     8.54327513000000     9.13505520000000
                H      5.90593607000000    10.38402139000000    10.33494414000000
                H      7.13592798000000     9.49245798000000    10.42905177000000
                H      9.92993327000000    10.91171374000000     8.08719778000000
                H      9.97214743000000     9.55801072000000     7.27491903000000
                H      9.18616373000000     5.86773897000000     8.40648222000000
                H     10.11270983000000     5.85949159000000     9.63740396000000
                H      5.80420954000000     9.02684141000000     6.94537844000000
                H      5.89975626000000     8.30493570000000     8.24209799000000
                H      7.34856673000000     4.62582933000000     7.10112906000000
                H      8.87657972000000     4.39553034000000     6.76285505000000
                H      5.63992197000000     8.48993898000000    11.85671186000000
                H      4.25092775000000     8.22644354000000    12.11848974000000
                H      4.32775576000000    12.11946966000000     9.73575926000000
                H      4.12015326000000    11.16826334000000     8.61309576000000
                H      6.13042719000000     8.91435362000000     4.62334776000000
                H      7.56886179000000     8.63735796000000     4.94625950000000
                H     10.34327646000000     9.45584131000000    12.27098454000000
                H     10.30605360000000     7.86078764000000    12.29638614000000
                H     12.39484674000000     7.40076792000000     7.09927382000000
                H     10.97104151000000     7.16573488000000     7.84728397000000
                H      5.94929010000000     6.13748194000000    12.04467060000000
                H      6.47470171000000     4.81976724000000    11.31844105000000
                H      8.31983396000000     6.41099478000000     5.71125092000000
                H      8.69735857000000     6.38881326000000     4.21445383000000
                H      8.11648447000000     9.25161482000000     8.62899508000000
                H      8.37755472000000     7.84520651000000     9.33359922000000
            """,
            basis="def2-tzvpd",
            unit="Angstrom",
            verbose=0,
        )

        dm = cp.ones((mol.nao, mol.nao))
        sorted_grids = gen_grid_gpu.Grids(mol)
        sorted_grids.atom_grid = (150, 974)
        sorted_grids.build(sort_grids=True)
        unsorted_grids = gen_grid_gpu.Grids(mol)
        unsorted_grids.atom_grid = (150, 974)
        unsorted_grids.build(sort_grids=False)

        n, exc, _ = NumInt().nr_rks(mol, sorted_grids, "revpbe", dm)
        n_s, e_s = float(n), float(exc)
        n, exc, _ = NumInt().nr_rks(mol, unsorted_grids, "revpbe", dm)
        n_u, e_u = float(n), float(exc)

        assert abs(n_s - n_u) < 2e-9
        assert abs(e_s - e_u) < 2e-9

if __name__ == "__main__":
    print("Full Tests for grids")
    unittest.main()
