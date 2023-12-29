# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import pyscf

import cupy
from pyscf import lib, scf
from pyscf.dft import Grids as Grids_cpu
from pyscf.dft.numint import NumInt as pyscf_numint
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf import dft
from gpu4pyscf.dft import Grids as Grids_gpu

def setUpModule():
    global mol, grids_cpu, grids_gpu
    mol = pyscf.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'ccpvdz',
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

if __name__ == "__main__":
    print("Full Tests for grids")
    unittest.main()