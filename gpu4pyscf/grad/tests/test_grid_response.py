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

import numpy as np
import pyscf
import cupy
import unittest
import pytest
from pyscf.dft import rks as cpu_rks
from gpu4pyscf.dft import rks as gpu_rks
from packaging import version

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas0='def2-tzvpp'
grids_level = 5
nlcgrids_level = 3
def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

class KnownValues(unittest.TestCase):

    def test_grids_response(self):
        mf = cpu_rks.RKS(mol_sph, xc='b3lyp')
        mf.kernel()

        grids_cpu = mf.grids

        coords_cpu = []
        w0_cpu = []
        w1_cpu = []
        from pyscf.grad.rks import grids_response_cc
        for coords, w0, w1 in grids_response_cc(grids_cpu):
            coords_cpu.append(coords)
            w0_cpu.append(w0)
            w1_cpu.append(w1)

        mf = cpu_rks.RKS(mol_sph, xc='b3lyp').to_gpu()
        mf.kernel()
        grids_gpu = mf.grids

        coords_gpu = []
        w0_gpu = []
        w1_gpu = []
        from gpu4pyscf.grad.rks import grids_response_cc
        for coords, w0, w1 in grids_response_cc(grids_gpu):
            coords_gpu.append(coords)
            w0_gpu.append(w0)
            w1_gpu.append(w1)
        
        for w0, w1 in zip(w0_gpu, w0_cpu):
            assert np.linalg.norm(w0.get() - w1) < 1e-10

        for w0, w1 in zip(w1_gpu, w1_cpu):
            assert np.linalg.norm(w0.get() - w1) < 1e-10

if __name__ == "__main__":
    print("Full Tests for grid response")
    unittest.main()
