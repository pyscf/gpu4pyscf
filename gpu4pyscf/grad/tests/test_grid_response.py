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
