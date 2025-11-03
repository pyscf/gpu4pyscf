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
from pyscf.dft.numint2c import NumInt2C as pyscf_numint2c
from gpu4pyscf.dft import Grids
from gpu4pyscf.dft import numint2c
from gpu4pyscf.dft.numint2c import NumInt2C
from gpu4pyscf import dft
from gpu4pyscf.dft import gen_grid

def setUpModule():
    global mol, grids_cpu, grids_gpu, dm, dm0, dm1, mo_occ, mo_coeff
    mol = pyscf.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'ccpvdz',
        charge = 1,
        spin = 1,  # = 2S = spin_up - spin_down
        output = '/dev/null'
        )

    np.random.seed(2)
    mf = scf.GHF(mol)
    mf.kernel()
    dm1 = mf.make_rdm1().copy()
    dm = dm1
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = (mo_coeff*mo_occ).dot(mo_coeff.T)

    grids_gpu = Grids(mol)
    grids_gpu.level = 1
    grids_gpu.build()

    grids_cpu = grids_gpu.to_cpu()
    grids_cpu.weights = cupy.asnumpy(grids_gpu.weights)
    grids_cpu.coords = cupy.asnumpy(grids_gpu.coords)

def tearDownModule():
    global mol, grids_cpu, grids_gpu
    mol.stdout.close()
    del mol, grids_cpu, grids_gpu

LDA = 'LDA_C_VWN'
GGA_PBE = 'GGA_C_PBE'
MGGA_M06 = 'MGGA_C_M06'

class KnownValues(unittest.TestCase):

    def test_eval_rho(self):
        np.random.seed(1)
        dm = np.random.random(dm0.shape)
        ni_gpu = NumInt2C()
        ni_cpu = pyscf_numint2c()
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho(mol, ao_gpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm, xctype=xctype, hermi=0, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

            rho = ni_gpu.eval_rho(mol, ao_gpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            ref = ni_cpu.eval_rho(mol, ao_cpu, dm0, xctype=xctype, hermi=1, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)

    def test_eval_rho2(self):
        np.random.seed(1)
        mo_coeff_test = np.random.random(mo_coeff.shape)
        ni_gpu = NumInt2C()
        ni_gpu.collinear='m'
        ni_cpu = pyscf_numint2c()
        ni_cpu.collinear='m'
        for xctype in ('LDA', 'GGA', 'MGGA'):
            deriv = 1
            if xctype == 'LDA':
                deriv = 0
            ao_gpu = ni_gpu.eval_ao(mol, grids_gpu.coords, deriv=deriv, transpose=False)
            ao_cpu = ni_cpu.eval_ao(mol, grids_cpu.coords, deriv=deriv)
            
            rho = ni_gpu.eval_rho2(mol, ao_gpu, mo_coeff_test, mo_occ, xctype=xctype, with_lapl=False)
            ref = ni_cpu.eval_rho2(mol, ao_cpu, mo_coeff_test, mo_occ, xctype=xctype, with_lapl=False)
            self.assertAlmostEqual(abs(rho[...,:grids_cpu.size].get() - ref).max(), 0, 10)


if __name__ == "__main__":
    print("Full Tests for dft numint2c")
    unittest.main()
