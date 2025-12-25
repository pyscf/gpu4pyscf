# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import pyscf
import numpy as np
import unittest
import pytest
import gpu4pyscf
from gpu4pyscf.dft import ucdft


atom = '''O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587'''

bas0 = 'def2tzvp'

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_grad_cons_orbital(self):
        charge_constraints = [ ['0 O 2py'], [1.2] ]
        mf = ucdft.CDFT_UKS(mol, 
                    charge_constraints=charge_constraints, 
                    projection_method='minao')
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.max_cycle = 100
        mf.conv_tol = 1e-12
        mf.kernel()

        g = mf.Gradients()
        grad_ana = g.kernel()

        ref_ana = np.array([
            [ 0.0000000000, -0.0000000000,  0.0213389061],
            [-0.0000000000,  0.0243185556, -0.0106696483],
            [-0.0000000000, -0.0243185556, -0.0106696483]])
        ref_fdiff = np.array([
            [ 0.0000000000,  7.10542736e-12,  2.13414587e-02],
            [-0.0000000000,  2.43216476e-02, -1.06705696e-02],
            [-0.0000000000, -2.43216476e-02, -1.06705697e-02]]) # delta = 0.005 bohr

        self.assertAlmostEqual(np.linalg.norm(grad_ana - ref_ana), 0, 6)
        self.assertAlmostEqual(np.linalg.norm(grad_ana - ref_fdiff), 0, 4)

    def test_grad_cons_atom(self):
        charge_constraints = [ [0, ['1 H 1s']], [8.1, 0.95] ]
        mf = ucdft.CDFT_UKS(mol, 
                    charge_constraints=charge_constraints, 
                    projection_method='minao')
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.max_cycle = 100
        mf.conv_tol = 1e-12
        mf.kernel()

        g = mf.Gradients()
        grad_ana = g.kernel()

        ref_ana = np.array([
            [ 0.0000000000, -0.0202717765,  0.1083333001],
            [-0.0000000000,  0.0673045836, -0.0620265075],
            [-0.0000000000, -0.0470328053, -0.0463071749]])
        ref_fdiff = np.array([
            [ 0.0000000000, -2.02720536e-02,  1.08335203e-01],
            [-0.0000000000,  6.73081168e-02, -6.20269967e-02],
            [-0.0000000000, -4.70360944e-02, -4.63077212e-02]]) # delta = 0.005 bohr

        self.assertAlmostEqual(np.linalg.norm(grad_ana - ref_ana), 0, 6)
        self.assertAlmostEqual(np.linalg.norm(grad_ana - ref_fdiff), 0, 4)

if __name__ == "__main__":
    print("Full Tests for UCDFT Gradient")
    unittest.main()
