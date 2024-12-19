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

import pyscf
import numpy as np
import unittest
from gpu4pyscf import scf
from gpu4pyscf.dft import rks, uks
from pyscf.geomopt.geometric_solver import optimize

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

xc='B3LYP'
bas='def2-tzvpp'
disp='d3bj'
grids_level = 8

def setUpModule():
    global mol, mol1
    mol = pyscf.M(atom=atom, basis=bas, output='/dev/null')
    mol.build()
    mol.verbose = 1

    mol1 = pyscf.M(atom=atom, basis=bas, output='/dev/null')
    mol1.charge = 1
    mol1.spin = 1
    mol1.build()
    mol1.verbose = 1

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1

eps = 1e-3

class KnownValues(unittest.TestCase):
    def test_rks_geomopt(self):
        mf = rks.RKS(mol, xc=xc)
        mf.disp = disp
        mf.grids.level = grids_level
        mf.kernel()
        mol_eq = optimize(mf, maxsteps=20)
        coords = mol_eq.atom_coords(unit='Ang')
        # reference from q-chem
        coords_qchem = np.array([
            [ 0.0000000000,     0.0000000000,     0.1164022656],
            [-0.7617088263,    -0.0000000000,    -0.4691011328],
            [0.7617088263,    -0.0000000000,    -0.4691011328]])

        assert np.linalg.norm(coords - coords_qchem) < 1e-4

    def test_rhf_geomopt(self):
        mf = scf.RHF(mol)
        mf.kernel()
        mol_eq = optimize(mf, maxsteps=20)
        coords = mol_eq.atom_coords(unit='Ang')
        # reference from q-chem
        coords_qchem = np.array([
            [0.0000000000,     0.0000000000,     0.1021249784],
            [-0.7519034531,    -0.0000000000,    -0.4619624892],
            [0.7519034531,    -0.0000000000,    -0.4619624892]])

        assert np.linalg.norm(coords - coords_qchem) < 1e-4

    def test_uks_geomopt(self):
        mf = uks.UKS(mol, xc=xc)
        mf.disp = disp
        mf.grids.level = grids_level
        mf.kernel()
        mol_eq = optimize(mf, maxsteps=20)
        coords = mol_eq.atom_coords(unit='Ang')
        # reference from q-chem
        coords_qchem = np.array([
            [ 0.0000000000,     0.0000000000,     0.1164022656],
            [-0.7617088263,    -0.0000000000,    -0.4691011328],
            [0.7617088263,    -0.0000000000,    -0.4691011328]])
        assert np.linalg.norm(coords - coords_qchem) < 1e-4

    def test_uhf_geomopt(self):
        mf = scf.UHF(mol)
        mf.kernel()
        mol_eq = optimize(mf, maxsteps=20)
        coords = mol_eq.atom_coords(unit='Ang')
        # reference from q-chem
        coords_qchem = np.array([
            [0.0000000000,     0.0000000000,     0.1021249784],
            [-0.7519034531,    -0.0000000000,    -0.4619624892],
            [0.7519034531,    -0.0000000000,    -0.4619624892]])

        assert np.linalg.norm(coords - coords_qchem) < 1e-4

if __name__ == "__main__":
    print("Full Tests for geometry optimization")
    unittest.main()
