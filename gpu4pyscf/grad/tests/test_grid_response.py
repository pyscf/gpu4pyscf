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
from gpu4pyscf.dft import rks

def setUpModule():
    global mol_sph, mol_cart, atom_grid_loose
    atom = '''
        O  0.0000  0.7375 -0.0528
        O  0.0000 -0.7375 -0.1528
        H  0.8190  0.8170  0.4220
        H -0.8190 -0.8170  1.4220
    '''

    bas0='def2-tzvpp'
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

    atom_grid_loose = (10,14)

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def numerical_gradient(mf):
    mol = mf.mol

    dx = 1e-5
    mol_copy = mol.copy()
    numerical_gradient = np.zeros([mol.natm, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            mf.grids.build()

            energy_p = mf.kernel()
            assert mf.converged

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            mf.grids.build()

            energy_m = mf.kernel()
            assert mf.converged

            numerical_gradient[i_atom, i_xyz] = (energy_p - energy_m) / (2 * dx)
    mf.reset(mol)
    mf.kernel()

    np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
    print(repr(numerical_gradient))
    return numerical_gradient

class KnownValues(unittest.TestCase):
    def test_grids_response_spherical(self):
        mf = rks.RKS(mol_sph, xc = 'PBE')
        mf = mf.density_fit()
        mf.grids.atom_grid = atom_grid_loose
        mf.conv_tol = 1e-12

        # ref_gradient = numerical_gradient(mf)
        ref_gradient = np.array([
            [ 0.0085868180121906, -0.0684450427002048,  0.0099872181635874],
            [ 0.0572144259081142,  0.0719846099173083, -0.0502120343526258],
            [-0.0385973308425491, -0.0158188314003382, -0.0227678512487728],
            [-0.0272039187620976,  0.012279252814551 ,  0.0629926674378112],
        ])

        mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        assert np.abs(np.max(test_gradient - ref_gradient)) < 1e-7

    def test_grids_response_cartesian(self):
        mf = rks.RKS(mol_cart, xc = 'r2SCAN')
        mf = mf.density_fit()
        mf.grids.atom_grid = atom_grid_loose
        mf.conv_tol = 1e-12

        # ref_gradient = numerical_gradient(mf)
        ref_gradient = np.array([
            [-0.0062197884176385, -0.0874558864438768,  0.0173943107029118],
            [ 0.0609035083698473,  0.0804387596531342, -0.0552899336980772],
            [-0.0247581013468334, -0.0047682860326859, -0.0322608016745107],
            [-0.029925608657777 ,  0.0117854000336592,  0.0701564246696762],
        ])

        mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        assert abs(test_gradient - ref_gradient).max() < 3e-6

if __name__ == "__main__":
    print("Full Tests for grid response")
    unittest.main()
