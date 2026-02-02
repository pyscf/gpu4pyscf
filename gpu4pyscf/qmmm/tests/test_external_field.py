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
from gpu4pyscf import dft, scf, qmmm
from gpu4pyscf.qmmm import external_field

def setUpModule():
    global mol, mol_charged
    mol = pyscf.M(atom='''
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.958602  0.000000  -0.504284
    ''',
    basis='ccpvdz',
    verbose=1,
    output = '/dev/null')
    mol.build()

    mol_charged = pyscf.M(atom='''
        O      0.199968    0.000000    0.000006
        H      1.174548   -0.000000   -0.000001
        H     -0.287258    0.844506   -0.000003
        H     -0.287258   -0.844506   -0.000003
    ''',
    basis='6-31g',
    verbose=0,
    charge=1,
    output = '/dev/null')
    mol_charged.build()

def tearDownModule():
    global mol, mol_charged
    mol.stdout.close()
    del mol
    mol_charged.stdout.close()
    del mol_charged

def emulate_field_with_charges(field, distance = 10000, origin = np.array([0,0,0])):
    # Notice: coords returned in Bohr, make sure to specify that!
    assert field.shape == (3,) and origin.shape == (3,)
    field_norm = np.linalg.norm(field)
    field_direction = field / field_norm
    coords = np.empty([2,3])
    coords[0] = origin + field_direction * distance
    coords[1] = origin - field_direction * distance
    charges = np.empty(2)
    charges[0] =  field_norm / 2 * distance**2
    charges[1] = -field_norm / 2 * distance**2
    return coords, charges

class KnownValues(unittest.TestCase):
    def test_energy_and_gradient(self):
        np.random.seed(10)

        field = ((np.random.random(3) - 0.5) * 2) * 0.01
        mm_coords, mm_charges = emulate_field_with_charges(field)

        mf = dft.RKS(mol, xc='pbe')
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges, unit = "Bohr")
        ref_energy = mf.kernel()
        assert mf.converged

        gobj = mf.nuc_grad_method()
        ref_gradient = gobj.kernel()

        ref_dipole = mf.dip_moment()

        ###

        mf = dft.RKS(mol, xc = 'pbe')
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = external_field.add_external_field(mf, field)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        test_dipole = mf.dip_moment()

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert np.max(np.abs(test_dipole - ref_dipole)) < 1e-6

    def test_shifted_origin(self):
        np.random.seed(10)

        field = ((np.random.random(3) - 0.5) * 2) * 0.01
        origin = np.array([1,2,0])
        mm_coords, mm_charges = emulate_field_with_charges(field, origin = origin)

        mf = scf.RHF(mol_charged)
        mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges, unit = "Bohr")
        ref_energy = mf.kernel()
        assert mf.converged

        gobj = mf.nuc_grad_method()
        ref_gradient = gobj.kernel()

        ref_dipole = mf.dip_moment()

        ###

        mf = scf.RHF(mol_charged)
        mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf = external_field.add_external_field(mf, field, origin = origin)
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        test_dipole = mf.dip_moment()

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6
        assert np.max(np.abs(test_dipole - ref_dipole)) < 1e-6

    def test_with_ecp(self):
        field = ((np.random.random(3) - 0.5) * 2) * 0.01
        mm_coords, mm_charges = emulate_field_with_charges(field)

        mol_with_ecp = pyscf.M(atom = '''
            K  1.0 0.0 0.0
            H -0.2 0.1 0.0
        ''',
        basis = 'LANL2DZ',
        ecp = 'LANL2DZ',
        charge = 0,
        verbose = 0)

        mf = dft.RKS(mol_with_ecp, xc = "pbe0")
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges, unit = "Bohr")
        mf = mf.density_fit()
        ref_energy = mf.kernel()
        assert mf.converged

        gobj = mf.nuc_grad_method()
        ref_gradient = gobj.kernel()

        ###

        mf = dft.RKS(mol_with_ecp, xc = "pbe0")
        mf.grids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = external_field.add_external_field(mf, field)
        mf = mf.density_fit()
        test_energy = mf.kernel()
        assert mf.converged

        gobj = mf.nuc_grad_method()
        test_gradient = gobj.kernel()

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_undo_external_field(self):
        field = np.array([0.01, 0.02, -0.03])

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf = mf.density_fit()
        mf = external_field.add_external_field(mf, field)
        energy_with_field = mf.kernel()
        assert mf.converged
        assert abs(energy_with_field - -75.95176870367526) < 1e-10

        mf = mf.undo_external_field()
        energy_without_field = mf.kernel()
        assert mf.converged
        assert abs(energy_without_field - -75.95594732431307) < 1e-10

if __name__ == "__main__":
    print("Full Tests for External Fields")
    unittest.main()
