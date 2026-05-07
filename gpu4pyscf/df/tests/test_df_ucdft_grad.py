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


def _run_cdft_energy(coords, charge_groups, spin_groups, charge_targets, spin_targets,
                         v_guess, dm_guess, projection_method):
    mol_tmp = pyscf.M(atom=atom, basis=bas0, verbose=0)
    mol_tmp.set_geom_(coords, unit='Bohr')
    charge_constraints = [charge_groups, charge_targets]
    spin_constraints = [spin_groups, spin_targets]
    
    mf_tmp = ucdft.CDFT_UKS(mol_tmp, 
                            charge_constraints=charge_constraints,
                            spin_constraints=spin_constraints,
                            projection_method=projection_method).density_fit()
    
    mf_tmp.xc = 'b3lyp'
    mf_tmp.grids.atom_grid = (99, 590)
    mf_tmp.conv_tol = 1e-12
    mf_tmp.micro_tol = 1e-6
    mf_tmp.max_cycle = 100
    
    if v_guess is not None:
        mf_tmp.v_lagrange = v_guess.copy()
    mf_tmp.kernel(dm0=dm_guess)
    return mf_tmp.e_tot


class KnownValues(unittest.TestCase):
    def check_single_component(self, mf, atom_id, axis_id, step_size=1e-3):
        g_obj = mf.Gradients()
        grad_ana_full = g_obj.kernel()
        grad_val_ana = grad_ana_full[atom_id, axis_id]
        
        mol_ref = mf.mol
        coords_base = mol_ref.atom_coords() # in Bohr
        
        charge_groups = mf.charge_groups
        spin_groups = mf.spin_groups
        charge_targets = mf.charge_targets
        spin_targets = mf.spin_targets
        
        v_conv = mf.v_lagrange
        dm_conv = mf.make_rdm1()
        proj_method = mf.projection_method

        coords_plus = coords_base.copy()
        coords_plus[atom_id, axis_id] += step_size
        e_plus = _run_cdft_energy(coords_plus, charge_groups, spin_groups, charge_targets, 
            spin_targets, v_conv, dm_conv, proj_method)
        
        coords_minus = coords_base.copy()
        coords_minus[atom_id, axis_id] -= step_size
        e_minus = _run_cdft_energy(coords_minus, charge_groups, spin_groups, charge_targets, 
            spin_targets, v_conv, dm_conv, proj_method)
        
        grad_val_num = (e_plus - e_minus) / (2.0 * step_size)
        
        self.assertAlmostEqual(grad_val_ana, grad_val_num, delta=1e-4)

    def test_grad_cons_orbital(self):
        charge_constraints = [ ['0 O 2py'], [1.2] ]
        mf = ucdft.CDFT_UKS(mol, 
                    charge_constraints=charge_constraints, 
                    projection_method='minao').density_fit()
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.max_cycle = 100
        mf.conv_tol = 1e-12
        mf.kernel()

        g = mf.Gradients()
        grad_ana = g.kernel()

        ref_ana = np.array([
            [ 3.02770017e-15, -2.69274027e-13,  2.13099280e-02],
            [-1.74921450e-15,  2.43064531e-02, -1.06551593e-02],
            [-1.23257607e-15, -2.43064531e-02, -1.06551593e-02],])

        self.assertAlmostEqual(np.linalg.norm(grad_ana - ref_ana), 0, 6)
        self.check_single_component(mf, atom_id=1, axis_id=1)
        self.check_single_component(mf, atom_id=0, axis_id=2)

    def test_grad_cons_atom(self):
        charge_constraints = [ [0, ['1 H 1s']], [8.1, 0.95] ]
        mf = ucdft.CDFT_UKS(mol, 
                    charge_constraints=charge_constraints, 
                    projection_method='minao').density_fit()
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.max_cycle = 100
        mf.conv_tol = 1e-12
        mf.kernel()

        g = mf.Gradients()
        grad_ana = g.kernel()

        ref_ana = np.array([
            [ 3.89206650e-16, -2.02770414e-02,  1.08305981e-01],
            [-2.51469979e-16,  6.72952283e-02, -6.20148892e-02],
            [ 4.39588255e-17, -4.70181852e-02, -4.62914741e-02],])

        self.assertAlmostEqual(np.linalg.norm(grad_ana - ref_ana), 0, 6)
        self.check_single_component(mf, atom_id=0, axis_id=2)
        self.check_single_component(mf, atom_id=1, axis_id=1)
        self.check_single_component(mf, atom_id=2, axis_id=1)

    def test_grad_cons_spin(self):
        spin_constraints = [ [0], [1.5] ]
        
        mf = ucdft.CDFT_UKS(mol, 
                    spin_constraints=spin_constraints, 
                    projection_method='minao').density_fit()
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-12
        mf.micro_tol = 1e-6
        mf.kernel()

        self.check_single_component(mf, atom_id=0, axis_id=0)
        self.check_single_component(mf, atom_id=1, axis_id=1)
        self.check_single_component(mf, atom_id=2, axis_id=2)

    def test_grad_cons_mixed(self):
        charge_constraints = [ [1], [0.8] ]
        spin_constraints = [ [0], [1.0] ]
        
        mf = ucdft.CDFT_UKS(mol, 
                    charge_constraints=charge_constraints,
                    spin_constraints=spin_constraints,
                    projection_method='minao').density_fit()
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-12
        mf.micro_tol = 1e-6
        mf.kernel()

        self.check_single_component(mf, atom_id=0, axis_id=0)
        self.check_single_component(mf, atom_id=1, axis_id=1)
        self.check_single_component(mf, atom_id=2, axis_id=2)

if __name__ == "__main__":
    print("Full Tests for density fitting UCDFT Gradient")
    unittest.main()
