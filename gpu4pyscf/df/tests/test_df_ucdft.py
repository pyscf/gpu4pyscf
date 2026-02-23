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

import cupy as cp
import unittest
import pyscf
from gpu4pyscf.dft import ucdft
from gpu4pyscf.dft.cdft_soscf import newton_cdft as newton_penalty
from gpu4pyscf.dft.cdft_soscf_full import newton_cdft

atom = '''
    O    0    0    0
    H    0.   -0.757   0.587
    H    0.   0.757    0.587
'''
bas='def2-tzvp'
charge_constraints_2 = [ [0, 1], [8.1, 0.95] ]
charge_constraints_2_ao = [ 
    [['0 O 1s', '0 O 2s', '0 O 2px', '0 O 2py', '0 O 2pz'], 1], 
    [8.1, 0.95] ]

def run_dft(mol, xc, method, charge_constraints, 
        soscf=False, penalty=None, projection_method='becke', with_df=True):
    if method == 'lagrange':
        assert penalty is None
    mf = ucdft.CDFT_UKS(mol, 
                       charge_constraints=charge_constraints, 
                       method=method, 
                       penalty_weight=penalty,
                       projection_method=projection_method
                        )
    if with_df:
        mf = mf.density_fit()
    mf.xc = xc
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1.0E-8
    mf.max_cycle = 100
    if method == 'penalty':
        mf = newton_penalty(mf)
        mf.kernel()
        v_lagrange = mf._scf.v_lagrange
    elif soscf:
        mf = newton_cdft(mf)
        mf.kernel()
        v_lagrange = mf._scf.v_lagrange
    else:
        mf.kernel()
        v_lagrange = mf.v_lagrange

    e_tot = mf.e_tot
    homo = mf.mo_energy[0][4]
    lumo = mf.mo_energy[0][5]

    dm = mf.make_rdm1()
    projs = mf.build_projectors()
    O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])
    H1_charge = cp.trace(dm[0] @ projs[1]) + cp.trace(dm[1] @ projs[1])

    return {
        'e_tot': float(e_tot),
        'homo': float(homo),
        'lumo': float(lumo),
        'O_charge': float(O_charge),
        'H1_charge': float(H1_charge),
        'v_lagrange': v_lagrange,
        'mf': mf
    }

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = pyscf.M(
            atom=atom,
            basis=bas,
            max_memory=32000,
            verbose = 1,
            spin = 0,
            charge = 0,
            output = '/dev/null'
        )
        cls.mol = mol
        cls.output_atom = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2, projection_method='minao')
        cls.output_ao = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2_ao, projection_method='minao')
        cls.output_atom_benchmark = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2, projection_method='minao', with_df=False)
        cls.output_ao_benchmark = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2_ao, projection_method='minao', with_df=False)

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_vs_direct(self):
        self.assertAlmostEqual(self.output_ao['e_tot'], self.output_ao_benchmark['e_tot'], 4)
        self.assertAlmostEqual(self.output_atom['e_tot'], self.output_atom_benchmark['e_tot'], 4)
        self.assertAlmostEqual(self.output_ao['homo'], self.output_ao_benchmark['homo'], 4)
        self.assertAlmostEqual(self.output_atom['homo'], self.output_atom_benchmark['homo'], 4)
        self.assertAlmostEqual(self.output_ao['lumo'], self.output_ao_benchmark['lumo'], 4)
        self.assertAlmostEqual(self.output_atom['lumo'], self.output_atom_benchmark['lumo'], 4)
        self.assertAlmostEqual(self.output_ao['O_charge'], self.output_ao_benchmark['O_charge'], 4)
        self.assertAlmostEqual(self.output_atom['O_charge'], self.output_atom_benchmark['O_charge'], 4)
        self.assertAlmostEqual(self.output_ao['H1_charge'], self.output_ao_benchmark['H1_charge'], 4)
        self.assertAlmostEqual(self.output_atom['H1_charge'], self.output_atom_benchmark['H1_charge'], 4)
        self.assertAlmostEqual(self.output_ao['v_lagrange'][0], self.output_ao_benchmark['v_lagrange'][0], 4)
        self.assertAlmostEqual(self.output_atom['v_lagrange'][0], self.output_atom_benchmark['v_lagrange'][0], 4)

if __name__ == "__main__":
    print("Full Tests for density fitting cdft")
    unittest.main()