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
from gpu4pyscf.dft import uks, ucdft
from gpu4pyscf.dft.cdft_soscf import newton_cdft as newton_penalty
from gpu4pyscf.dft.cdft_soscf_full import newton_cdft

atom = '''
    O    0    0    0
    H    0.   -0.757   0.587
    H    0.   0.757    0.587
'''
bas='def2-tzvp'
charge_constraints_2 = [ [0, 1], [8.1, 0.95] ]
charge_constraints_3 = [ [0, 1, 2], [8.1, 0.95, 0.95] ]
charge_constraints_2_ao = [ 
    [['0 O 1s', '0 O 2s', '0 O 2px', '0 O 2py', '0 O 2pz'], 1], 
    [8.1, 0.95] ]

def run_dft(mol, xc, method, charge_constraints, 
        soscf=False, penalty=None, projection_method='becke'):
    if method == 'lagrange':
        assert penalty is None
    mf = ucdft.CDFT_UKS(mol, 
                       charge_constraints=charge_constraints, 
                       method=method, 
                       penalty_weight=penalty,
                       projection_method=projection_method
                        )
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
        'v_lagrange': v_lagrange
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
        cls.output_penalty_cons2 = run_dft(mol, 'b3lyp', 'penalty', 
            charge_constraints_2, penalty=10, projection_method='becke')
        cls.output_lagrange_soscf_cons2 = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2, soscf=True, projection_method='becke')
        cls.output_lagrange_soscf_cons3 = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_3, soscf=True, projection_method='becke')
        cls.output_lagrange_nested_cons2 = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2, projection_method='becke')
        cls.output_lagrange_soscf_cons2_minao = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2, soscf=True, projection_method='minao')
        cls.output_lagrange_soscf_cons2_minao_ao = run_dft(mol, 'b3lyp', 'lagrange', 
            charge_constraints_2_ao, soscf=True, projection_method='minao')

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_energy(self):
        ref = -76.37687528310500 # ref from becke partition, 0.1e-10 penalty weight
        self.assertAlmostEqual(self.output_penalty_cons2['e_tot'], ref, 1)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['e_tot'], ref, 7)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons3['e_tot'], ref, 7)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['e_tot'], ref, 7)

    def test_homo_lumo(self):
        homo_ref = -0.59463327378979
        lumo_ref = -0.08711895811466
        self.assertAlmostEqual(self.output_penalty_cons2['homo'], homo_ref, 1)
        self.assertAlmostEqual(self.output_penalty_cons2['lumo'], lumo_ref, 1)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['homo'], homo_ref, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['lumo'], lumo_ref, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['homo'], homo_ref, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['lumo'], lumo_ref, 6)

    def test_charge(self):
        charge_ref_O = 8.1
        charge_ref_H1 = 0.95

        self.assertAlmostEqual(self.output_penalty_cons2['O_charge'], charge_ref_O, 1)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['O_charge'], charge_ref_O, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons3['O_charge'], charge_ref_O, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['O_charge'], charge_ref_O, 6)
        self.assertAlmostEqual(self.output_penalty_cons2['H1_charge'], charge_ref_H1, 2)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['H1_charge'], charge_ref_H1, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons3['H1_charge'], charge_ref_H1, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['H1_charge'], charge_ref_H1, 6)

    def test_multiplier(self):
        ref_O = -4.50487015E-01
        ref_H = -2.37985800E-08

        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['v_lagrange'][0], ref_O, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2['v_lagrange'][1], ref_H, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['v_lagrange'][0], ref_O, 6)
        self.assertAlmostEqual(self.output_lagrange_nested_cons2['v_lagrange'][1], ref_H, 6)


    def test_shift(self):
        shift_homo = self.output_lagrange_soscf_cons3['homo'] - self.output_lagrange_nested_cons2['homo']
        shift_lumo = self.output_lagrange_soscf_cons3['lumo'] - self.output_lagrange_nested_cons2['lumo']
        O_multiplier_con3 = self.output_lagrange_soscf_cons3['v_lagrange'][0]
        H1_multiplier_con3 = self.output_lagrange_soscf_cons3['v_lagrange'][1]
        H2_multiplier_con3 = self.output_lagrange_soscf_cons3['v_lagrange'][2]
        O_multiplier_cons2 = self.output_lagrange_soscf_cons2['v_lagrange'][0]
        H1_multiplier_cons2 = self.output_lagrange_soscf_cons2['v_lagrange'][1]

        assert abs(shift_homo - H2_multiplier_con3) < 1e-6
        assert abs(shift_lumo - H2_multiplier_con3) < 1e-6
        assert abs(O_multiplier_con3-H2_multiplier_con3-O_multiplier_cons2) < 1e-6
        assert abs(H1_multiplier_con3-H2_multiplier_con3-H1_multiplier_cons2) < 1e-6

    def test_minao_projection(self):
        ref_energy = -76.3639776639758
        ref_homo = -0.123418866057557
        ref_lumo = 0.058258581369145
        ref_O_charge = 8.1
        ref_H1_charge = 0.95
        ref_O_multiplier = 0.31278893
        ref_H_multiplier = -0.05760844
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['e_tot'], ref_energy, 7)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['homo'], ref_homo, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['lumo'], ref_lumo, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['O_charge'], ref_O_charge, 4)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['H1_charge'], ref_H1_charge, 4)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['v_lagrange'][0], ref_O_multiplier, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao['v_lagrange'][1], ref_H_multiplier, 6)

        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['e_tot'], ref_energy, 7)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['homo'], ref_homo, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['lumo'], ref_lumo, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['O_charge'], ref_O_charge, 4)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['H1_charge'], ref_H1_charge, 4)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['v_lagrange'][0], ref_O_multiplier, 6)
        self.assertAlmostEqual(self.output_lagrange_soscf_cons2_minao_ao['v_lagrange'][1], ref_H_multiplier, 6)
        
    def test_minao_projection_1ao(self):
        mf = ucdft.CDFT_UKS(self.mol, 
                       charge_constraints=[ ['0 O 2pz'], [1.2] ], 
                       method='lagrange', 
                       projection_method='minao'
                        )
        mf.xc = 'b3lyp'
        mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1.0E-8
        mf.max_cycle = 100
        mf = newton_cdft(mf)
        mf.kernel()
        v_lagrange = mf._scf.v_lagrange
        e_tot = mf.e_tot
        homo = mf.mo_energy[0][4]
        lumo = mf.mo_energy[0][5]
        dm = mf.make_rdm1()
        projs = mf.build_projectors()
        O_charge = cp.trace(dm[0] @ projs[0]) + cp.trace(dm[1] @ projs[0])

        self.assertAlmostEqual(float(O_charge), 1.2, 6)
        self.assertAlmostEqual(float(e_tot), -76.3840250035048, 7)
        self.assertAlmostEqual(float(homo), -0.236743012466074, 6)
        self.assertAlmostEqual(float(lumo), 0.0494326184909096, 6)
        self.assertAlmostEqual(float(v_lagrange[0]), 0.288396, 6)


if __name__ == "__main__":
    print("Full Tests for cdft")
    unittest.main()

