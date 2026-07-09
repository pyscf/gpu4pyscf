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

import pickle
import numpy as np
import unittest
import pyscf
from gpu4pyscf.dft import rks
try:
    from gpu4pyscf.dispersion import dftd3, dftd4
except ImportError:
    dftd3 = dftd4 = None

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-tzvpp'
grids_level = 5
nlcgrids_level = 2

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose=1,
        output='/dev/null'
    )
    mol_cart = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose=1,
        cart=1,
        output = '/dev/null'
    )

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def run_dft(xc, mol, disp=None):
    mf = rks.RKS(mol, xc=xc)
    mf.disp = disp
    mf.grids.level = grids_level
    mf.nlcgrids.level = nlcgrids_level
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    '''
    known values are obtained by Q-Chem
    '''
    def test_rks_lda(self):
        print('------- LDA ----------------')
        mf = mol_sph.RKS(xc='LDA,vwn5').to_gpu()
        mf.grids.level = grids_level
        mf.nlcgrids.level = nlcgrids_level
        e_tot = mf.kernel()
        e_ref = -75.9046410402
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

        # test serialization
        mf1 = pickle.loads(pickle.dumps(mf))
        assert mf1.e_tot == e_tot

    def test_rks_pbe(self):
        print('------- PBE ----------------')
        e_tot = run_dft('PBE', mol_sph)
        e_ref = -76.3800182418
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_b3lyp(self):
        print('-------- B3LYP -------------')
        e_tot = run_dft('B3LYP', mol_sph)
        e_ref = -76.4666495594
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_m06(self):
        print('--------- M06 --------------')
        e_tot = run_dft("M06", mol_sph)
        e_ref = -76.4265870634
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_wb97(self):
        print('-------- wB97 --------------')
        e_tot = run_dft("HYB_GGA_XC_WB97", mol_sph)
        e_ref = -76.4486274326
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_vv10(self):
        print("------- wB97m-v -------------")
        e_tot = run_dft('HYB_MGGA_XC_WB97M_V', mol_sph)
        e_ref = -76.4334218842
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_cart(self):
        print("-------- cart ---------------")
        e_tot = run_dft('b3lyp', mol_cart)
        e_ref = -76.4672144985
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    @unittest.skipIf(dftd3 is None, "dftd3 not available")
    def test_rks_d3bj(self):
        print('-------- B3LYP with d3bj -------------')
        e_tot = run_dft('B3LYP', mol_sph, disp='d3bj')
        e_ref = -76.4672233969
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    @unittest.skipIf(dftd4 is None, "dftd4 not available")
    def test_rks_d4(self):
        print('-------- B3LYP with d4 -------------')
        e_tot = run_dft('B3LYP', mol_sph, disp='d4')
        e_ref = -76.4669590803
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    @unittest.skipIf(dftd3 is None, "dftd3 not available")
    def test_rks_b3lyp_d3bj(self):
        print('-------- B3LYP-d3bj -------------')
        e_tot = run_dft('B3LYP-d3bj', mol_sph)
        e_ref = -76.4672233969
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    @unittest.skipIf(dftd3 is None, "dftd3 not available")
    def test_rks_wb97x_d3bj(self):
        print('-------- wb97x-d3bj -------------')
        e_tot = run_dft('wb97x-d3bj', mol_sph)
        e_ref = -76.47761276450566
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    @unittest.skipIf(dftd3 is None, "dftd3 not available")
    def test_rks_wb97m_d3bj(self):
        print('-------- wb97m-d3bj -------------')
        e_tot = run_dft('wb97m-d3bj', mol_sph)
        e_ref = -76.47675948061112
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    @unittest.skipIf(dftd4 is None, "dftd4 not available")
    def test_rks_b3lyp_d4(self):
        print('-------- B3LYP with d4 -------------')
        e_tot = run_dft('B3LYP-d4', mol_sph)
        e_ref = -76.4669590803
        print('| CPU - GPU |:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5 #-76.4728129216)

    def test_rks_hf(self):
        print('-------- HF -------------')
        mf = mol_sph.RKS(xc='hf')
        e_cpu = mf.kernel()

        mf_gpu = mf.to_gpu()
        e_gpu = mf_gpu.kernel()
        print('| CPU - GPU |:', e_cpu - e_gpu)
        assert np.abs(e_cpu - e_gpu) < 1e-5

    def test_roks(self):
        mol = pyscf.M(
            atom='''
            C 0.00000000 0.00000000 -0.60298508
            O 0.00000000 0.00000000 0.60539399
            H 0.00000000 0.93467313 -1.18217476
            H 0.00000000 -0.93467313 -1.18217476''',
            charge=1, spin=1, unit='B')
        mf = mol.ROKS(xc='b3lyp').to_gpu().run()
        self.assertAlmostEqual(mf.e_tot, -108.14711706818548, 8)
        ref = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_ghost_atom_close_to_real_atom(self):
        mol = pyscf.M(
            atom = """
            O      0.000000     0.000000     0.000000
            H      0.960000     0.000000     0.000000
            H     -0.240000     0.930000     0.000000
            ghost:H     -0.240000    -0.310000     0.880000
            """,
            basis = "def2-TZVPD", # Large and diffuse basis required to reproduce the bug
            verbose = 0,
        )

        mf = rks.RKS(mol, xc = "pbe").density_fit(auxbasis = "def2-universal-jkfit")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-12

        test_energy = mf.kernel()

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        ### Q-Chem reference input
        # $molecule
        # 0 1
        #     O      0.000000     0.000000     0.000000
        #     H      0.960000     0.000000     0.000000
        #     H     -0.240000     0.930000     0.000000
        #     @H     -0.240000    -0.310000     0.880000
        # $end

        # $rem
        # JOBTYPE force
        # METHOD PBE
        # XC_GRID       000099000590
        # BASIS def2-TZVPD
        # SYMMETRY      FALSE
        # SYM_IGNORE    TRUE
        # MAX_SCF_CYCLES 100
        # PURECART 1111
        # SCF_CONVERGENCE 10
        # THRESH        14
        # ri_j        True
        # ri_k        True
        # aux_basis RIJK-def2-TZVP
        # $end
        ref_energy = -76.3811987938
        ref_gradient = np.array([
            [ 0.0084456, -0.0114582,  0.0030382, -0.0000256],
            [ 0.0102732,  0.0003065, -0.0105467, -0.0000329],
            [ 0.0000791, -0.0000347, -0.0000343, -0.0000100],
        ]).T

        ### The check threshold reflects the difference between Q-Chem and GPU4PySCF without ghost atom
        assert np.abs(test_energy - ref_energy) < 1e-7
        assert np.max(np.abs(test_gradient - ref_gradient)) < 3e-6

if __name__ == "__main__":
    print("Full Tests for dft")
    unittest.main()
