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
from unittest.mock import patch
import numpy as np
import cupy as cp
from gpu4pyscf.sem.gto.params import load_sem_params
from gpu4pyscf.sem.gto.params import SEMParams
from gpu4pyscf.sem.gto.params import build_task_instructions

def compute_single_multipole_angular_factors(self):
    ch = np.zeros((45, 3, 5), dtype=np.float64)
    
    def set_ch(i, l, m, v):
        ch[i, l, m+2] = v
    set_ch(0,0,0, 1.0)
    set_ch(1,1,0, 1.0)
    set_ch(2,1,1, 1.0)
    set_ch(3,1,-1,1.0)
    set_ch(4,2,0, 1.15470054)
    set_ch(5,2,1, 1.0)
    set_ch(6,2,-1,1.0)
    set_ch(7,2,2, 1.0)
    set_ch(8,2,-2,1.0)
    set_ch(9,0,0,1.0)
    set_ch(9,2,0,1.33333333)
    set_ch(10,2,1,1.0)
    set_ch(11,2,-1,1.0)
    set_ch(12,1,0,1.15470054)
    set_ch(13,1,1,1.0)
    set_ch(14,1,-1,1.0)
    set_ch(17,0,0,1.0)
    set_ch(17,2,0,-0.66666667)
    set_ch(17,2,2,1.0)
    set_ch(18,2,-2,1.0)
    set_ch(19,1,1,-0.57735027)
    set_ch(20,1,0,1.0)
    set_ch(22,1,1,1.0)
    set_ch(23,1,-1,1.0)
    set_ch(24,0,0,1.0)
    set_ch(24,2,0,-0.66666667)
    set_ch(24,2,2,-1.0)
    set_ch(25,1,-1,-0.57735027)
    set_ch(27,1,0,1.0)
    set_ch(28,1,-1,-1.0)
    set_ch(29,1,1,1.0)
    set_ch(30,0,0,1.0)
    set_ch(30,2,0,1.33333333)
    set_ch(31,2,1,0.57735027)
    set_ch(32,2,-1,0.57735027)
    set_ch(33,2,2,-1.15470054)
    set_ch(34,2,-2,-1.15470054)
    set_ch(35,0,0,1.0)
    set_ch(35,2,0,0.66666667)
    set_ch(35,2,2,1.0)
    set_ch(36,2,-2,1.0)
    set_ch(37,2,1,1.0)
    set_ch(38,2,-1,1.0)
    set_ch(39,0,0,1.0)
    set_ch(39,2,0,0.66666667)
    set_ch(39,2,2,-1.0)
    set_ch(40,2,-1,-1.0)
    set_ch(41,2,1,1.0)
    set_ch(42,0,0,1.0)
    set_ch(42,2,0,-1.33333333)
    set_ch(44,0,0,1.0)
    set_ch(44,2,0,-1.33333333)
    
    self.multipole_angular_factors = ch


REFERENCE_VALUES = {
    "uss_C":  -51.089653,   # Carbon's energy_core_s (uss6[5])
    "zs_C":   2.047558,   # Carbon's exponent_s (zs6[5])
    
    "tore_C": 4.0,    # Carbon's core charge / valence electrons (should be 4.0)
    "tore_Fe": 8.0,   # Iron (Z=26) core charge (should be 8.0 for PM6)
    
    "natorb_H":  1,   # Hydrogen (Z=1) orbitals (1s)
    "natorb_C":  4,   # Carbon (Z=6) orbitals (s, px, py, pz)
    "natorb_Fe": 9,   # Iron (Z=26) orbitals (s, p*3, d*5)
    
    "eheat_H":  52.102, # Hydrogen's heat of formation ref
    
    # alpha_bond[5, 0] (C-H interaction)
    "alpb_CH": 1.027806, 
}
# =========================================================================

class TestSEMParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = load_sem_params('PM6')

    def test_singleton(self):
        params2 = load_sem_params('PM6')
        self.assertIs(self.params, params2, "load_sem_params should return a singleton instance")

    def test_atomic_parameters_exist(self):
        expected_keys = ['energy_core_s', 'exponent_s', 'beta_s']
        for key in expected_keys:
            try:
                val = self.params.get_parameter(key, to_gpu=False)
                self.assertEqual(val.shape, (107,), f"Parameter {key} has wrong shape")
            except KeyError:
                self.fail(f"Parameter '{key}' failed to load from atomic module")

    def test_values_consistency(self):
        idx_C = 5 
        
        if REFERENCE_VALUES["uss_C"] is not None:
            uss = self.params.get_parameter('energy_core_s', to_gpu=False)
            self.assertAlmostEqual(uss[idx_C], REFERENCE_VALUES["uss_C"], places=6, 
                                   msg="USS for Carbon does not match reference")
            
        if REFERENCE_VALUES["zs_C"] is not None:
            zs = self.params.get_parameter('exponent_s', to_gpu=False)
            self.assertAlmostEqual(zs[idx_C], REFERENCE_VALUES["zs_C"], places=6,
                                   msg="Exponent S for Carbon does not match reference")

    def test_core_charges_reconstruction(self):
        core_charges = self.params.get_core_charges()
        
        idx_C = 5
        self.assertEqual(core_charges[idx_C], REFERENCE_VALUES["tore_C"], 
                         f"Core charge for Carbon should be {REFERENCE_VALUES['tore_C']}")
        
        if REFERENCE_VALUES["tore_Fe"] is not None:
            idx_Fe = 25
            self.assertEqual(core_charges[idx_Fe], REFERENCE_VALUES["tore_Fe"],
                             f"Core charge for Iron should be {REFERENCE_VALUES['tore_Fe']}")

    def test_natorb_logic(self):
        natorb = self.params.get_natorb_table()
        
        self.assertEqual(natorb[0], REFERENCE_VALUES["natorb_H"]) # H index 0
        self.assertEqual(natorb[5], REFERENCE_VALUES["natorb_C"]) # C index 5
        self.assertEqual(natorb[25], REFERENCE_VALUES["natorb_Fe"]) # Fe index 25
        
        if self.params.get_parameter('exponent_s', to_gpu=False)[106] == 0.0:
            self.assertEqual(natorb[106], 0, "Empty element slot should have 0 orbitals")

    def test_binary_matrices_loading(self):
        try:
            alpb = self.params.get_parameter('alpha_bond', to_gpu=False)
            self.assertEqual(alpb.shape, (107, 107))
            
            if REFERENCE_VALUES["alpb_CH"] is not None:
                self.assertAlmostEqual(alpb[5, 0], REFERENCE_VALUES["alpb_CH"], places=6)
        except KeyError:
            self.fail("Binary parameter 'alpha_bond' failed to load")

    def test_gpu_lazy_loading(self):
        key = 'energy_core_s'
        
        if key in self.params._gpu_cache:
            del self.params._gpu_cache[key]
            
        gpu_data = self.params.get_parameter(key, to_gpu=True)
        
        self.assertTrue(isinstance(gpu_data, cp.ndarray), "Should return cupy array")
        
        self.assertIn(key, self.params._gpu_cache, "Data should be cached after loading")
        
        cpu_data = self.params.get_parameter(key, to_gpu=False)
        cp.testing.assert_allclose(gpu_data, cpu_data)

    @patch('gpu4pyscf.sem.gto.params.SEMParams._compute_multipole_angular_factors', new=compute_single_multipole_angular_factors)
    def test_new_parameters_init(self):
        self.assertEqual(self.params.principal_quantum_number_s.shape, (107,))
        self.assertEqual(self.params.principal_quantum_number_s[0], 1)
        self.assertEqual(self.params.principal_quantum_number_s[5], 2)
        
        if REFERENCE_VALUES["eheat_H"] is not None:
            self.assertAlmostEqual(self.params.heat_formation_ref[0], REFERENCE_VALUES["eheat_H"])

        ch = self.params.multipole_angular_factors
        self.assertEqual(ch.shape, (45, 3, 5))
        
        self.assertAlmostEqual(ch[0, 0, 2], 1.0)
        
        self.assertEqual(ch[15, 0, 2], 0.0)
        self.assertEqual(ch[16, 0, 2], 0.0)
        
        self.assertAlmostEqual(ch[17, 0, 2], 1.0)

    def test_eri2c2e_params(self):
        TASK_ACTION, TASK_TARGET, TASK_IJ, TASK_KL, TASK_LI, TASK_LJ, TASK_LK, TASK_LL, IND2, INDEXD = build_task_instructions()
        task_action_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1,
            1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1,
            1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 1, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1,
            1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1,
            1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 2,
            1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
            1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
            2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 2, 2, 1, 3, 1, 1, 3, 3, 1, 2, 1, 2, 3, 2, 2, 3, 3, 2, 2, 2,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 1, 3,
            1, 1, 3, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
            3, 2, 2, 3, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
        task_target_ref = np.array([  0,   4,  10,  11,  11,   1,   5,  12,  13,  13,   2,   7,  15,
            17,  17,   6,  14,   9,  19,   3,   8,  16,  18,  20,   6,  14,
            9,  19,  21,   3,   8,  16,  20,  18,  -1,  -1,  -1,  -1,  -1,
            37,  38,  -1,  41,  -1,  -1,  -1,  -1,  -1,  46,  47,  -1,  50,
            -1,  -1,  -1,  -1,  -1,  55,  56,  -1,  59,  -1,  -1,  -1,  -1,
            -1,  -1,  65,  66,  -1,  -1,  -1,  -1,  -1,  -1,  73,  74,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  87,  61,  62,
            63,  64,  65,  66,  65,  66,  69,  70,  71,  72,  73,  74,  73,
            74,  85,  85,  -1,  84,  85,  86,  77,  78,  79,  82,  83,  80,
            81,  84,  85,  86,  87,  87,  -1,  -1,  -1,  -1, 126,  -1,  -1,
            -1,  -1,  -1, 131, 132,  -1, 135,  -1,  -1,  -1,  -1, 140,  -1,
            -1,  -1,  -1,  -1, 145, 146,  -1, 149,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1, 157, 158, 151, 152, 153, 154, 155, 156, 157, 158,
            157, 158,  -1,  -1,  -1,  -1, 174,  -1,  -1,  -1,  -1,  -1, 179,
            180,  -1, 183,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 191, 192,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 201, 202,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            220,  -1, 218, 218,  -1, 217, 218, 219,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1, 235, 236,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 254, 185, 186, 187, 188,
            189, 190, 191, 192, 191, 192, 195, 196, 197, 198, 199, 200, 201,
            202, 201, 202, 222, 218, 218, 225, 217, 218, 219, 205, 206, 207,
            209, 208, 210, 211, 212, 215, 216, 213, 214, 217, 218, 219, 220,
            220, 229, 230, 231, 232, 233, 234, 235, 236, 235, 236,  -1, 252,
            252,  -1, 251, 252, 253, 239, 240, 241, 243, 242, 244, 245, 246,
            249, 250, 247, 248, 251, 252, 253, 254, 254,  -1, 334,  -1,  -1,
            336, 337,  -1, 336,  -1, 222, 222, 218, 225, 218, 225, 217, 218,
            219,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 352, 353,
            354, 355, 356, 357, 358, 359, 360, 361,  -1, 372,  -1,  -1, 374,
            375,  -1, 374,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,  -1,  -1,
            -1,  -1, 404,  -1,  -1,  -1,  -1,  -1, 409, 410,  -1,  -1, 334,
            336, 336, 337, 340, 336, 342, 222, 218, 218, 225, 217, 218, 219,
            352, 353, 354, 355, 356, 357, 360, 361, 358, 359, 352, 353, 354,
            355, 356, 357, 360, 361, 358, 359, 372, 374, 374, 375, 378, 374,
            380, 381, 382, 383, 384, 385, 386, 389, 390, 387, 388, 381, 382,
            383, 384, 385, 386, 389, 390, 387, 388,  -1, 401, 402, 403, 404,
            404, 406, 407, 408, 409, 410, 409, 410, 414, 413], dtype=np.int32)
        task_ij_ref = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  9,  9,  9,  9,  9,  9,  9,  9,  9,  2,  2,  2,  2,  2,  2,  2,
            2, 10, 10, 10, 10, 10, 10, 10, 10, 17, 17, 17, 17, 17, 17, 17, 17,
            17, 17, 17, 17,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11, 11, 11, 11,
            11, 11, 11, 18, 18, 18, 18, 18, 18, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
            4, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 19, 19,
            19, 19, 19, 19, 19, 19, 19, 19, 25, 25, 25, 25, 25, 25, 25, 25, 25,
            25, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,  5,  5,
            5,  5,  5,  5,  5,  5,  5,  5, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
            20, 26, 26, 26, 26, 26, 26, 26, 31, 31, 31, 31, 31, 31, 31, 31, 31,
            31, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
            35,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 14, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 21, 21, 21, 21, 21, 21, 21, 27, 27, 27, 27, 27, 27,
            27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 36, 36, 36, 36, 36, 36, 36, 39, 39, 39, 39, 39, 39,
            39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,  7,  7,  7,  7,  7,  7,
            7,  7,  7, 15, 15, 15, 15, 15, 15, 15, 15, 15, 22, 22, 22, 22, 22,
            22, 22, 22, 22, 22, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 33, 33,
            33, 33, 33, 33, 33, 33, 33, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
            40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 42, 42, 42, 42, 42, 42, 42,
            42, 42, 42, 42, 42, 42, 42,  8,  8,  8,  8,  8,  8,  8, 16, 16, 16,
            16, 16, 16, 16, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 29, 29, 29,
            29, 29, 29, 29, 29, 29, 29, 34, 34, 34, 34, 34, 34, 34, 38, 38, 38,
            38, 38, 38, 38, 38, 38, 38, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
            43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44],
            dtype=np.int32)
        ind2_ref = np.array([ 0,  1, -1, -1, 34, -1, -1, -1, -1,  2, -1, -1, 35, -1, -1, -1, -1,
            3, -1, -1, 37, -1, -1, -1,  4, -1, -1, 39, -1, -1, 36, -1, -1, -1,
            -1, 38, -1, -1, -1, 40, -1, -1, 41, -1, 42,  5,  6, -1, -1, 43],
            dtype=np.int32)
        indexd_ref = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
            [ 1,  9, 10, 11, 12, 13, 14, 15, 16],
            [ 2, 10, 17, 18, 19, 20, 21, 22, 23],
            [ 3, 11, 18, 24, 25, 26, 27, 28, 29],
            [ 4, 12, 19, 25, 30, 31, 32, 33, 34],
            [ 5, 13, 20, 26, 31, 35, 36, 37, 38],
            [ 6, 14, 21, 27, 32, 36, 39, 40, 41],
            [ 7, 15, 22, 28, 33, 37, 40, 42, 43],
            [ 8, 16, 23, 29, 34, 38, 41, 43, 44]], dtype=np.int32)
        
        assert np.abs(TASK_ACTION-task_action_ref).max()<1.0E-13
        assert np.abs(TASK_TARGET-task_target_ref).max()<1.0E-13
        assert np.abs(TASK_IJ-task_ij_ref).max()<1.0E-13
        assert np.abs(IND2.flatten()[:50]-ind2_ref).max()<1.0E-13
        assert np.abs(INDEXD-indexd_ref).max()<1.0E-13

if __name__ == '__main__':
    print("Full tests for semi-empirical parameters reading function.")
    unittest.main()