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
import cupy as cp
from gpu4pyscf.sem.gto.params import load_sem_params


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

if __name__ == '__main__':
    print("Full tests for semi-empirical parameters reading function.")
    unittest.main()