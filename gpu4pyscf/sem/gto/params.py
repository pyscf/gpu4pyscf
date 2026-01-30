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

import os
import numpy as np
import cupy as cp
from gpu4pyscf.sem.data import atomic
from gpu4pyscf.sem.data import electron_repulsion
from gpu4pyscf.sem.data import corrections

class SEMParams:
    """
    Semi-Empirical Parameters Container.
    
    This class manages the loading, storage, and CPU-to-GPU transfer of 
    semi-empirical parameters.
    
    Attributes:
        method (str): The method name (e.g., 'PM6').
        natorb (np.ndarray): Table of number of orbitals per atom (indexed by Z).
        core_charges (np.ndarray): Table of core charges/valence electrons (indexed by Z).
    """
    def __init__(self, method='PM6'):
        self.method = method.upper()
        self._data = {}
        self._gpu_cache = {}
        self._check_method_supported()

        self._load_module_params(atomic)
        self._load_module_params(electron_repulsion)
        self._load_module_params(corrections)
        self._load_binary_matrices()

        self.natorb = self._compute_natorb()
        self.core_charges = self._extract_core_charges()

    def _check_method_supported(self):
        """
        Check if the method is supported.
        """
        supported_methods = ['PM6']
        if self.method not in supported_methods:
            raise ValueError(f"Method {self.method} is not supported. "
                             f"Supported methods are: {supported_methods}")

    def _load_module_params(self, module):
        for name in dir(module):
            if name.startswith("_"): 
                continue
            
            val = getattr(module, name)
            if isinstance(val, np.ndarray):
                self._data[name] = val

    def _load_binary_matrices(self):
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        prefix = self.method.lower()
        files_map = {
            "alpha_bond": f"{prefix}_alpha_bond.npy",
            "x_factor":   f"{prefix}_x_factor.npy"
        }
        
        for key, filename in files_map.items():
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                self._data[key] = np.load(path)
            else:
                pass

    def _compute_natorb(self):
        """
        Infer the number of orbitals (0, 1, 4, or 9) for each element 
        based on the existence of s/p/d exponents.
        
        Returns:
            np.ndarray: Array of shape (107,) containing 0, 1, 4, or 9.
        """
        zs = self._data.get('exponent_s', np.zeros(107))
        zp = self._data.get('exponent_p', np.zeros(107))
        zd = self._data.get('exponent_d', np.zeros(107))
        
        natorb = np.zeros(107, dtype=np.int32)
        
        threshold = 1e-6
        has_s = zs > threshold
        has_p = zp > threshold
        has_d = zd > threshold
        
        natorb[has_s] = 1
        natorb[has_s & has_p] = 4
        natorb[has_s & has_p & has_d] = 9
        
        return natorb

    def _extract_core_charges(self):
        """
        Extract the core charge (valence electrons) table.
        This corresponds to 'tore' in the original code.
        """
        # Try new name first, then fall back to legacy name
        if 'core_charge' in self._data:
            return self._data['core_charge']
        elif 'tore' in self._data:
            return self._data['tore']
        else:
            # Critical error: Core charges are required for electron counting
            raise RuntimeError(f"Critical parameter 'core_charge' (or 'tore') "
                               f"not found in {self.method} library.")

    def get_natorb_table(self):
        return self.natorb

    def get_core_charges(self):
        return self.core_charges

    def get_parameter(self, key, to_gpu=True):
        """
        Retrieve a parameter by name, optionally moving it to GPU memory.
        
        Args:
            key (str): Parameter name (e.g., 'energy_core_s', 'g_ss').
            to_gpu (bool): If True, returns a cupy array (cached).
                           If False, returns the numpy array.
        
        Returns:
            np.ndarray or cp.ndarray
        """
        if key not in self._data:
            raise KeyError(f"Parameter '{key}' not found in {self.method} library.")
        
        data_cpu = self._data[key]
        
        if not to_gpu:
            return data_cpu
            
        if key not in self._gpu_cache:
            arr_contiguous = np.ascontiguousarray(data_cpu, dtype=np.float64)
            self._gpu_cache[key] = cp.asarray(arr_contiguous)
            
        return self._gpu_cache[key]

# ===========================================================
# Cache Mechanism (Singleton Pattern)
# ===========================================================
_PARAM_CACHE = {}

def load_sem_params(method='PM6'):
    """
    Factory function to load and cache parameters.
    
    Args:
        method (str): Method name (e.g., 'PM6').
        
    Returns:
        SEMParams: The parameter object.
    """
    method = method.upper()
    if method in _PARAM_CACHE:
        return _PARAM_CACHE[method]
    
    # Initialize and cache
    params = SEMParams(method)
    _PARAM_CACHE[method] = params
    return params