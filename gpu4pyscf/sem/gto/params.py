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
from pyscf.lib import logger
from gpu4pyscf.sem.data import atomic
from gpu4pyscf.sem.data import electron_repulsion
from gpu4pyscf.sem.data import corrections

class SEMParams:
    """
    Semi-Empirical Parameters Container.
    
    This class manages the loading, storage, and CPU-to-GPU transfer of 
    semi-empirical parameters. It also reconstructs derived parameters 
    (like core charges) that are typically calculated on-the-fly.
    
    Attributes:
        method (str): The method name (e.g., 'PM6').
        natorb (np.ndarray): Table of number of orbitals per atom (indexed by Z).
                             0 means the element is not supported.
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
        self.core_charges = self._compute_core_charges()

    def _check_method_supported(self):
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
                raise FileNotFoundError(f"Could not find file {filename} in {data_dir}")

    def _compute_natorb(self):
        """
        Infer the number of orbitals (0, 1, 4, or 9) for each element 
        based on the existence of s/p/d exponents.
        
        Returns:
            np.ndarray: Array of shape (107,) containing 0, 1, 4, or 9.
            0 indicates the element is not supported/has no basis.
        """
        natorb = np.zeros(107, dtype=np.int32)
        
        zs = self._data.get('exponent_s', np.zeros(107))
        zp = self._data.get('exponent_p', np.zeros(107))
        zd = self._data.get('exponent_d', np.zeros(107))
        
        threshold = 1e-6
        
        has_s = zs > threshold
        has_p = zp > threshold
        has_d = zd > threshold
        
        natorb[has_s] = 1
        natorb[has_s & has_p] = 4
        natorb[has_s & has_p & has_d] = 9
        
        return natorb

    def _compute_core_charges(self):
        """
        Compute core charges (tore) from occupation numbers (nocc_s, nocc_p, nocc_d).
        This reconstructs the logic from model_initialization.py.
        """
        if 'core_charge' in self._data:
            return self._data['core_charge']
        if 'tore' in self._data:
            return self._data['tore']

        def rep(n, v): 
            return [v]*n

        # s-electrons (ios)
        nocc_s = []
        nocc_s += [1, 2]                              # 1..2 (H-He)
        nocc_s += [1, 2] + [2, 2, 2, 2, 2, 0]         # 3..10 (Li-Ne)
        nocc_s += [1, 2] + [2, 2, 2, 2, 2, 0]         # 11..18 (Na-Ar)
        nocc_s += ([1, 2, 2] +                        # 19..21 (K-Sc)
                [2, 2, 1, 2, 2, 2, 2, 1, 2] +         # 22..30 (Ti-Zn)
                [2, 2, 2, 2, 2, 0])                   # 31..36 (Ga-Kr)
        nocc_s += ([1, 2, 2] +                        # 37..39 (Rb-Y)
                [2, 1, 1, 2, 1, 1, 0, 1, 2] +         # 40..48 (Zr-Cd)
                [2, 2, 2, 2, 2, 0])                   # 49..54 (In-Xe)
        nocc_s += ([1, 2, 2] +                        # 55..57 (Cs-La)
                rep(5, 2) + rep(3, 2) + rep(6, 2)+    # 58..71 (Ce..Lu)
                [2, 2, 1, 2, 2, 2, 1, 1, 2] +         # 72..80 (Hf..Hg)
                [2, 2, 2, 2, 2, 0])                   # 81..86 (Tl..Rn)
        nocc_s += ([1, 1, 2, 4, 2, 2] +                  # 87..92
                [2, 2, 2, 2, 2, 1, 0, 3, -3] +        # 93..101
                [1, 2, 1, -2, -1, 0])                 # 102..107
        # ! it should be noted that there is negative occupancy for some elements

        # p-electrons (iop)
        nocc_p = []
        nocc_p += [0, 0]
        nocc_p += [0, 0] + [1, 2, 3, 4, 5, 6]
        nocc_p += [0, 0] + [1, 2, 3, 4, 5, 6]
        nocc_p += [0, 0, 0] + rep(9, 0) + [1,2,3,4,5,6]
        nocc_p += [0, 0, 0] + rep(9, 0) + [1,2,3,4,5,6]
        nocc_p += [0, 0, 0] + rep(14, 0) + [0]*9 + [1,2,3,4,5,6]
        nocc_p += rep(21, 0)

        # d-electrons (iod)
        nocc_d = []
        nocc_d += [0, 0]
        nocc_d += rep(8, 0)
        nocc_d += rep(8, 0)
        nocc_d += [0, 0, 1, 2, 3, 5, 5, 6, 7, 8, 10, 0] + rep(6, 0)
        nocc_d += [0, 0, 1, 2, 4, 5, 5, 7, 8, 10,10, 0] + rep(6, 0)
        nocc_d += [0, 0, 1] + rep(13, 1) + [1, 2, 3, 5, 5, 6, 7, 9, 10] + rep(7, 0)
        nocc_d += [0, 0, 1] + rep(9, 0) + rep(9, 0)

        tore = (nocc_s + nocc_p + nocc_d).astype(np.float64)
        
        return tore

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
# This dictionary prevents reloading the heavy parameters (like alpb/xfac)
# multiple times if the user creates multiple Mole objects.
_PARAM_CACHE = {}

def load_sem_params(method='PM6'):
    """
    Factory function to load and cache parameters.
    
    Args:
        method (str): Method name (e.g., 'PM6').
        
    Returns:
        SEMParams: The parameter object (singleton per method).
    """
    method = method.upper()
    if method in _PARAM_CACHE:
        return _PARAM_CACHE[method]
    
    try:
        params = SEMParams(method)
        _PARAM_CACHE[method] = params
        return params
    except Exception as e:
        if method in _PARAM_CACHE:
            del _PARAM_CACHE[method]
        raise e