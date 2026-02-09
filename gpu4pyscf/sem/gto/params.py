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
        norbitals_per_atom (np.ndarray): Table of number of orbitals per atom (indexed by Z).
                             0 means the element is not supported. (formerly natorb)
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
        self.norbitals_per_atom = self._compute_natorb()
        self._compute_core_charges()
        
        self._init_principal_quantum_numbers()
        self._init_electronic_configuration_metadata()
        self._init_reference_heats()
        self._compute_multipole_angular_factors()

        self.cutoff_radius = 15.0

        zd = self._data.get('exponent_d', np.zeros(107))
        self.has_d_orbitals = (self.principal_quantum_number_d > 0) & (zd > 1.0e-8)

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

        nocc_s = np.array(nocc_s, dtype=int)
        nocc_p = np.array(nocc_p, dtype=int)
        nocc_d = np.array(nocc_d, dtype=int)

        core_charges = (nocc_s + nocc_p + nocc_d).astype(np.float64)
        
        self.nocc_s = nocc_s
        self.nocc_p = nocc_p
        self.nocc_d = nocc_d
        self.core_charges = core_charges

    def _init_principal_quantum_numbers(self):
        """
        Initialize Principal Quantum Numbers (PQN).
        Formerly: iii -> principal_quantum_number_s, i
        iid -> principal_quantum_number_d, 
        npq -> principal_quantum_number_matrix
        """
        # PQN for s/p orbitals (107 elements)
        self.principal_quantum_number_s = np.array(
            [1]*2 + [2]*8 + [3]*8 + [4]*18 + [5]*18 + [6]*32 + [0]*21, 
            dtype=int
        )
        
        # PQN for d orbitals (107 elements)
        self.principal_quantum_number_d = np.array(
            [3]*30 + [4]*18 + [5]*32 + [6]*6 + [0]*21, 
            dtype=int
        )

        def rep(n, v): 
            return [v]*n
        
        npq_s = []
        npq_s += [1, 1]
        npq_s += [2, 2] + rep(5, 2) + [3]
        npq_s += [3, 3] + rep(5, 3) + [4]
        npq_s += rep(17, 4) + [5]
        npq_s += rep(17, 5) + [6]
        npq_s += rep(31, 6) + [7]
        npq_s = np.array(npq_s + [0]*(107-len(npq_s)), dtype=int)

        npq_p = []
        npq_p += [1, 2]
        npq_p += [2]*8
        npq_p += [3]*8
        npq_p += [4]*18
        npq_p += [5]*18
        npq_p += [6]*32
        npq_p += [0]*21
        npq_p = np.array(npq_p, dtype=int)

        npq_d = []
        npq_d += [0, 0] + rep(8, 0) # 1-10
        npq_d += [3, 3] + [3]*5 + [4] # 11-18
        npq_d += [3, 3] + rep(9, 3) + [4]*6 + [5] # 19-36
        npq_d += [4, 4] + rep(9, 4) + [5]*6 + [6] # 37-54
        npq_d += [5, 5] + rep(14, 5) + [5]*9 + [6]*6 + [7] # 55-86
        npq_d = np.array(npq_d + [0]*(107-len(npq_d)), dtype=int)

        self.principal_quantum_number_matrix = np.stack((npq_s, npq_p, npq_d), axis=-1)

    def _init_electronic_configuration_metadata(self):
        """
        Initialize metadata regarding electronic configuration.
        Formerly: ndelec, main_group
        """
        def rep(n, v): return [v]*n
        
        # d-shell occupation reference (ndelec)
        self.d_shell_occupation_ref = np.array(
            rep(20, 0) +
            [0, 0, 2, 2, 4, 4, 6, 8, 10, 10] +
            rep(8, 0) +
            [0, 0, 2, 2, 4, 4, 6, 8, 10, 10] +
            rep(22, 0) +
            [0, 0, 2, 2, 4, 4, 6, 8, 10, 10] +
            rep(27, 0),
            dtype=int
        )

        # Main group flag (main_group)
        self.is_main_group = np.array(
            [True]*2 +                    
            [True]*8 +                    
            [True]*8 +                    
            [True]*2 + [False]*9 + [True]*7 +   
            [True]*2 + [False]*9 + [True]*7 +   
            [True]*2 + [False]*23 + [True]*7 +  
            [True]*21,
            dtype=bool
        )

    def _init_reference_heats(self):
        """
        Initialize experimental Heat of Formation data.
        Formerly: eheat, eheat_sparkles
        """
        self.heat_formation_ref = np.zeros(107, dtype=np.float64)
        data_pairs = {
            1:  52.102,  3:  38.410,  4:  76.960,  5: 135.700, 6: 170.890,  7: 113.000,
            8:  59.559,  9:  18.890, 11:  25.650, 12:  35.000, 13:  79.490, 14: 108.390,
            15:  75.570, 16:  66.400, 17:  28.990, 19:  21.420, 20:  42.600, 21:  90.300,
            22: 112.300, 23: 122.900, 24:  95.000, 25:  67.700, 26:  99.300, 27: 102.400,
            28: 102.800, 29:  80.700, 30:  31.170, 31:  65.400, 32:  89.500, 33:  72.300,
            34:  54.300, 35:  26.740, 37:  19.600, 38:  39.100, 39: 101.500, 40: 145.500,
            41: 172.400, 42: 157.300, 43: 162.000, 44: 155.500, 45: 133.000, 46:  90.000,
            47:  68.100, 48:  26.720, 49:  58.000, 50:  72.200, 51:  63.200, 52:  47.000,
            53:  25.517, 55:  18.700, 56:  42.500, 57: 103.011, 58: 101.004, 59:  84.990,
            60:  78.298, 61:  83.174, 62:  49.402, 63:  41.898, 64:  95.007, 65:  92.902,
            66:  69.407, 67:  71.893, 68:  75.791, 69:  55.500, 70:  36.358, 71: 102.199,
            72: 148.000, 73: 186.900, 74: 203.100, 75: 185.000, 76: 188.000, 77: 160.000,
            78: 135.200, 79:  88.000, 80:  14.690, 81:  43.550, 82:  46.620, 83:  50.100,
            90: 1674.64, 102: 207.000
        }
        for n, val in data_pairs.items():
            self.heat_formation_ref[n-1] = val # formerly eheat

        self.heat_formation_sparkles_ref = np.zeros(107, dtype=np.float64)
        sparkles_pairs = {
            57:  928.90, 58:  944.70, 59:  952.90, 60:  962.80, 61:  976.90,
            62:  974.40, 63: 1006.60, 64:  991.37, 65:  999.00, 66: 1001.30,
            67: 1009.60, 68: 1016.15, 69: 1022.06, 70: 1039.03, 71: 1031.20,
        }
        for n, val in sparkles_pairs.items():
            self.heat_formation_sparkles_ref[n-1] = val

    def _compute_multipole_angular_factors(self):
        """
        Compute 'multipole_angular_factors' (formerly ch).
        This extracts purely the coefficient logic, discarding index arrays.
        
        Returns:
            np.ndarray: Shape (45, 3, 5). 
        """
        ch = np.zeros((45, 3, 5), dtype=np.float64)
        
        def set_ch(i_1b, l, m, v): 
            # i_1b is 1-based index from original code
            # ! i_1b is 0-based index in numpy in this code!
            ch[i_1b, l, m+2] = v

        set_ch(0,0,0, 1.0)
        set_ch(1,1,0, 1.0)
        set_ch(2,1,1, 1.0)
        set_ch(3,1,-1,1.0)
        set_ch(4,2,0, 1.15470054)
        set_ch(5,2,1, 1.0)
        set_ch(6,2,-1,1.0)
        set_ch(7,2,2, 1.0)
        set_ch(8,2,-2,1.0)
        set_ch(9,0,0,1.0); set_ch(9,2,0,1.33333333)
        set_ch(10,2,1,1.0)
        set_ch(11,2,-1,1.0)
        set_ch(12,1,0,1.15470054)
        set_ch(13,1,1,1.0)
        set_ch(14,1,-1,1.0)
        set_ch(17,0,0,1.0); set_ch(17,2,0,-0.66666667); set_ch(17,2,2,1.0)
        set_ch(18,2,-2,1.0)
        set_ch(19,1,1,-0.57735027)
        set_ch(20,1,0,1.0)
        set_ch(22,1,1,1.0)
        set_ch(23,1,-1,1.0)
        set_ch(24,0,0,1.0); set_ch(24,2,0,-0.66666667); set_ch(24,2,2,-1.0)
        set_ch(25,1,-1,-0.57735027)
        set_ch(27,1,0,1.0)
        set_ch(28,1,-1,-1.0)
        set_ch(29,1,1,1.0)
        set_ch(30,0,0,1.0); set_ch(30,2,0,1.33333333)
        set_ch(31,2,1,0.57735027)
        set_ch(32,2,-1,0.57735027)
        set_ch(33,2,2,-1.15470054)
        set_ch(34,2,-2,-1.15470054)
        set_ch(35,0,0,1.0); set_ch(35,2,0,0.66666667); set_ch(35,2,2,1.0)
        set_ch(36,2,-2,1.0)
        set_ch(37,2,1,1.0)
        set_ch(38,2,-1,1.0)
        set_ch(39,0,0,1.0); set_ch(39,2,0,0.66666667); set_ch(39,2,2,-1.0)
        set_ch(40,2,-1,-1.0)
        set_ch(41,2,1,1.0)
        set_ch(42,0,0,1.0); set_ch(42,2,0,-1.33333333)
        set_ch(44,0,0,1.0); set_ch(44,2,0,-1.33333333)
        
        self.multipole_angular_factors = ch

    def get_natorb_table(self):
        return self.norbitals_per_atom

    def get_core_charges(self):
        return self.core_charges

    def get_parameter(self, key, to_gpu=True):
        if (key not in self._data) and (key not in self.__dict__.keys()):
            raise KeyError(f"Parameter '{key}' not found in {self.method} library.")
        
        if key in self._data:
            data_cpu = self._data[key]
        else:
            data_cpu = self.__dict__[key]
        
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