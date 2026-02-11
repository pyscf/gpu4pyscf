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

import ctypes
import os
import numpy as np
import cupy as cp

def _load_cuda_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'liberi_2c2e_kernel.so'
    
    lib_path = os.path.join(curr_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)
    
    lib.launch_multipole_eval_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p
    ]

    return lib

_eri2c2e_MODULE = _load_cuda_library()

def multipole_eval(r, l1, l2, m, da, db, add):
    """
    Compute multipole interaction potential V(r) on GPU.
    All inputs must be CuPy arrays of shape (N,).
    
    Args:
        r: Distance (Bohr)
        l1, l2: Angular momentum (0=s, 1=p, 2=d)
        m: Magnetic quantum number
        da, db: Multipole lengths
        add: Klopman-Ohno squared term
        
    Returns:
        out: (N,) Interaction potential
    """

    r = cp.ascontiguousarray(r, dtype=cp.float64)
    l1 = cp.ascontiguousarray(l1, dtype=cp.int32)
    l2 = cp.ascontiguousarray(l2, dtype=cp.int32)
    m = cp.ascontiguousarray(m, dtype=cp.int32)
    da = cp.ascontiguousarray(da, dtype=cp.float64)
    db = cp.ascontiguousarray(db, dtype=cp.float64)
    add = cp.ascontiguousarray(add, dtype=cp.float64)

    n_pair = r.shape[0]
    out = cp.zeros(n_pair, dtype=cp.float64)
    _eri2c2e_MODULE.launch_multipole_eval_kernel_c(
        ctypes.c_int(n_pair),
        ctypes.c_void_p(r.ctypes.data),
        ctypes.c_void_p(l1.ctypes.data),
        ctypes.c_void_p(l2.ctypes.data),
        ctypes.c_void_p(m.ctypes.data),
        ctypes.c_void_p(da.ctypes.data),
        ctypes.c_void_p(db.ctypes.data),
        ctypes.c_void_p(add.ctypes.data),
        ctypes.c_void_p(out.ctypes.data)
    )

    return out

