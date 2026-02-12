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

_MAX_FAC = 30
_FACT_CPU = np.ones(_MAX_FAC, dtype=np.float64)
_FACT_CPU[1:] = np.cumprod(np.arange(1, _MAX_FAC, dtype=np.float64))
_FACT_GPU = cp.asarray(_FACT_CPU)

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
        ctypes.c_void_p(r.data.ptr),
        ctypes.c_void_p(l1.data.ptr),
        ctypes.c_void_p(l2.data.ptr),
        ctypes.c_void_p(m.data.ptr),
        ctypes.c_void_p(da.data.ptr),
        ctypes.c_void_p(db.data.ptr),
        ctypes.c_void_p(add.data.ptr),
        ctypes.c_void_p(out.data.ptr)
    )

    return out


def a_function_ijl(z1, z2, n1, n2, l):
    """
    This function calculate the function defined in S3 of 10.1002/qua.25799
    Which will be used in determining the distance of the multipole interaction.
    
    **WARNING**
    The MOPAC code is as follows:
    double precision function aijl (z1, z2, n1, n2, l)
      double precision, intent(in) :: z1
      double precision, intent(in) :: z2
      integer, intent(in) :: n1
      integer, intent(in) :: n2
      integer, intent(in) :: l
      double precision :: zz
      zz = z1 + z2 + 1.d-20
      aijl = fx(n1+n2+l+1)/sqrt(fx(2*n1+1)*fx(2*n2+1))*(2*z1/zz)**n1*sqrt(2&
        *z1/zz)*(2*z2/zz)**n2*sqrt(2*z2/zz)*2**l/zz**l
      return
    end function aijl

    ** INCONSISTENT WITH THE PAPER **
    """

    zz = z1 + z2 + 1e-20
    
    idx1 = (n1 + n2 + l).astype(cp.int32)
    idx2 = (2 * n1).astype(cp.int32)
    idx3 = (2 * n2).astype(cp.int32)
    
    t1 = _FACT_GPU[idx1] / cp.sqrt(_FACT_GPU[idx2] * _FACT_GPU[idx3])
    t2 = cp.power(2.0 * z1 / zz, n1 + 0.5) * cp.power(2.0 * z2 / zz, n2 + 0.5)
    t3 = cp.power(2.0 / zz, l)
    
    return t1 * t2 * t3

