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

import ctypes
import os
import numpy as np
import cupy as cp
from scipy.special import comb


_MAX_FAC = 30
_FACT_CPU = np.ones(_MAX_FAC, dtype=np.float64)
_FACT_CPU[1:] = np.cumprod(np.arange(1, _MAX_FAC, dtype=np.float64))
_FACT_GPU = cp.asarray(_FACT_CPU)


_MAX_GRID = 30
_n_grid = np.arange(_MAX_GRID).reshape(-1, 1)
_k_grid = np.arange(_MAX_GRID).reshape(1, -1)
_BINOMIALS_CPU = comb(_n_grid, _k_grid)
BINOMIALS_GPU = cp.asarray(_BINOMIALS_CPU)
BINOMIALS_GPU_FLAT = cp.asarray(_BINOMIALS_CPU.ravel(), dtype=np.float64)


def _load_cuda_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'liberi_1c2e_kernel.so'
    
    lib_path = os.path.join(curr_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)

    lib.launch_rsc_kernel_c.argtypes = [
        ctypes.c_int, ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p
    ]

    return lib


_eri1c2e_MODULE = _load_cuda_library()


def rsc(k, na, ea, nb, eb, nc, ec, nd, ed, HARTREE2EV=27.211386245988):
    """
    Compute Reduced Single Center (RSC) one-center two-electron integrals.
    
    This function calculates one-center two-electron integrals for semi-empirical methods
    based on Slater-type orbitals and Klopman-Ohno approximation. It uses CUDA kernels
    for GPU-accelerated computation.
    
    Args:
        k (int or cupy.ndarray): Angular momentum quantum number or array of angular momenta
        na, nb, nc, nd (cupy.ndarray): Quantum numbers for the four orbitals
        ea, eb, ec, ed (cupy.ndarray): Orbital exponents for the four orbitals
        HARTREE2EV (float): Conversion factor from Hartree to electron volts, default 27.211386245988
        
    Returns:
        cupy.ndarray: Array of computed one-center two-electron integral values, shape (n_size,)
        
    Note:
        - All input arrays must have the same length n_size
        - Scalar k will be converted to an array of length n_size internally
        - Uses CUDA kernel for parallel computation, suitable for large-scale integral calculations
        - Results are in Hartree units, can be converted to eV using HARTREE2EV parameter
    """
    if isinstance(k, int):
        n_size = na.shape[0]
        k_arr = cp.full(n_size, k, dtype=cp.int32)
    else:
        k_arr = cp.ascontiguousarray(k, dtype=cp.int32)
        n_size = k_arr.shape[0]

    na = cp.ascontiguousarray(na, dtype=cp.int32)
    nb = cp.ascontiguousarray(nb, dtype=cp.int32)
    nc = cp.ascontiguousarray(nc, dtype=cp.int32)
    nd = cp.ascontiguousarray(nd, dtype=cp.int32)
    
    ea = cp.ascontiguousarray(ea, dtype=cp.float64)
    eb = cp.ascontiguousarray(eb, dtype=cp.float64)
    ec = cp.ascontiguousarray(ec, dtype=cp.float64)
    ed = cp.ascontiguousarray(ed, dtype=cp.float64)
    
    out = cp.zeros(n_size, dtype=cp.float64)
    
    _eri1c2e_MODULE.launch_rsc_kernel_c(
        ctypes.c_int(n_size),
        ctypes.c_double(HARTREE2EV),
        ctypes.c_void_p(k_arr.data.ptr),
        ctypes.c_void_p(na.data.ptr), ctypes.c_void_p(ea.data.ptr),
        ctypes.c_void_p(nb.data.ptr), ctypes.c_void_p(eb.data.ptr),
        ctypes.c_void_p(nc.data.ptr), ctypes.c_void_p(ec.data.ptr),
        ctypes.c_void_p(nd.data.ptr), ctypes.c_void_p(ed.data.ptr),
        ctypes.c_void_p(_FACT_GPU.data.ptr),
        ctypes.c_void_p(BINOMIALS_GPU_FLAT.data.ptr),
        ctypes.c_void_p(out.data.ptr)
    )
    
    return out