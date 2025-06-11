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

import dpnp
import ctypes

# Load your custom SYCL-backed shared library
# Define oneMKL function prototypes
libonemkl = ctypes.CDLL('/lus/flare/projects/NWChemEx_aesp_CNDA/abagusetty/gpu4pyscf/gpu4pyscf/gpu4pyscf/lib/libdpnp_helper.so')

# Define ctypes prototype
# extern "C" void onemkl_trsm(double* a, double* b,
#                             int m, int n, int lda, int ldb,
#                             int lower, int trans, int unit_diagonal)
libonemkl.onemkl_trsm.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # B
    ctypes.c_int,                     # m
    ctypes.c_int,                     # n
    ctypes.c_int,                     # lda
    ctypes.c_int,                     # ldb
    ctypes.c_int,                     # lower
    ctypes.c_int,                     # trans
    ctypes.c_int                      # unit_diagonal
]
libonemkl.onemkl_trsm.restype = None

def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=False):
    """
    Solve the equation a x = b for x, assuming a is a triangular matrix using dpnp + oneMKL.

    Args:
        a (dpnp.ndarray): The matrix with dimension (M, M).
        b (dpnp.ndarray): The matrix with dimension (M,) or (M, N).
        lower (bool): Use lower triangle if True, otherwise upper.
        trans ('N'|'T'|'C'): Solve transposed systems.
        unit_diagonal (bool): If True, assumes diagonal elements are all 1.
        overwrite_b (bool): Unused in dpnp version.
        check_finite (bool): Whether to check for NaNs or Infs.

    Returns:
        dpnp.ndarray: Solution x with same shape as b.
    """

    # Check shapes
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix 'a' must be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Dimensions of 'a' and 'b' do not align.")

    trans_flag = 0
    if trans in [1, 'T']:
        trans_flag = 1
    elif trans in [2, 'C']:
        raise NotImplementedError("Hermitian transpose not supported")

    # Type promotion
    if a.dtype.char in 'fdFD':
        dtype = a.dtype
    else:
        dtype = dpnp.promote_types(a.dtype.char, 'f')

    # Promote and convert to Fortran order (required by MKL)
    a = dpnp.array(a, dtype=dtype, order='F', copy=False)
    b = dpnp.array(b, dtype=dtype, order='F', copy=(not overwrite_b))

    if check_finite:
        if a.dtype.kind == 'f' and not dpnp.isfinite(a).all():
            raise ValueError(
                'A array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not dpnp.isfinite(b).all():
            raise ValueError(
                'B array must not contain infs or NaNs')

    # Dimensions
    m, n = (b.size, 1) if b.ndim == 1 else b.shape
    # m = a.shape[0]
    # n = b.shape[1] if b.ndim == 2 else 1
    lda = a.shape[1]
    ldb = b.shape[1] if b.ndim == 2 else 1

    # Raw pointers
    a_ptr = ctypes.c_void_p(a.__sycl_usm_array_interface__["data"][0])
    b_ptr = ctypes.c_void_p(b.__sycl_usm_array_interface__["data"][0])

    # Call oneMKL trsm
    libonemkl.onemkl_trsm(A_ptr, B_ptr,
                          ctypes.c_int(m), ctypes.c_int(n),
                          ctypes.c_int(lda), ctypes.c_int(ldb),
                          ctypes.c_int(lower), ctypes.c_int(trans_flag), ctypes.c_int(unit_diagonal))

    return b
