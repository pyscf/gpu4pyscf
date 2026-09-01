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

"""oneMKL-backed stand-in for gpu4pyscf.lib.cusolver on the SYCL backend.

cusolver.py calls cuSOLVER through `cupy_backends.cuda.libs.cusolver`, which
does not exist under SYCL. gpu4pyscf/cupy/__init__.py therefore redirects both
`cupy_backends.cuda.libs.cusolver` and `gpu4pyscf.lib.cusolver` to this module,
so anything importing either name (x2c/x2c.py, lib/cupy_helper.py) reaches
oneMKL instead.

This module must mirror cusolver.py's public surface exactly -- `eigh`,
`cholesky`, `LinAlgError` and `MAX_EIGH_DIM` -- with the same signatures and
the same failure behaviour, since callers are shared, unmodified upstream code.

Only the routines with no dpnp equivalent are routed through libonemkl_helper:
`dsygvd`/`zhegvd` for the generalized problem Hx = (lambda)Sx (dpnp.linalg.eigh
takes a single matrix) and `trsm`. `potrf` is kept here too so that Cholesky
failures raise this module's LinAlgError rather than dpnp's, matching what the
CUDA path does.
"""

import numpy as np
import dpnp
import dpctl
import ctypes
import os

# workspace size (lwork) provided by the cusolver*_bufferSize is an 32-bit
# integer. For arrays above this dimension, the workspace size would overflow.
MAX_EIGH_DIM = 23150

CUSOLVER_EIG_TYPE_1 = 1


_lib_dir = os.path.dirname(os.path.abspath(__file__))
libonemkl = ctypes.CDLL(os.path.join(_lib_dir, 'libonemkl_helper.so'))

libonemkl.onemkl_dsygvd_scratchpad_size.argtypes = [
    ctypes.c_int, # itype
    ctypes.c_int, # n
    ctypes.c_int, # lda
    ctypes.c_int, # ldb
    ctypes.c_void_p # *scratchpad_size
]
libonemkl.onemkl_zhegvd_scratchpad_size.argtypes = [
    ctypes.c_int,    # itype
    ctypes.c_int,    # n
    ctypes.c_int,    # lda
    ctypes.c_int,    # ldb
    ctypes.c_void_p  # *scratchpad_size
]


libonemkl.onemkl_dsygvd.argtypes = [
    ctypes.c_int,    # itype
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p, # *scratchpad
    ctypes.c_int     # scratchpad_size
]
libonemkl.onemkl_zhegvd.argtypes = [
    ctypes.c_int,    # itype
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p, # *scratchpad
    ctypes.c_int     # scratchpad_size
]


libonemkl.onemkl_dpotrf_scratchpad_size.argtypes = [
    ctypes.c_int, # n
    ctypes.c_int # lda
]
libonemkl.onemkl_dpotrf_scratchpad_size.restype = ctypes.c_int64
libonemkl.onemkl_dpotrf.restype = ctypes.c_int
libonemkl.onemkl_zpotrf.restype = ctypes.c_int
libonemkl.onemkl_zpotrf_scratchpad_size.argtypes = [
    ctypes.c_int, # n
    ctypes.c_int # lda
]
libonemkl.onemkl_zpotrf_scratchpad_size.restype = ctypes.c_int64

libonemkl.onemkl_dpotrf.argtypes = [
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *scratchpad
    ctypes.c_int     # scratchpad_size
]
libonemkl.onemkl_zpotrf.argtypes = [
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *scratchpad
    ctypes.c_int     # scratchpad_size
]

_buffersize = {}
def eigh(h, s, overwrite=False):
    """
    Solve the generalized eigenvalue problem Hx = λ Sx using oneMKL.
    """
    assert h.dtype == s.dtype
    assert h.dtype in (np.float64, np.complex128)
    n = h.shape[0]
    if h.dtype == np.complex128 and h.flags.c_contiguous:
        # zhegvd requires the matrices in F-order. For hermitian matrices,
        # .T.copy() is equivalent to .conj()
        A = h.conj()
        B = s.conj()
    elif overwrite:
        A = h
        B = s
    else:
        A = h.copy()
        B = s.copy()

    # Create buffers for A, B, and w
    # https://github.com/IntelPython/dpctl/issues/888
    w = dpnp.zeros(n)

    # TODO: reuse workspace
    if (h.dtype, n) in _buffersize:
        lwork = _buffersize[h.dtype, n]
    else:
        lwork = ctypes.c_int(0)
        if h.dtype == np.float64:
            fn = libonemkl.onemkl_dsygvd_scratchpad_size
        else:
            fn = libonemkl.onemkl_zhegvd_scratchpad_size
        fn(
            CUSOLVER_EIG_TYPE_1,
            n,
            n,
            n,
            ctypes.byref(lwork)
        )
        lwork = lwork.value
        _buffersize[h.dtype, n] = lwork

    if h.dtype == np.float64:
        fn = libonemkl.onemkl_dsygvd
    else:
        fn = libonemkl.onemkl_zhegvd
    #Allocate work-space
    work_buf = dpnp.empty((lwork,), dtype=h.dtype)
    fn(CUSOLVER_EIG_TYPE_1,
       n,
       ctypes.cast(A.data.ptr, ctypes.c_void_p),
       n,
       ctypes.cast(B.data.ptr, ctypes.c_void_p),
       n,
       ctypes.cast(w.data.ptr, ctypes.c_void_p),
       ctypes.cast(work_buf.data.ptr, ctypes.c_void_p),
       lwork)
    return w, A.T

def cholesky(A):
    """
    Compute the Cholesky decomposition of a Hermitian positive-definite matrix.

    Args:
        A: Hermitian positive-definite matrix

    Returns:
        Lower triangular matrix L such that A = L * L.T
    """
    n = len(A)
    # cusolver.py transposes an F-contiguous input and copies to C order;
    # do the same rather than asserting, so callers passing a transposed
    # view behave identically on both backends.
    if A.flags.f_contiguous:
        A = A.T
    x = A.copy(order='C')
    if A.dtype == np.float64:
        potrf = libonemkl.onemkl_dpotrf
        potrf_bufferSize = libonemkl.onemkl_dpotrf_scratchpad_size
    else:
        potrf = libonemkl.onemkl_zpotrf
        potrf_bufferSize = libonemkl.onemkl_zpotrf_scratchpad_size
    scratchpad_size = potrf_bufferSize(n, n)
    scratchpad = dpnp.empty(scratchpad_size, dtype=A.dtype)
    info = potrf(n,
                 ctypes.cast(x.data.ptr, ctypes.c_void_p),
                 n,
                 ctypes.cast(scratchpad.data.ptr, ctypes.c_void_p),
                 scratchpad_size)
    # cusolver.py raises on a non-zero dev_info; df.py and grad/rhf.py catch
    # RuntimeError to fall back to an eigendecomposition for singular j2c.
    if info != 0:
        raise LinAlgError('failed to perform Cholesky Decomposition')
    x = dpnp.tril(x, k=0)
    return x


class LinAlgError(RuntimeError):
    """Mirrors cusolver.LinAlgError, which also derives from RuntimeError."""
    pass
