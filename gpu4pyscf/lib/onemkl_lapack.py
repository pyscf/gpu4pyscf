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

import numpy as np
import dpnp
import dpctl
import ctypes

def tril_dpnp(x, k=0):
    rows, cols = x.shape
    mask = dpnp.arange(rows).reshape(-1, 1) >= (dpnp.arange(cols) - k)
    return x * mask

CUSOLVER_EIG_TYPE_1 = 1

# Define oneMKL function prototypes
libonemkl = ctypes.CDLL('/lus/flare/projects/NWChemEx_aesp_CNDA/abagusetty/gpu4pyscf/gpu4pyscf/gpu4pyscf/lib/libdpnp_helper.so')

# Define the function signatures (for sygvd)
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
    ctypes.c_int,                                                   # itype
    ctypes.c_int,                                                   # n
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # *A
    ctypes.c_int,                                                   # lda
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # *B
    ctypes.c_int,                                                   # ldb
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # *w
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # *scratchpad
    ctypes.c_int                                                    # scratchpad_size
]
libonemkl.onemkl_zhegvd.argtypes = [
    ctypes.c_int,                                                      # itype
    ctypes.c_int,                                                      # n
    np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"), # *A
    ctypes.c_int,                                                      # lda
    np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"), # *B
    ctypes.c_int,                                                      # ldb
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),    # *w
    np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"), # *scratchpad
    ctypes.c_int  # scratchpad_size
]


# Define the function signatures (for sygvd)
libonemkl.onemkl_dpotrf_scratchpad_size.argtypes = [
    ctypes.c_int, # n
    ctypes.c_int, # lda
    ctypes.c_void_p # *scratchpad_size
]
libonemkl.onemkl_zpotrf_scratchpad_size.argtypes = [
    ctypes.c_int, # n
    ctypes.c_int, # lda
    ctypes.c_void_p # *scratchpad_size
]

libonemkl.onemkl_dpotrf.argtypes = [
    ctypes.c_int, # n
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # *A
    ctypes.c_int, # lda
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"), # *scratchpad
    ctypes.c_int  # scratchpad_size
]
libonemkl.onemkl_zpotrf.argtypes = [
    ctypes.c_int, # n
    np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"), # *A
    ctypes.c_int, # lda
    np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"), # *scratchpad
    ctypes.c_int  # scratchpad_size
]

_buffersize = {}

def eigh(h, s):
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
    else:
        A = h.copy()
        B = s.copy()

    # Create buffers for A, B, and w
    # https://github.com/IntelPython/dpctl/issues/888
    A_buf = dpctl.tensor.from_numpy(A)
    B_buf = dpctl.tensor.from_numpy(B)
    w_buf = dpctl.tensor.empty((n,), dtype=h.dtype)

    # TODO: reuse workspace
    if (h.dtype, n) in _buffersize:
        lwork = _buffersize[h.dtype, n]
    else:
        lwork = ctypes.c_int(0)
        if h.dtype == np.float64:
            fn = libonemkl.onemkl_dsygvd_scratchpad_size
        else:
            fn = libonemkl.onemkl_zhegvd_scratchpad_size
        status = fn(
            _handle,
            CUSOLVER_EIG_TYPE_1,
            n,
            n,
            n,
            ctype.byref(lwork)
        )
        lwork = lwork.value
        _buffersize[h.dtype, n] = lwork

    if h.dtype == np.float64:
        fn = libonemkl.onemkl_dsygvd
    else:
        fn = libonemkl.onemkl_zhegvd
    #Allocate work-space
    work_buf = dpctl.tensor.empty((lwork,), dtype=h.dtype)
    fn(CUSOLVER_EIG_TYPE_1,
       n,
       A_buf.data,
       n,
       B_buf.data,
       n,
       w_buf.data,
       work_buf.data,
       lwork
       )

    # Retrieve results
    w = w_buf.get()
    V = A_buf.get().T  # Transpose of A as eigenvectors
    return w, V

def cholesky(A):
    n = len(A)
    assert A.flags['C_CONTIGUOUS']
    x = A.copy()
    x_buf = dpctl.tensor.from_numpy(x)
    if A.dtype == np.float64:
        potrf = libonemkl.onemkl_dpotrf
        potrf_bufferSize = libonemkl.onemkl_dpotrf_scratchpad_size
    else:
        potrf = libonemkl.onemkl_zpotrf
        potrf_bufferSize = libonemkl.onemkl_zpotrf_scratchpad_size
    potrf_bufferSize(n, n, ctype.byref(buffersize))
    buffersize = buffersize.value
    workspace_buf = dpctl.tensor.empty((buffersize,), dtype=A.dtype)
    potrf(n, x_buf.data, n, workspace_buf.data, buffersize)

    tril_dpnp(x)
    return x
