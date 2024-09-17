# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import cupy
import ctypes
from cupy_backends.cuda.libs import cusolver
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device

_handle = device.get_cusolver_handle()
libcusolver = ctypes.CDLL('libcusolver.so')

CUSOLVER_EIG_TYPE_1 = 1
CUSOLVER_EIG_TYPE_2 = 2
CUSOLVER_EIG_TYPE_3 = 3

CUSOLVER_EIG_MODE_NOVECTOR = 0
CUSOLVER_EIG_MODE_VECTOR = 1

libcusolver.cusolverDnDsygvd_bufferSize.restype = int
libcusolver.cusolverDnDsygvd.restype = int

_buffersize = {}

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnDsygvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnDsygvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

def eigh(h, s):
    '''
    solve generalized eigenvalue problem
    '''
    n = h.shape[0]
    w = cupy.zeros(n)
    A = h.copy()
    B = s.copy()
    
    # TODO: reuse workspace
    if n in _buffersize:
        lwork = _buffersize[n]
    else:
        lwork = ctypes.c_int()
        status = libcusolver.cusolverDnDsygvd_bufferSize(
            _handle,
            CUSOLVER_EIG_TYPE_1,
            CUSOLVER_EIG_MODE_VECTOR,
            cublas.CUBLAS_FILL_MODE_LOWER,
            n,
            A.data.ptr,
            n,
            B.data.ptr,
            n,
            w.data.ptr,
            ctypes.byref(lwork)
        )
        lwork = lwork.value
    
    work = cupy.empty(lwork)
    devInfo = cupy.empty(1, dtype=np.int32)
    status = libcusolver.cusolverDnDsygvd(
        _handle,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        cublas.CUBLAS_FILL_MODE_LOWER,
        n,
        A.data.ptr,
        n,
        B.data.ptr,
        n,
        w.data.ptr,
        work.data.ptr,
        lwork,
        devInfo.data.ptr
    )
    
    if status != 0:
        raise RuntimeError("failed in eigh kernel")
    return w, A.T

def cholesky(A):
    n = len(A)
    assert A.flags['C_CONTIGUOUS']
    x = A.copy()
    handle = device.get_cusolver_handle()
    potrf = cusolver.dpotrf
    potrf_bufferSize = cusolver.dpotrf_bufferSize
    buffersize = potrf_bufferSize(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize)
    dev_info = cupy.empty(1, dtype=np.int32)
    potrf(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)

    if dev_info[0] != 0:
        raise RuntimeError('failed to perform Cholesky Decomposition')
    cupy.linalg._util._tril(x,k=0)
    return x