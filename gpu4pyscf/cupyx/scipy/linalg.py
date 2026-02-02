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
libonemkl = ctypes.CDLL('/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/lib/libonemkl_helper.so')

libonemkl.onemkl_trsm.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_int,     # m
    ctypes.c_int,     # n
    ctypes.c_int,     # lda
    ctypes.c_int,     # ldb
    ctypes.c_int,     # lower
    ctypes.c_int,     # trans
    ctypes.c_int      # unit_diagonal
]
libonemkl.onemkl_trsm.restype = None

###########################################################################################################

def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=False):
    # print("inputs from a in linalg.py: ", a)
    # print("inputs from b in linalg.py: ", b)
    """
    Solve the equation a x = b for x, assuming a is a triangular matrix using dpnp + oneMKL.

    Args:
        a (dpnp.ndarray): The matrix with dimension (M, M).
        b (dpnp.ndarray): The matrix with dimension (M,) or (M, N).
        lower (bool): Use lower triangle if True, otherwise upper.
        trans (0, 1, 2, 'N', 'T', 'C'): Type of system to solve:
            - 0 or 'N' -- a x = b
            - 1 or 'T' -- a^T x = b
            - 2 or 'C' -- a^H x = b
        unit_diagonal (bool): If True, assumes diagonal elements are all 1.
        overwrite_b (bool): Allow overwriting data in b (may enhance performance).
        check_finite (bool): Whether to check for NaNs or Infs.

    Returns:
        dpnp.ndarray: Solution x with same shape as b.
    """

    # Check shapes
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix 'a' must be square.")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Dimensions of 'a' and 'b' do not align.")

    # Handle trans parameter
    trans_flag = 0
    if trans in [1, 'T']:
        trans_flag = 1
    elif trans in [2, 'C']:
        raise NotImplementedError("Hermitian transpose not supported")

    # Type promotion - cast to float32 or float64
    if a.dtype.char in 'fdFD':
        dtype = a.dtype
    else:
        dtype = dpnp.promote_types(a.dtype.char, 'f')

    # FIX: Remove copy=False to allow dpnp to copy when necessary
    # If conversion to F-order or dtype change is needed, dpnp will copy automatically
    a = dpnp.array(a, dtype=dtype, order='F')

    # For b, handle overwrite_b properly
    # If overwrite_b=True and no conversion needed, don't copy
    # Otherwise, copy as needed
    if overwrite_b:
        # Try to avoid copy, but allow it if necessary
        b = dpnp.asarray(b, dtype=dtype)
        # Convert to F-order if needed (may copy)
        if not b.flags['F_CONTIGUOUS']:
            b = dpnp.asfortranarray(b)
    else:
        # Always make a copy
        b = dpnp.array(b, dtype=dtype, order='F', copy=True)

    if check_finite:
        if a.dtype.kind == 'f' and not dpnp.isfinite(a).all():
            raise ValueError('A array must not contain infs or NaNs')
        if b.dtype.kind == 'f' and not dpnp.isfinite(b).all():
            raise ValueError('B array must not contain infs or NaNs')

    m, n = (b.size, 1) if b.ndim == 1 else b.shape

    libonemkl.onemkl_trsm(ctypes.cast(a.data.ptr, ctypes.c_void_p),
                          ctypes.cast(b.data.ptr, ctypes.c_void_p),
                          ctypes.c_int(m), ctypes.c_int(n),
                          ctypes.c_int(m), ctypes.c_int(m),
                          ctypes.c_int(lower), ctypes.c_int(trans_flag),
                          ctypes.c_int(unit_diagonal))
    return b

###########################################################################################################


def block_diag(*arrs):
    """Create a block diagonal matrix from provided arrays.

    Given the inputs ``A``, ``B``, and ``C``, the output will have these
    arrays arranged on the diagonal::

        [A, 0, 0]
        [0, B, 0]
        [0, 0, C]

    Args:
        A, B, C, ... (cupy.ndarray): Input arrays. A 1-D array of length ``n``
            is treated as a 2-D array with shape ``(1,n)``.

    Returns:
        (cupy.ndarray): Array with ``A``, ``B``, ``C``, ... on the diagonal.
        Output has the same dtype as ``A``.

    .. seealso:: :func:`scipy.linalg.block_diag`
    """
    if not arrs:
        return dpnp.empty((1, 0))

    # --- NEW: unwrap gpu4pyscf wrappers like DPNPArrayWithTag ---
    def _unwrap_dpnp_like(a):
        base = getattr(a, "array", None)
        if isinstance(base, dpnp.ndarray):
            return base
        return a

    arrs = tuple(_unwrap_dpnp_like(a) for a in arrs)
    # --- END NEW ---

    # Convert to 2D and check
    if len(arrs) == 1:
        arrs = (dpnp.atleast_2d(*arrs),)
    else:
        arrs = dpnp.atleast_2d(*arrs)
    if any(a.ndim != 2 for a in arrs):
        bad = [k for k in range(len(arrs)) if arrs[k].ndim != 2]
        raise ValueError('arguments in the following positions have dimension '
                         'greater than 2: {}'.format(bad))

    shapes = tuple(a.shape for a in arrs)
    shape = tuple(sum(x) for x in zip(*shapes))
    out = dpnp.zeros(shape, dtype=dpnp.result_type(*arrs))
    r, c = 0, 0
    for arr in arrs:
        rr, cc = arr.shape
        out[r:r + rr, c:c + cc] = arr
        r += rr
        c += cc
    return out

###########################################################################################################

def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    cupyx.scipy.linalg.lu_factor(a, overwrite_a=False, check_finite=True)

    Thin wrapper that forwards to dpnp.linalg.lu_factor with the same
    semantics and defaults you pasted from dpnp.
    """
    # Forward directly; dpnp will do device/type checks and finiteness checks
    return _dpnp_lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """
    cupyx.scipy.linalg.lu_solve((lu, piv), b, trans=0, overwrite_b=False, check_finite=True)

    Thin wrapper that forwards to dpnp.linalg.lu_solve.
    """
    lu, piv = lu_and_piv
    return _dpnp_lu_solve(
        lu,
        piv,
        b,
        trans=trans,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )

###########################################################################################################

# Source: https://github.com/cupy/cupy/blob/main/cupyx/scipy/linalg/_matfuncs.py#L45

import math
th13 = 5.37

b = [64764752532480000.,
     32382376266240000.,
     7771770303897600.,
     1187353796428800.,
     129060195264000.,
     10559470521600.,
     670442572800.,
     33522128640.,
     1323241920.,
     40840800.,
     960960.,
     16380.,
     182.,
     1.,]

def expm(a):
    """Compute the matrix exponential.

    Parameters
    ----------
    a : dpnp.ndarray, 2D

    Returns
    -------
    matrix exponential of `a`

    Notes
    -----
    Uses (a simplified) version of Algorithm 2.3 of [1]_:
    a [13 / 13] Pade approximant with scaling and squaring.

    Simplifications:

        * we always use a [13/13] approximate
        * no matrix balancing

    References
    ----------
    .. [1] N. Higham, SIAM J. MATRIX ANAL. APPL. Vol. 26(4), p. 1179 (2005)
       https://doi.org/10.1137/04061101X

    """
    if a.size == 0:
        return dpnp.zeros((0, 0), dtype=a.dtype)

    n = a.shape[0]

    # follow scipy.linalg.expm dtype handling
    a_dtype = a.dtype if dpnp.issubdtype(
        a.dtype, dpnp.inexact) else dpnp.float64

    # try reducing the norm
    mu = dpnp.diag(a).sum() / n
    A = a - dpnp.eye(n, dtype=a_dtype) * mu

    # scale factor
    nrmA = dpnp.linalg.norm(A, ord=1).item()

    scale = nrmA > th13
    if scale:
        s = int(math.ceil(math.log2(float(nrmA) / th13))) + 1
    else:
        s = 1

    A /= 2**s

    # compute [13/13] Pade approximant
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4

    E = dpnp.eye(A.shape[0], dtype=a_dtype)
    bb = dpnp.asarray(b, dtype=a_dtype)

    u1, u2, v1, v2 = _expm_inner(E, A, A2, A4, A6, bb)
    u = A @ (A6 @ u1 + u2)
    v = A6 @ v1 + v2

    r13 = dpnp.linalg.solve(-u + v, u + v)

    # squaring
    x = r13
    for _ in range(s):
        x = x @ x

    # undo preprocessing
    emu = cmath.exp(mu) if dpnp.issubdtype(
        mu.dtype, dpnp.complexfloating) else math.exp(mu)
    x *= emu

    return x

def _expm_inner(E, A, A2, A4, A6, b):
    u1 = b[13]*A6 + b[11]*A4 + b[9]*A2
    u2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*E

    v1 = b[12]*A6 + b[10]*A4 + b[8]*A
    v2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*E
    return u1, u2, v1, v2
###########################################################################################################
