# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

"""GPU (CuPy) versions of the array helpers in pyscf.gw.utils.gw_np_helper.

Same functions, signatures and in-place semantics as the CPU module; the
arrays live on the device.  mkslice is host-side index bookkeeping and also
accepts CuPy index arrays.
"""

import cupy as cp
import numpy as np


def array_scale(arr, alpha):
    """Scale an array in place by a scalar.
    arr <- alpha * arr

    A single elementwise kernel handles contiguous and strided arrays alike,
    so there is no BLAS/numexpr split as in the CPU version.

    Parameters
    ----------
    arr : cupy.ndarray
        Array to be scaled (modified in place).
    alpha : scalar
        Scale factor.
    """
    arr *= alpha


def addto_diagonal(arr, x):
    """Add a scalar or vector to the diagonal of a matrix.

    Parameters
    ----------
    arr : (M, M) cupy.ndarray
        Square matrix (will be modified)
    x : (M,) array_like | scalar
        Vector or scalar.

    Returns
    -------
    arr : (M, M) cupy.ndarray
        arr[i, i] <- arr[i, i] + x[i].
    """
    if not (np.isscalar(x) or isinstance(x, cp.ndarray)):
        x = cp.asarray(x)
    idx = cp.arange(min(arr.shape))
    arr[idx, idx] += x
    return arr


def get_id_minus_pi(Pi):
    """Calculate I - Pi in place.

    Parameters
    ----------
    Pi : (M, M) cupy.ndarray
        Input matrix.

    Returns
    -------
    id_minus_pi : (M, M) cupy.ndarray
        (I - Pi)
    """
    array_scale(Pi, -1.0)
    addto_diagonal(Pi, 1.0)
    return Pi


def get_id_minus_pi_inv(Pi, overwrite_input=False):
    """Calculate (I - Pi)^-1, given Pi

    As in the CPU version, Pi itself is always overwritten with I - Pi.
    cuSOLVER has no in-place inverse, so with overwrite_input=True the inverse
    is copied back into Pi's buffer and Pi is returned, preserving the CPU
    version's aliasing at the cost of one device copy.

    Parameters
    ----------
    Pi : (M, M) cupy.ndarray
        Input matrix. Must be C-contiguous.

    Returns
    -------
    Pi_inv : (M, M) cupy.ndarray
        (I - Pi)^-1
    """

    assert Pi.flags.c_contiguous
    id_minus_pi = get_id_minus_pi(Pi)
    id_minus_pi_inv = cp.linalg.inv(id_minus_pi)
    if overwrite_input:
        id_minus_pi[...] = id_minus_pi_inv
        id_minus_pi_inv = id_minus_pi
    return id_minus_pi_inv


def get_id_minus_pi_inv_minus_id(Pi, overwrite_input=False):
    """Calculate (I - Pi)^-1 - I.

    Parameters
    ----------
    Pi : (M, M) cupy.ndarray
        Input matrix. Must be C-contiguous.

    Returns
    -------
    Pi_inv : (M, M) cupy.ndarray
        (I - Pi)^-1 - I.
    """
    id_minus_pi_inv = get_id_minus_pi_inv(Pi, overwrite_input=overwrite_input)
    return addto_diagonal(id_minus_pi_inv, -1.0)


def mkslice(l):
    """
    Try to make a slice from a list of integers.
    If this is not possible, return the input unchanged.

    Index bookkeeping stays on the host; CuPy index arrays are accepted too.
    Two cases the CPU version gets wrong are handled: a descending run whose
    next element would be negative (the CPU version builds stop < 0, which
    Python reads as counting from the end, giving an empty slice), and
    repeated indices (a stride of 0 is not a valid slice step).

    Parameters
    ----------
    l : slice | list | range | ndarray
        Various ways of representing a list of integer indices

    Returns
    -------
    slice | list | ndarray
        slice if possible, otherwise returns the input unchanged
    """
    # If l is already a slice, return it
    if isinstance(l, slice):
        return l

    if l is None or (isinstance(l, tuple) and l == ()):
        return slice(None)

    # range to slice is easy
    if isinstance(l, range):
        return slice(l.start, l.stop, l.step)

    if len(l) < 2:
        return l

    idx = cp.asnumpy(l) if isinstance(l, cp.ndarray) else np.asarray(l)
    strides = np.diff(idx)
    if not np.all(strides == strides[0]) or strides[0] == 0:
        return l
    start = int(idx[0])
    step = int(strides[0])
    stop = int(idx[-1]) + step
    if stop < 0:
        stop = None
    return slice(start, stop, step)
