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

import sys
import types
import numpy as np
import dpnp
from dpnp.dpnp_array import dpnp_array
import dpctl.tensor as dpt

# --- cupy ndarray alias (callable + isinstance) ---
from abc import ABCMeta

def _resolve_dpnp_impl():
    try:
        import dpnp.dpnp_array as _mod
        return getattr(_mod, "dpnp_array", None)
    except Exception:
        return None

_DPNP_ARRAY_IMPL = _resolve_dpnp_impl()


########################################################################################

 # Issue: https://github.com/IntelPython/dpnp/issues/2641

 # at top
def _construct_from_memptr(shape, dtype, memptr):
    """
    CuPy-compatible constructor:
    Given a device array-like `memptr` (dpnp array or dpctl usm_ndarray),
    return a dpnp.ndarray that views the **first prod(shape)** elements
    (starting at the current view offset), reshaped to `shape` in C-order,
    without copying.
    """
    # Normalize to a dpnp array view (preserves USM base+offset)
    if hasattr(memptr, "__sycl_usm_array_interface__"):
        arr = dpnp.asarray(memptr)  # no copy; keeps offset
    else:
        # Fallback: allow dpctl usm_ndarray
        try:
            u = dpt.asarray(memptr, copy=False)
            arr = dpnp.asarray(u)   # wrap to dpnp
        except Exception:
            # Last resort: let dpnp try (may copy)
            arr = dpnp.asarray(memptr)

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)

    # Number of elements to expose
    needed = int(np.prod(shape))
    total  = int(arr.size)
    if needed > total:
        raise ValueError(f"Cannot construct array of shape {shape} "
                         f"from buffer with only {total} elements")

    # Make a C-contiguous 1D view **from the current view start**,
    # then take the first `needed` elements and reshape.
    flat = dpnp.ravel(arr, order="C")[:needed]   # view, no copy
    return flat.reshape(shape, order="C")        # view, no copy

# then in your meta-class __call__:
class _CuPyNdarrayMeta(ABCMeta):
    def __call__(cls, shape, dtype=np.float64, memptr=None):
        if memptr is not None and hasattr(memptr, "get_array"):
            memptr = memptr.get_array()
        if memptr is not None:
            return _construct_from_memptr(shape, dtype, memptr)
        return dpnp.ndarray(shape, dtype=dtype)

  # once the above issue is fixed, delete this section between ### and re-enable the next
  # class __CuPyNdarrayMeta's __call__ method
########################################################################################
# class _CuPyNdarrayMeta(ABCMeta):
#     # Make cupy.ndarray((shape), dtype=..., memptr=...) construct dpnp.ndarray
#     def __call__(cls, shape, dtype=np.float64, memptr=None):
#         if memptr is not None and hasattr(memptr, "get_array"):
#             memptr = memptr.get_array()
#         if memptr is not None:
#             return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
#         return dpnp.ndarray(shape, dtype=dtype)

    # isinstance(x, cupy.ndarray) -> True for:
    # - dpnp.ndarray
    # - dpnp.dpnp_array.dpnp_array (some versions)
    # - duck-typed wrappers that expose .array as a dpnp.ndarray (e.g. DPNPArrayWithTag)
    def __instancecheck__(cls, obj):
        if isinstance(obj, dpnp.ndarray):
            return True
        if _DPNP_ARRAY_IMPL and isinstance(obj, _DPNP_ARRAY_IMPL):
            return True
        if hasattr(obj, "array") and isinstance(getattr(obj, "array"), dpnp.ndarray):
            return True
        return False

    def __subclasscheck__(cls, sub):
        try:
            bases = [dpnp.ndarray]
            if _DPNP_ARRAY_IMPL:
                bases.append(_DPNP_ARRAY_IMPL)
            # treat “has .array of dpnp.ndarray” as acceptable duck-subclass: cannot be checked reliably here
            return any(issubclass(sub, b) for b in bases)
        except TypeError:
            return False

class _CuPyNdarray(metaclass=_CuPyNdarrayMeta):
    """Alias type for CuPy ndarray over dpnp arrays and wrappers."""
    pass

# --- Setup fake cupy module ---
cupy_fake = types.ModuleType("cupy")
cupy_fake.ndarray = _CuPyNdarray

####################################################
# the logic in this section is to support snippets like
# exc -= cupy.einsum('ij,ji', dm, vk).real * .25, where dm is
# of type <class 'gpu4pyscf.lib.dpnp_helper.DPNPArrayWithTag'> while
# vk is of type: <class 'dpnp.dpnp_array.dpnp_array'>

# ---- helpers ----
def _unwrap_dpnp(x):
    # unwrap objects that carry a dpnp array in `.array`
    if hasattr(x, "array") and isinstance(getattr(x, "array"), dpnp.ndarray):
        return x.array
    return x

# ---- safe asarray (unwrap then coerce) ----
def _cupy_asarray(a, *args, **kwargs):
    a = _unwrap_dpnp(a)
    return dpnp.asarray(a, *args, **kwargs)

# ---- safe einsum (unwrap all operands, coerce to dpnp) ----
def _cupy_einsum(subscripts, *operands, **kwargs):
    ops = []
    for op in operands:
        op = _unwrap_dpnp(op)
        # If someone passes a tuple (rare in einsum), unwrap its members too
        if isinstance(op, tuple):
            op = tuple(_unwrap_dpnp(t) for t in op)
        # Ensure dpnp dtype
        if not isinstance(op, dpnp.ndarray):
            op = dpnp.asarray(op)
        ops.append(op)
    return dpnp.einsum(subscripts, *ops, **kwargs)

def _cupy_asnumpy(a, *args, **kwargs):
    a = _unwrap_dpnp(a)
    return dpnp.asnumpy(a, *args, **kwargs)

# install overrides (must be AFTER the bulk setattr() loop)
cupy_fake.asarray = _cupy_asarray
cupy_fake.einsum  = _cupy_einsum
cupy_fake.asnumpy = _cupy_asnumpy

# Here is a work around for another `DPNPArrayWithTag` using dot()
# from DPNP.
_original_dpnp_dot = dpnp.dot
_original_ndarray_dot = dpnp.ndarray.dot

def _cupy_dot(a, b, out=None):
    """dpnp.dot with DPNPArrayWithTag support"""
    a = _unwrap_dpnp(a)
    b = _unwrap_dpnp(b)
    if out is not None:
        out = _unwrap_dpnp(out)
    return _original_dpnp_dot(a, b, out=out)

def _ndarray_dot_method(self, b, out=None):
    """ndarray.dot() method with DPNPArrayWithTag support"""
    b = _unwrap_dpnp(b)
    if out is not None:
        out = _unwrap_dpnp(out)
    return _original_ndarray_dot(self, b, out=out)

# Install patches
dpnp.dot = _cupy_dot
dpnp.ndarray.dot = _ndarray_dot_method
cupy_fake.dot = _cupy_dot

####################################################

# Memory Pool Stub (dpnp doesn't have memory pools)
# Make get_default_memory_pool() a no-op to match cupy API

class _DummyMemoryPool:
    """No-op memory pool stub for dpnp (which has no memory pool concept)"""

    def free_all_blocks(self):
        """No-op: dpnp manages memory automatically"""
        pass

    def free_all_free(self):
        """No-op: dpnp manages memory automatically"""
        pass

    def used_bytes(self):
        """Return 0 since we can't query dpnp memory usage"""
        return 0

    def total_bytes(self):
        """Return 0 since we can't query dpnp memory allocation"""
        return 0

    def n_free_blocks(self):
        """Return 0 since dpnp has no block concept"""
        return 0

# Create a singleton instance
_dummy_pool = _DummyMemoryPool()

def _get_default_memory_pool():
    """Return dummy memory pool (no-op for dpnp)"""
    return _dummy_pool

def _set_allocator(allocator=None):
    """No-op: dpnp memory allocation is managed by SYCL"""
    pass

# Install into cupy_fake module
cupy_fake.get_default_memory_pool = _get_default_memory_pool
cupy_fake.set_allocator = _set_allocator

##########################################################################

def patched_cupy_array(a, *args, **kwargs):
    unwrapped = getattr(a, "array", a)
    if isinstance(unwrapped, dpnp.ndarray) and kwargs.get("copy") is False:
        kwargs.pop("copy", None)
    return dpnp.array(unwrapped, *args, **kwargs)
cupy_fake.array = patched_cupy_array


# Populate other dpnp functions as cupy attributes
for attr in [
        "append", "max", "dot", "linalg", "concatenate", "asarray", "zeros", "ones",
        "empty", "eye", "view", "empty_like", "copyto", "cumsum", "any", "matmul",
        "vstack", "full", "arange", "stack", "expand_dims", "unique", "double",
        "sqrt", "argsort", "count_nonzero", "where", "split", "take", "tril", "log",
        "complex128", "uint8", "int32", "int64", "float64", "ravel", "random", "sum", "exp",
        "outer", "ix_", "pi", "square", "multiply", "diag_indices", "repeat", "diag",
        "tril_indices_from", "ceil", "newaxis", "ascontiguousarray", "nonzero",
        "array_equal"
]:
    try:
        setattr(cupy_fake, attr, getattr(dpnp, attr))
    except AttributeError:
        pass

# Optional: cupy.cuda submodule stub
try:
    from . import cuda
    cupy_fake.cuda = cuda
except ImportError as e:
    print(f"Could not import .cuda: {e}")

# Register in sys.modules
sys.modules["cupy"] = cupy_fake

#####################################################################

# [COMMENT]: The next few lines is the older version (commented out)
# _original_setitem = dpnp.ndarray.__setitem__
# def safe_setitem(self, key, value):
#     if isinstance(key, tuple):
#         key = tuple(dpnp.asarray(k) if isinstance(k, np.ndarray) else k for k in key)
#     return _original_setitem(self, key, value)
# dpnp.ndarray.__setitem__ = safe_setitem

# [WORKAROUND]: To address indexing np.ndarray in tuples, list
# Similar to the issue with getitem() as described in https://github.com/IntelPython/dpnp/issues/2622
#
# Error: dpctl.tensor._usmarray._basic_slice_meta
#IndexError: Only integers, slices (`:`), ellipsis (`...`), dpctl.tensor.newaxis (`None`) and integer and boolean arrays are valid indices.

_original_setitem = dpnp.ndarray.__setitem__
def safe_setitem(self, key, value):
    """Handle list/array indexing that DPNP doesn't support natively."""
    def _convert_index(k):
        # Python list of ints -> convert to dpnp array
        if isinstance(k, list):
            return dpnp.asarray(k, dtype=dpnp.intp)
        # NumPy array -> move to device
        if isinstance(k, np.ndarray) and k.dtype.kind in ("b", "i", "u"):
            return dpnp.asarray(k)
        return k

    if isinstance(key, tuple):
        key = tuple(_convert_index(k) for k in key)
    else:
        key = _convert_index(key)

    return _original_setitem(self, key, value)
dpnp.ndarray.__setitem__ = safe_setitem

#####################################################################

# Add `.set()`, `.get()` method to dpnp_array to mimic CuPy behavior
def _dpnp_set(self, host_array):
    self[...] = host_array
dpnp.dpnp_array.dpnp_array.set = _dpnp_set

#commenting this since there is a bug with "order" arg:
# see: https://github.com/IntelPython/dpctl/issues/2138
# def _dpnp_get(self, order='C'):
#     try:
#         return dpnp.asnumpy(self, order=order)
#     except TypeError:
#         return dpnp.asnumpy(self)
# dpnp.dpnp_array.dpnp_array.get = _dpnp_get

def _dpnp_get(self, order='C'):
    # 1) device -> host (order ignored by dpnp for dpnp_array)
    host = self.asnumpy()

    # 2) enforce requested order like CuPy
    if order == 'C':
        return np.ascontiguousarray(host)          # copies only if needed
    if order == 'F':
        return np.asfortranarray(host)             # copies only if needed
    if order == 'A':
        # 'F' if strictly Fortran-only, else 'C' (matches NumPy semantics)
        if host.flags['F_CONTIGUOUS'] and not host.flags['C_CONTIGUOUS']:
            return np.asfortranarray(host)
        return np.ascontiguousarray(host)
    if order == 'K':
        # keep strides; avoid extra copy
        return np.array(host, order='K', copy=False)

    # default fallback (behave like 'C')
    return np.ascontiguousarray(host)

dpnp.dpnp_array.dpnp_array.get = _dpnp_get

##########################################################################

# this serves as a WA for cupy/dpnp differences where for eg:
# `rows = cp.hstack(rows)`, where `rows` on RHS is a np.array
# This works in cupy but not for dpnp. So makes dpnp also work
# by casting to dpnp

def _to_dpnp_seq(seq):
    out = []
    for s in seq:
        s = getattr(s, "array", s)  # unwrap optional .array
        if isinstance(s, np.ndarray) and not isinstance(s, dpnp.ndarray):
            out.append(dpnp.asarray(s))
        else:
            out.append(s)
    return out

# Match DPNP signatures and forward kwargs directly
def _hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = _to_dpnp_seq(tup)
    return dpnp.hstack(arrs, dtype=dtype, casting=casting)

def _vstack(tup, *, dtype=None, casting="same_kind"):
    arrs = _to_dpnp_seq(tup)
    return dpnp.vstack(arrs, dtype=dtype, casting=casting)

# def _stack(arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind"):
#     arrs = _to_dpnp_seq(arrays)
#     return dpnp.stack(arrs, axis=axis, out=out, dtype=dtype, casting=casting)

# def _concatenate(arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind"):
#     arrs = _to_dpnp_seq(arrays)
#     return dpnp.concatenate(arrs, axis=axis, out=out, dtype=dtype, casting=casting)

# Install into your CuPy-compatible namespace
cupy_fake.hstack = _hstack
cupy_fake.vstack = _vstack
#cupy_fake.stack = _stack
#cupy_fake.concatenate = _concatenate

##########################################################################
# Wrappers for array creation functions to handle positional dtype argument
# CuPy: zeros(shape, dtype, order) - dtype can be positional
# DPNP: zeros(shape, dtype=None, order='C') - dtype must be keyword

def _cupy_zeros(shape, dtype=None, order='C'):
    """Wrapper to match CuPy's zeros signature"""
    return dpnp.zeros(shape, dtype=dtype, order=order)

cupy_fake.zeros = _cupy_zeros

##########################################################################

# section to support DPNP zeros_like() API, when np.ndarray is passed as
# argument. Works with cupy but not with dpnp. Hence the patch.
# "zeros_like" entry in the attributes is removed to support the following

def _norm_order(order):
    # dpnp supports 'C'/'F'; treat CuPy's 'K'/'A' as 'C'
    return 'C' if order in (None, 'K', 'A') else order

def _shape_dtype_from(a, shape=None, dtype=None):
    if shape is None:
        try:
            shape = a.shape
        except Exception:
            shape = np.asarray(a).shape
    if dtype is None:
        try:
            dtype = a.dtype
        except Exception:
            dtype = np.asarray(a).dtype
    # ensure plain Python ints for dpnp
    shape = tuple(int(s) for s in shape)
    return shape, np.dtype(dtype)


def _zeros_like(a, dtype=None, order='K', subok=False, shape=None):
    if isinstance(a, np.ndarray):
        shape, dtype = _shape_dtype_from(a, shape, dtype)
        return dpnp.zeros(shape, dtype=dtype, order=_norm_order(order))
    return dpnp.zeros_like(a, dtype=dtype, order=_norm_order(order))

def _empty_like(a, dtype=None, order='K', subok=False, shape=None):
    if isinstance(a, np.ndarray):
        shape, dtype = _shape_dtype_from(a, shape, dtype)
        return dpnp.empty(shape, dtype=dtype, order=_norm_order(order))
    return dpnp.empty_like(a, dtype=dtype, order=_norm_order(order))

cupy_fake.zeros_like = _zeros_like
cupy_fake.empty_like = _empty_like

# ##########################################################################
# # # SECTION: Monkey-patch NumPy to handle dpnp arrays
# # Implement __array_function__ protocol for DPNP arrays
# # This allows NumPy functions to work with DPNP arrays like CuPy does
# #  When PySCF code calls numpy.zeros_like(dpnp_array), create dpnp array instead
# #
# # # Errors:
# # #   File "/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/scf/hf.py", line 273, in _kernel
# # #     mo_occ = mf.get_occ(mo_energy, mo_coeff)
# # #   File "/home/abagusetty/gpu4pyscf-testing/mygpu4pyscf_pip_aurora/lib/python3.10/site-packages/pyscf/scf/uhf.py", line 292, in get_occ
# # #     mo_occ = numpy.zeros_like(mo_energy)
# # #   File "/home/abagusetty/gpu4pyscf-testing/mygpu4pyscf_pip_aurora/lib/python3.10/site-packages/numpy/_core/numeric.py", line 128, in zeros_like
# # #     res = empty_like(
# # #   File "/home/abagusetty/gpu4pyscf-testing/dpnp/dpnp/dpnp_array.py", line 142, in __array__
# # #     raise TypeError(
# # # TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.asnumpy()` to construct a NumPy array explicitly.

# _original_dpnp_array_function = getattr(dpnp.ndarray, '__array_function__', None)

# def _dpnp_array_function(self, func, types, args, kwargs):
#     """
#     Implement NumPy's __array_function__ protocol for DPNP.
#     Routes numpy.zeros_like, etc. to dpnp equivalents.
#     """
#     # Map NumPy functions to DPNP equivalents
#     HANDLED_FUNCTIONS = {
#         np.zeros_like: dpnp.zeros_like,
#         np.empty_like: dpnp.empty_like,
#         np.ones_like: dpnp.ones_like,
#         np.full_like: dpnp.full_like,
#     }

#     if func in HANDLED_FUNCTIONS:
#         return HANDLED_FUNCTIONS[func](*args, **kwargs)

#     # Fallback to original implementation if it exists
#     if _original_dpnp_array_function is not None:
#         return _original_dpnp_array_function(self, func, types, args, kwargs)

#     # If we can't handle it, return NotImplemented so NumPy tries other methods
#     return NotImplemented

# # Monkey-patch DPNP's ndarray class
# dpnp.ndarray.__array_function__ = _dpnp_array_function

# ##########################################################################

#Issue[CLOSED]: https://github.com/IntelPython/dpnp/issues/2566
# There is a difference in behaviors with cupy.allclose and dpnp.allclose.
# dpnp.allclose doesnt work with scalars given the tight restrictions.
# To navigate this is the workarond:

# --- cupy.allclose that accept scalars gracefully ---
def _cupy_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # 1) Both are plain scalars -> NumPy scalar path
    if np.isscalar(a) and np.isscalar(b):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    a_is_dp = isinstance(a, dpnp.ndarray)
    b_is_dp = isinstance(b, dpnp.ndarray)
    a_is_np = isinstance(a, np.ndarray)
    b_is_np = isinstance(b, np.ndarray)

    # 2) Any NumPy array present → compare on host
    if (a_is_np or b_is_np) and not (a_is_dp and b_is_dp):
        # pull dpnp operand to host only if needed
        if a_is_dp and b_is_np:
            return bool(np.allclose(a.asnumpy(), b, rtol=rtol, atol=atol, equal_nan=equal_nan))
        if a_is_np and b_is_dp:
            return bool(np.allclose(a, b.asnumpy(), rtol=rtol, atol=atol, equal_nan=equal_nan))
        # both numpy
        return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    # 3) Both dpnp arrays → device path
    if a_is_dp and b_is_dp:
        return bool(dpnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    # # 4) Mixed array + scalar → route to numpy (safe & simple)
    # return bool(np.allclose(
    #     a.asnumpy() if a_is_dp else a,
    #     b.asnumpy() if b_is_dp else b,
    #     rtol=rtol, atol=atol, equal_nan=equal_nan
    # ))

# Override the earlier attribute that pointed to dpnp
cupy_fake.allclose = _cupy_allclose

##########################################################################

# [WORKAROUND], np.allclose(A,B). When A or B is a dpnp-array and an other
# one is an numpy.ndarray. Where as cupy-array is not an issue with np.allclose

_numpy_allclose_original = np.allclose

def _numpy_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Wrapper for numpy.allclose that handles dpnp arrays.
    Converts dpnp arrays to numpy arrays when detected in either argument.
    """
    a_is_dpnp = isinstance(a, dpnp.ndarray)
    b_is_dpnp = isinstance(b, dpnp.ndarray)

    # If either argument is a dpnp array, convert to numpy
    if a_is_dpnp or b_is_dpnp:
        a_numpy = a.asnumpy() if a_is_dpnp else a
        b_numpy = b.asnumpy() if b_is_dpnp else b
        return _numpy_allclose_original(a_numpy, b_numpy, rtol=rtol, atol=atol, equal_nan=equal_nan)

    # Otherwise, use original numpy.allclose
    return _numpy_allclose_original(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

# Monkey-patch numpy.allclose
np.allclose = _numpy_allclose


# [WORKAROUND], np.einsum(inputs). Error such as below:
# Traceback (most recent call last):
#   File "/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/./gpu4pyscf/scf/tests/test_fermi_smearing.py", line 57, in test_df_uhf_gradient
#     gpu_mf = mol.UHF().to_gpu().density_fit().smearing(sigma=0.1).run()
#   File "/home/abagusetty/gpu4pyscf-testing/mygpu4pyscf_pip_aurora/lib/python3.10/site-packages/pyscf/lib/misc.py", line 638, in run
#     self.kernel(*args)
#   File "/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/scf/hf.py", line 360, in scf
#     _kernel(mf, mf.conv_tol, mf.conv_tol_grad,
#   File "/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/scf/hf.py", line 196, in _kernel
#     dm0 = mf.get_init_guess(mol, mf.init_guess)
#   File "/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/lib/dpnp_helper.py", line 352, in filter_ret
#     ret = fn(*args, **kwargs)
#   File "/home/abagusetty/gpu4pyscf-testing/mygpu4pyscf_pip_aurora/lib/python3.10/site-packages/pyscf/scf/uhf.py", line 847, in get_init_guess
#     nelec =(numpy.einsum('ij,ji', dm[0], s).real,
#   File "/home/abagusetty/gpu4pyscf-testing/mygpu4pyscf_pip_aurora/lib/python3.10/site-packages/numpy/_core/einsumfunc.py", line 1423, in einsum
#     return c_einsum(*operands, **kwargs)
#   File "/home/abagusetty/gpu4pyscf-testing/dpnp/dpnp/dpnp_array.py", line 144, in __array__
#     raise TypeError(
# TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.asnumpy()` to construct a NumPy array explicitly.
#
# ISSUE: When GPU arrays (cupy/dpnp) are passed to numpy.einsum(), cupy is file because of the support for
# __array_function__, but not with DPNP arrays since it is more restrictive with implicit conversions.

# WORKAROUND to patch numpy.einsum to handle dpnp.ndarrays
_original_numpy_einsum = np.einsum

def _numpy_einsum_with_dpnp(*args, **kwargs):
    """Wrapper for numpy.einsum that handles dpnp arrays"""
    # Check if any args are dpnp arrays
    has_dpnp = any(isinstance(arg, dpnp.ndarray) for arg in args if hasattr(arg, '__class__'))

    if has_dpnp:
        # Convert all arrays to dpnp and use dpnp.einsum
        converted_args = []
        for arg in args:
            if isinstance(arg, str):  # subscript string
                converted_args.append(arg)
            elif isinstance(arg, np.ndarray) and not isinstance(arg, dpnp.ndarray):
                converted_args.append(dpnp.asarray(arg))
            else:
                converted_args.append(arg)
        result = dpnp.einsum(*converted_args, **kwargs)
        # Return as dpnp array (will be converted by decorator if needed)
        return result
    else:
        # All numpy, use original
        return _original_numpy_einsum(*args, **kwargs)

# Monkey-patch numpy.einsum
np.einsum = _numpy_einsum_with_dpnp


# WORKAROUND to patch numpy.dot to handle dpnp.ndarrays
_original_numpy_dot = np.dot

def _numpy_dot_with_dpnp(*args, **kwargs):
    """Wrapper for numpy.dot that handles dpnp arrays"""
    # Check if any args are dpnp arrays
    has_dpnp = any(isinstance(arg, dpnp.ndarray) for arg in args if hasattr(arg, '__class__'))

    if has_dpnp:
        # Convert all arrays to dpnp and use dpnp.dot
        converted_args = []
        for arg in args:
            if isinstance(arg, str):  # subscript string
                converted_args.append(arg)
            elif isinstance(arg, np.ndarray) and not isinstance(arg, dpnp.ndarray):
                converted_args.append(dpnp.asarray(arg))
            else:
                converted_args.append(arg)
        result = dpnp.dot(*converted_args, **kwargs)
        # Return as dpnp array (will be converted by decorator if needed)
        return result
    else:
        # All numpy, use original
        return _original_numpy_dot(*args, **kwargs)

# Monkey-patch numpy.dot
np.dot = _numpy_dot_with_dpnp

##########################################################################

# This is a workaround to address the issue[OPEN]: https://github.com/IntelPython/dpnp/issues/2622

_original_getitem = getattr(dpnp.ndarray, "__getitem__", None)
def _to_device_index(x):
    """
    Convert supported host-side indexers into dpnp device arrays when appropriate.
    Only converts integer/bool lists/tuples/ndarrays; leaves slices/ints/... alone.
    """
    # Already a device array -> good
    if isinstance(x, dpnp.ndarray) and x.dtype.kind in ("b", "i", "u"):
        return x

    # Pure Python lists/tuples -> try to see if they are integer/bool-like
    if isinstance(x, (list, tuple)):
        # Heuristic: accept nested sequences of ints/bools
        def _all_int_bool(seq):
            for el in seq:
                if isinstance(el, (list, tuple, np.ndarray, dpnp.ndarray)):
                    if not _all_int_bool(el):
                        return False
                elif not isinstance(el, (bool, int, np.bool_, np.integer)):
                    return False
            return True
        if _all_int_bool(x):
            return dpnp.asarray(x, dtype=dpnp.intp)  # or bool_ when you detect bools
        return x  # not an int/bool indexer -> leave it

    # NumPy array -> move to device if integer/bool typed
    if isinstance(x, np.ndarray) and x.dtype.kind in ("b", "i", "u"):
        return dpnp.asarray(x)

    # Everything else unchanged (slice, int, None, Ellipsis, dpnp float arrays, etc.)
    return x

def _safe_getitem(self, key):
    """
    Normalize the key so that any advanced indexing arrays are device arrays.
    """
    if _original_getitem is None:
        raise AttributeError("__getitem__ not found on dpnp.ndarray")

    # Normalize to tuple for uniform handling
    if not isinstance(key, tuple):
        key = (key,)

    # Convert each component of the index if needed
    fixed = []
    for k in key:
        fixed.append(_to_device_index(k))

    return _original_getitem(self, tuple(fixed))

# Monkeypatch dpnp.ndarray
dpnp.ndarray.__getitem__ = _safe_getitem

##########################################################################

# To address issue related to passing
# ```
#   File "/home/abagusetty/gpu4pyscf-testing/gpu4pyscf/gpu4pyscf/pbc/gto/int1e.py", line 146, in generate_shl_pairs
#     ijsh = ijsh[cp.tril_indices(ish1-ish0)]
#   File "/home/abagusetty/gpu4pyscf-testing/dpnp/dpnp/dpnp_iface_indexing.py", line 2444, in tril_indices
#     tri_ = dpnp.tri(
#   File "/home/abagusetty/gpu4pyscf-testing/dpnp/dpnp/dpnp_iface_arraycreation.py", line 3702, in tri
#     raise TypeError(f"`N` must be a integer data type, but got {type(N)}")
# TypeError: `N` must be a integer data type, but got <class 'numpy.int64'>
# ```
# --- cupy.tril_indices shim: accept numpy.int64 etc. ---
def _cupy_tril_indices(n, k=0, m=None):
    n = int(n)
    k = int(k)
    m = None if m is None else int(m)
    return dpnp.tril_indices(n, k=k, m=m)

cupy_fake.tril_indices = _cupy_tril_indices

##########################################################################

import sys
from types import ModuleType

class _LazyModule(ModuleType):
    """
    A module that defers importing the real implementation until first attribute access.
    This avoids circular imports during package initialization.
    """
    def __init__(self, name, loader_func):
        super().__init__(name)
        self._loader_func = loader_func
        self._loaded = False
        self._real_module = None
        self.__path__ = []

    def _load(self):
        if not self._loaded:
            self._real_module = self._loader_func()
            self._loaded = True
        return self._real_module

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        real = self._load()
        if real is None:
            raise AttributeError(f"module has no attribute '{name}'")
        return getattr(real, name)

    def __dir__(self):
        real = self._load()
        if real is None:
            return []
        return dir(real)


def _load_onemkl_lapack():
    """Lazy loader - only called when cusolver attributes are actually accessed"""
    try:
        from gpu4pyscf.lib import onemkl_lapack
        return onemkl_lapack
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import onemkl_lapack: {e}")
        return None


def _setup_cupy_backends():
    """Setup cupy_backends mock for Intel GPU with lazy loading"""

    if 'cupy_backends' in sys.modules:
        return

    # Create module hierarchy immediately (no imports needed here)
    cupy_backends = ModuleType('cupy_backends')
    cupy_backends.__path__ = []

    cuda = ModuleType('cupy_backends.cuda')
    cuda.__path__ = []
    cupy_backends.cuda = cuda

    libs = ModuleType('cupy_backends.cuda.libs')
    libs.__path__ = []
    cuda.libs = libs

    # Mock cublas (just constants, no lazy loading needed)
    cublas = ModuleType('cupy_backends.cuda.libs.cublas')
    cublas.CUBLAS_FILL_MODE_LOWER = 0
    cublas.CUBLAS_FILL_MODE_UPPER = 1
    cublas.CUBLAS_OP_N = 0
    cublas.CUBLAS_OP_T = 1
    cublas.CUBLAS_OP_C = 2

    # Create lazy cusolver - will load onemkl_lapack on first attribute access
    cusolver = _LazyModule('cupy_backends.cuda.libs.cusolver', _load_onemkl_lapack)

    # Assign to module hierarchy
    libs.cusolver = cusolver
    libs.cublas = cublas

    # Register all modules in sys.modules
    sys.modules['cupy_backends'] = cupy_backends
    sys.modules['cupy_backends.cuda'] = cuda
    sys.modules['cupy_backends.cuda.libs'] = libs
    sys.modules['cupy_backends.cuda.libs.cusolver'] = cusolver
    sys.modules['cupy_backends.cuda.libs.cublas'] = cublas

    # Also register gpu4pyscf.lib.cusolver as lazy alias
    gpu4pyscf_cusolver = _LazyModule('gpu4pyscf.lib.cusolver', _load_onemkl_lapack)
    sys.modules['gpu4pyscf.lib.cusolver'] = gpu4pyscf_cusolver

_setup_cupy_backends()
del _setup_cupy_backends


##########################################################################
# the below needs to be uncommented for scf/tests/test_fermi_smearing.py
# scf/tests/test_soscf.py: test_with_df, test_secondary_auxbasis

# ROBUST DPNP strides patch - auto-detects byte vs element strides
# https://github.com/IntelPython/dpnp/issues/2640

_original_dpnp_strides_property = dpnp.ndarray.strides

def _get_strides_in_bytes(self):
    """
    Get array strides in bytes (like NumPy/CuPy) instead of elements (DPNP default).

    Auto-detects whether DPNP is returning byte or element strides by checking
    if the reported strides are consistent with the array shape and itemsize.
    """
    raw_strides = _original_dpnp_strides_property.fget(self)

    if raw_strides is None or len(self.shape) == 0:
        return raw_strides

    itemsize = self.dtype.itemsize

    # For contiguous C-order array, last dimension stride should equal itemsize
    # Calculate expected minimum stride (accounting for size-1 dimensions)
    min_expected_stride = itemsize

    # Check if raw_strides look like they're already in bytes
    # Heuristic: if smallest stride >= itemsize, likely already bytes
    min_stride = min(raw_strides) if raw_strides else 0

    if min_stride >= itemsize:
        # Strides are likely already in bytes
        # This happens for some DPNP bugs with size-1 dimensions

        # Additional check: for size-1 dimensions, all strides should be itemsize
        # if the array is contiguous
        has_size1_dims = sum(1 for dim in self.shape if dim == 1)

        if has_size1_dims >= 2:  # e.g., shape (N, 1, 1)
            # For shape like (18, 1, 1), contiguous strides should be (8, 8, 8)
            # If DPNP reports (64, 64, 64), it's a bug - normalize it
            if all(s > itemsize for s in raw_strides) and len(set(raw_strides)) == 1:
                # All strides are identical and > itemsize - likely DPNP bug
                # Normalize to itemsize for size-1 dimensions
                return tuple(itemsize for _ in raw_strides)

        # Otherwise, assume already in bytes, return as-is
        return raw_strides

    # Strides look like element strides - multiply by itemsize
    byte_strides = tuple(stride * itemsize for stride in raw_strides)
    return byte_strides

# Replace the strides property
dpnp.ndarray.strides = property(_get_strides_in_bytes)

##########################################################################
