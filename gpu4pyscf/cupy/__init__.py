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

# # ── Fix dpnp .data.ptr not accounting for USM offsets on views ──
# # Workaround for issue: https://github.com/IntelPython/dpnp/issues/2781

# _dpnp_array_cls = dpnp.dpnp_array.dpnp_array
# _orig_data_fget = _dpnp_array_cls.__dict__['data'].fget

# class _OffsetMemory:
#     __slots__ = ('_base', '_byte_offset')
#     def __init__(self, base_memory, byte_offset):
#         self._base = base_memory
#         self._byte_offset = byte_offset
#     @property
#     def ptr(self):
#         return self._base.ptr + self._byte_offset
#     def __getattr__(self, name):
#         return getattr(self._base, name)

# def _fixed_data(self):
#     orig = _orig_data_fget(self)
#     iface = self.__sycl_usm_array_interface__
#     offset = iface.get('offset', 0)
#     if offset != 0:
#         base_ptr = iface['data'][0]
#         if orig.ptr == base_ptr:
#             return _OffsetMemory(orig, offset * self.itemsize)
#     return orig

# _dpnp_array_cls.data = property(_fixed_data)

########################################################################################

# this entire class and the below methods are needed as a work-around
# to support code like
# `out=cupy.ndarray((comp,nao_sub,ip1-ip0), memptr=buf.data))`
# where the `memptr` is an argument not supported under dpnp.
class _CuPyNdarrayMeta(ABCMeta):
    def __call__(cls, shape, dtype=np.float64, memptr=None):
        if memptr is not None and hasattr(memptr, "get_array"):
            memptr = memptr.get_array()
        if isinstance(shape, (tuple, list)):
            shape = tuple(int(s) for s in shape)
        else:
            shape = (int(shape),)
        if memptr is not None:
            return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
        return dpnp.ndarray(shape, dtype=dtype)

    def __instancecheck__(cls, obj):
        if isinstance(obj, dpnp.ndarray):
            return True
        if _DPNP_ARRAY_IMPL and isinstance(obj, _DPNP_ARRAY_IMPL):
            return True
        return False

    def __subclasscheck__(cls, sub):
        try:
            bases = [dpnp.ndarray]
            if _DPNP_ARRAY_IMPL:
                bases.append(_DPNP_ARRAY_IMPL)
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
# Simple cupy shims — no CPArrayWithTag unwrapping needed since
# CPArrayWithTag is now a dpnp.ndarray subclass.

cupy_fake.asarray = dpnp.asarray
cupy_fake.einsum  = dpnp.einsum
cupy_fake.asnumpy = dpnp.asnumpy

####################################################

# dot — keep the shape-mismatch fixup for ndarray.dot(out=...) edge case
_original_dpnp_dot = dpnp.dot
_original_ndarray_dot = dpnp.ndarray.dot

def _ndarray_dot_method(self, b, out=None):
    if out is None:
        return _original_ndarray_dot(self, b, out=None)

    result = _original_ndarray_dot(self, b, out=None)

    if result.shape != out.shape:
        if result.size == out.size:
            result = result.squeeze()
            if result.shape != out.shape:
                result = result.reshape(out.shape)
        else:
            raise ValueError(f"Cannot fit result {result.shape} into {out.shape}")

    out[:] = result
    return out

dpnp.ndarray.dot = _ndarray_dot_method
cupy_fake.dot = _original_dpnp_dot

####################################################

# Memory Pool Stub (dpnp doesn't have memory pools)
class _DummyMemoryPool:
    """No-op memory pool stub for dpnp (which has no memory pool concept)"""

    def free_all_blocks(self):
        pass

    def free_all_free(self):
        pass

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0

    def n_free_blocks(self):
        return 0

    def set_limit(self, size=None, fraction=None):
        pass

    def get_limit(self):
        return 0

    def free_bytes(self):
        return 0

_dummy_pool = _DummyMemoryPool()

def _get_default_memory_pool():
    return _dummy_pool

def _set_allocator(allocator=None):
    pass

cupy_fake.get_default_memory_pool = _get_default_memory_pool
cupy_fake.set_allocator = _set_allocator

##########################################################################

cupy_fake.array = dpnp.array

# Populate other dpnp functions as cupy attributes
for attr in [
        "append", "max", "linalg", "concatenate", "zeros", "ones",
        "empty", "eye", "view", "empty_like", "copyto", "cumsum", "any", "matmul",
        "vstack", "full", "arange", "stack", "expand_dims", "unique", "double", "sign",
        "argsort", "count_nonzero", "where", "split", "take", "tril", "log",
        "complex128", "uint8", "int32", "int64", "float32", "float64", "ravel", "random", "sum", "exp",
        "outer", "ix_", "pi", "square", "multiply", "diag_indices", "repeat", "diag",
        "tril_indices_from", "ceil", "newaxis", "ascontiguousarray", "nonzero",
        "array_equal", "isinf", "isnan", "dtype", "asfortranarray", "abs", "shape",
        "argmax"
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

if hasattr(cupy_fake, 'cuda'):
    cupy_fake.cuda.PinnedMemoryPool = _DummyMemoryPool

sys.modules["cupy"] = cupy_fake

#####################################################################

# [WORKAROUND]: To address indexing np.ndarray in tuples, list
# Issue: https://github.com/IntelPython/dpnp/issues/2622

_original_setitem = dpnp.ndarray.__setitem__
def safe_setitem(self, key, value):
    """Handle list/array indexing that DPNP doesn't support natively."""
    def _convert_index(k):
        if isinstance(k, list):
            return dpnp.asarray(k, dtype=dpnp.intp)
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

def _dpnp_get(self, order='C'):
    host = self.asnumpy()
    if order == 'C':
        return np.ascontiguousarray(host)
    if order == 'F':
        return np.asfortranarray(host)
    if order == 'A':
        if host.flags['F_CONTIGUOUS'] and not host.flags['C_CONTIGUOUS']:
            return np.asfortranarray(host)
        return np.ascontiguousarray(host)
    if order == 'K':
        return np.array(host, order='K', copy=False)
    return np.ascontiguousarray(host)

dpnp.dpnp_array.dpnp_array.get = _dpnp_get

##########################################################################

# hstack/vstack: cast np.ndarray inputs to dpnp (cupy does this, dpnp doesn't)

def _to_dpnp_seq(seq):
    out = []
    for s in seq:
        if isinstance(s, (np.ndarray, np.generic)) and not isinstance(s, dpnp.ndarray):
            out.append(dpnp.asarray(s))
        else:
            out.append(s)
    return out

def _hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = _to_dpnp_seq(tup)
    return dpnp.hstack(arrs, dtype=dtype, casting=casting)

def _vstack(tup, *, dtype=None, casting="same_kind"):
    arrs = _to_dpnp_seq(tup)
    return dpnp.vstack(arrs, dtype=dtype, casting=casting)

cupy_fake.hstack = _hstack
cupy_fake.vstack = _vstack

##########################################################################
# Wrappers for array creation functions to handle positional dtype argument

def _cupy_zeros(shape, dtype=None, order='C'):
    return dpnp.zeros(shape, dtype=dtype, order=order)

cupy_fake.zeros = _cupy_zeros

##########################################################################

# zeros_like / empty_like: handle np.ndarray input (works with cupy, not dpnp)

def _norm_order(order):
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

##########################################################################

# cupy.allclose that accepts scalars (dpnp.allclose doesn't)
# Issue: https://github.com/IntelPython/dpnp/issues/2566

def _cupy_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if np.isscalar(a) and np.isscalar(b):
        return bool(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    a_is_dp = isinstance(a, dpnp.ndarray)
    b_is_dp = isinstance(b, dpnp.ndarray)
    a_is_np = isinstance(a, np.ndarray)
    b_is_np = isinstance(b, np.ndarray)

    if (a_is_np or b_is_np) and not (a_is_dp and b_is_dp):
        if a_is_dp and b_is_np:
            return bool(np.allclose(a.asnumpy(), b, rtol=rtol, atol=atol, equal_nan=equal_nan))
        if a_is_np and b_is_dp:
            return bool(np.allclose(a, b.asnumpy(), rtol=rtol, atol=atol, equal_nan=equal_nan))
        return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    if a_is_dp and b_is_dp:
        return bool(dpnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))

cupy_fake.allclose = _cupy_allclose

##########################################################################

# dpnp.sqrt doesn't handle Python scalars
# Issue: https://github.com/IntelPython/dpnp/issues/2566

_orig_dpnp_sqrt = dpnp.sqrt
_SCALAR_TYPES = frozenset({int, float, complex, bool})

def _patched_dpnp_sqrt(x, **kwargs):
    if type(x) in _SCALAR_TYPES:
        x = dpnp.array(x)
    return _orig_dpnp_sqrt(x, **kwargs)

dpnp.sqrt = _patched_dpnp_sqrt
cupy_fake.sqrt = _patched_dpnp_sqrt

##########################################################################

# Patch numpy.einsum to handle dpnp arrays
_original_numpy_einsum = np.einsum

def _numpy_einsum_with_dpnp(*args, **kwargs):
    has_dpnp = any(isinstance(arg, dpnp.ndarray) for arg in args if hasattr(arg, '__class__'))
    if has_dpnp:
        converted_args = []
        for arg in args:
            if isinstance(arg, str):
                converted_args.append(arg)
            elif isinstance(arg, np.ndarray) and not isinstance(arg, dpnp.ndarray):
                converted_args.append(dpnp.asarray(arg))
            else:
                converted_args.append(arg)
        return dpnp.einsum(*converted_args, **kwargs)
    else:
        return _original_numpy_einsum(*args, **kwargs)

np.einsum = _numpy_einsum_with_dpnp

# Patch numpy.dot to handle dpnp arrays
_original_numpy_dot = np.dot

def _numpy_dot_with_dpnp(*args, **kwargs):
    has_dpnp = any(isinstance(arg, dpnp.ndarray) for arg in args if hasattr(arg, '__class__'))
    if has_dpnp:
        converted_args = []
        for arg in args:
            if isinstance(arg, str):
                converted_args.append(arg)
            elif isinstance(arg, np.ndarray) and not isinstance(arg, dpnp.ndarray):
                converted_args.append(dpnp.asarray(arg))
            else:
                converted_args.append(arg)
        return dpnp.dot(*converted_args, **kwargs)
    else:
        return _original_numpy_dot(*args, **kwargs)

np.dot = _numpy_dot_with_dpnp

##########################################################################

# Workaround for __getitem__ with host-side indexers
# Issue: https://github.com/IntelPython/dpnp/issues/2622

_original_getitem = getattr(dpnp.ndarray, "__getitem__", None)

def _to_device_index(x):
    if isinstance(x, dpnp.ndarray) and x.dtype.kind in ("b", "i", "u"):
        return x
    if isinstance(x, (list, tuple)):
        def _all_int_bool(seq):
            for el in seq:
                if isinstance(el, (list, tuple, np.ndarray, dpnp.ndarray)):
                    if not _all_int_bool(el):
                        return False
                elif not isinstance(el, (bool, int, np.bool_, np.integer)):
                    return False
            return True
        if _all_int_bool(x):
            return dpnp.asarray(x, dtype=dpnp.intp)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind in ("b", "i", "u"):
        return dpnp.asarray(x)
    return x

def _safe_getitem(self, key):
    if _original_getitem is None:
        raise AttributeError("__getitem__ not found on dpnp.ndarray")
    if not isinstance(key, tuple):
        return _original_getitem(self, _to_device_index(key))
    fixed = tuple(_to_device_index(k) for k in key)
    return _original_getitem(self, fixed)

dpnp.ndarray.__getitem__ = _safe_getitem

##########################################################################

# cupy.tril_indices: accept numpy.int64 etc.
def _cupy_tril_indices(n, k=0, m=None):
    n = int(n)
    k = int(k)
    m = None if m is None else int(m)
    return dpnp.tril_indices(n, k=k, m=m)

cupy_fake.tril_indices = _cupy_tril_indices

##########################################################################

class _LazyModule(types.ModuleType):
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
    try:
        from gpu4pyscf.lib import onemkl_lapack
        return onemkl_lapack
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import onemkl_lapack: {e}")
        return None


def _setup_cupy_backends():
    if 'cupy_backends' in sys.modules:
        return

    cupy_backends = types.ModuleType('cupy_backends')
    cupy_backends.__path__ = []

    cuda = types.ModuleType('cupy_backends.cuda')
    cuda.__path__ = []
    cupy_backends.cuda = cuda

    libs = types.ModuleType('cupy_backends.cuda.libs')
    libs.__path__ = []
    cuda.libs = libs

    cublas = types.ModuleType('cupy_backends.cuda.libs.cublas')
    cublas.CUBLAS_FILL_MODE_LOWER = 0
    cublas.CUBLAS_FILL_MODE_UPPER = 1
    cublas.CUBLAS_OP_N = 0
    cublas.CUBLAS_OP_T = 1
    cublas.CUBLAS_OP_C = 2

    cusolver = _LazyModule('cupy_backends.cuda.libs.cusolver', _load_onemkl_lapack)

    libs.cusolver = cusolver
    libs.cublas = cublas

    sys.modules['cupy_backends'] = cupy_backends
    sys.modules['cupy_backends.cuda'] = cuda
    sys.modules['cupy_backends.cuda.libs'] = libs
    sys.modules['cupy_backends.cuda.libs.cusolver'] = cusolver
    sys.modules['cupy_backends.cuda.libs.cublas'] = cublas

    gpu4pyscf_cusolver = _LazyModule('gpu4pyscf.lib.cusolver', _load_onemkl_lapack)
    sys.modules['gpu4pyscf.lib.cusolver'] = gpu4pyscf_cusolver

_setup_cupy_backends()
del _setup_cupy_backends

##########################################################################

# Redefined with actual SYCL memory reporting
class _DummyMemoryPool:
    """Memory pool stub for dpnp that reports actual SYCL memory usage."""

    def free_all_blocks(self):
        pass

    def free_all_free(self):
        pass

    def used_bytes(self):
        try:
            from . import cuda
            return cuda.get_total_memory() - cuda.get_free_memory()
        except Exception:
            return 0

    def free_bytes(self):
        return 0

    def total_bytes(self):
        try:
            from . import cuda
            return cuda.get_total_memory()
        except Exception:
            return 0

    def n_free_blocks(self):
        return 0

    def set_limit(self, size=None, fraction=None):
        pass

    def get_limit(self):
        return 0
