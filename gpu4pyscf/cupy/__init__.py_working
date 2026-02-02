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

# --- Simplified cupy ndarray wrapper ---
# for eg. cupy.ndarray((comp,nao_sub,ip1-ip0), memptr=buf.data))
# to allow `memptr=` as args to cupy.ndarray construction to map
# to dpnp.ndarray
class CuPyNdarrayWrapper:
    def __call__(self, shape, dtype=np.float64, memptr=None):
        if memptr is not None:
            if hasattr(memptr, 'get_array'):
                memptr = memptr.get_array()
            return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
        else:
            return dpnp.ndarray(shape, dtype=dtype)

    def __instancecheck__(self, instance):
        return isinstance(instance, dpnp.dpnp_array.dpnp_array)

    def __subclasscheck__(self, subclass):
        return issubclass(subclass, dpnp.dpnp_array.dpnp_array)

# class CuPyNdarrayWrapper:
#     def __call__(self, shape, dtype=np.float64, memptr=None):
#         if memptr is not None:
#             if isinstance(memptr, DataWithPtr):
#                 memptr = memptr._usm_array
#             elif hasattr(memptr, 'get_array'):
#                 memptr = memptr.get_array()
#             return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
#         else:
#             return dpnp.ndarray(shape, dtype=dtype)

#     def __instancecheck__(self, instance):
#         import gpu4pyscf.lib.dpnp_helper as helper
#         return isinstance(instance, dpnp.dpnp_array.dpnp_array) or isinstance(instance, helper.DPNPArrayWithTag)

#     def __subclasscheck__(self, subclass):
#         return issubclass(subclass, dpnp.dpnp_array.dpnp_array)

# class CuPyNdarrayWrapper:
#     def __call__(self, shape, dtype=np.float64, memptr=None):
#         if memptr is not None:
#             # Unwrap DataWithPtr to get the actual usm_ndarray
#             if isinstance(memptr, DataWithPtr):
#                 memptr = memptr._usm_array
#             elif hasattr(memptr, 'get_array'):
#                 memptr = memptr.get_array()
#             return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
#         else:
#             return dpnp.ndarray(shape, dtype=dtype)

#     def __instancecheck__(self, instance):
#         return isinstance(instance, dpnp.dpnp_array.dpnp_array)

#     def __subclasscheck__(self, subclass):
#         return issubclass(subclass, dpnp.dpnp_array.dpnp_array)

# # --- Patch dpnp_array to have .data return underlying usm_ndarray ---
# # Mimic CuPy-style .data.ptr -> get_array()._pointer
# class DataWithPtr:
#     def __init__(self, usm_array):
#         self._usm_array = usm_array

#     @property
#     def ptr(self):
#         return self._usm_array._pointer  # same as cupy.data.ptr

#     def __getattr__(self, name):
#         # Forward other attribute accesses to the underlying usm_ndarray
#         return getattr(self._usm_array, name)

#     def __array__(self):
#         return np.asarray(self._usm_array)  # numpy compatibility

# @property
# def dpnp_data_property(self):
#     """Return USM array wrapped with .ptr access."""
#     return DataWithPtr(self.get_array())

# # Patch it into dpnp_array
# dpnp.dpnp_array.dpnp_array.data = dpnp_data_property    

# --- Setup fake cupy module ---
cupy_fake = types.ModuleType("cupy")
cupy_fake.ndarray = CuPyNdarrayWrapper()

def patched_cupy_array(a, *args, **kwargs):
    from gpu4pyscf.lib.dpnp_helper import DPNPArrayWithTag
    unwrapped_a = a.array if isinstance(a, DPNPArrayWithTag) else a
    # Drop copy=False if it causes problems with dpnp
    if isinstance(unwrapped_a, dpnp.ndarray) and kwargs.get("copy") is False:
        kwargs.pop("copy")  # Let dpnp handle default (copy=True)
    return dpnp.array(unwrapped_a, *args, **kwargs)
cupy_fake.array = patched_cupy_array


# Populate other dpnp functions as cupy attributes
for attr in [
        "max", "dot", "linalg", "concatenate", "asarray", "zeros", "ones",
        "empty", "eye", "einsum", "hstack", "view", "empty_like", "copyto",
        "vstack", "full", "arange", "asnumpy", "stack", "expand_dims", "unique", "double",
        "sqrt", "zeros_like", "argsort", "count_nonzero", "where", "split", "take", "log",
        "int32", "int64"
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
    
# Optional get/set compatibility
def get(x):
    return dpnp.asnumpy(x)
cupy_fake.get = get

# Register fake cupy module in sys.modules
sys.modules["cupy"] = cupy_fake


_original_setitem = dpnp.ndarray.__setitem__
def safe_setitem(self, key, value):
    if isinstance(key, tuple):
        key = tuple(dpnp.asarray(k) if isinstance(k, np.ndarray) else k for k in key)
    return _original_setitem(self, key, value)
dpnp.ndarray.__setitem__ = safe_setitem

# Add `.set()`, `.get()` method to dpnp_array to mimic CuPy behavior
def _dpnp_set(self, host_array):
    self[...] = host_array
dpnp.dpnp_array.dpnp_array.set = _dpnp_set

def _dpnp_get(self, order='C'):
    try:
        return dpnp.asnumpy(self, order=order)
    except TypeError:
        return dpnp.asnumpy(self)
dpnp.dpnp_array.dpnp_array.get = _dpnp_get

# # this is used to create a view() in DPNP since the functionality is
# # not yet supported: https://github.com/IntelPython/dpnp/issues/2486
# def dpnp_view_like(a, dtype):
#     return dpnp_array(
#         a.shape,
#         dtype=dtype,
#         buffer=a,
#         strides=a.strides,
#         usm_type=a.usm_type,
#         sycl_queue=a.sycl_queue,
#     )
# cupy_fake.dpnp_view_like = dpnp_view_like
