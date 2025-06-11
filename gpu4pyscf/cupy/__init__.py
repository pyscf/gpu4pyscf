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

# v3
import sys
import types
import numpy as np
import dpnp
import dpctl.tensor as dpt

# --- Simplified CuPy ndarray wrapper ---
class CuPyNdarrayWrapper:
    def __call__(self, shape, dtype=np.float64, memptr=None):
        if memptr is not None:
            # Unwrap DataWithPtr to get the actual usm_ndarray
            if isinstance(memptr, DataWithPtr):
                memptr = memptr._usm_array
            elif hasattr(memptr, 'get_array'):
                memptr = memptr.get_array()
            return dpnp.ndarray(shape, dtype=dtype, buffer=memptr)
        else:
            return dpnp.ndarray(shape, dtype=dtype)

    def __instancecheck__(self, instance):
        return isinstance(instance, dpnp.dpnp_array.dpnp_array)

    def __subclasscheck__(self, subclass):
        return issubclass(subclass, dpnp.dpnp_array.dpnp_array)

# --- Patch dpnp_array to have .data return underlying usm_ndarray ---
# Mimic CuPy-style .data.ptr → get_array()._pointer
class DataWithPtr:
    def __init__(self, usm_array):
        self._usm_array = usm_array

    @property
    def ptr(self):
        return self._usm_array._pointer  # same as cupy.data.ptr

    def __getattr__(self, name):
        # Forward other attribute accesses to the underlying usm_ndarray
        return getattr(self._usm_array, name)

    def __array__(self):
        return np.asarray(self._usm_array)  # numpy compatibility

@property
def dpnp_data_property(self):
    """Return USM array wrapped with .ptr access."""
    return DataWithPtr(self.get_array())

# Patch it into dpnp_array
dpnp.dpnp_array.dpnp_array.data = dpnp_data_property    

# # Add .ptr to usm_ndarray if not already present
# if not hasattr(dpt.usm_ndarray, "ptr"):
#     @property
#     def ptr(self):
#         return self._pointer  # Expose raw device pointer

#     dpt.usm_ndarray.ptr = ptr

# --- Setup fake cupy module ---
cupy_fake = types.ModuleType("cupy")
cupy_fake.ndarray = CuPyNdarrayWrapper()
cupy_fake.array = dpnp.array

# Populate other dpnp functions as cupy attributes
for attr in [
        "max", "dot", "linalg", "concatenate", "asarray", "zeros", "ones",
        "empty", "eye", "einsum", "hstack", "view", "empty_like", "copyto",
        "vstack", "full", "arange", "asnumpy", "stack", "expand_dims", "unique", "double",
        "sqrt"
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

def _dpnp_set(self, host_array):
    self[...] = host_array

dpnp.dpnp_array.dpnp_array.set = _dpnp_set

cupy_fake.get = get

# Register fake cupy module in sys.modules
sys.modules["cupy"] = cupy_fake


_original_setitem = dpnp.ndarray.__setitem__
def safe_setitem(self, key, value):
    if isinstance(key, tuple):
        key = tuple(dpnp.asarray(k) if isinstance(k, np.ndarray) else k for k in key)
    return _original_setitem(self, key, value)
dpnp.ndarray.__setitem__ = safe_setitem



# def patched_getitem(self, key):
#     try:
#         # Try standard DPNP indexing
#         return self._array_obj[key]
#     except IndexError as e:
#         # Fallback to host-side NumPy for fancy indexing
#         if "Only integers, slices" in str(e):
#             return dpnp.array(np.asarray(self._array_obj)[key])
#         else:
#             raise

# # Patch dpnp_array.__getitem__
# dpnp.dpnp_array.dpnp_array.__getitem__ = patched_getitem

# # v2
# print("Inside custom cupy/__init__.py")

# import sys
# import types
# import ctypes
# import numpy as np
# import dpnp
# import dpctl
# import dpctl.memory as dpmem
# import dpctl.tensor as dpt

# # --- Combined constructor + type-check wrapper ---
# class CuPyNdarrayWrapper:
#     def __call__(self, shape, dtype=np.float64, memptr=None):
#         if memptr is not None:
#             # Expecting memptr to be a dpctl.memory.MemoryUSMDevice or similar
#             if isinstance(memptr, dpctl.memory.MemoryUSMDevice):
#                 usm_arr = dpt.usm_ndarray(shape=shape, dtype=dtype, buffer=memptr)
#                 return dpnp.asarray(usm_arr)
#             else:
#                 raise TypeError("memptr must be a dpctl.memory.MemoryUSMDevice object, not raw pointer")
#         else:
#             return dpnp.ndarray(shape, dtype=dtype)
#     # def __call__(self, shape, dtype=np.float64, memptr=None):
#     #     if memptr is not None:
#     #         itemsize = np.dtype(dtype).itemsize
#     #         strides = tuple(
#     #             s * itemsize for s in reversed(np.cumprod((1,) + shape[:-1])[::-1])
#     #         )

#     #         if isinstance(memptr, ctypes.c_void_p):
#     #             memptr = memptr.value
#     #         elif hasattr(memptr, "ptr"):  # e.g., MemoryPointer
#     #             memptr = int(memptr.ptr)
#     #         elif isinstance(memptr, np.ndarray):
#     #             memptr = memptr.ctypes.data

#     #         usm_arr = dpt.usm_ndarray(
#     #             shape=shape,
#     #             dtype=dtype,
#     #             buffer=memptr,
#     #             strides=strides
#     #         )
#     #         return dpnp.asarray(usm_arr)
#     #     else:
#     #         return dpnp.ndarray(shape, dtype=dtype)

#     def __instancecheck__(self, instance):
#         return isinstance(instance, dpnp.dpnp_array.dpnp_array)

#     def __subclasscheck__(self, subclass):
#         return issubclass(subclass, dpnp.dpnp_array.dpnp_array)

# # --- Set up fake `cupy` module ---
# cupy_fake = types.ModuleType("cupy")
# cupy_fake.ndarray = CuPyNdarrayWrapper()  # ✅ now both callable and isinstance()-friendly
# cupy_fake.array = dpnp.array

# # Populate other dpnp-based functionality
# for attr in [
#         "max", "dot", "linalg", "concatenate", "asarray", "zeros", "ones",
#         "empty", "eye", "einsum", "hstack", "view", "empty_like", "copyto",
#         "vstack", "full", "arange", "asnumpy", "stack", "expand_dims", "unique", "double"
# ]:
#     try:
#         setattr(cupy_fake, attr, getattr(dpnp, attr))
#     except AttributeError:
#         print(f"dpnp does not have {attr}, skipping.")

# # Optional get/set compatibility
# def get(x):
#     return dpnp.asnumpy(x)

# def _dpnp_set(self, host_array):
#     self[...] = host_array

# dpnp.dpnp_array.dpnp_array.set = _dpnp_set

# # def set(x, host_array):
# #     x[...] = host_array

# cupy_fake.get = get
# #cupy_fake.set = set

# # Optional: cupy.cuda submodule stub
# try:
#     from . import cuda
#     cupy_fake.cuda = cuda
# except ImportError as e:
#     print(f"Could not import .cuda: {e}")

# # Register in sys.modules
# sys.modules["cupy"] = cupy_fake

# # Mimic CuPy's .data.ptr structure on dpnp_array
# # class MemoryPointer:
# #     def __init__(self, ptr):
# #         self.ptr = ptr
# #     def __int__(self):
# #         return self.ptr


# @property
# def dpnp_data_property(self):
#     try:
#         iface = self.__sycl_usm_array_interface__
#         ptr = iface['data'][0]
#         nbytes = np.prod(self.shape) * self.dtype.itemsize
#         return dpmem.MemoryUSMDevice(ptr, nbytes, queue=dpctl.SyclQueue())
#     except Exception as e:
#         raise AttributeError(f"Cannot extract USM memory from dpnp_array: {e}")

# # @property
# # def dpnp_data_property(self):
# #     try:
# #         # Get raw USM pointer from DPNP array via __sycl_usm_array_interface__
# #         iface = self.__sycl_usm_array_interface__
# #         ptr = iface['data'][0]  # data is (ptr, read_only)
# #         return MemoryPointer(ptr)
# #     except AttributeError:
# #         raise AttributeError("dpnp_array does not expose USM pointer")

# # Patch dpnp_array with `.data` property
# dpnp.dpnp_array.dpnp_array.data = dpnp_data_property

# _original_setitem = dpnp.ndarray.__setitem__
# def safe_setitem(self, key, value):
#     if isinstance(key, tuple):
#         key = tuple(dpnp.asarray(k) if isinstance(k, np.ndarray) else k for k in key)
#     return _original_setitem(self, key, value)
# dpnp.ndarray.__setitem__ = safe_setitem


# print("Inside custom cupy/__init__.py")

# import sys
# import types
# import ctypes
# import numpy as np
# import dpnp
# import dpnp.dpnp_array
# import dpctl.tensor as dpt
# import dpctl

# # --- Custom ndarray constructor that supports memptr ---

# def cupy_ndarray(shape, dtype=np.float64, memptr=None):
#     if memptr is not None:
#         itemsize = np.dtype(dtype).itemsize
#         # Create C-style strides
#         strides = tuple(s * itemsize for s in reversed(np.cumprod((1,) + shape[:-1])[::-1]))

#         if isinstance(memptr, ctypes.c_void_p):
#             memptr = memptr.value
#         elif isinstance(memptr, np.ndarray):
#             memptr = memptr.ctypes.data
#         elif hasattr(memptr, 'ptr'):
#             memptr = int(memptr.ptr)

#         usm_arr = dpt.usm_ndarray(
#             shape=shape,
#             dtype=dtype,
#             buffer=memptr,
#             strides=strides,
#             usm_type="device",
#             queue=dpctl.SyclQueue()
#         )
#         return dpnp.asarray(usm_arr)
#     else:
#         return dpnp.ndarray(shape, dtype=dtype)

# # --- Create fake "cupy" module ---

# cupy_fake = types.ModuleType("cupy")
# cupy_fake.ndarray = cupy_ndarray
# print("Set cupy.ndarray as conditional constructor with memptr support")

# # Copy selected dpnp functions into cupy
# for attr in [
#     "max", "dot", "linalg", "concatenate", "asarray", "zeros", "ones", "array",
#     "empty", "eye", "einsum", "hstack", "view", "empty_like", "copyto", "vstack",
#     "full", "arange", "asnumpy", "stack", "expand_dims", "unique", "double"
# ]:
#     try:
#         setattr(cupy_fake, attr, getattr(dpnp, attr))
#         print(f"Set cupy.{attr} from dpnp.{attr}")
#     except AttributeError:
#         print(f"dpnp does not have {attr}, skipping.")

# # --- Add get() and set() mimicking CuPy behavior ---

# def get(x):
#     return dpnp.asnumpy(x)

# def set(x, host_array):
#     x[...] = host_array

# cupy_fake.get = get
# cupy_fake.set = set

# # --- Optional: Add cupy.cuda shim if available ---

# try:
#     from . import cuda
#     cupy_fake.cuda = cuda
# except ImportError as e:
#     print(f"Could not import .cuda: {e}")

# # --- Patch dpnp.ndarray to add get/set methods ---

# def _dpnp_get(self):
#     return dpnp.asnumpy(self)

# def _dpnp_set(self, host_array):
#     self[...] = host_array

# dpnp.ndarray.get = _dpnp_get
# dpnp.ndarray.set = _dpnp_set

# # --- Inject fake module into sys.modules ---

# print("Before sys.modules['cupy'] =", sys.modules.get("cupy", "NOT FOUND"))
# sys.modules["cupy"] = cupy_fake
# print("After sys.modules['cupy'] =", sys.modules["cupy"])


#v0
# print("Inside custom cupy/__init__.py")

# import sys
# import types
# import abc
# import dpnp
# import numpy as np
# import dpnp.dpnp_array
# import dpctl.tensor as dpt

# # Create a API specifically for `memptr` arg that is not
# # supported from DPNP APIs
# # class FakeCupyNdarray(dpnp.ndarray, abc.ABC):
# #     def get(self):
# #         return dpnp.asnumpy(self)

# #     def set(self, host_array):
# #         self[...] = host_array

# #     def __new__(cls, shape, dtype=np.float64, memptr=None):
# #         if memptr is None:
# #             obj = dpnp.ndarray.__new__(cls, shape, dtype=dtype)
# #         else:
# #             itemsize = np.dtype(dtype).itemsize
# #             strides = tuple(s * itemsize for s in reversed(np.cumprod((1,) + shape[:-1])[::-1]))
# #             if isinstance(memptr, ctypes.c_void_p):
# #                 memptr = memptr.value
# #             usm_arr = dpt.usm_ndarray(
# #                 shape=shape,
# #                 dtype=dtype,
# #                 buffer=memptr,
# #                 strides=strides,
# #                 usm_type="device",
# #                 queue=dpctl.SyclQueue()
# #             )
# #             obj = dpnp.asarray(usm_arr).view(cls)
# #         return obj

# # # Register dpnp array class as virtual subclass
# # FakeCupyNdarray.register(dpnp.dpnp_array.dpnp_array)

# # # Set up fake cupy module
# # cupy_fake = types.ModuleType("cupy")
# # cupy_fake.ndarray = FakeCupyNdarray

# class FakeCupyNdarray(dpnp.ndarray):
#     def get(self):
#         return dpnp.asnumpy(self)

#     def set(self, host_array):
#         self[...] = host_array

#     def __new__(cls, shape, dtype=np.float64, memptr=None):
#         if memptr is None:
#             # Regular dpnp allocation
#             obj = dpnp.ndarray.__new__(cls, shape, dtype=dtype)
#         else:
#             # Use USM pointer from memptr
#             itemsize = np.dtype(dtype).itemsize
#             strides = tuple(s * itemsize for s in reversed(np.cumprod((1,) + shape[:-1])[::-1]))
#             if isinstance(memptr, ctypes.c_void_p):
#                 memptr = memptr.value
#             usm_arr = dpt.usm_ndarray(
#                 shape=shape,
#                 dtype=dtype,
#                 buffer=memptr,
#                 strides=strides,
#                 usm_type="device",
#                 queue=dpctl.SyclQueue()
#             )
#             obj = dpnp.asarray(usm_arr).view(cls)
#         return obj

# # Create a new module object to act as "cupy"
# cupy_fake = types.ModuleType("cupy")


# # Populate it with selected dpnp functions
# for attr in ["ndarray", "max", "dot", "linalg", "concatenate", "asarray", "zeros", "ones", "array", "empty", "eye", "einsum", "hstack", "view", "empty_like", "copyto", "vstack", "full", "arange", "asnumpy", "stack", "expand_dims", "unique", "double"]:

#     try:
#         if attr == "ndarray":
#             setattr(cupy_fake, attr, FakeCupyNdarray)
#             print("Set cupy.ndarray to custom constructor with memptr support")
#         else:
#             setattr(cupy_fake, attr, getattr(dpnp, attr))
#             print(f"Set cupy.{attr} from dpnp.{attr}")
#     except AttributeError:
#         print(f"dpnp does not have {attr}, skipping.")

# # Define get() and set() methods that mimic CuPy behavior
# def _dpnp_get(self):
#     """Mimics CuPy's ndarray.get()"""
#     return dpnp.asnumpy(self)

# def _dpnp_set(self, host_array):
#     """Mimics CuPy's ndarray.set()"""
#     self[...] = host_array

# # Inject as methods on dpnp.ndarray
# dpnp.ndarray.get = _dpnp_get
# dpnp.ndarray.set = _dpnp_set

# # Also provide module-level get(x) and set(x, host_array) as alternatives
# def get(x):
#     return x.get() if isinstance(x, dpnp.ndarray) else x

# def set(x, host_array):
#     if isinstance(x, dpnp.ndarray):
#         x.set(host_array)
#     else:
#         raise TypeError(f"set() only supports dpnp arrays, got {type(x)}")

# cupy_fake.get = get
# cupy_fake.set = set

# # (Optional) add submodules like `cuda` if needed
# try:
#     from . import cuda
#     cupy_fake.cuda = cuda
# except ImportError as e:
#     print(f"Could not import .cuda: {e}")

# # Show before injecting
# print("Before sys.modules['cupy'] =", sys.modules.get("cupy", "NOT FOUND"))

# # Register this fake module
# sys.modules["cupy"] = cupy_fake

# # After injection
# print("After sys.modules['cupy'] =", sys.modules["cupy"])
# print("cupy.einsum =", getattr(sys.modules["cupy"], "einsum", "NOT FOUND"))
