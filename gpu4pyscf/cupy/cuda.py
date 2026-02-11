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

import dpctl
from dpctl import SyclEvent
import time
import ctypes, os

# Load your shared lib (adjust path if needed)
lib_path = os.path.join(os.path.dirname(__file__), "../lib/libgint.so")
lib_path = os.path.abspath(lib_path)
libgpu = ctypes.CDLL(lib_path)

# Existing function to get default current queue
libgpu.sycl_get_queue_ptr.restype = ctypes.c_void_p

# New function to get nth queue
libgpu.sycl_get_queue_ptr_nth.argtypes = [ctypes.c_int]
libgpu.sycl_get_queue_ptr_nth.restype = ctypes.c_void_p

# Existing function to set device for current thread
libgpu.sycl_set_device.argtypes = [ctypes.c_int]
libgpu.sycl_set_device.restype = None

libgpu.sycl_get_device_id.restype = ctypes.c_int

libgpu.sycl_get_device_count.argtypes = []
libgpu.sycl_get_device_count.restype = ctypes.c_int

libgpu.sycl_get_total_memory.argtypes = []
libgpu.sycl_get_total_memory.restype = ctypes.c_size_t

libgpu.sycl_get_shared_memory.argtypes = []
libgpu.sycl_get_shared_memory.restype = ctypes.c_size_t

libgpu.sycl_get_free_memory.argtypes = []
libgpu.sycl_get_free_memory.restype = ctypes.c_size_t

# Bind to sycl_queue_synchronize(void*)
libgpu.sycl_queue_synchronize.argtypes = [ctypes.c_void_p]
libgpu.sycl_queue_synchronize.restype = None

# bind to sycl_memcpy
libgpu.sycl_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
libgpu.sycl_memcpy.restype  = None

class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, obj, owner):
        return self.fget(owner)

class Stream:
    def __init__(self, device_id=None):
        if device_id is not None:
            libgpu.sycl_set_device(device_id)
            ptr = libgpu.sycl_get_queue_ptr_nth(device_id)
            if ptr is None:
                raise ValueError(f"Invalid device_id {device_id} - out of range")
        else:
            ptr = libgpu.sycl_get_queue_ptr()

        self._ptr = ptr

    @property
    def ptr(self):
        return self._ptr

    def __int__(self):
        return self._ptr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def synchronize(self):
        libgpu.sycl_queue_synchronize(self._ptr)

    @classproperty
    def null(cls):
        return get_current_stream()

# --- CuPy-compatible stream namespace ---------------------------------
class _StreamNS:
    # expose the Stream class under cp.cuda.stream.Stream
    Stream = Stream

    @staticmethod
    def get_current_stream():
        return get_current_stream()

    # # optional: provide a convenient alias like CuPy's null stream
    # @property
    # def null(self):
    #     return Stream.null

# Expose as cp.cuda.stream
stream = _StreamNS()


# class Stream:
#     def __init__(self, device_id=None):
#         if device_id is not None:
#             # Optionally set the thread device ID if you want
#             libgpu.sycl_set_device(device_id)
#             ptr = libgpu.sycl_get_queue_ptr_nth(device_id)
#             if ptr is None:
#                 raise ValueError(f"Invalid device_id {device_id} - out of range")
#         else:
#             ptr = libgpu.sycl_get_queue_ptr()

#         self._ptr = ptr

#     @property
#     def ptr(self):
#         return self._ptr

#     def __int__(self):
#         return self._ptr

#     def __enter__(self):
#         # Push stream context if needed
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Pop stream context if needed
#         pass

#     def synchronize(self):
#         """Wait for all operations in the stream to finish."""
#         libgpu.sycl_queue_synchronize(self._ptr)

def _init_streams(devices):
    # devices: list of device IDs (ints)
    # Create a Stream for each device id
    return [Stream(device_id=dev) for dev in devices]

def get_current_stream():
    # Default Stream for current default device (no device_id passed)
    return Stream()

# Class-level property injection
#Stream.null = staticmethod(get_current_stream)

def get_device_count():
    return libgpu.sycl_get_device_count()

def get_total_memory():
    return libgpu.sycl_get_total_memory()

def get_shared_memory():
    return libgpu.sycl_get_shared_memory()

def get_free_memory():
    return libgpu.sycl_get_free_memory()

################################################################################

# # Cache all available SYCL devices
# _cached_sycl_devices = dpctl.get_devices()

# class Stream:
#     def __init__(self, queue=None):
#         from dpctl._sycl_device_factory import _cached_default_device as get_default_cached_device
#         from dpctl._sycl_queue_manager import get_device_cached_queue

#         if queue is not None:
#             self.queue = queue
#             self.dev = queue.get_sycl_device()
#         else:
#             self.dev = get_default_cached_device()
#             self.queue = get_device_cached_queue(self.dev)

#     def addressof_ref(self):
#         return self.queue.addressof_ref()

#     def is_in_order(self):
#         return self.queue.is_in_order()

#     def __enter__(self):
#         # Optionally push stream to a global context
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Optionally pop from context
#         pass

# def _init_streams(devices):
#     return [Stream(dpctl.SyclQueue(dev, property='in_order')) for dev in devices]

# def get_current_stream():
#     return Stream()

class Device:
    def __init__(self, device=None):
        if device is None:
            # Use current thread's device (do not change anything)
            self._id = libgpu.sycl_get_device_id()
        elif isinstance(device, int):
            count = libgpu.sycl_get_device_count()
            if device < 0 or device >= count:
                raise ValueError(f"Device index {device} out of range. Available devices: {count}")
            libgpu.sycl_set_device(device)
            self._id = device
        else:
            raise TypeError("device must be None or an integer device ID")

    @classmethod
    def get_device_id(cls) -> int:
        return int(libgpu.sycl_get_device_id())

    @property
    def id(self):
        return self._id

    def __enter__(self):
        # Set the device for current thread context
        libgpu.sycl_set_device(self._id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Could restore previous device context if you wanted to track it
        pass

device = Device

# class Device:
#     def __init__(self, device=None):
#         if device is None:
#             self._dev = get_default_cached_device()
#         elif isinstance(device, SyclDevice):
#             self._dev = device
#         elif isinstance(device, str):
#             self._dev = SyclDevice(device)
#         elif isinstance(device, int):
#             try:
#                 self._dev = _cached_sycl_devices[device]
#             except IndexError:
#                 raise ValueError(f"Device index {device} out of range. Available devices: {len(_cached_sycl_devices)}")
#         else:
#             raise TypeError(
#                 "device must be None, a str filter selector, an int index, or a SyclDevice instance"
#             )
#     # def __init__(self, device=None):
#     #     if device is None:
#     #         self._dev = SyclDevice()
#     #     else:
#     #         self._dev = SyclDevice(device)

#     @property
#     def id(self):
#         return self._dev.get_device_id()

#     def __enter__(self):
#         # Optionally push this device context (e.g., set some global state)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         # Clean up or restore previous state if needed
#         pass

# class Event:
#     def __init__(self):
#         self._event = None
#         self._timestamp = None

#     def record(self, stream=None):
#         """Record the event using a SYCL in-order queue barrier."""
#         if stream is None:
#             stream = get_current_stream()
#         queue = stream.queue

#         # Record timestamp (optional, for elapsed_time)
#         self._timestamp = time.perf_counter()

#         # Record an actual event using a barrier (works only on in_order queues)
#         self._event = queue.submit_barrier()

#     def synchronize(self):
#         """Wait for the event to complete."""
#         if isinstance(self._event, SyclEvent):
#             self._event.wait()

#     def query(self):
#         """Returns True if the event has completed, False otherwise."""
#         if self._event is None:
#             return False
#         return self._event.get_info("command_execution_status") == "complete"

#     def elapsed_time(self, end_event=None):
#         """Estimate elapsed wall-clock time (in milliseconds) between this and another event."""
#         if self._timestamp is None:
#             return None
#         end_time = (
#             end_event._timestamp if isinstance(end_event, Event) and end_event._timestamp
#             else time.perf_counter()
#         )
#         return (end_time - self._timestamp) * 1000.0  # milliseconds

class Event:
    def __init__(self):
        self._handle = None
        self._timestamp = None

    def record(self, stream=None):
        # Note: stream is unused since queue context is thread-bound
        self._timestamp = time.perf_counter()
        self._handle = libgpu.sycl_record_event()

    def synchronize(self):
        if self._handle:
            libgpu.sycl_wait_event(self._handle)
            self._handle = None  # avoid reuse

    def __del__(self):
        self.synchronize()

    # def elapsed_time(self, other):
    #     """Return elapsed time (in milliseconds) between two events."""
    #     if self._timestamp is None or other._timestamp is None:
    #         raise RuntimeError("Both events must be recorded.")
    #     return (other._timestamp - self._timestamp) * 1000  # ms

    # def get_event(self):
    #     return self._event

def get_elapsed_time(start_event, end_event):
    """Returns elapsed time between two recorded events in milliseconds.

    Arguments:
        start_event (Event): The starting event.
        end_event (Event): The ending event.

    Returns:
        float: Elapsed time in milliseconds.
    """
    if not isinstance(start_event, Event) or not isinstance(end_event, Event):
        raise TypeError("Both arguments must be Event instances.")

    if start_event._timestamp is None or end_event._timestamp is None:
        raise ValueError("Both events must be recorded before calling get_elapsed_time.")

    return (end_event._timestamp - start_event._timestamp) * 1000.0  # milliseconds

#############################################################
# runtime shim

def _addr_of(obj) -> int:
    """Return an integer address for ints, NumPy/DPNP arrays, or USM objects."""
    # Raw int or c_void_p
    if isinstance(obj, int):
        return obj
    if isinstance(obj, ctypes.c_void_p):
        return int(obj.value)

    # dpnp/dpctl USM arrays expose __sycl_usm_array_interface__
    ai = getattr(obj, "__sycl_usm_array_interface__", None)
    if isinstance(ai, dict) and "data" in ai:
        return int(ai["data"][0])

    # NumPy ndarray
    ai = getattr(obj, "__array_interface__", None)
    if isinstance(ai, dict) and "data" in ai:
        return int(ai["data"][0])

    # dpctl MemoryUSM* objects are int()-able
    try:
        return int(obj)
    except Exception:
        pass

    # NumPy ctypes bridge
    if hasattr(obj, "ctypes") and hasattr(obj.ctypes, "data"):
        try:
            return int(obj.ctypes.data)
        except Exception:
            pass

    raise TypeError(f"Cannot obtain address from object of type {type(obj)}")

class _Runtime:
    # ---- CUDA-compatible memcpy kind constants ----
    memcpyHostToHost     = 0
    memcpyHostToDevice   = 1
    memcpyDeviceToHost   = 2
    memcpyDeviceToDevice = 3
    memcpyDefault        = 4

    # Host allocation flags (CUDA compatibility)
    hostAllocMapped      = 0x02

    # Cache GPU devices
    _gpu_devices = None

    @classmethod
    def _get_gpu_devices(cls):
        """Get cached list of GPU devices."""
        if cls._gpu_devices is None:
            try:
                cls._gpu_devices = dpctl.get_devices(backend='level_zero', device_type='gpu')
            except Exception:
                # Fallback to any available GPU devices
                try:
                    cls._gpu_devices = dpctl.get_devices(device_type='gpu')
                except Exception:
                    cls._gpu_devices = dpctl.get_devices()
            if not cls._gpu_devices:
                cls._gpu_devices = dpctl.get_devices()
        return cls._gpu_devices

    @staticmethod
    def getDeviceCount() -> int:
        return get_device_count()

    @staticmethod
    def memGetInfo():
        """Return (free_memory, total_memory) tuple (CuPy-compatible)."""
        free_mem = get_free_memory()
        total_mem = get_total_memory()
        return (free_mem, total_mem)

    @staticmethod
    def memcpy(dst, src, nbytes, kind):
        n = int(nbytes)
        dst_addr = _addr_of(dst)
        src_addr = _addr_of(src)
        libgpu.sycl_memcpy(ctypes.c_void_p(dst_addr), ctypes.c_void_p(src_addr), ctypes.c_size_t(n))

    @staticmethod
    def getDeviceProperties(device_id: int) -> dict:
        """
        Return device properties dict compatible with CuPy/CUDA runtime.

        Maps SYCL device properties to CUDA-style property names.
        """
        devices = _Runtime._get_gpu_devices()

        if not devices or device_id < 0 or device_id >= len(devices):
            # Return default properties if device not found
            return {
                'totalGlobalMem': get_total_memory(),
                'sharedMemPerBlock': get_shared_memory(),
                'sharedMemPerBlockOptin': get_shared_memory(),
                'name': 'Unknown SYCL Device',
                'maxThreadsPerBlock': 1024,
                'maxWorkGroupSize': 1024,
                'maxComputeUnits': 1,
                'major': 8,
                'minor': 0,
                'warpSize': 32,
                'multiProcessorCount': 1,
            }

        dev = devices[device_id]

        # Get memory info
        total_mem = dev.global_mem_size
        local_mem = dev.local_mem_size

        # SYCL doesn't have direct equivalent to sharedMemPerBlockOptin
        # Use local_mem_size for both
        shared_mem = local_mem
        shared_mem_optin = local_mem

        # Get subgroup size (warp equivalent)
        try:
            warp_size = dev.sub_group_sizes[0] if dev.sub_group_sizes else 32
        except Exception:
            warp_size = 32

        # Build CUDA-compatible properties dict
        props = {
            # Memory properties
            'totalGlobalMem': total_mem,
            'sharedMemPerBlock': shared_mem,
            'sharedMemPerBlockOptin': shared_mem_optin,

            # Device info
            'name': dev.name,
            'maxThreadsPerBlock': dev.max_work_group_size,
            'maxWorkGroupSize': dev.max_work_group_size,
            'maxComputeUnits': dev.max_compute_units,

            # Placeholders for CUDA properties (approximate mappings)
            'major': 8,  # Fake compute capability for compatibility
            'minor': 0,
            'warpSize': warp_size,
            'multiProcessorCount': dev.max_compute_units,

            # Additional SYCL-specific info
            'localMemSize': local_mem,
            'globalMemSize': total_mem,
        }

        return props

    @staticmethod
    def deviceCanAccessPeer(src: int, dst: int) -> bool:
        """
        Check if device src can access memory on device dst.

        With SYCL USM (especially shared memory), P2P access is generally
        handled transparently by the runtime. Return True.
        """
        # With USM shared/host memory, cross-device access is handled by the runtime
        return True

runtime = _Runtime()

#############################################################
# this section support the usecase of cupy.cuda.alloc_pinned_memory() APIs
# using SYCL

import numpy as _np
import dpctl, dpctl.memory as dpmem

def _queue_from_native():
    """Recreate the SYCL queue we use in native code; fallback to default."""
    try:
        q_ptr = int(libgpu.sycl_get_queue_ptr())
        # Some dpctl versions expose _create_from_ptr; fall back to default queue if absent.
        return dpctl.SyclQueue._create_from_ptr(q_ptr)  # type: ignore[attr-defined]
    except Exception:
        return dpctl.SyclQueue()

# ---- CuPy-compatible pinned allocator ----
def alloc_pinned_memory(nbytes, flags=None):
    """
    CuPy API: cupy.cuda.alloc_pinned_memory(nbytes) -> buffer-like object.
    We return a USM allocation that NumPy can view via buffer=...
    By default we use USM Shared (closest to cudaHostAllocMapped semantics).
    """
    nbytes = int(nbytes)
    q = _queue_from_native()

    # If caller ever passes flags and DOESN'T request mapping, pick Host instead.
    # This keeps compatibility with code that might someday pass hostAllocMapped.
    mapped = True
    if flags is not None:
        try:
            mapped = bool(flags & runtime.hostAllocMapped)
        except Exception:
            mapped = True

    Mem = dpmem.MemoryUSMShared if mapped else dpmem.MemoryUSMHost
    return Mem(nbytes, queue=q)

#############################################################
