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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# modified by Xiaojie Wu <wxj6000@gmail.com>; Zhichen Pu <hoshishin@163.com>


import dpctl
from dpctl import SyclEvent
import time

################################################################################

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

libgpu.sycl_get_free_memory.argtypes = []
libgpu.sycl_get_free_memory.restype = ctypes.c_size_t

# Bind to sycl_queue_synchronize(void*)
libgpu.sycl_queue_synchronize.argtypes = [ctypes.c_void_p]
libgpu.sycl_queue_synchronize.restype = None

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
