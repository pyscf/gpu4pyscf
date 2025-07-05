# Copyright 2025 The PySCF Developers. All Rights Reserved.
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


from concurrent.futures import ThreadPoolExecutor
import functools
import cupy as cp
import numpy as np
from pyscf.lib import prange
from gpu4pyscf.lib.memcpy import p2p_transfer
from gpu4pyscf.__config__ import num_devices

__all__ = [
    'run', 'map', 'reduce', 'array_reduce', 'array_broadcast', 'lru_cache'
]

def run(func, args=(), kwargs={}, non_blocking=False):
    '''Execute a function on each GPU.

    Kwargs:
        non_blocking: If `True`, functions are executed in parallel using multi-threads.
    '''
    if num_devices == 1:
        return [func(*args, *kwargs)]

    cp.cuda.Stream.null.synchronize()

    def proc(device_id):
        with cp.cuda.Device(device_id):
            return func(*args, **kwargs)

    if not non_blocking:
        return [proc(i) for i in range(num_devices)]

    with ThreadPoolExecutor(max_workers=num_devices) as ex:
        futures = [ex.submit(proc, i) for i in range(num_devices)]
        return [fut.result() for fut in futures]

def map(func, tasks, args=(), kwargs={}, schedule='dynamic') -> list:
    '''Distributes tasks to multiple GPU devices for parallel computation.

    Kwargs:
        schedule: controls how the tasks are distributed. Can be 'static' or 'dynamic'.
            If 'static', tasks are distributed in the round-robin fashion;
            If 'dynamic', tasks are scheduled dynamically, with better load balance.
    '''
    if num_devices == 1:
        return [func(t, *args, *kwargs) for t in tasks]

    tasks = list(enumerate(tasks))
    result = [None] * len(tasks)

    def consumer():
        if schedule == 'dynamic':
            stream = cp.cuda.stream.get_current_stream()
            while tasks:
                try:
                    key, t = tasks.pop()
                except IndexError:
                    return
                result[key] = func(t, *args, **kwargs)
                stream.synchronize()
        else:
            device_id = cp.cuda.device.get_device_id()
            for key, t in tasks[device_id::num_devices]:
                result[key] = func(t, *args, **kwargs)

    run(consumer, non_blocking=True)
    return result

def reduce(func, tasks, args=(), kwargs={}, schedule='dynamic'):
    '''Processes tasks on multiple GPU devices and returns the sum of the results.
    '''
    result = map(func, tasks, args, kwargs)
    dtype = cp.result_type(*result)
    if num_devices == 1:
        out = result[0].astype(dtype=dtype, copy=False)
        for r in result[1:]:
            out += r
        return out

    groups = [None] * num_devices
    for r in result:
        device_id = r.device.id
        if groups[device_id] is None:
            groups[device_id] = r.astype(dtype, copy=False)
        else:
            groups[device_id] += r

    for i in num_devices:
        if groups[i] is None:
            groups[i] = cp.zeros(result[0].shape, dtype=dtype)
    return array_reduce(groups, inplace=True)

def array_broadcast(a):
    '''Broadcast a cupy ndarray to all devices, return a list of cupy ndarrays.
    '''
    if num_devices == 1:
        return [a]

    out = [None] * num_devices
    out[0] = a

    # Tree broadcast
    step = num_devices >> 1
    while step > 0:
        for device_id in range(0, num_devices, 2*step):
            if device_id + step < num_devices:
                with cp.cuda.Device(device_id+step):
                    out[device_id+step] = dst = cp.empty_like(a)
                    p2p_transfer(dst, a)
        step >>= 1
    return out

def array_reduce(array_list, inplace=False):
    '''The sum of cupy ndarrays from all devices to device 0.
    '''
    assert len(array_list) == num_devices
    if num_devices == 1:
        return array_list[0]

    a0 = array_list[0]
    out_shape = a0.shape
    size = a0.size
    dtype = a0.dtype
    assert all(x.dtype == dtype for x in array_list)

    array_list = list(array_list)
    for device_id in range(num_devices):
        with cp.cuda.Device(device_id):
            if inplace or device_id % 2 == 1:
                array_list[device_id] = array_list[device_id].ravel()
            else:
                array_list[device_id] = array_list[device_id].copy().ravel()

    blksize = 1024*1024*1024 // dtype.itemsize # 1GB
    # Tree-reduce
    step = 1
    while step < num_devices:
        for device_id in range(0, num_devices, 2*step):
            if device_id + step < num_devices:
                with cp.cuda.Device(device_id):
                    dst = array_list[device_id]
                    src = array_list[device_id+step]
                    buf = cp.empty_like(dst[:blksize])
                    for p0, p1 in prange(0, size, blksize):
                        dst[p0:p1] += p2p_transfer(buf[:p1-p0], src[p0:p1])
        step *= 2
    return array_list[0].reshape(out_shape)

def lru_cache(size):
    '''LRU cache for multiple devices'''
    def to_cache(fn):
        @functools.lru_cache(size)
        def fn_with_device_id(device_id, *args, **kwargs):
            return fn(*args, **kwargs)
        @functools.wraps(fn)
        def fn_on_device(*args, **kwargs):
            device_id = cp.cuda.Device().id
            return fn_with_device_id(device_id, *args, **kwargs)
        return fn_on_device
    return to_cache
