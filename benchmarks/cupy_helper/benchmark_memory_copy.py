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

import numpy as np
import cupy as cp
from cupyx import profiler
from gpu4pyscf.lib.cupy_helper import copy_array

'''
Benchmark different ways of transfering data from pinned memory to device
'''

# Host array
host_array = cp.cuda.alloc_pinned_memory(512*512*512 * 8)
big_host_data = np.ndarray(512**3, dtype=cp.float64, buffer=host_array)
big_host_data = big_host_data.reshape(512,512,512)
big_host_data += np.random.rand(512,512,512)

# Device array
big_device_data = cp.empty_like(big_host_data)

# Create views on both arrays
host_view = big_host_data[:, 128:]  # Non-contiguous view on the host
device_view = big_device_data[:, 128:]  # Non-contiguous view on the device

print("Host View Shape:", host_view.shape)
print("Device View Shape:", device_view.shape)
'''
print("------ Benchmark device to host transfer ----------")
size = host_view.nbytes
perf_custom = profiler.benchmark(copy_array, (host_view, device_view), n_repeat=100, n_warmup=3)
t_kernel = perf_custom.gpu_times.mean()
bandwidth = size / t_kernel / 1e9
print('using custom function', t_kernel)
print(f"Effective PCIe Bandwidth: {bandwidth:.2f} GB/s")

def cupy_copy(c, out):
    out[:] = cp.asarray(c)
    return out
perf_cupy = profiler.benchmark(cupy_copy, (host_view, device_view), n_repeat=100, n_warmup=3)
t_kernel = perf_cupy.gpu_times.mean()
bandwidth = size / t_kernel / 1e9
print('using cupy function', t_kernel)
print(f"Effective PCIe Bandwidth: {bandwidth:.2f} GB/s")

print("------- Benchmark host to device transfer ---------")
size = host_view.nbytes
perf_custom = profiler.benchmark(copy_array, (device_view, host_view), n_repeat=100, n_warmup=3)
t_kernel = perf_custom.gpu_times.mean()
bandwidth = size / t_kernel / 1e9
print('using custom function', t_kernel)
print(f"Effective PCIe Bandwidth: {bandwidth:.2f} GB/s")

def cupy_copy(c, out):
    out[:] = c.get()
    return out
perf_cupy = profiler.benchmark(cupy_copy, (device_view, host_view), n_repeat=100, n_warmup=3)
t_kernel = perf_cupy.gpu_times.mean()
bandwidth = size / t_kernel / 1e9
print('using cupy function', t_kernel)
print(f"Effective PCIe Bandwidth: {bandwidth:.2f} GB/s")
'''
with cp.cuda.Device(0):
    a = cp.random.rand(512,512,512)
    device0_view = a[:,128:, 128:]
with cp.cuda.Device(1):
    b = cp.random.rand(512,512,512)
    device1_view = b[:,128:, 128:]
perf_cupy = profiler.benchmark(copy_array, (device0_view, device1_view), n_repeat=100, n_warmup=3)
t_kernel = perf_cupy.gpu_times.mean()
bandwidth = device0_view.nbytes / t_kernel / 1e9
print('using custom function', t_kernel)
print(f"Effective PCIe Bandwidth: {bandwidth:.2f} GB/s")

assert np.linalg.norm(device0_view.get() - device1_view.get()) < 1e-10

def cupy_copy(c, out):
    with cp.cuda.Device(out.device):
        out[:] = cp.asarray(c.get())
    return out
perf_cupy = profiler.benchmark(cupy_copy, (device0_view, device1_view), n_repeat=100, n_warmup=3)
t_kernel = perf_cupy.gpu_times.mean()
bandwidth = device0_view.nbytes / t_kernel / 1e9
print('using cupy function', t_kernel)
print(f"Effective PCIe Bandwidth: {bandwidth:.2f} GB/s")
