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

# import cupy

# num_devices = cupy.cuda.runtime.getDeviceCount()

# # TODO: switch to non_blocking stream (currently blocked by libxc)
# _streams = [None] * num_devices
# for device_id in range(num_devices):
#     with cupy.cuda.Device(device_id):
#         _streams[device_id] = cupy.cuda.stream.Stream(non_blocking=False)

# props = cupy.cuda.runtime.getDeviceProperties(0)
# GB = 1024*1024*1024
# min_ao_blksize = 256        # maxisum batch size of AOs
# min_grid_blksize = 128*128  # maximum batch size of grids for DFT
# ao_aligned = 32             # global AO alignment for slicing
# grid_aligned = 256          # 256 alignment for grids globally

# # Use smaller blksize for old gaming GPUs
# if props['totalGlobalMem'] < 16 * GB:
#     min_ao_blksize = 64
#     min_grid_blksize = 64*64

# # Use 90% of the global memory for CuPy memory pool
# mem_fraction = 0.9
# cupy.get_default_memory_pool().set_limit(fraction=mem_fraction)

# if props['sharedMemPerBlockOptin'] > 65536:
#     shm_size = props['sharedMemPerBlockOptin']
# else:
#     shm_size = props['sharedMemPerBlock']

# # Check P2P data transfer is available
# _p2p_access = True
# if num_devices > 1:
#     for src in range(num_devices):
#         for dst in range(num_devices):
#             if src != dst:
#                 can_access_peer = cupy.cuda.runtime.deviceCanAccessPeer(src, dst)
#                 _p2p_access &= can_access_peer

import dpctl
from gpu4pyscf.cupy.cuda import Stream  # avoids circular import of full Stream


# Get all available SYCL GPU devices
gpu_devices = dpctl.get_devices(backend='level_zero', device_type="gpu")
num_devices = len(gpu_devices)
if num_devices == 0:
    raise RuntimeError("No Intel GPU (Level Zero) devices found!")

# Initializes streams using helper
_streams = [Stream(device_id=i) for i in range(num_devices)]

props = {
    'multiProcessorCount': gpu_devices[0].max_compute_units
}

# Memory and alignment settings
GB = 1024 * 1024 * 1024
min_ao_blksize = 256         # max batch size of AOs
min_grid_blksize = 128 * 128 # max batch size of grids
ao_aligned = 32              # global AO alignment
grid_aligned = 256           # global grid alignment

# Adjust blksize for lower-memory GPUs
for i, dev in enumerate(gpu_devices):
    total_mem = dev.global_mem_size
    if total_mem < 16 * GB:
        min_ao_blksize = 64
        min_grid_blksize = 64 * 64

mem_fraction = 0.9
        
# Note: No CuPy-style memory pool setting in dpnp/dpctl,
# but memory usage can be tracked or controlled manually via USM if needed.

shm_size = gpu_devices[0].local_mem_size

# Check for peer-to-peer (P2P) access (not directly exposed in SYCL runtime)
# Assume it's handled by SYCL runtime — can't enforce manually via dpctl
_p2p_access = True
                
