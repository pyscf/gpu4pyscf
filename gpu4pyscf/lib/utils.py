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

import os
import sys
import time
import platform
import h5py
import functools
import cupy
import numpy
import scipy
import pyscf
from pyscf import lib
from pyscf.lib import parameters as param

__all__ = ['load_library', 'format_sys_info', 'to_cpu']

@functools.lru_cache
def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        raise

class _OmniObject:
    '''Class with default attributes. When accessing an attribute that is not
    initialized, a default value will be returned than raising an AttributeError.
    '''
    verbose = 0
    max_memory = param.MAX_MEMORY
    stdout = sys.stdout

    def __init__(self, default_factory=None):
        self._default = default_factory

    def __getattr__(self, key):
        return self._default

omniobj = _OmniObject()
omniobj.mol = omniobj
omniobj._scf = omniobj
omniobj.base = omniobj

def to_cpu(method, out=None):
    if method.__module__.startswith('pyscf'):
        return method

    if out is None:
        import pyscf

        if isinstance(method, (lib.SinglePointScanner, lib.GradScanner)):
            method = method.undo_scanner()

        from importlib import import_module
        mod = import_module(method.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, method.__class__.__name__)
        out = method.view(cls)

    cls_keys = set.union(*[getattr(cls, '_keys', ()) for cls in out.__class__.__mro__[:-1]])
    gpu_keys = set.union(*[getattr(cls, '_keys', ()) for cls in method.__class__.__mro__[:-1]])
    # Discards keys that are only defined in GPU classes
    discards = gpu_keys.difference(cls_keys)
    for k in discards:
        out.__dict__.pop(k, None)

    for key, val in method.__dict__.items():
        # Convert only the keys that are defined in the corresponding GPU class
        if key in cls_keys:
            if hasattr(val, 'to_cpu'):
                val = val.to_cpu()
            elif isinstance(val, cupy.ndarray):
                val = val.get()
        setattr(out, key, val)

    for key in ['_scf', '_numint']:
        val = getattr(method, key, None)
        if hasattr(val, 'to_cpu'):
            setattr(out, key, val.to_cpu())

    if hasattr(out, 'reset'):
        try:
            out.reset()
        except NotImplementedError:
            pass
    return out

def to_gpu(method, device=None):
    return method

@property
def device(obj):
    if 'gpu4pyscf' in obj.__class__.__module__:
        return 'gpu'
    else:
        return 'cpu'

def format_sys_info():
    '''Format a list of system information for printing.'''
    import gpu4pyscf    
    from gpu4pyscf.__config__ import num_devices, mem_fraction, props as device_props

    pyscf_info = lib.repo_info(pyscf.__file__)
    gpu4pyscf_info = lib.repo_info(os.path.join(__file__, '..', '..'))

    from importlib.util import find_spec
    has_dpctl = find_spec("dpctl")
    if not has_dpctl:
        from cupyx._runtime import get_runtime_info
        cuda_version = cupy.cuda.runtime.runtimeGetVersion()
        cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

        runtime_info = get_runtime_info()
        result = [
            f'System: {platform.uname()}  Threads {lib.num_threads()}',
            f'Python {sys.version}',
            f'numpy {numpy.__version__}  scipy {scipy.__version__}  '
            f'h5py {h5py.__version__}',
            f'Date: {time.ctime()}',
            f'PySCF version {pyscf.__version__}',
            f'PySCF path  {pyscf_info["path"]}',
            'CUDA Environment',
            f'    CuPy {runtime_info.cupy_version}',
            f'    CUDA Path {runtime_info.cuda_path}',
            f'    CUDA Build Version {runtime_info.cuda_build_version}',
            f'    CUDA Driver Version {runtime_info.cuda_driver_version}',
            f'    CUDA Runtime Version {runtime_info.cuda_runtime_version}',
            'CUDA toolkit',
            f'    cuSolver {runtime_info.cusolver_version}',
            f'    cuBLAS {runtime_info.cublas_version}',
            f'    cuTENSOR {runtime_info.cutensor_version}',
            'Device info',
            f'    Device name {device_props["name"]}',
            f'    Device global memory {device_props["totalGlobalMem"] / 1024**3:.2f} GB',
            f'    CuPy memory fraction {mem_fraction}',
            f'    Num. Devices {num_devices}',
            f'GPU4PySCF {gpu4pyscf.__version__}',
            f'GPU4PySCF path  {gpu4pyscf_info["path"]}'
        ]
    else:
        import dpnp, dpctl
        # DPCTL device info: pick default device or first GPU device
        try:
            device = dpctl.get_devices(device_type='gpu')[0]
        except IndexError:
            device = dpctl.get_default_device()  # fallback to any device

        # Get device properties
        dev_name = device.name
        dev_driver_version = device.driver_version if hasattr(device, 'driver_version') else 'Unknown'
#        dev_platform_version = device.platform.version if hasattr(device.platform, 'version') else 'Unknown'
        dev_global_mem_bytes = device.global_mem_size

        result = [
            f'System: {platform.uname()}  Threads {lib.num_threads()}',
            f'Python {sys.version}',
            f'numpy {numpy.__version__}  scipy {scipy.__version__}  '
            f'h5py {h5py.__version__}',
            f'Date: {time.ctime()}',
            f'PySCF version {pyscf.__version__}',
            f'PySCF path  {pyscf_info["path"]}',
            'SYCL / DPNP / DPCTL Environment',
            f'    Device name {dev_name}',
#            f'    Device platform version {dev_platform_version}',
            f'    Device driver version {dev_driver_version}',
            f'    Device max alloc size {dev_global_mem_bytes / 1024**3:.2f} GB',
            f'    DPNP version {dpnp.__version__}',
            f'    DPCTL version {dpctl.__version__}',
            f'    DPNP memory fraction {mem_fraction}',
            f'    Num. Devices {num_devices}',
            f'GPU4PySCF {gpu4pyscf.__version__}',
            f'GPU4PySCF path  {gpu4pyscf_info["path"]}'
        ]

    if 'git' in pyscf_info:
        result.append(pyscf_info['git'])
    return result

def splits_by_blocksize(cum, block_size):
    '''
    Given a cumulative array, split its indices so that each segment spans
    approximately a given block size.

    Returns:
        splits: split points in cum, starting with 0 and ending with len(cum)-1.
    '''
    bound = block_size
    tot = cum[-1]
    splits = [0]
    i = 0
    while bound < tot:
        i += max(numpy.searchsorted(cum[i:], bound, side='right') - 1, 1)
        splits.append(i)
        bound = cum[i] + block_size
    splits.append(len(cum) - 1)
    return splits
