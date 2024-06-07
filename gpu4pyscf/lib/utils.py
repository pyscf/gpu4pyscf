# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
import gpu4pyscf

def patch_cpu_kernel(cpu_kernel):
    '''Generate a decorator to patch cpu function to gpu function'''
    def patch(gpu_kernel):
        @functools.wraps(cpu_kernel)
        def hybrid_kernel(method, *args, **kwargs):
            if getattr(method, 'device', 'cpu') == 'gpu':
                return gpu_kernel(method, *args, **kwargs)
            else:
                return cpu_kernel(method, *args, **kwargs)
        hybrid_kernel.__package__ = 'gpu4pyscf'
        return hybrid_kernel
    return patch

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

        # A temporary CPU instance. This ensures to initialize private
        # attributes that are only available for CPU code.
        out = cls(omniobj)

    # Convert only the keys that are defined in the corresponding CPU class
    cls_keys = [getattr(cls, '_keys', ()) for cls in out.__class__.__mro__[:-1]]
    out_keys = set(out.__dict__).union(*cls_keys)
    # Only overwrite the attributes of the same name.
    keys = set(method.__dict__).intersection(out_keys)
    for key in keys:
        val = getattr(method, key)
        if isinstance(val, cupy.ndarray):
            val = val.get()
        elif hasattr(val, 'to_cpu'):
            val = val.to_cpu()
        setattr(out, key, val)
    out.reset()
    return out

def to_gpu(method, device=None):
    return method

@property
def device(obj):
    if 'gpu4pyscf' in obj.__class__.__module__:
        return 'gpu'
    else:
        return 'cpu'

#@patch_cpu_kernel(lib.misc.format_sys_info)
def format_sys_info():
    '''Format a list of system information for printing.'''
    from cupyx._runtime import get_runtime_info

    pyscf_info = lib.repo_info(pyscf.__file__)
    gpu4pyscf_info = lib.repo_info(os.path.join(__file__, '..', '..'))
    cuda_version = cupy.cuda.runtime.runtimeGetVersion()
    cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

    runtime_info = get_runtime_info()
    device_props = cupy.cuda.runtime.getDeviceProperties(0)
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
        f'GPU4PySCF {gpu4pyscf.__version__}',
        f'GPU4PySCF path  {gpu4pyscf_info["path"]}'
    ]
    if 'git' in pyscf_info:
        result.append(pyscf_info['git'])
    return result
