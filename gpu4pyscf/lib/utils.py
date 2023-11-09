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

import functools
import cupy

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

def to_cpu(method):
    # Search for the class in pyscf closest to the one defined in gpu4pyscf
    for pyscf_cls in method.__class__.__mro__:
        if 'gpu4pyscf' not in pyscf_cls.__module__:
            break
    method = method.view(pyscf_cls)
    keys = []
    for cls in pyscf_cls.__mro__[:-1]:
        if hasattr(cls, '_keys'):
            keys.extend(cls._keys)
    if keys:
        keys = set(keys).intersection(method.__dict__)

    for key in keys:
        val = getattr(method, key)
        if isinstance(val, cupy.ndarray):
            setattr(method, key, cupy.asnumpy(val))
        elif hasattr(val, 'to_cpu'):
            setattr(method, key, val.to_cpu())
    return method

def to_gpu(method, device=None):
    return method

@property
def device(obj):
    if 'gpu4pyscf' in obj.__class__.__module__:
        return 'gpu'
    else:
        return 'cpu'
