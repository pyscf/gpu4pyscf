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

import sys
import functools
import cupy
import numpy
from pyscf import lib
from pyscf.lib import parameters as param

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
