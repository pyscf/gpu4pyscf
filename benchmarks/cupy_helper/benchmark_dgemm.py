# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

import ctypes
import numpy as np
import cupy
from cupyx import profiler
from gpu4pyscf.lib.cupy_helper import libcupy_helper

def _dot(a, b):
    n = a.shape[0]
    c = cupy.empty([n,n])
    stream = cupy.cuda.get_current_stream()
    libcupy_helper.dgemm(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(a.data.ptr, ctypes.c_void_p),
        ctypes.cast(b.data.ptr, ctypes.c_void_p),
        ctypes.cast(c.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )

    return c

n = 2048
a = cupy.random.rand(n,n)
b = cupy.random.rand(n,n)

print(cupy.linalg.norm(_dot(a,b) - a.dot(b)))
assert cupy.linalg.norm(_dot(a,b) - a.dot(b)) < 1e-6

from cupyx import profiler
perf = profiler.benchmark(_dot, (a, b), n_repeat=20, n_warmup=3)
flops = 2*np.prod(a.shape) * b.shape[0]
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

from cupyx import profiler
perf = profiler.benchmark(cupy.dot, (a, b), n_repeat=20, n_warmup=3)
flops = 2*np.prod(a.shape) * b.shape[0]
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')
