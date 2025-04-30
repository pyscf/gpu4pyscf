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

import numpy as np
import cupy
from cupyx import profiler
from gpu4pyscf.lib.cutensor import contract

print('benchmarking tensor contraction')
a = cupy.random.random([512,512,512])
b = cupy.random.random([512,512])
perf = profiler.benchmark(contract, ('ijk,lk->ijl', a, b), n_repeat=20, n_warmup=3)
flops = 2*np.prod(a.shape) * b.shape[0]
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

print('benchmarking tensor contraction with stride')
a0 = a[64:480,:,64:480]
b0 = b[:,64:480]
perf = profiler.benchmark(contract, ('ijk,lk->ijl', a0, b0), n_repeat=20, n_warmup=3)
flops = 2*np.prod(a0.shape) * b0.shape[0]
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

print('benchmarking tensor contraction with stride')
a0 = a[64:480,:,:128]
b0 = b[:,64:480]
perf = profiler.benchmark(contract, ('kji,lk->ijl', a0, b0), n_repeat=20, n_warmup=3)
flops = 2*np.prod(a0.shape) * b0.shape[0]
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

print('benchmarking tensor contraction with stride')
a0 = cupy.random.random([320,128*128])
b0 = cupy.random.random([320,128*128])
perf = profiler.benchmark(contract, ('jk,jk->j', a0, b0), n_repeat=20, n_warmup=3)
flops = a0.nbytes/1e9
print(flops/perf.gpu_times.mean(), 'GFLOPS')

perf = profiler.benchmark(cupy.sum, (a0,), n_repeat=20, n_warmup=3)
flops = a0.nbytes/1e9
print(flops/perf.gpu_times.mean(), 'GFLOPS')

@cupy.fuse()
def _contract(a0):
    c = a0 * a0
    return cupy.sum(c, axis=-1)
perf = profiler.benchmark(_contract, (a0,), n_repeat=20, n_warmup=0)
print(perf.gpu_times)
flops = a0.nbytes/1e9
print(flops/perf.gpu_times.mean(), 'GFLOPS')

a0 = cupy.random.random([20,320,320])
b0 = cupy.random.random([320,54])
perf = profiler.benchmark(contract, ('ijk,jo->iok', a0, b0), n_repeat=20, n_warmup=3)
flops = 20*320*320*54/1e9
print(flops/perf.gpu_times.mean(), 'GFLOPS')

perf = profiler.benchmark(cupy.dot, (b0.T, a0), n_repeat=20, n_warmup=3)
print(flops/perf.gpu_times.mean(), 'GFLOPS')

import cupy as cp
from cupy.cuda import cublas
import ctypes
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas #NOQA

libcublas = ctypes.CDLL('libcublas.so')
_handle = device.get_cublas_handle()

print(cupy.matmul(b0.T,a0).shape)
#handle = cublas.create()
perf = profiler.benchmark(cupy.matmul, (b0.T,a0), n_repeat=20, n_warmup=3)
print(flops/perf.gpu_times.mean(), 'GFLOPS')
