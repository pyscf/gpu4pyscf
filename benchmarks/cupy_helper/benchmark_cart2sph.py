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
from gpu4pyscf.lib.cupy_helper import cart2sph, cart2sph_cutensor

print('benchmarking cart2sph when ang=2')
a = cupy.random.random([512,6*128,512])
b = cupy.random.random([512,5*128,512])
perf_kernel = profiler.benchmark(cart2sph, (a,1,2,b), n_repeat=20, n_warmup=3)
perf_cutensor = profiler.benchmark(cart2sph_cutensor, (a,1,2), n_repeat=20, n_warmup=3)
t_kernel = perf_kernel.gpu_times.mean()
t_cutensor = perf_cutensor.gpu_times.mean()
print('kernel:', t_kernel)
print('cutensor:', t_cutensor)
print('memory bandwidth:',(a.nbytes+b.nbytes)/t_kernel/1024**3, 'GB/s')

print('benchmarking cart2sph when ang=3')
a = cupy.random.random([512,10*128,512])
b = cupy.random.random([512,7*128,512])
perf_kernel = profiler.benchmark(cart2sph, (a,1,3,b), n_repeat=20, n_warmup=3)
perf_cutensor = profiler.benchmark(cart2sph_cutensor, (a,1,3), n_repeat=20, n_warmup=3)
t_kernel = perf_kernel.gpu_times.mean()
t_cutensor = perf_cutensor.gpu_times.mean()
print('kernel:', t_kernel)
print('cutensor:', t_cutensor)
print('memory bandwidth:',(a.nbytes+b.nbytes)/t_kernel/1024**3, 'GB/s')

print('benchmarking cart2sph when ang=4')
a = cupy.random.random([512,15*128,512])
b = cupy.random.random([512,9*128,512])
perf_kernel = profiler.benchmark(cart2sph, (a,1,4,b), n_repeat=20, n_warmup=3)
perf_cutensor = profiler.benchmark(cart2sph_cutensor, (a,1,4), n_repeat=20, n_warmup=3)
t_kernel = perf_kernel.gpu_times.mean()
t_cutensor = perf_cutensor.gpu_times.mean()
print('kernel:', t_kernel)
print('cutensor:', t_cutensor)
print('memory bandwidth:',(a.nbytes+b.nbytes)/t_kernel/1024**3, 'GB/s')

print('benchmarking cart2sph when ang=5')
a = cupy.random.random([512,21*128,512])
b = cupy.random.random([512,11*128,512])
perf_kernel = profiler.benchmark(cart2sph, (a,1,5,b), n_repeat=20, n_warmup=3)
perf_cutensor = profiler.benchmark(cart2sph_cutensor, (a,1,5), n_repeat=20, n_warmup=3)
t_kernel = perf_kernel.gpu_times.mean()
t_cutensor = perf_cutensor.gpu_times.mean()
print('kernel:', t_kernel)
print('cutensor:', t_cutensor)
print('memory bandwidth:',(a.nbytes+b.nbytes)/t_kernel/1024**3, 'GB/s')

print('benchmarking cart2sph when ang=6')
a = cupy.random.random([512,28*128,512])
b = cupy.random.random([512,13*128,512])
perf_kernel = profiler.benchmark(cart2sph, (a,1,6,b), n_repeat=20, n_warmup=3)
perf_cutensor = profiler.benchmark(cart2sph_cutensor, (a,1,6), n_repeat=20, n_warmup=3)
t_kernel = perf_kernel.gpu_times.mean()
t_cutensor = perf_cutensor.gpu_times.mean()
print('kernel:', t_kernel)
print('cutensor:', t_cutensor)
print('memory bandwidth:',(a.nbytes+b.nbytes)/t_kernel/1024**3, 'GB/s')
