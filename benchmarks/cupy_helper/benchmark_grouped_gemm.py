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

import cupy
from cupyx import profiler
from gpu4pyscf.lib.cupy_helper import grouped_dot, grouped_gemm

# ----------------- grouped dot ---------------------------
count = 108
m = n = 256
k = 128*128
As = []
Bs = []
for i in range(count):
    As.append(cupy.random.random([m,k]))
    Bs.append(cupy.random.random([n,k]))

print('---------- benchmarking ik,jk->ij ----------')
perf = profiler.benchmark(grouped_dot, (As, Bs), n_repeat=20, n_warmup=3)
flops = 2*m*n*k*count
print('grouped dot')
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

def grouped_cupy_dot(As,Bs):
    Cs = []
    for a, b in zip(As, Bs):
        Cs.append(cupy.dot(a,b.T))
    return Cs
perf = profiler.benchmark(grouped_cupy_dot, (As, Bs), n_repeat=20, n_warmup=3)
flops = 2*m*n*k*count
print('cupy dot')
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

# ------------------ grouped DGEMM -------------------------
print('---------- benchmarking ki,kj->ij ----------')
count = 200
m = 68
n = 128*128
k = 256
As = []
Bs = []
for i in range(count):
    As.append(cupy.random.random([k,n]))
    Bs.append(cupy.random.random([k,m]))

perf = profiler.benchmark(grouped_gemm, (Bs, As), n_repeat=20, n_warmup=3)
flops = 2*m*n*k*count
print('grouped gemm')
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')

def grouped_cupy_dot(As,Bs):
    Cs = []
    for a, b in zip(As, Bs):
        Cs.append(cupy.dot(b.T,a))
    return Cs
perf = profiler.benchmark(grouped_cupy_dot, (As, Bs), n_repeat=20, n_warmup=3)
flops = 2*m*n*k*count
print('cupy dot')
print(flops/perf.gpu_times.mean()/1024**3, 'GFLOPS')