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
