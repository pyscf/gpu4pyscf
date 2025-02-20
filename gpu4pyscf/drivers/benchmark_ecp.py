import os
import time
import numpy as np
import cupy as cp
from pyscf import gto, lib
from gpu4pyscf.gto.ecp import get_ecp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
basis = os.path.join(CURRENT_DIR, 'basis_vDZP_NWCHEM.dat')
ecp = os.path.join(CURRENT_DIR, 'ecp_vDZP_NWCHEM.dat')

lib.num_threads(8)
mol = gto.M(
    atom='../../benchmarks/molecules/organic/020_Vitamin_C.xyz',
    basis=basis,
    ecp=ecp)

runs = 20
warmup = 3
times = []
for i in range(runs):
    print(f'{i}th run on CPU ...')
    start_time = time.perf_counter()
    h1_cpu = mol.intor('ECPscalar')
    end_time = time.perf_counter()
    times.append(end_time - start_time)
print(f"average time with CPU: {sum(times[warmup:])/runs}")

times = []
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()
for i in range(runs):
    print(f'{i}th run on GPU ...')
    start_event.record()
    h1_gpu = get_ecp(mol)
    end_event.record()
    end_event.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
    times.append(elapsed_time)

avg_time = (sum(times[warmup:]) / runs)/1000
print(f"average time with GPU: {avg_time}")

assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-7