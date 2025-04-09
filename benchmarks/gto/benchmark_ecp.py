import os
import time
import numpy as np
import cupy as cp
from pyscf import gto, lib, scf
from gpu4pyscf.gto.ecp import get_ecp, get_ecp_ip, get_ecp_ipip

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
basis = os.path.join(CURRENT_DIR, '../../gpu4pyscf/drivers/basis_vDZP_NWCHEM.dat')
ecp = os.path.join(CURRENT_DIR, '../../gpu4pyscf/drivers/ecp_vDZP_NWCHEM.dat')

lib.num_threads(8)
mol = gto.M(
    atom='../molecules/organic/020_Vitamin_C.xyz',
    basis=basis,
    cart=1,
    ecp=ecp)

def measure_cpu_time(fn, mol):
    runs = 10
    warmup = 3
    times = []
    for i in range(runs):
        #print(f'{i}th run on CPU ...')
        start_time = time.perf_counter()
        h1_cpu = fn(mol)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return h1_cpu, np.mean(times[warmup:])

def measure_gpu_time(fn, mol):
    runs = 20
    warmup = 3
    times = []
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    for i in range(runs):
        #print(f'{i}th run on GPU ...')
        start_event.record()
        h1_gpu = fn(mol)#get_ecp(mol)
        end_event.record()
        end_event.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_time)
    return h1_gpu, np.mean(times[warmup:])/1000

# Benchmark ECPscalar
print("Benchmarking ECPscalar")
fn = lambda x: x.intor('ECPscalar')
h1_cpu, cpu_time = measure_cpu_time(fn, mol)
h1_gpu, gpu_time = measure_gpu_time(get_ecp, mol)
assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-7
print(f"cpu time: {cpu_time}")
print(f"gpu time: {gpu_time}")
print(f"Speedup of ECPscalar: {cpu_time/gpu_time:.3f}")
print()

# Benchmark ECPscalar_ipnuc
print("Benchmarking ECPscalar_ipnuc")
fn = lambda x: x.intor('ECPscalar_ipnuc')
h1_cpu, cpu_time = measure_cpu_time(fn, mol)
h1_gpu, gpu_time = measure_gpu_time(get_ecp_ip, mol)
assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-7
print(f"cpu time: {cpu_time}")
print(f"gpu time: {gpu_time}")
print(f"Speedup of ECPscalar_ipnuc: {cpu_time/gpu_time:.3f}")
print()

# Benchmark ECPscalar_ipnucip
print("Benchmarking ECPscalar_ipnucip")
fn = lambda x: x.intor('ECPscalar_ipnucip', comp=9)
h1_cpu, cpu_time = measure_cpu_time(fn, mol)
fn = lambda x: get_ecp_ipip(x, ip_type='ipvip')
h1_gpu, gpu_time = measure_gpu_time(fn, mol)
assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-7
print(f"cpu time: {cpu_time}")
print(f"gpu time: {gpu_time}")
print(f"Speedup of ECPscalar_ipnucip: {cpu_time/gpu_time:.3f}")
print()

# Benchmark ECPscalar_ipipnuc
print("Benchmarking ECPscalar_ipipnuc")
fn = lambda x: x.intor('ECPscalar_ipipnuc', comp=9)
h1_cpu, cpu_time = measure_cpu_time(fn, mol)
fn = lambda x: get_ecp_ipip(x, ip_type='ipipv')
h1_gpu, gpu_time = measure_gpu_time(fn, mol)
assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-7
print(f"cpu time: {cpu_time}")
print(f"gpu time: {gpu_time}")
print(f"Speedup of ECPscalar_ipipnuc: {cpu_time/gpu_time:.3f}")
