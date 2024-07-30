import pyscf
from gpu4pyscf.scf import hf

mol = pyscf.M(atom = 'molecules/water_clusters/008.xyz')
mol.basis = {
    'H': [[0, (0.01,  0.0070011547)]],
    'O': [[0, (0.01,  0.0070011547)]]}
mol.build()

vhfopt = hf._VHFOpt(mol, 'int2e').build()

from gpu4pyscf.lib.cupy_helper import load_library
libgdft = load_library('libgdft')

import numpy as np
import cupy
from gpu4pyscf.dft.gen_grid import Grids
grids = Grids(mol)
grids.build()
coords = cupy.asarray(grids.coords[:256*256*16], order='F')#.copy()

ngrids, _ = coords.shape

ao_loc = mol.ao_loc_nr()
ao_loc = cupy.asarray(ao_loc, dtype=np.int32)
nao = mol.nao
dm = cupy.random.rand(nao,nao)
dm += dm.T
print(dm[:4,:4])
print(vhfopt.bas_pair2shls[:,:1000])

import ctypes
from pyscf import gto
from gpu4pyscf.dft import numint
ni = numint.NumInt().build(mol, coords)
env = cupy.asarray(ni.gdftopt._sorted_mol._env)
ish, jsh = vhfopt.bas_pair2shls
diag_idx = ish == jsh
npairs = len(ish)
bas = ni.gdftopt._sorted_mol._bas
exp_sparse = cupy.empty([npairs, 2], order='F')
bas_exp = bas[ish, gto.PTR_EXP]
exp_sparse[:,0] = cupy.asarray(env[bas_exp])
bas_exp = bas[jsh, gto.PTR_EXP]
exp_sparse[:,1] = cupy.asarray(env[bas_exp])

coef_sparse = cupy.empty([npairs, 2], order='F')
bas_coef = bas[ish, gto.PTR_COEFF]
coef_sparse[:,0] = cupy.asarray(env[bas_coef])
bas_coef = bas[jsh, gto.PTR_COEFF]
coef_sparse[:,1] = cupy.asarray(env[bas_coef])

atm_coords = cupy.asarray(mol.atom_coords())
coord_pairs = cupy.empty([npairs, 6], order='F')
coord_pairs[:,:3] = atm_coords[ish,:]
coord_pairs[:,3:] = atm_coords[jsh,:]

dm_sparse = dm[ish, jsh] * coef_sparse[:,0] * coef_sparse[:,1]
dm_sparse[diag_idx] *= 0.5

coords_fp32 = coords.astype(np.float32)
exp_sparse_fp32 = exp_sparse.astype(np.float32)
coef_sparse_fp32 = coef_sparse.astype(np.float32)
coord_pairs_fp32 = coord_pairs.astype(np.float32)
dm_sparse_fp32 = dm_sparse.astype(np.float32)

def eval_rho0(dm):
    with ni.gdftopt.gdft_envs_cache():
        rho = cupy.empty([4,ngrids])
        libgdft.eval_rho_fp32(
            vhfopt.bpcache,
            ctypes.cast(coords.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.cast(rho.data.ptr, ctypes.c_void_p),
            ctypes.cast(exp_sparse_fp32.data.ptr, ctypes.c_void_p),
            ctypes.cast(coef_sparse_fp32.data.ptr, ctypes.c_void_p),
            ctypes.cast(coord_pairs_fp32.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nao),
            ctypes.cast(dm_sparse_fp32.data.ptr, ctypes.c_void_p))
        return rho
    return eval_rho0

def eval_rho1(dm):
    rho = cupy.empty([4,ngrids])
    with ni.gdftopt.gdft_envs_cache():
        libgdft.eval_rho_fp64(
            vhfopt.bpcache,
            ctypes.cast(coords.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.cast(rho.data.ptr, ctypes.c_void_p),
            ctypes.cast(exp_sparse.data.ptr, ctypes.c_void_p),
            ctypes.cast(coef_sparse.data.ptr, ctypes.c_void_p),
            ctypes.cast(coord_pairs.data.ptr, ctypes.c_void_p),
            ctypes.c_int(nao),
            ctypes.cast(dm_sparse.data.ptr, ctypes.c_void_p))
    return rho

def eval_rho2(dm):
    ao = numint.eval_ao(ni, ni.gdftopt._sorted_mol, coords, deriv=1)
    rho0 = numint.eval_rho(ni.gdftopt._sorted_mol, ao, dm, xctype='GGA')
    return rho0

from cupyx import profiler
perf = profiler.benchmark(eval_rho0, (dm_sparse,), n_repeat=20, n_warmup=3)
print('with eval_rho0', perf.gpu_times.mean())
# 0.130s on V100
# 0.115s on A10

perf = profiler.benchmark(eval_rho1, (dm_sparse,), n_repeat=20, n_warmup=3)
print('with eval_rho1', perf.gpu_times.mean())
# 0.324s on V100
# 5.401s on A10

perf = profiler.benchmark(eval_rho2, (dm,), n_repeat=20, n_warmup=3)
print('with eval_rho2', perf.gpu_times.mean())
# 0.0858s on v100
# 0.723s on A10

rho0 = eval_rho0(dm)
rho1 = eval_rho1(dm)
print(rho0[:4,:3])
print(rho1[:4,:3])

print(cupy.max(rho0 - rho1), cupy.max(rho0))
