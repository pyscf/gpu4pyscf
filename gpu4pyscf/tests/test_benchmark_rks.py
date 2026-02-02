# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pyscf
import pytest
from gpu4pyscf.dft import rks
CUDA_VISIBLE_DEVICES=0
# Any task taking more than 1000s will be marked as 'slow'

# How to run
# 1. run test only
# pytest test_benchmark_rks.py --benchmark-disable -s -v -m "not slow" --durations=20

# 2. benchmark less expensive tasks
# pytest test_benchmark_rks.py -v -m "not slow"

# 3. benchmark all the tests
# pytest test_benchmark_rks.py -v

# 4. save benchmark results
# pytest test_benchmark_rks.py -s -v -m "not slow and not high_memory" --benchmark-save=v1.4.0_rks_1v100

# 5. compare benchmark results, fail if performance regresses by more than 10%
# pytest test_benchmark_rks.py -s -v -m "not slow and not high_memory" --benchmark-compare-fail=min:10% --benchmark-compare=1v100 --benchmark-storage=benchmark_results/

current_folder = os.path.dirname(os.path.abspath(__file__))
small_mol = os.path.join(current_folder, '020_Vitamin_C.xyz')
medium_mol = os.path.join(current_folder, '057_Tamoxifen.xyz')
large_mol = os.path.join(current_folder, '095_Azadirachtin.xyz')

def run_rks(atom, basis, xc, with_df, with_solvent, disp=None):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = rks.RKS(mol, xc=xc)
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    if disp is not None:
        mf.disp = disp
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    return mf.kernel()

def run_rks_grad(atom, basis, xc, with_df, with_solvent, disp=None):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = rks.RKS(mol, xc=xc)
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    if disp is not None:
        mf.disp = disp
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return g

def run_rks_hessian(atom, basis, xc, with_df, with_solvent, disp=None):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = rks.RKS(mol, xc=xc)
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    if disp is not None:
        mf.disp = disp
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-6
    mf.kernel()
    hobj = mf.Hessian()
    if with_df:
        hobj.auxbasis_response = 2
    h = hobj.kernel()
    return h

#######
# DF
#######
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp(benchmark):
    e = benchmark(run_rks, small_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp')
    assert np.isclose(np.linalg.norm(e), 684.9998712035579, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_grad(benchmark):
    g = benchmark(run_rks_grad, small_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp grad')
    assert np.isclose(np.linalg.norm(g), 0.17435941081837686, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_df_rb3lyp_hessian(benchmark):
    h = benchmark(run_rks_hessian, small_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp hessian')
    assert np.isclose(np.linalg.norm(h), 3.7587394873290885, atol=1e-4, rtol=1e-16)

################
# Direct SCF
################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp(benchmark):
    e = benchmark(run_rks, small_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp')
    assert np.isclose(np.linalg.norm(e), 684.999735850967, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_grad(benchmark):
    g = benchmark(run_rks_grad, small_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp grad')
    assert np.isclose(np.linalg.norm(g), 0.1744127474130983, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rb3lyp_hessian(benchmark):
    h = benchmark(run_rks_hessian, small_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp hessian')
    assert np.isclose(np.linalg.norm(h), 3.7588443634477833, atol=1e-4, rtol=1e-16)

####################
# Medium molecule
####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_medium(benchmark):
    e = benchmark(run_rks, medium_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp medium')
    assert np.isclose(np.linalg.norm(e), 1138.371390377773, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_grad_medium(benchmark):
    g = benchmark(run_rks_grad, medium_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp grad medium')
    assert np.isclose(np.linalg.norm(g), 0.26010545073602614, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_df_rb3lyp_hessian_medium(benchmark):
    h = benchmark(run_rks_hessian, medium_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp hessian medium')
    assert np.isclose(np.linalg.norm(h), 6.31265424196621, atol=1e-4, rtol=1e-16)

@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rb3lyp_medium(benchmark):
    e = benchmark(run_rks, medium_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp medium')
    assert np.isclose(np.linalg.norm(e), 1138.3710752128077, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rb3lyp_grad_medium(benchmark):
    g = benchmark(run_rks_grad, medium_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp grad medium')
    assert np.isclose(np.linalg.norm(g), 0.2601443836937988, atol=1e-5, rtol=1e-16)
@pytest.mark.slow
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rb3lyp_hessian_medium(benchmark):
    h = benchmark(run_rks_hessian, medium_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp hessian medium')
    assert np.isclose(np.linalg.norm(h), 6.312714778020796, atol=1e-4, rtol=1e-16)

####################
# large molecule
####################
@pytest.mark.high_memory
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_large(benchmark):
    e = benchmark(run_rks, large_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp large')
    assert np.isclose(np.linalg.norm(e), 2564.198712152175, atol=1e-7, rtol=1e-16)
@pytest.mark.high_memory
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_grad_large(benchmark):
    g = benchmark(run_rks_grad, large_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp grad large')
    assert np.isclose(np.linalg.norm(g), 0.3784358687859323, atol=1e-5, rtol=1e-16)
@pytest.mark.high_memory
@pytest.mark.slow
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_df_rb3lyp_hessian_large(benchmark):
    h = benchmark(run_rks_hessian, large_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp hessian large')
    assert np.isclose(np.linalg.norm(h), 7.583208736873523, atol=1e-4, rtol=1e-16)
@pytest.mark.slow
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_large(benchmark):
    e = benchmark(run_rks, large_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp large')
    assert np.isclose(np.linalg.norm(e), 2564.198099576358, atol=1e-7, rtol=1e-16)
@pytest.mark.slow
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_grad_large(benchmark):
    g = benchmark(run_rks_grad, large_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp grad large')
    assert np.isclose(np.linalg.norm(g), 0.3784664384209763, atol=1e-5, rtol=1e-16)

# Hessian for large molecule with large basis set is too slow
'''
@pytest.mark.slow
@pytest.mark.benchmark
def test_rb3lyp_hessian_large(benchmark):
    h = benchmark(run_rks_hessian, large_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing rb3lyp hessian large')
    print(np.linalg.norm(h))
'''

#####################
# Small basis set
#####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs(benchmark):
    e = benchmark(run_rks, small_mol, '6-31gs', 'b3lyp', True, False)
    print('testing df rb3lyp 631gs')
    assert np.isclose(np.linalg.norm(e), 684.6646008642876, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_grad(benchmark):
    g = benchmark(run_rks_grad, small_mol, '6-31gs', 'b3lyp', True, False)
    print('testing df rb3lyp 631gs grad')
    assert np.isclose(np.linalg.norm(g), 0.17530687343398219, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_df_rb3lyp_631gs_hessian(benchmark):
    h = benchmark(run_rks_hessian, small_mol, '6-31gs', 'b3lyp', True, False)
    print('testing df rb3lyp 631gs hessian')
    assert np.isclose(np.linalg.norm(h), 3.9071846157996553, atol=1e-4, rtol=1e-16)

#########################################
# Small basis set for large molecule
#########################################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_631gs_large(benchmark):
    e = benchmark(run_rks, large_mol, '6-31gs', 'b3lyp', False, False)
    print('testing rb3lyp 631gs large')
    assert np.isclose(np.linalg.norm(e), 2563.1171191823423, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_631gs_grad_large(benchmark):
    g = benchmark(run_rks_grad, large_mol, '6-31gs', 'b3lyp', False, False)
    print('testing df rb3lyp 631gs grad large')
    assert np.isclose(np.linalg.norm(g), 0.37778228700247984, atol=1e-5, rtol=1e-16)
@pytest.mark.slow
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rb3lyp_631gs_hessian_large(benchmark):
    h = benchmark(run_rks_hessian, large_mol, '6-31gs', 'b3lyp', False, False)
    print('testing df rb3lyp 631gs hessian large')
    assert np.isclose(np.linalg.norm(h), 7.920764634100053, atol=1e-4, rtol=1e-16)

###################
# Solvent model
###################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_solvent(benchmark):
    e = benchmark(run_rks, small_mol, '6-31gs', 'b3lyp', True, True)
    print('testing df rb3lyp 631gs solvent')
    assert np.isclose(np.linalg.norm(e), 684.6985561053816, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_solvent_grad(benchmark):
    g = benchmark(run_rks_grad, small_mol, '6-31gs', 'b3lyp', True, True)
    print('testing df rb3lyp 631gs solvent grad')
    assert np.isclose(np.linalg.norm(g), 0.16956999476137297, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_df_rb3lyp_631gs_solvent_hessian(benchmark):
    h = benchmark(run_rks_hessian, small_mol, '6-31gs', 'b3lyp', True, True)
    print('testing df rb3lyp 631gs solvent hessian')
    assert np.isclose(np.linalg.norm(h), 3.8991230592666737, atol=1e-4, rtol=1e-16)

# No need to test d3bj generally
'''
# b3lyp d3bj
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_d3bj(benchmark):
    e = benchmark(run_rks, small_mol, '6-31gs', 'b3lyp', True, True, 'd3bj')
    print('testing df rb3lyp 631gs solvent')
    assert np.isclose(np.linalg.norm(e), 684.7313814096565, atol=1e-7)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_d3bj_grad(benchmark):
    g = benchmark(run_rks_grad, small_mol, '6-31gs', 'b3lyp', True, True, 'd3bj')
    print('testing df rb3lyp 631gs solvent grad')
    assert np.isclose(np.linalg.norm(g), 0.17010044498887264, atol=1e-5)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_d3bj_hessian(benchmark):
    h = benchmark(run_rks_hessian, small_mol, '6-31gs', 'b3lyp', True, True, 'd3bj')
    print('testing df rb3lyp 631gs solvent hessian')
    assert np.isclose(np.linalg.norm(h), 3.902367554157861, atol=1e-4)
'''

###############################
# Medium molecule with wB97M-V
###############################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rwb97mv_medium(benchmark):
    e = benchmark(run_rks, medium_mol, 'def2-tzvpp', 'wb97m-v', True, False)
    print('testing df rwb97mv medium')
    assert np.isclose(np.linalg.norm(e), 1137.8935922602527, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rwb97mv_grad_medium(benchmark):
    g = benchmark(run_rks_grad, medium_mol, 'def2-tzvpp', 'wb97m-v', True, False)
    print('testing df rwb97mv grad medium')
    assert np.isclose(np.linalg.norm(g), 0.25882527440752034, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_df_rwb97mv_hessian_small(benchmark):
    h = benchmark(run_rks_hessian, small_mol, 'def2-tzvpp', 'wb97m-v', True, False)
    print('testing df rwb97mv hessian small')
    assert np.isclose(np.linalg.norm(h), 3.8459983082385696, atol=1e-4, rtol=1e-16)

@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rwb97mv_medium(benchmark):
    e = benchmark(run_rks, medium_mol, 'def2-tzvpp', 'wb97m-v', False, False)
    print('testing rwb97mv medium')
    assert np.isclose(np.linalg.norm(e), 1137.8932216907351, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rwb97mv_grad_medium(benchmark):
    g = benchmark(run_rks_grad, medium_mol, 'def2-tzvpp', 'wb97m-v', False, False)
    print('testing rwb97mv grad medium')
    assert np.isclose(np.linalg.norm(g), 0.25886924645878, atol=1e-5, rtol=1e-16)
@pytest.mark.slow
@pytest.mark.benchmark(warmup=False, min_rounds=1)
def test_rwb97mv_hessian_small(benchmark):
    h = benchmark(run_rks_hessian, small_mol, 'def2-tzvpp', 'wb97m-v', False, False)
    print('testing rwb97mv hessian small')
    print(np.linalg.norm(h))
    assert np.isclose(np.linalg.norm(h), 3.8461100556365104, atol=1e-4, rtol=1e-16)
