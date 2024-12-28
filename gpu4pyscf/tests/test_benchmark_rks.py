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
import unittest
import numpy as np
import pyscf
import pytest
import cupy
from gpu4pyscf.dft import rks, uks

# Any task taking more than 1000s will be marked as 'slow'

# How to run
# 1. run test only
# pytest test_rks.py --benchmark-disable -s -v -m "not slow" --durations=20
# 2. benchmark less expensive tasks
# pytest test_rks.py -v -m "not slow"
# 3. benchmark all the tests
# pytest test_rks.py -v

current_folder = os.path.dirname(os.path.abspath(__file__))
small_mol = os.path.join(current_folder, '020_Vitamin_C.xyz')
median_mol = os.path.join(current_folder, '057_Tamoxifen.xyz')
large_mol = os.path.join(current_folder, '095_Azadirachtin.xyz')

def run_rb3lyp(atom, basis, with_df, with_solvent, disp=None):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = rks.RKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    if disp is not None:
        mf.disp = disp
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    return mf.kernel()

def run_rb3lyp_grad(atom, basis, with_df, with_solvent, disp=None):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = rks.RKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    if disp is not None:
        mf.disp = disp
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return g

def run_rb3lyp_hessian(atom, basis, with_df, with_solvent, disp=None):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = rks.RKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    if disp is not None:
        mf.disp = disp
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-6
    mf.kernel()
    h = mf.Hessian().kernel()
    return h

# DF
@pytest.mark.benchmark
def test_df_rb3lyp(benchmark):
    e = benchmark(run_rb3lyp, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp')
    assert np.isclose(np.linalg.norm(e), 684.9998712035579, atol=1e-7)
@pytest.mark.benchmark
def test_df_rb3lyp_grad(benchmark):
    g = benchmark(run_rb3lyp_grad, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp grad')
    assert np.isclose(np.linalg.norm(g), 0.17435941081837686, atol=1e-5)
@pytest.mark.slow
@pytest.mark.benchmark
def test_df_rb3lyp_hessian(benchmark):
    h = benchmark(run_rb3lyp_hessian, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp hessian')
    assert np.isclose(np.linalg.norm(h), 3.7668761221997764, atol=1e-4)

# Direct SCF
@pytest.mark.benchmark
def test_rb3lyp(benchmark):
    e = benchmark(run_rb3lyp, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp')
    assert np.isclose(np.linalg.norm(e), 684.999735850967, atol=1e-7)
@pytest.mark.benchmark
def test_rb3lyp_grad(benchmark):
    g = benchmark(run_rb3lyp_grad, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp grad')
    assert np.isclose(np.linalg.norm(g), 0.1744127474130983, atol=1e-5)
@pytest.mark.benchmark
def test_rb3lyp_hessian(benchmark):
    h = benchmark(run_rb3lyp_hessian, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp hessian')
    assert np.isclose(np.linalg.norm(h), 3.7588443634477833, atol=1e-4)

# median molecule
@pytest.mark.benchmark
def test_df_rb3lyp_median(benchmark):
    e = benchmark(run_rb3lyp, median_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp median')
    assert np.isclose(np.linalg.norm(e), 1138.371390377773, atol=1e-7)
@pytest.mark.benchmark
def test_df_rb3lyp_grad_median(benchmark):
    g = benchmark(run_rb3lyp_grad, median_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp grad median')
    assert np.isclose(np.linalg.norm(g), 0.26010545073602614, atol=1e-4)
@pytest.mark.benchmark
def test_df_rb3lyp_hessian_median(benchmark):
    h = benchmark(run_rb3lyp_hessian, median_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp hessian median')
    assert np.isclose(np.linalg.norm(h), 6.32514169232998, atol=1e-4)

@pytest.mark.benchmark
def test_rb3lyp_median(benchmark):
    e = benchmark(run_rb3lyp, median_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp median')
    assert np.isclose(np.linalg.norm(e), 1138.3710752128077, atol=1e-7)
@pytest.mark.benchmark
def test_rb3lyp_grad_median(benchmark):
    g = benchmark(run_rb3lyp_grad, median_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp grad median')
    assert np.isclose(np.linalg.norm(g), 0.2601443836937988, atol=1e-5)

@pytest.mark.slow
@pytest.mark.benchmark
def test_rb3lyp_hessian_median(benchmark):
    h = benchmark(run_rb3lyp_hessian, median_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp hessian median')
    print(np.linalg.norm(h))
    assert np.isclose(np.linalg.norm(h))

# large molecule
@pytest.mark.benchmark
def test_df_rb3lyp_large(benchmark):
    e = benchmark(run_rb3lyp, large_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp large')
    assert np.isclose(np.linalg.norm(e), 2564.198712152175, atol=1e-7)
@pytest.mark.benchmark
def test_df_rb3lyp_grad_large(benchmark):
    g = benchmark(run_rb3lyp_grad, large_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp grad large')
    assert np.isclose(np.linalg.norm(g), 0.3784358687859323, atol=1e-5)
@pytest.mark.high_memory
@pytest.mark.slow
@pytest.mark.benchmark
def test_df_rb3lyp_hessian_large(benchmark):
    h = benchmark(run_rb3lyp_hessian, large_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp hessian large')
    assert np.isclose(np.linalg.norm(h), 7.583208736873523, atol=1e-4)
@pytest.mark.slow
@pytest.mark.benchmark
def test_rb3lyp_large(benchmark):
    e = benchmark(run_rb3lyp, large_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp large')
    assert np.isclose(np.linalg.norm(e), 2564.198099576358, atol=1e-7)
@pytest.mark.slow
@pytest.mark.benchmark
def test_rb3lyp_grad_large(benchmark):
    g = benchmark(run_rb3lyp_grad, large_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp grad large')
    assert np.isclose(np.linalg.norm(g), 0.3784664384209763, atol=1e-5)

# Hessian for large molecule with large basis set is too slow
'''
@pytest.mark.slow
@pytest.mark.benchmark
def test_rb3lyp_hessian_large(benchmark):
    h = benchmark(run_rb3lyp_hessian, large_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp hessian large')
    print(np.linalg.norm(h))
'''
# small basis set
@pytest.mark.benchmark
def test_df_rb3lyp_631gs(benchmark):
    e = benchmark(run_rb3lyp, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs')
    assert np.isclose(np.linalg.norm(e), 684.6646008642876, atol=1e-7)

@pytest.mark.benchmark
def test_df_rb3lyp_631gs_grad(benchmark):
    g = benchmark(run_rb3lyp_grad, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs grad')
    assert np.isclose(np.linalg.norm(g), 0.17530687343398219, atol=1e-5)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_hessian(benchmark):
    h = benchmark(run_rb3lyp_hessian, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs hessian')
    assert np.isclose(np.linalg.norm(h), 3.908874851569459, atol=1e-4)

# small basis set for large molecule
@pytest.mark.benchmark
def test_rb3lyp_631gs_large(benchmark):
    e = benchmark(run_rb3lyp, large_mol, '6-31gs', False, False)
    print('testing rb3lyp 631gs large')
    assert np.isclose(np.linalg.norm(e), 2563.1171191823423, atol=1e-7)
@pytest.mark.benchmark
def test_rb3lyp_631gs_grad_large(benchmark):
    g = benchmark(run_rb3lyp_grad, large_mol, '6-31gs', False, False)
    print('testing df rb3lyp 631gs grad large')
    assert np.isclose(np.linalg.norm(g), 0.37778228700247984, atol=1e-5)
@pytest.mark.slow
@pytest.mark.benchmark
def test_rb3lyp_631gs_hessian_large(benchmark):
    h = benchmark(run_rb3lyp_hessian, large_mol, '6-31gs', False, False)
    print('testing df rb3lyp 631gs hessian large')
    assert np.isclose(np.linalg.norm(h), 7.920764634100053, atol=1e-4)

#solvent model
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_solvent(benchmark):
    e = benchmark(run_rb3lyp, small_mol, '6-31gs', True, True)
    print('testing df rb3lyp 631gs solvent')
    assert np.isclose(np.linalg.norm(e), 684.6985561053816, atol=1e-7)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_solvent_grad(benchmark):
    g = benchmark(run_rb3lyp_grad, small_mol, '6-31gs', True, True)
    print('testing df rb3lyp 631gs solvent grad')
    assert np.isclose(np.linalg.norm(g), 0.16956999476137297, atol=1e-5)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_solvent_hessian(benchmark):
    h = benchmark(run_rb3lyp_hessian, small_mol, '6-31gs', True, True)
    print('testing df rb3lyp 631gs solvent hessian')
    assert np.isclose(np.linalg.norm(h), 3.9008165041707294, atol=1e-4)

# b3lyp d3bj
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_d3bj(benchmark):
    e = benchmark(run_rb3lyp, small_mol, '6-31gs', True, True, 'd3bj')
    print('testing df rb3lyp 631gs solvent')
    assert np.isclose(np.linalg.norm(e), 684.7313814096565, atol=1e-7)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_d3bj_grad(benchmark):
    g = benchmark(run_rb3lyp_grad, small_mol, '6-31gs', True, True, 'd3bj')
    print('testing df rb3lyp 631gs solvent grad')
    assert np.isclose(np.linalg.norm(g), 0.17010044498887264, atol=1e-5)
@pytest.mark.benchmark
def test_df_rb3lyp_631gs_d3bj_hessian(benchmark):
    h = benchmark(run_rb3lyp_hessian, small_mol, '6-31gs', True, True, 'd3bj')
    print('testing df rb3lyp 631gs solvent hessian')
    assert np.isclose(np.linalg.norm(h), 3.902367554157861, atol=1e-4)

