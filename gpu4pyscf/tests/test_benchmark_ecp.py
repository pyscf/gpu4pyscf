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

# Any task taking more than 1000s will be marked as 'slow'

# How to run
# 1. run test only
# pytest test_benchmark_ecp.py --benchmark-disable -s -v -m "not slow" --durations=20

# 2. benchmark less expensive tasks
# pytest test_benchmark_ecp.py -v -m "not slow"

# 3. benchmark all the tests
# pytest test_benchmark_ecp.py -v

# 4. save benchmark results
# pytest test_benchmark_ecp.py -s -v -m "not slow and not high_memory" --benchmark-save=v1.4.0_ecp_1v100

# 5. compare benchmark results, fail if performance regresses by more than 10%
# pytest test_benchmark_ecp.py -s -v -m "not slow and not high_memory" --benchmark-compare-fail=min:10% --benchmark-compare=1v100 --benchmark-storage=benchmark_results/

current_folder = os.path.dirname(os.path.abspath(__file__))
# A molecule from MOBH35 dataset (https://pubs.acs.org/doi/10.1021/acs.jpca.9b01546)
small_mol = os.path.join(current_folder, 'r14.xyz')

def run_rb3lyp(atom, basis, ecp, with_df):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0, ecp=ecp)
    mf = rks.RKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    return mf.kernel()

def run_rb3lyp_grad(atom, basis, ecp, with_df):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0, ecp=ecp)
    mf = rks.RKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return g

def run_rb3lyp_hessian(atom, basis, ecp, with_df):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0, ecp=ecp)
    mf = rks.RKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    mf.grids.atom_grid = (99,590)
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
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=5)
def test_df_ecp_rb3lyp(benchmark):
    e = benchmark(run_rb3lyp, small_mol, 'def2-tzvpp', 'def2-tzvpp', True)
    print('testing df rb3lyp')
    print(np.linalg.norm(e))
    assert np.isclose(np.linalg.norm(e), 1166.5960170669186, atol=1e-7, rtol=1e-16)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=5)
def test_df_ecp_rb3lyp_grad(benchmark):
    g = benchmark(run_rb3lyp_grad, small_mol, 'def2-tzvpp', 'def2-tzvpp', True)
    print('testing df rb3lyp grad')
    print(np.linalg.norm(g))
    assert np.isclose(np.linalg.norm(g), 0.06662495031714166, atol=1e-5, rtol=1e-16)
@pytest.mark.benchmark(warmup=False, min_rounds=3)
def test_df_ecp_rb3lyp_hessian(benchmark):
    h = benchmark(run_rb3lyp_hessian, small_mol, 'def2-tzvpp', 'def2-tzvpp', True)
    print('testing df rb3lyp hessian')
    print(np.linalg.norm(h))
    assert np.isclose(np.linalg.norm(h), 4.82148828357935, atol=1e-4, rtol=1e-16)
