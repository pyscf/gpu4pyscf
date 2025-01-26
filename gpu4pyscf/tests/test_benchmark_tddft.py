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
from gpu4pyscf import tdscf
CUDA_VISIBLE_DEVICES=0
# Any task taking more than 1000s will be marked as 'slow'

# How to run
# 1. run test only
# pytest test_benchmark_tddft.py --benchmark-disable -s -v -m "not slow" --durations=20

# 2. benchmark less expensive tasks
# pytest test_benchmark_tddft.py -v -m "not slow"

# 3. benchmark all the tests
# pytest test_benchmark_tddft.py -v

# 4. save benchmark results
# pytest test_benchmark_tddft.py -s -v -m "not slow and not high_memory" --benchmark-save=v1.3.0_tddft_1v100

# 5. compare benchmark results, fail if performance regresses by more than 10%
# pytest test_benchmark_tddft.py -s -v -m "not slow and not high_memory" --benchmark-compare-fail=min:10% --benchmark-compare=1v100 --benchmark-storage=benchmark_results/

current_folder = os.path.dirname(os.path.abspath(__file__))
small_mol = os.path.join(current_folder, '020_Vitamin_C.xyz')
medium_mol = os.path.join(current_folder, '057_Tamoxifen.xyz')

def run_rb3lyp_tddft(atom, basis, with_df, with_solvent, disp=None):
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

    td = mf.TDDFT().set(nstates=5)
    assert td.device == 'gpu'
    e = td.kernel()[0]

    return e


def run_rb3lyp_tda(atom, basis, with_df, with_solvent, disp=None):
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

    td = mf.TDA().set(nstates=5)
    assert td.device == 'gpu'
    e = td.kernel()[0]

    return e


#######
# DF
#######
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tddft(benchmark):
    e = benchmark(run_rb3lyp_tddft, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tddft')
    assert np.allclose([0.1652341, 0.19442237, 0.21143863, 0.21809428, 0.22871117], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tda(benchmark):
    e = benchmark(run_rb3lyp_tda, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tda')
    assert np.allclose([0.16602878, 0.2008427, 0.21245292, 0.22157353, 0.2294365], e)


################
# Direct SCF
################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tddft(benchmark):
    e = benchmark(run_rb3lyp_tddft, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tddft')
    assert np.allclose([0.16523407, 0.19442038, 0.21143666, 0.21809283, 0.22870855], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tda(benchmark):
    e = benchmark(run_rb3lyp_tda, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tda')
    assert np.allclose([0.16602873, 0.20084131, 0.21245116, 0.22157196, 0.2294339 ], e)


####################
# Medium molecule
####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tddft_medium(benchmark):
    e = benchmark(run_rb3lyp_tddft, medium_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tddft medium')
    assert np.allclose([0.1652341, 0.19442237, 0.21143863, 0.21809428, 0.22871117], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tda_medium(benchmark):
    e = benchmark(run_rb3lyp_tda, medium_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tda medium')
    assert np.allclose([0.1652341, 0.19442237, 0.21143863, 0.21809428, 0.22871117], e)

@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tddft_medium(benchmark):
    e = benchmark(run_rb3lyp_tddft, medium_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tddft medium')
    assert np.allclose([0.1652341, 0.19442237, 0.21143863, 0.21809428, 0.22871117], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tda_medium(benchmark):
    e = benchmark(run_rb3lyp_tda, medium_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tda medium')
    assert np.allclose([0.1652341, 0.19442237, 0.21143863, 0.21809428, 0.22871117], e)



#####################
# Small basis set
#####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_tddft(benchmark):
    e = benchmark(run_rb3lyp_tddft, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs tddft')
    assert np.allclose([0.16324444, 0.19561619, 0.20816938, 0.21759079, 0.22805259], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_tda(benchmark):
    e = benchmark(run_rb3lyp_tda, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs tda')
    assert np.allclose([0.16397102, 0.20123098, 0.20907196, 0.22170831, 0.22844169], e)


