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
    td.lindep = 1e-6
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
    assert np.allclose([0.16523319, 0.19442242, 0.21143836, 0.21809395, 0.22871087], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tda(benchmark):
    e = benchmark(run_rb3lyp_tda, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tda')
    assert np.allclose([0.16602786, 0.20084277, 0.21245269, 0.2215731,  0.2294362], e)


################
# Direct SCF
################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tddft(benchmark):
    e = benchmark(run_rb3lyp_tddft, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tddft')
    assert np.allclose([0.16523316, 0.19442043, 0.21143639, 0.2180925,  0.22870825], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tda(benchmark):
    e = benchmark(run_rb3lyp_tda, small_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tda')
    assert np.allclose([0.16602782, 0.20084139, 0.21245093, 0.22157153, 0.2294336], e)


####################
# Medium molecule
####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tddft_medium(benchmark):
    e = benchmark(run_rb3lyp_tddft, medium_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tddft medium')
    assert np.allclose([0.14359969, 0.15114103, 0.15593607, 0.16176117, 0.16484172], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_tda_medium(benchmark):
    e = benchmark(run_rb3lyp_tda, medium_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp tda medium')
    assert np.allclose([0.14642032, 0.15194237, 0.15696979, 0.16456322, 0.16519566], e)

@pytest.mark.slow
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tddft_medium(benchmark):
    e = benchmark(run_rb3lyp_tddft, medium_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tddft medium')
    assert np.allclose([0.14359864, 0.15114158, 0.15593616, 0.16176195, 0.16483943], e)
@pytest.mark.slow
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_rb3lyp_tda_medium(benchmark):
    e = benchmark(run_rb3lyp_tda, medium_mol, 'def2-tzvpp', False, False)
    print('testing rb3lyp tda medium')
    assert np.allclose([0.14641966, 0.15194219, 0.15697108, 0.16456213, 0.16519457], e)



#####################
# Small basis set
#####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_tddft(benchmark):
    e = benchmark(run_rb3lyp_tddft, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs tddft')
    assert np.allclose([0.16324379, 0.19561657, 0.20816873, 0.21759055, 0.22805238], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_631gs_tda(benchmark):
    e = benchmark(run_rb3lyp_tda, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 631gs tda')
    assert np.allclose([0.16397038, 0.20123123, 0.20907137, 0.22170817, 0.2284415], e)


