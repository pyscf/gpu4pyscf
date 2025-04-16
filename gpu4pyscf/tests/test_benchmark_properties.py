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
from gpu4pyscf.properties import polarizability, shielding
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

def run_rb3lyp_nmr(atom, basis, with_df, with_solvent, disp=None):
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

    msc_d, msc_p = shielding.eval_shielding(mf)
    msc = (msc_d + msc_p).get()
    isotropic_msc = [msc[i].trace()/3 for i in range(mol.natm)]

    return np.array(isotropic_msc)


def run_rb3lyp_polarizability(atom, basis, with_df, with_solvent, disp=None):
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

    polar_gpu = polarizability.eval_polarizability(mf)

    return polar_gpu



#######
# DF
#######
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_nmr(benchmark):
    isotropic_msc = benchmark(run_rb3lyp_nmr, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp NMR')
    print(isotropic_msc)
    # assert np.allclose([0.16523319, 0.19442242, 0.21143836, 0.21809395, 0.22871087], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_polarizability(benchmark):
    polarizability= benchmark(run_rb3lyp_polarizability, small_mol, 'def2-tzvpp', True, False)
    print('testing df rb3lyp polarizability')
    print(polarizability)
    # assert np.allclose([0.16602786, 0.20084277, 0.21245269, 0.2215731,  0.2294362], e)


################
# Direct SCF
################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_nmr(benchmark):
    isotropic_msc = benchmark(run_rb3lyp_nmr, small_mol, 'def2-tzvpp', False, False)
    print('testing direct rb3lyp NMR')
    print(isotropic_msc)
    # assert np.allclose([0.16523319, 0.19442242, 0.21143836, 0.21809395, 0.22871087], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_polarizability(benchmark):
    polarizability= benchmark(run_rb3lyp_polarizability, small_mol, 'def2-tzvpp', False, False)
    print('testing direct rb3lyp polarizability')
    print(polarizability)
    # assert np.allclose([0.16602786, 0.20084277, 0.21245269, 0.2215731,  0.2294362], e)


#####################
# Small basis set
#####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_nmr(benchmark):
    isotropic_msc = benchmark(run_rb3lyp_nmr, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 6-31gs NMR')
    print(isotropic_msc)
    # assert np.allclose([0.16523319, 0.19442242, 0.21143836, 0.21809395, 0.22871087], e)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_polarizability(benchmark):
    polarizability= benchmark(run_rb3lyp_polarizability, small_mol, '6-31gs', True, False)
    print('testing df rb3lyp 6-31gs polarizability')
    print(polarizability)
    # assert np.allclose([0.16602786, 0.20084277, 0.21245269, 0.2215731,  0.2294362], e)


