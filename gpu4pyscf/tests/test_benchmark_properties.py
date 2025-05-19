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
# pytest test_benchmark_properties.py --benchmark-disable -s -v -m "not slow" --durations=20

# 2. benchmark less expensive tasks
# pytest test_benchmark_properties.py -v -m "not slow"

# 3. benchmark all the tests
# pytest test_benchmark_properties.py -v

# 4. save benchmark results
# pytest test_benchmark_properties.py -s -v -m "not slow and not high_memory" --benchmark-save=v1.4.0_properties_1v100

# 5. compare benchmark results, fail if performance regresses by more than 10%
# pytest test_benchmark_properties.py -s -v -m "not slow and not high_memory" --benchmark-compare-fail=min:10% --benchmark-compare=1v100 --benchmark-storage=benchmark_results/

current_folder = os.path.dirname(os.path.abspath(__file__))
small_mol = os.path.join(current_folder, '020_Vitamin_C.xyz')

def run_rks_nmr(atom, basis, xc, with_df, with_solvent, disp=None):
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

    msc_d, msc_p = shielding.eval_shielding(mf)
    msc = (msc_d + msc_p).get()
    isotropic_msc = [msc[i].trace()/3 for i in range(mol.natm)]

    return np.array(isotropic_msc)


def run_rks_polarizability(atom, basis, xc, with_df, with_solvent, disp=None):
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

    polar_gpu = polarizability.eval_polarizability(mf, max_cycle=40)

    return polar_gpu



#######
# DF
#######
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_nmr(benchmark):
    isotropic_msc = benchmark(run_rks_nmr, small_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp NMR')
    ref = [  98.23139214,  71.3565165 ,  -12.69350159,   39.67757168,    3.22702007,
             27.66270797, 262.48311365,   29.59649503,  179.93849665,   27.40947391,
           -182.78062449, 104.58489902,   27.85197309,  114.13310739,   26.89714614,
             28.56425129, 284.08891676,   29.53540855,  308.58172738,   31.16494053]
    assert np.allclose(ref, isotropic_msc)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rb3lyp_polarizability(benchmark):
    polar= benchmark(run_rks_polarizability, small_mol, 'def2-tzvpp', 'b3lyp', True, False)
    print('testing df rb3lyp polarizability')
    ref = [[113.85255878,  13.2027686 ,  14.61812779],
           [ 13.2027686 ,  94.61158119,  18.79964077],
           [ 14.61812779,  18.79964077,  76.78038446],]
    assert np.allclose(ref, polar)


################
# Direct SCF
################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_direct_rb3lyp_nmr(benchmark):
    isotropic_msc = benchmark(run_rks_nmr, small_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing direct rb3lyp NMR')
    ref = [  98.24407781,  71.34910533, -12.68969181,  39.6833743 ,    3.23016859,
             27.66185633, 262.48145299,  29.59588728, 179.93276709,   27.40815127,
           -182.78223744, 104.58743097,  27.84954477, 114.12604115,   26.89481303,
             28.5632408 , 284.09239341,  29.53477888, 308.57785874,   31.16414777]
    assert np.allclose(ref, isotropic_msc)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_direct_rb3lyp_polarizability(benchmark):
    polar= benchmark(run_rks_polarizability, small_mol, 'def2-tzvpp', 'b3lyp', False, False)
    print('testing direct rb3lyp polarizability')
    ref = [[113.85420856, 13.2037831 ,  14.61909362],
           [ 13.2037831 , 94.61286669,  18.8008085 ],
           [ 14.61909362, 18.8008085 ,  76.77967724]]
    assert np.allclose(polar, ref)


#####################
# Small basis set
#####################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_small_rb3lyp_nmr(benchmark):
    isotropic_msc = benchmark(run_rks_nmr, small_mol, '6-31gs', 'b3lyp', True, False)
    print('testing df rb3lyp 6-31gs NMR')
    ref = [ 113.7096066 , 100.15894088,   11.98277223,   60.31002233,   27.92836304,
             28.46690645, 269.08190586,   30.48012605,  199.99645123,   29.32942155,
           -149.63329562, 119.4816839,    28.49788251,  127.21836872,   28.61607066,
             29.44243821, 291.07404077,   30.64420464,  312.50520443,   32.35149868]
    assert np.allclose(isotropic_msc, ref)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_small_rb3lyp_polarizability(benchmark):
    polar= benchmark(run_rks_polarizability, small_mol, '6-31gs', 'b3lyp', True, False)
    print('testing df rb3lyp 6-31gs polarizability')
    ref = [[100.00109918, 13.66936311, 14.7179826 ],
           [ 13.66936311, 81.69064061, 18.53488172],
           [ 14.7179826 , 18.53488172, 62.87327742]]
    assert np.allclose(polar, ref)



##########################
# wB97M-V polarizability
##########################
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_df_rwb97mv_polarizability(benchmark):
    polar= benchmark(run_rks_polarizability, small_mol, 'def2-tzvpp', 'wb97m-v', True, False)
    print('testing df rwb97mv polarizability')
    ref = [[112.23094268,  13.2731982 ,  14.44019449],
           [ 13.2731982 ,  93.68456945,  18.19620343],
           [ 14.44019449,  18.19620343,  76.53716955]]
    assert np.allclose(ref, polar)
@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=3)
def test_direct_rwb97mv_polarizability(benchmark):
    polar= benchmark(run_rks_polarizability, small_mol, 'def2-tzvpp', 'wb97m-v', False, False)
    print('testing direct rwb97mv polarizability')
    ref = [[112.2321085 ,  13.27420481,  14.44118473],
           [ 13.27420481,  93.68513707,  18.1974098 ],
           [ 14.44118473,  18.1974098 ,  76.53589836]]
    assert np.allclose(polar, ref)

