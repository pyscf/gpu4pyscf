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

current_folder = os.path.dirname(os.path.abspath(__file__))
small_mol = os.path.join(current_folder, '020_Vitamin_C.xyz')

def run_ub3lyp(atom, basis, with_df, with_solvent):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = uks.UKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    return mf.kernel()

def run_ub3lyp_grad(atom, basis, with_df, with_solvent):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = uks.UKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method().kernel()
    return g

def run_ub3lyp_hessian(atom, basis, with_df, with_solvent):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = uks.UKS(mol, xc='b3lyp')
    if with_df:
        mf = mf.density_fit()
    if with_solvent:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
    mf.grids.atom_grid = (99,590)
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-6
    mf.kernel()
    h = mf.Hessian().kernel()
    return h


# UKS
@pytest.mark.benchmark
def test_df_ub3lyp(benchmark):
    e = benchmark(run_ub3lyp, small_mol, 'def2-tzvpp', True, False)
    print('testing df ub3lyp')
    assert np.isclose(np.linalg.norm(e), 684.9998712035856, atol=1e-7)
@pytest.mark.benchmark
def test_df_ub3lyp_grad(benchmark):
    g = benchmark(run_ub3lyp_grad, small_mol, 'def2-tzvpp', True, False)
    print('testing df ub3lyp grad')
    assert np.isclose(np.linalg.norm(g), 0.17435842214665462, atol=1e-5)
@pytest.mark.benchmark
def test_df_ub3lyp_hessian(benchmark):
    h = benchmark(run_ub3lyp_hessian, small_mol, 'def2-tzvpp', True, False)
    print('testing df ub3lyp hessian')
    assert np.isclose(np.linalg.norm(h), 3.7669464279078064, atol=1e-4)
@pytest.mark.benchmark
def test_ub3lyp(benchmark):
    e = benchmark(run_ub3lyp, small_mol, 'def2-tzvpp', False, False)
    print('testing ub3lyp')
    assert np.isclose(np.linalg.norm(e), 684.9997358509884, atol=1e-7)
@pytest.mark.benchmark
def test_ub3lyp_grad(benchmark):
    g = benchmark(run_ub3lyp_grad, small_mol, 'def2-tzvpp', False, False)
    print('testing ub3lyp grad')
    assert np.isclose(np.linalg.norm(g), 0.17441176110160253, atol=1e-5)
@pytest.mark.benchmark
def test_ub3lyp_hessian(benchmark):
    h = benchmark(run_ub3lyp_hessian, small_mol, 'def2-tzvpp', False, False)
    print('testing ub3lyp hessian')
    assert np.isclose(np.linalg.norm(h), 3.758916526520172, atol=1e-4)