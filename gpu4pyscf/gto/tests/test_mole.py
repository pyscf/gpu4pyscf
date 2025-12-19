# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf.gto import mole as mole_gpu

def test_basis_seg_contraction():
    mol = pyscf.M(
        atom='C 0 0 0; O 0 1 1',
        basis=('ccpvdz', [[2, [9, .1, .1], [3, 1, .5], [1, .5, 1]]]),
    )
    pmol, c = mole_gpu.basis_seg_contraction(mol)
    ref = mol.intor('int1e_ovlp')
    c = c.get()
    dat = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
    assert abs(dat - ref).max() < 1e-15

    pmol, c = mole_gpu.basis_seg_contraction(mol, allow_replica=True)
    c = c.get()
    dat = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
    assert abs(dat - ref).max() < 1e-15

    pmol, c = mole_gpu.basis_seg_contraction(mol, allow_replica=False)
    c = c.get()
    dat = c.T.dot(pmol.intor('int1e_ovlp')).dot(c)
    assert abs(dat - ref).max() < 1e-15

def test_general_contraction():
    basis = '''
S
    14. .1   .1
    9.  .8   .2
    3.  .4   .5
    1.  .3   .8
S
    1.5  1.
S
    .5  1.
P
    53. .1  .0
    48. .1  .0
    43. .1  .0
    38. .1  .0
    33. .1  .0
    28. .1  .0
    23. .1  .05
    19. .7  .08
    13. .4  .14
    11. .2  .33
    9.  .1  .8
    3.  .1  .4
    1.  .1  .3
P
    19. .08
    13. .14
    11. .33
    9.  .8 
    3.  .4 
    1.  .3 
P
    13. .14
    11. .33
    9.  .8 
    3.  .4 
    1.  .3 
P
    .5  1.
D
    9.  .8   .1
    3.  .4   .5
    1.  .3   .8
D
    11. .33
    3.  .5
    1.  .8
D
    .5  1.
'''
    mol = pyscf.M(
        atom='C 0 0 0; C 0 .5 1',
        basis=basis, cart=True)

    def _check(mol, sorted_mol):
        nao_sorted = sorted_mol.nao
        nao = mol.nao
        c = sorted_mol.C_dot_mat(cp.eye(nao))
        assert abs(c - sorted_mol.CT_dot_mat(cp.eye(nao_sorted)).T).max() < 1e-12
        assert abs(c - sorted_mol.mat_dot_C(cp.eye(nao_sorted))).max() < 1e-12
        assert abs(c - sorted_mol.mat_dot_CT(cp.eye(nao)).T).max() < 1e-12

        s1 = cp.random.rand(nao_sorted, nao_sorted)
        s1 = s1 + s1.T
        assert abs(sorted_mol.CT_dot_mat(s1) - sorted_mol.mat_dot_C(s1).T).max() < 1e-12
        assert abs(sorted_mol.CT_dot_mat(s1) - c.T.dot(s1)).max() < 1e-12
        assert abs(sorted_mol.mat_dot_C(s1) - s1.dot(c)).max() < 1e-12

        s2 = cp.random.rand(nao, nao)
        s2 = s2 + s2.T
        assert abs(sorted_mol.mat_dot_CT(s2) - sorted_mol.C_dot_mat(s2).T).max() < 1e-12
        assert abs(sorted_mol.mat_dot_CT(s2) - s2.dot(c.T)).max() < 1e-12
        assert abs(sorted_mol.C_dot_mat(s2) - c.dot(s2)).max() < 1e-12

        s0 = cp.asarray(mol.intor('int1e_ovlp'))
        s1 = cp.asarray(sorted_mol.intor('int1e_ovlp'))
        assert abs(sorted_mol.apply_CT_mat_C(s1) - s0).max() < 1e-12
        assert abs(sorted_mol.apply_C_mat_CT(s0) - c.dot(s0).dot(c.T)).max() < 1e-12

    sorted_mol = mole_gpu.SortedMole.from_mol(
        mol, allow_replica=True, allow_split_seg_contraction=False)
    mol.cart = True
    _check(mol, sorted_mol)
    mol.cart = False
    _check(mol, sorted_mol)

    sorted_mol = mole_gpu.SortedMole.from_mol(
        mol, allow_replica=-1, allow_split_seg_contraction=True)
    mol.cart = True
    _check(mol, sorted_mol)
    mol.cart = False
    _check(mol, sorted_mol)

    sorted_mol = mole_gpu.SortedMole.from_mol(
        mol, allow_replica=1, allow_split_seg_contraction=True)
    mol.cart = True
    _check(mol, sorted_mol)
    mol.cart = False
    _check(mol, sorted_mol)
