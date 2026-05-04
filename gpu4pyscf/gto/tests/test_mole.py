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

    sorted_mol = mole_gpu.SortedMole.from_mol(mol, decontract=False)
    mol.cart = True
    _check(mol, sorted_mol)
    mol.cart = False
    _check(mol, sorted_mol)

    sorted_mol = mole_gpu.SortedMole.from_mol(
        mol, decontract=True, diffuse_cutoff=0.1)
    mol.cart = True
    _check(mol, sorted_mol)
    mol.cart = False
    _check(mol, sorted_mol)

    sorted_mol = mole_gpu.SortedMole.from_mol(
        mol, decontract=True, diffuse_cutoff=0.3)
    mol.cart = True
    _check(mol, sorted_mol)
    mol.cart = False
    _check(mol, sorted_mol)

def test_apply_C_dot():
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'})
    nao = mol.nao
    cp.random.seed(9)
    c = cp.random.rand(3,nao,9)+.2j*cp.random.rand(3,nao,9)
    mol = mole_gpu.SortedGTO.from_mol(mol)
    ref = cp.einsum('pq,sqi->spi', mol.ctr_coeff, c)
    dat = mol.apply_C_dot(c, axis=1)
    assert abs(ref - dat).max() < 1e-15

def test_basis_recontraction():
    def check(mol, decontract=True, diffuse_cutoff=None):
        ref = mol.intor('int1e_ovlp')
        mol = mole_gpu.SortedGTO.from_mol(mol, decontract=decontract,
                                          diffuse_cutoff=diffuse_cutoff)
        c = mol.ctr_coeff.get()
        assert abs(c.T.dot(mol.intor('int1e_ovlp')).dot(c) - ref).max() < 1e-14
        return mol

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    1.49  0.00  0.36  0.00
    0.71  0.00  0.21  0.00
    0.24  1.00  0.81  0.00
    0.18  0.00  0.23  1.00'''))
    assert mol.nbas == 3

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    64.71  0.10  0.02  0.00
    21.06  0.27  0.06  0.00
     7.49  0.44  0.15  0.00
     2.79  0.28  0.12  0.00
     0.52  0.04  0.54  1.00'''))
    assert mol.nbas == 5

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    3.11  0.00  1.00  0.00
    2.26  0.00  0.00  0.00
    1.96  0.00  0.00  1.00
    0.44  1.00  0.00  0.00'''))
    assert mol.nbas == 3

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    2.98  0.00  0.05
    1.21  0.70  0.52
    0.58  0.00  0.46
    0.26  0.40  0.00'''), decontract=False)
    assert mol.nbas == 2

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    2.98  0.10
    1.21  0.70
    0.18  0.08
    0.06  0.40'''), diffuse_cutoff=0.4)
    assert mol.nbas == 3

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    2.98  0.10  0.00
    1.21  0.70  0.00
    0.58  0.33  0.41
    0.06  0.40  0.80'''), diffuse_cutoff=0.1)
    assert mol.nbas == 3

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    8.00  0.71  0.00
    5.00  0.71  0.00
    2.98  0.00  0.10
    1.21  0.00  0.70
    0.58  0.00  0.41'''))
    # Compared to fully decontraction, this is preferred to be decontracted to 2
    # shells. Current implementation does not recognize this pattern
    assert mol.nbas == 5

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    2.98  0.10  0.00
    1.21  0.70  1.00
    0.08  0.20  0.00
    0.46  0.40  0.00'''), diffuse_cutoff=0.1)
    assert mol.nbas == 2

    mol = check(pyscf.M(
        atom='''C   1.3    .2       .3 ''',
        basis='''
C S
    2.98  0.10  0.00
    1.21  0.70  0.00
    0.46  0.40  1.00
    0.08  0.20  0.00'''), diffuse_cutoff=0.1)
    assert mol.nbas == 2
