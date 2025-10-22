# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf.scf import j_engine, jk
from pyscf.scf.hf import get_jk

def test_j_engine():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)

    vj1 = j_engine.get_j(mol, dm).get()
    ref = get_jk(mol, dm, with_k=False)[0]
    assert abs(lib.fp(vj1) - -2327.4715195591784) < 1e-9
    assert abs(vj1 - ref).max() < 1e-9

    mol.omega = 0.2
    vj1 = j_engine.get_j(mol, dm).get()
    ref = get_jk(mol, dm, hermi=1, with_k=False)[0]
    assert abs(vj1 - ref).max() < 1e-9
    assert abs(lib.fp(vj1) -  1163.932604635460) < 5e-10

    mol.omega = -0.2
    vj1 = j_engine.get_j(mol, dm).get()
    ref = get_jk(mol, dm, hermi=1, with_k=False)[0]
    assert abs(vj1 - ref).max() < 5e-9
    assert abs(lib.fp(vj1) - -3491.404124194866) < 5e-10

def test_j_engine_8fold_symmetry():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)

    old_scheme = j_engine._md_j_engine_quartets_scheme
    # break alignment between tilex and tiley to test 8-fold symmetry.
    def custom_scheme(*args, **kwargs):
        out = list(old_scheme(*args, **kwargs))
        out[0] = out[0] // 2
        out[2] = out[2] * 2
        out[3] = max(out[3] // 15 * 3 + 1, 5)
        out[4] = max(out[3] - 3, 2)
        return tuple(out)

    try:
        j_engine._md_j_engine_quartets_scheme = custom_scheme
        vj = j_engine.get_j(mol, dm)
    finally:
        j_engine._md_j_engine_quartets_scheme = old_scheme
    vj1 = vj.get()
    ref = get_jk(mol, dm, with_k=False)[0]
    assert abs(lib.fp(vj1) - -2327.4715195591784) < 1e-9
    assert abs(vj1 - ref).max() < 1e-9

def test_j_engine_multiple_dms():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    for n in range(2, 10):
        dm = np.random.rand(n, nao, nao)
        dm = dm + dm.transpose(0, 2, 1)
        vj = j_engine.get_j(mol, dm)
        vj1 = vj.get()
        ref = get_jk(mol, dm, with_k=False)[0]
        assert abs(vj1 - ref).max() < 1e-9

    mol.omega = 0.2
    dm = np.random.rand(2, nao, nao)
    dm = dm + dm.transpose(0, 2, 1)
    vj = j_engine.get_j(mol, dm)
    vj1 = vj.get()
    ref = get_jk(mol, dm, with_k=False)[0]
    assert abs(vj1 - ref).max() < 1e-9

    mol.omega = -0.2
    dm = np.random.rand(2, nao, nao)
    dm = dm + dm.transpose(0, 2, 1)
    vj = j_engine.get_j(mol, dm)
    vj1 = vj.get()
    ref = get_jk(mol, dm, with_k=False)[0]
    assert abs(vj1 - ref).max() < 1e-9

def test_j_engine_integral_screen():
    basis = ([[0,[2**x,1]] for x in range(-1, 5)] +
             [[1,[2**x,1]] for x in range(-1, 2)] +
             [[3,[2**x,1]] for x in range(-1, 1)]
            )
    mol = pyscf.M(
        atom = '''
O  -9.2037 -0.1259  6.4262 
H -11.7768  0.2184  7.9561 
H -11.7819 -1.0073  7.9636
H -11.2190 -0.1224  5.3389
N  -9.2130 -0.1182  8.6103 
C  -7.7662 -0.1219  8.6103 
C  -7.2447 -0.1180 10.0438 
O  -7.9744 -0.1125 11.0321 
H  -7.4164 -1.0206  8.0911 
H  -7.4110  0.2317  8.0835  
H  -9.6852 -0.1162  9.5099 
N  -5.9251 -0.1205 10.2766 
C  -5.4305 -0.1166 11.6362 
C  -3.9051 -0.1205 11.6362 
O  -3.2258 -0.1262 10.6126 
H  -5.7987  0.2177 12.1423  
H  -5.8042 -1.0067 12.1503 
        ''',
        basis=basis,
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)*.1 - .05
    dm = dm.dot(dm.T)
    ref = jk.get_jk(mol, dm)[0].get()

    vj = j_engine.get_j(mol, dm)
    vj1 = vj.get()
    assert abs(vj1 - ref).max() < 1e-9

def test_sparse_dm():
    basis = ([[0,[2**x,1]] for x in range(-1, 5)] +
             [[1,[2**x,1]] for x in range(-1, 3)] +
             [[3,[2**x,1]] for x in range(-1, 3)]
            )
    mol = pyscf.M(
        atom = '''
O  -9.2037 -0.1259  6.4262
H -11.7768  0.2184  7.9561
H -11.7819 -1.0073  7.9636
H -11.2190 -0.1224  5.3389
N  -9.2130 -0.1182  8.6103
C  -7.7662 -0.1219  8.6103
C  -7.2447 -0.1180 10.0438
O  -7.9744 -0.1125 11.0321
H  -7.4164 -1.0206  8.0911
H  -7.4110  0.2317  8.0835
H  -9.6852 -0.1162  9.5099
N  -5.9251 -0.1205 10.2766
C  -5.4305 -0.1166 11.6362
C  -3.9051 -0.1205 11.6362
O  -3.2258 -0.1262 10.6126
H  -5.7987  0.2177 12.1423
H  -5.8042 -1.0067 12.1503
        ''',
        basis=basis,
        unit='B',)

    dm = np.eye(mol.nao)
    ref = jk.get_jk(mol, dm)[0].get()

    vj1 = j_engine.get_j(mol, dm).get()
    assert abs(vj1 - ref).max() < 1e-9

    dm = np.array([dm, dm])
    vj1 = j_engine.get_j(mol, dm).get()
    assert abs(vj1 - ref).max() < 1e-9

    mol.cart = True
    mol.build(0, 0)
    dm = np.eye(mol.nao)
    ref = jk.get_jk(mol, dm)[0].get()

    vj1 = j_engine.get_j(mol, dm).get()
    assert abs(vj1 - ref).max() < 1e-9

def test_general_contraction():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        ''',
        basis=('ccpvdz', [[3, [2., 1., .5], [1., .5, 1.]]]),
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)

    vj = j_engine.get_j(mol, dm)
    ref = jk.get_jk(mol, dm)[0]
    assert abs(vj - ref).max() < 1e-9
