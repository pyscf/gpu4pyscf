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

import unittest
import numpy as np
import pyscf
from pyscf import lib, gto
from gpu4pyscf.scf import jk
from pyscf.scf.hf import get_jk

def test_jk_hermi1():
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

    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj1 - ref[0]).max() < 1e-9
    assert abs(vk1 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj1) - -2327.4715195591784) < 5e-10
    assert abs(lib.fp(vk1) - -4069.3170008260583) < 5e-10

    vj = jk.get_j(mol, dm, hermi=1).get()
    assert abs(vj - ref[0]).max() < 1e-9
    assert abs(lib.fp(vj) - -2327.4715195591784) < 5e-10

    mol.omega = 0.2
    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj2 = vj.get()
    vk2 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj2 - ref[0]).max() < 1e-9
    assert abs(vk2 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj2) -  1163.932604635460) < 5e-10
    assert abs(lib.fp(vk2) - -1269.969109438691) < 5e-10

    mol.omega = -0.2
    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj3 = vj.get()
    vk3 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj3 - ref[0]).max() < 1e-8
    assert abs(vk3 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj3) - -3491.404124194866) < 5e-10
    assert abs(lib.fp(vk3) - -2799.347891387202) < 5e-10

    assert abs(vj2+vj3 - vj1).max() < 1e-9
    assert abs(vk2+vk3 - vk1).max() < 1e-9

def test_jk_hermi1_cart():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        cart=True,
        basis='def2-tzvp',
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)*.1 - .03
    dm = dm.dot(dm.T)

    vj, vk = jk.get_jk(mol, dm, hermi=1)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=1)
    assert abs(vj1 - ref[0]).max() < 1e-10
    assert abs(vk1 - ref[1]).max() < 1e-10
    assert abs(lib.fp(vj1) - 88.88500592206657) < 1e-10
    assert abs(lib.fp(vk1) - 48.57434458906684) < 1e-10

    vj = jk.get_j(mol, dm, hermi=1).get()
    assert abs(vj - ref[0]).max() < 1e-10

def test_jk_hermi0():
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

    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj1 = vj.get()
    vk1 = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj1 - ref[0]).max() < 5e-10
    assert abs(vk1 - ref[1]).max() < 5e-10
    assert abs(lib.fp(vj1) - -53.489298042359046) < 5e-10
    assert abs(lib.fp(vk1) - -115.11792498085259) < 5e-10
    
    vj = jk.get_j(mol, dm, hermi=0).get()
    assert abs(vj - ref[0]).max() < 1e-9
    assert abs(lib.fp(vj) - -53.489298042359046) < 5e-10
    
    mol.omega = 0.2
    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj2 = vj.get()
    vk2 = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj2 - ref[0]).max() < 1e-9
    assert abs(vk2 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj2) -  24.18519249677608) < 5e-10
    assert abs(lib.fp(vk2) - -34.15933205656134) < 5e-10

    mol.omega = -0.2
    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj3 = vj.get()
    vk3 = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj3 - ref[0]).max() < 1e-8
    assert abs(vk3 - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj3) - -77.67449053914103) < 5e-10
    assert abs(lib.fp(vk3) - -80.95859292428769) < 5e-10

    assert abs(vj2+vj3 - vj1).max() < 1e-9
    assert abs(vk2+vk3 - vk1).max() < 1e-9

def test_jk_hermi0_l5():
    mol = pyscf.M(
        atom = '''
        O   0.000   -0.    0.1174
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        basis={'default': 'def2-tzvp', 'O': [[5, [1., 1.]]]},
        unit='B',)

    np.random.seed(9)
    nao = mol.nao
    dm = np.random.rand(nao, nao)
    vj, vk = jk.get_jk(mol, dm, hermi=0)
    vj = vj.get()
    vk = vk.get()
    ref = get_jk(mol, dm, hermi=0)
    assert abs(vj - ref[0]).max() < 1e-9
    assert abs(vk - ref[1]).max() < 1e-9
    assert abs(lib.fp(vj) - -61.28856847097108) < 1e-9
    assert abs(lib.fp(vk) - -76.38373664249241) < 1e-9

    vj = jk.get_j(mol, dm, hermi=0).get()
    assert abs(vj - ref[0]).max() < 1e-9
    assert abs(lib.fp(vj) - -61.28856847097108) < 1e-9
