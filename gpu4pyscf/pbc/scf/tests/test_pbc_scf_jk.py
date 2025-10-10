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
import numpy as cp
import pyscf
from pyscf import lib, gto
from gpu4pyscf.pbc.scf import rsjk
from pyscf.pbc.scf.rsjk import RangeSeparationJKBuilder

def test_sr_vk_hermi1_gamma_point_vs_cpu():
    from gpu4pyscf.scf.jk import get_k as mol_k
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        C   1.      1.    0.
        ''',
        a=np.eye(3)*7.,
        basis=('ccpvdz', [[3, [.5, 1]]]),
    )

    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)
    vk = rsjk.get_k(cell, dm, hermi=1).get()
    omega = rsjk.OMEGA
    with_rsjk = RangeSeparationJKBuilder(cell)
    with_rsjk.exclude_dd_block = False
    with_rsjk.allow_drv_nodddd = False
    ref = with_rsjk.build(omega)._get_jk_sr(
        dm, hermi=1, kpts=np.zeros((1,3)), with_j=False)
    print( abs(vk - ref).max() )

def test_sr_vk_hermi1_kpts_vs_cpu():
    from gpu4pyscf.scf.jk import get_k as mol_k
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        C   1.      1.    0.
        ''',
        a=np.eye(3)*7.,
        basis=('ccpvdz', [[3, [.5, 1]]]),
    )

    kpts = cell.make_kpts([3,2,1])
    nkpts = len(kpts)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nkpts, nao, nao) + np.random.rand(nkpts, nao, nao) * .5j
    dm = dm + dm.transpose(0, 2, 1).conj()
    vk = rsjk.get_k(cell, dm, hermi=1, kpts=kpts).get()
    omega = rsjk.OMEGA
    with_rsjk = RangeSeparationJKBuilder(cell)
    with_rsjk.exclude_dd_block = False
    with_rsjk.allow_drv_nodddd = False
    ref = with_rsjk.build(omega)._get_jk_sr(
        dm, hermi=1, kpts=kpts, with_j=False)
    print( abs(vk - ref).max() )

def test_sr_vk_hermi1_gamma_point_vs_fft():
    from gpu4pyscf.pbc.df import fft
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        a=np.eye(3)*4.,
        basis=[[0, [.25, 1]], [1, [.3, 1]]],
    )
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)
    vk = rsjk.get_k(cell, dm, hermi=1).get()

    omega = cell.omega = -rsjk.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, with_j=False)[1].get()

    s = cell.pbc_intor('int1e_ovlp')
    w = cell.get_Gv_weights()[2]
    coulG0_SR = np.pi / omega**2
    ref += s.dot(dm).dot(s) * (w*coulG0_SR)
    print( abs(vk - ref).max() )

def test_sr_vk_hermi1_kpts_vs_fft():
    pass

def test_sr_vk_hermi0_gamma_point_vs_fft():
    pass

def test_sr_vk_hermi0_kpts_vs_fft():
    pass
