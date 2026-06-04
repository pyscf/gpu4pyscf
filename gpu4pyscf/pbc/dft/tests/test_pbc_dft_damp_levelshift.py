# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

import numpy as np
from pyscf import lib
from pyscf.pbc import gto as pgto
from gpu4pyscf.pbc.dft.kucdft import CDFT_KUKS

import pytest

@pytest.fixture
def cell():
    L = 4.
    cell = pgto.Cell()
    cell.a = np.eye(3)*L
    cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
    cell.basis = [[0, (3.0, 1.0)], [0, (1.0, 1.0)]]
    cell.verbose = 6
    cell.output = '/dev/null'
    cell.build()
    return cell

def test_rks_damp(cell):
    mf = cell.RKS(xc='pbe').to_gpu()

    mf.damp = 0.5
    mf.diis_start_cycle = 3
    mf.run()

def test_rks_levelshift(cell):
    mf = cell.RKS(xc='pbe').to_gpu()

    mf.level_shift = 0.2
    mf.diis_start_cycle = 3
    mf.run()

def test_uks_damp(cell):
    mf = cell.UKS(xc='pbe').to_gpu()

    mf.damp = 0.5
    mf.diis_start_cycle = 3
    mf.run()

def test_uks_levelshift(cell):
    mf = cell.UKS(xc='pbe').to_gpu()

    mf.level_shift = 0.2
    mf.diis_start_cycle = 3
    mf.run()

def test_krks_damp(cell):
    kpts = cell.make_kpts([1,1,1])
    mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()

    mf.damp = 0.5
    mf.diis_start_cycle = 3
    mf.run()

def test_krks_levelshift(cell):
    kpts = cell.make_kpts([1,1,1])
    mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()

    mf.level_shift = 0.2
    mf.diis_start_cycle = 3
    mf.run()

def test_kuks_damp(cell):
    kpts = cell.make_kpts([1,1,1])
    mf = cell.KUKS(xc='pbe', kpts=kpts).to_gpu()

    mf.damp = 0.5
    mf.diis_start_cycle = 3
    mf.run()

def test_kuks_levelshift(cell):
    kpts = cell.make_kpts([1,1,1])
    mf = cell.KUKS(xc='pbe', kpts=kpts).to_gpu()

    mf.level_shift = 0.2
    mf.diis_start_cycle = 3
    mf.run()

def test_kucdft_damp(cell):
    kpts = cell.make_kpts([1,1,1])
    mf = CDFT_KUKS(cell, kpts).to_gpu()
    mf.xc = 'pbe'

    mf.damp = 0.5
    mf.diis_start_cycle = 3
    mf.kernel()