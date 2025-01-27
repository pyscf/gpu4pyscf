# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2


def test_int3c2e_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[3, [1.1, 1.]],
                      [4, [2., 1.]]],
               'C2': 'ccpvdz'},
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    P
    102.9917624900           1.0000000000
C    P
     28.1325940100           1.0000000000
C    P
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = -0.2
    dat = sr_aux_e2(cell, auxcell, omega).get()

    cell.precision=1e-10
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-8

def test_int3c2e_kpoints():
    cell = pyscf.M(
        atom='''H1   1.3    .2       .3
                H2   .19   .1      1.1
        ''',
        basis='ccpvdz',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*4)
    auxcell = cell.copy()
    auxcell.basis = [[0, [3.5, 1.]],
                     [0, [1.1, 1.]],
                     [1, [0.7, 1.]],
                     [2, [1.5, 1.]]]
    auxcell.build()
    kpts = cell.make_kpts([5,1,1])
    omega = -0.2
    dat = sr_aux_e2(cell, auxcell, omega, kpts).get()

    cell.precision=1e-10
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell, kpts).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-8

def test_minor_diffused_basis():
    cell = pyscf.M(
        atom='''H   1.3    .2       .3
                H   .19   .1      1.1
        ''',
        basis='''
C    S
      7.5                    0.40
      2.6                    0.90
      0.5                    0.08''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3)
    auxcell = cell.copy()
    auxcell.basis = '''
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 '''
    auxcell.build()
    omega = -0.2
    dat = sr_aux_e2(cell, auxcell, omega).get()

    cell.precision=1e-12
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-8

def test_ignorable_diffused_basis():
    cell = pyscf.M(
        atom='''H   1.3    .2       .3
                H   .19   .1      1.1
        ''',
        basis='''
C    S
      7.5                    0.4000000
      2.6                    0.9000000
      0.5                    0.0000002''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3)
    auxcell = cell.copy()
    auxcell.basis = '''
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.1995412500           1.0000000000 '''
    auxcell.build()
    omega = -0.2
    cell.verbose = 6
    dat = sr_aux_e2(cell, auxcell, omega).get()

    cell.basis='''
C S
      7.5                    0.4000000
      2.6                    0.9000000'''
    cell.build()
    df = rsdf_builder._RSGDFBuilder(cell, auxcell).build(omega=abs(omega))
    int3c = df.gen_int3c_kernel('int3c2e', aosym='s1', return_complex=True)
    ref = int3c().reshape(dat.shape)
    assert abs(dat - ref).max() < 1e-6
