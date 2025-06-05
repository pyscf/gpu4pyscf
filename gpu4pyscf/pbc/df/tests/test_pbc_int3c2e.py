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
import ctypes
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.df.int3c2e import sr_aux_e2, sr_int2c2e, fill_triu_bvk_conj
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
from gpu4pyscf.pbc.df.ft_ao import libpbc


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

def test_aopair_fill_triu():
    bvk_ncells, nao = 6, 13
    out = cp.random.rand(bvk_ncells,nao,nao)
    conj_mapping = cp.asarray(conj_images_in_bvk_cell([bvk_ncells,1,1]), dtype=np.int32)
    ix, iy = cp.tril_indices(nao, -1)
    ref = out.copy()
    for k, ck in enumerate(conj_mapping):
        ref[ck,iy,ix] = ref[k,ix,iy]
    out = fill_triu_bvk_conj(out, nao, [bvk_ncells,1,1])
    assert abs(out-ref).max() == 0.

def test_sr_int2c2e():
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3,
        basis='def2-universal-jkfit')
    omega = 0.2
    dat = sr_int2c2e(cell, -omega).get()[0]

    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    auxcell_sr = cell.copy()
    auxcell_sr.precision = 1e-14
    auxcell_sr.rcut = 50
    with auxcell_sr.with_short_range_coulomb(omega):
        ref = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=kpts)
    assert abs(dat - ref[0]).max() < 1e-10

    dat = sr_int2c2e(cell, -omega, kpts=kpts, bvk_kmesh=kmesh).get()
    assert abs(dat - ref).max() < 1e-10

    cell = cell.copy()
    cell.basis = {
        'C1':'''
C    S
     12.9917624900           1.0000000000
C    S
      2.1325940100           1.0000000000
C    P
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.5769010900           1.0000000000
C    D
      0.6                    1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    cell.build()
    omega = 0.2
    dat = sr_int2c2e(cell, -omega).get()[0]
    auxcell_sr = cell.copy()
    auxcell_sr.precision = 1e-14
    auxcell_sr.rcut = 50
    with auxcell_sr.with_short_range_coulomb(omega):
        ref = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=kpts)
    assert abs(dat - ref[0]).max() < 1e-10

    dat = sr_int2c2e(cell, -omega, kpts=kpts, bvk_kmesh=kmesh).get()
    assert abs(dat - ref).max() < 1e-10
