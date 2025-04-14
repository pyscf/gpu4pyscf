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

import tempfile
import numpy as np
import cupy as cp
import pyscf
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.df import _load3c
from gpu4pyscf.pbc.df.rsdf_builder import build_cderi
from gpu4pyscf.pbc.df import rsdf_builder

def test_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[0, [1.1, 1.]],
                      [1, [2., 1.]]],
               'C2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
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
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = 0.3
    gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts=None, omega=omega)

    cell.precision = 1e-10
    auxcell.precision = 1e-10
    kpts = cell.make_kpts([1,1,1])
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.omega = omega
    dfbuilder.j2c_eig_always = False
    dfbuilder.fft_dd_block = True
    dfbuilder.exclude_d_aux = True
    naux = auxcell.nao
    nao = cell.nao
    with tempfile.NamedTemporaryFile() as tmpf:
        dfbuilder.make_j3c(tmpf.name, aosym='s1')
        with _load3c(tmpf.name, 'j3c', kpts[[0,0]]) as cderi:
            ref = abs(cderi[:].reshape(naux,nao,nao))
            dat = abs(gpu_dat[0,0].get())
            assert abs(dat - ref).max() < 1e-8

def test_kpts():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[0, [1.1, 1.]],
                      [1, [2., 1.]]],
               'C2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
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
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = 0.3
    kmesh = [6,1,1]
    kpts = cell.make_kpts(kmesh)
    gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts, omega=omega)

    cell.precision = 1e-10
    auxcell.precision = 1e-10
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.omega = omega
    dfbuilder.j2c_eig_always = False
    dfbuilder.fft_dd_block = True
    dfbuilder.exclude_d_aux = True
    naux = auxcell.nao
    nao = cell.nao
    with tempfile.NamedTemporaryFile() as tmpf:
        dfbuilder.make_j3c(tmpf.name, aosym='s1')
        for ki, kj in gpu_dat:
            with _load3c(tmpf.name, 'j3c', kpts[[ki,kj]]) as cderi:
                ref = abs(cderi[:].reshape(naux,nao,nao))
                dat = abs(gpu_dat[ki,kj].get())
                print(ki,kj)
                assert abs(dat - ref).max() < 1e-8

def test_kpts_j_only():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[0, [1.1, 1.]],
                      [1, [2., 1.]]],
               'C2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
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
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = 0.3
    kmesh = [1,3,4]
    kpts = cell.make_kpts(kmesh)
    gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts, omega=omega, j_only=True)

    cell.precision = 1e-10
    auxcell.precision = 1e-10
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.j_only = True
    dfbuilder.omega = omega
    dfbuilder.j2c_eig_always = False
    dfbuilder.fft_dd_block = True
    dfbuilder.exclude_d_aux = True
    naux = auxcell.nao
    nao = cell.nao
    with tempfile.NamedTemporaryFile() as tmpf:
        dfbuilder.make_j3c(tmpf.name, aosym='s1', j_only=True)
        for ki, kj in gpu_dat:
            with _load3c(tmpf.name, 'j3c', kpts[[ki,kj]]) as cderi:
                ref = abs(cderi[:].reshape(naux,nao,nao))
                dat = abs(gpu_dat[ki,kj].get())
                print(ki,kj)
                assert abs(dat - ref).max() < 1e-8

def test_gamma_point_compressed():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[0, [1.1, 1.]],
                      [1, [2., 1.]]],
               'C2': 'ccpvdz'},
        a=np.diag([2.5, 1.9, 2.2])*3)

    auxcell = cell.copy()
    auxcell.basis = {
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
      0.1995412500           1.0000000000 ''',
        'C2':[[0, [.5, 1.]]],
    }
    auxcell.build()
    omega = 0.3
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_gamma_point(cell, auxcell, omega=omega)
    nao = cell.nao
    i, j, diag = idx
    naux = auxcell.nao
    out = cp.zeros((naux,nao,nao))
    out[:,j,i] = dat
    out[:,i,j] = dat

    ref = build_cderi(cell, auxcell, omega=omega)[0]
    assert abs(ref[0,0] - out).max() < 1e-14
