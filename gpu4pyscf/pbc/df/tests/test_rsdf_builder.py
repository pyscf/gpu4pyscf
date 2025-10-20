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
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.df import _load3c
from gpu4pyscf.pbc.df.rsdf_builder import build_cderi
from gpu4pyscf.pbc.df import rsdf_builder
import pytest

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
        basis={'C1': ('ccpvdz',
                      [[2, [1.1, 1.]],
                      [3, [2., 1.]]]),
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
    ij, diag = idx
    i, j = divmod(ij, nao)
    naux = auxcell.nao
    out = cp.zeros((naux,nao,nao))
    out[:,j,i] = dat[0]
    out[:,i,j] = dat[0]

    ref = build_cderi(cell, auxcell, omega=omega)[0]
    assert abs(ref[0,0] - out).max() < 1e-12

def test_sr_gamma_point_compressed():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[2, [1.1, 1.]],
                      [3, [2., 1.]]]),
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
    cell.omega = auxcell.omega = -omega
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_gamma_point(
        cell, auxcell, omega=omega, with_long_range=False)
    nao = cell.nao
    ij, diag = idx
    i, j = divmod(ij, nao)
    naux = auxcell.nao
    out = cp.zeros((naux,nao,nao))
    out[:,j,i] = dat[0]
    out[:,i,j] = dat[0]

    ref = build_cderi(cell, auxcell, omega=omega)[0]
    assert abs(ref[0,0] - out).max() < 1e-12

def test_kpts_compressed():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[2, [1.1, 1.]],
                      [3, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision=1e-10,
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
    nao = cell.nao
    omega = 0.3
    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(cell, auxcell, kpts, omega=omega)
    ref = build_cderi(cell, auxcell, kpts, omega=omega)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder.unpack_cderi(dat[kp], idx, kp, kk_conserv, expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 1e-11

def test_kpts_compressed1():
    from pyscf.pbc.df import df as df_cpu
    cell = pyscf.M(
        atom = 'He 1. .5 .5;C .1 1.3 2.1',
        basis = {'He': [(0, (1., 1)), (1, (.4, 1))],
                 'C' :[[0, [1., 1]]],},
        a = np.eye(3) * 2.5,
    )
    auxcell = df_cpu.make_auxcell(cell)

    nao = cell.nao
    kmesh = [1,3,1]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(cell, auxcell, kpts)
    ref = build_cderi(cell, auxcell, kpts)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder.unpack_cderi(dat[kp], idx, kp, kk_conserv, expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 1e-10

def test_kpts_compressed_general_contraction():
    cell = pyscf.M(
        atom='''C   1.3    .2       .3
                C   .19   .1      1.1
        ''',
        basis='''
        C  D
           173    0.27   -0.03
           5.8    0.8    -0.26
           1.9    0.1     0.81
        ''',
        a=np.eye(3)*6)

    auxcell = cell.copy()
    auxcell.basis = '''
C  S
    2.00   1.
C  D
    0.59   1.''',
    auxcell.build()
    nao = cell.nao
    omega = 0.3
    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(cell, auxcell, kpts, omega=omega)
    ref = build_cderi(cell, auxcell, kpts, omega=omega)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder.unpack_cderi(dat[kp], idx, kp, kk_conserv, expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 1e-11

@pytest.mark.skip('Must include gamma point')
def test_kpts_compressed2():
    from pyscf.pbc.df import df as df_cpu
    cell = pyscf.M(
        atom = 'He 1. .5 .5;C .1 1.3 2.1',
        basis = {'He': [(0, (1., 1)), (1, (.4, 1))],
                 'C' :[[0, [1., 1]]],},
        a = np.eye(3) * 2.5,
    )
    auxcell = df_cpu.make_auxcell(cell)

    nao = cell.nao
    kmesh = [2,3,1]
    kpts = cell.make_kpts(kmesh, with_gamma_point=False)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(cell, auxcell, kpts)
    ref = build_cderi(cell, auxcell, kpts)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder.unpack_cderi(dat[kp], idx, kp, kk_conserv, expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 1e-10

def test_sr_kpts_compressed():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[2, [1.1, 1.]],
                      [3, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision=1e-10,
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
    nao = cell.nao
    omega = 0.3
    cell.omega = auxcell.omega = -omega
    kmesh = [3,1,1]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(
        cell, auxcell, kpts, omega=omega, with_long_range=False)
    ref = build_cderi(cell, auxcell, kpts, omega=omega)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder.unpack_cderi(dat[kp], idx, kp, kk_conserv, expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 1e-11

def test_j_only_compressed():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[2, [1.1, 1.]],
                      [3, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision=1e-10,
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
    nao = cell.nao
    omega = 0.3
    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_j_only(cell, auxcell, kpts, omega=omega)
    ref = build_cderi(cell, auxcell, kpts, omega=omega)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))

    out = rsdf_builder.unpack_cderi(dat[0], idx, 0, kk_conserv, expLk, nao)
    for ki in range(nkpts):
        _ref = ref[ki, ki]
        assert abs(_ref - out[ki]).max() < 1e-11

def test_sr_j_only_compressed():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[2, [1.1, 1.]],
                      [3, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision=1e-10,
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
    nao = cell.nao
    omega = 0.3
    cell.omega = auxcell.omega = -omega
    kmesh = [3,1,1]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_j_only(
        cell, auxcell, kpts, omega=omega, with_long_range=False)
    ref = build_cderi(cell, auxcell, kpts, omega=omega)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))

    out = rsdf_builder.unpack_cderi(dat[0], idx, 0, kk_conserv, expLk, nao)
    for ki in range(nkpts):
        _ref = ref[ki, ki]
        assert abs(_ref - out[ki]).max() < 1e-11

def _get_2c2e_slow(auxcell, uniq_kpts, omega, with_long_range=True):
    from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
    from pyscf.pbc.lib.kpts_helper import is_zero
    from gpu4pyscf.gto.mole import extract_pgto_params
    from gpu4pyscf.pbc.df import ft_ao
    from gpu4pyscf.pbc.df.rsdf_builder import _weighted_coulG_LR
    # j2c ~ (-kpt_ji | kpt_ji) => hermi=1
    precision = auxcell.precision ** 1.5
    aux_exps, aux_cs = extract_pgto_params(auxcell, 'diffused')
    aux_exp = aux_exps.min()
    theta = 1./(2./aux_exp + omega**-2)
    rad = auxcell.vol**(-1./3) * auxcell.rcut + 1
    surface = 4*np.pi * rad**2
    lattice_sum_factor = 2*np.pi*auxcell.rcut/(auxcell.vol*theta) + surface
    rcut_sr = (np.log(lattice_sum_factor / precision + 1.) / theta)**.5
    auxcell_sr = auxcell.copy()
    auxcell_sr.rcut = rcut_sr
    with auxcell_sr.with_short_range_coulomb(omega):
        j2c = auxcell_sr.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    if not with_long_range:
        return j2c

    ke = estimate_ke_cutoff_for_omega(auxcell, omega, precision)
    mesh = auxcell.cutoff_to_mesh(ke)
    mesh = auxcell.symmetrize_mesh(mesh)

    Gv, Gvbase, kws = auxcell.get_Gv_weights(mesh)

    if uniq_kpts is None:
        j2c = cp.asarray(j2c)
        coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws)
        auxG = ft_ao.ft_ao(auxcell, Gv).T
        j2c += (auxG.conj() * coulG_LR).dot(auxG.T).real
        j2c = [j2c.real.get()]
    else:
        for k, kpt in enumerate(uniq_kpts):
            j2c_k = cp.asarray(j2c[k])
            coulG_LR = _weighted_coulG_LR(auxcell, Gv, omega, kws, kpt)
            gamma_point = is_zero(kpt)

            auxG = ft_ao.ft_ao(auxcell, Gv, kpt=kpt).T
            if gamma_point:
                j2c_k += (auxG.conj() * coulG_LR).dot(auxG.T).real
            else:
                j2c_k += (auxG.conj() * coulG_LR).dot(auxG.T)
            auxG = None
            j2c[k] = j2c_k.get()
    return j2c

def test_2c2e():
    cell = pyscf.M(
        atom='''C  1.3    .2       .3
                C  .19   .1      1.1
                C  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*3,
        basis='def2-universal-jkfit')
    omega = 0.2
    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    dat = rsdf_builder._get_2c2e(cell, kpts, omega, with_long_range=True)
    ref = _get_2c2e_slow(cell, kpts, omega, with_long_range=True)
    assert abs(dat - cp.asarray(ref)).max() < 1e-10

def test_kpts_compressed_linear_dep():
    from pyscf.pbc.df import df as df_cpu
    cell = pyscf.M(
        atom='''
        C 0.0 0.0 0.0
        C 0.0 1.8 1.8
        C 1.8 0.0 1.8
        C 1.8 1.8 0.0''', a=np.eye(3) * 3.6,
        basis=[[0, [4., 1.]],
               [0, [.1, 1.]],
               [0, [.035, 1.]]])
    auxcell = df_cpu.make_auxcell(cell)
    nao = cell.nao
    kmesh = [2, 1, 1]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(
        cell, auxcell, kpts=kpts)
    ref = build_cderi(cell, auxcell, kpts)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder.unpack_cderi(dat[kp], idx, kp, kk_conserv, expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            _ref = np.einsum('pij,plk->ijkl', _ref, _ref.conj(), optimize=True)
            _dat = np.einsum('pij,plk->ijkl', out[ki], out[ki].conj(), optimize=True)
            print(ki, kj)
            assert abs(_ref - _dat).max() < 3e-7
