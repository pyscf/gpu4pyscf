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
from pyscf import lib
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.df.df import _load3c
from gpu4pyscf.pbc.df.rsdf_builder import build_cderi
from gpu4pyscf.pbc.df import rsdf_builder
from gpu4pyscf.pbc.lib.kpts_helper import conj_images_in_bvk_cell
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
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = 0.3
    with lib.temporary_env(rsdf_builder, PREFER_ED=False):
        gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts=None)

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
            assert abs(lib.fp(ref) - -0.6376070572) < 1e-8
            assert abs(dat - ref).max() < 3e-8

def test_kpts():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': [[0, [1.1, 1.]],
                      [1, [2., 1.]],
                      [2, [1., 1.]]],
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
        'C2':[[0, [.5, 1.]], [2, [.4, 1.]]],
    }
    auxcell.build()
    omega = 0.3
    kmesh = [6,1,1]
    kpts = cell.make_kpts(kmesh)
    with lib.temporary_env(rsdf_builder, PREFER_ED=False):
        gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts)

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
                      [1, [2., 1.]],
                      [2, [1., 1.]]],
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
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    kmesh = [1,3,4]
    kpts = cell.make_kpts(kmesh)
    with lib.temporary_env(rsdf_builder, PREFER_ED=False):
        gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts, j_only=True)

    cell.precision = 1e-10
    auxcell.precision = 1e-10
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.j_only = True
    dfbuilder.omega = 0.2
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
                assert abs(dat - ref).max() < 3e-8

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
    omega = 0.2
    with lib.temporary_env(rsdf_builder, PREFER_ED=False):
        dat, dat_neg, idx = rsdf_builder.compressed_cderi_gamma_point(
            cell, auxcell, omega=-omega)
    cp.cuda.get_current_stream().synchronize()

    nao = cell.nao
    naux = auxcell.nao
    out = rsdf_builder._unpack_cderi_v2(
        dat[0], idx[0], [0], [0], cp.ones((1, 1), dtype=np.complex128), nao)[0]
    assert abs(lib.fp(abs(out.get())) - -4.663003306619004) < 1e-8

    auxcell.omega = cell.omega = -omega
    cell.precision = 1e-10
    auxcell.precision = 1e-10
    auxcell.rcut = 35.0
    kpts = cell.make_kpts([1,1,1])
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.j2c_eig_always = False
    dfbuilder.fft_dd_block = False
    dfbuilder.exclude_d_aux = False
    naux = auxcell.nao
    nao = cell.nao
    with tempfile.NamedTemporaryFile() as tmpf:
        dfbuilder.make_j3c(tmpf.name, aosym='s1')
        with _load3c(tmpf.name, 'j3c', kpts[[0,0]]) as cderi:
            ref = abs(cderi[:].reshape(naux,nao,nao))
            dat = abs(out.get())
            assert abs(dat - ref).max() < 3e-8

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
    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(cell, auxcell, kpts)
    ref = build_cderi(cell, auxcell, kpts)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder._unpack_cderi_v2(
            dat[kp], idx[0], np.where(kk_conserv == kp)[1],
            conj_images_in_bvk_cell(kmesh), expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 3e-12

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
        out = rsdf_builder._unpack_cderi_v2(
            dat[kp], idx[0], np.where(kk_conserv == kp)[1],
            conj_images_in_bvk_cell(kmesh), expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 5e-11

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
    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(cell, auxcell, kpts)
    ref = build_cderi(cell, auxcell, kpts)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder._unpack_cderi_v2(
            dat[kp], idx[0], np.where(kk_conserv == kp)[1],
            conj_images_in_bvk_cell(kmesh), expLk, nao)
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
        out = rsdf_builder._unpack_cderi_v2(
            dat[kp], idx[0], np.where(kk_conserv == kp)[1],
            conj_images_in_bvk_cell(kmesh), expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            print(ki, kj)
            assert abs(_ref - out[ki]).max() < 1e-10

def test_sr_kpts():
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
    omega = 0.2
    kmesh = [3,1,1]
    kpts = cell.make_kpts(kmesh)
    with lib.temporary_env(rsdf_builder, PREFER_ED=False):
        gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts, omega=omega)

    auxcell.omega = cell.omega = -omega
    cell.precision = 1e-10
    auxcell.precision = 1e-10
    auxcell.rcut = 35.0
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
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
    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dat, dat_neg, idx = rsdf_builder.compressed_cderi_j_only(cell, auxcell, kmesh)
    ref = build_cderi(cell, auxcell, kpts)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))

    out = rsdf_builder._unpack_cderi_v2(
        dat[0], idx[0], np.where(kk_conserv == 0)[1],
        conj_images_in_bvk_cell(kmesh), expLk, nao)
    for ki in range(nkpts):
        _ref = ref[ki, ki]
        assert abs(_ref - out[ki]).max() < 1e-11

def test_sr_j_only():
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
    omega = 0.2
    kmesh = [3,1,1]
    kpts = cell.make_kpts(kmesh)
    with lib.temporary_env(rsdf_builder, PREFER_ED=False):
        gpu_dat, dat_neg = build_cderi(cell, auxcell, kpts, omega=omega, j_only=True)

    auxcell.omega = cell.omega = -omega
    cell.precision = 1e-10
    auxcell.precision = 1e-10
    auxcell.rcut = 35.0
    dfbuilder = _RSGDFBuilder(cell, auxcell, kpts)
    dfbuilder.j_only = True
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

def _get_2c2e_slow(auxcell, uniq_kpts, omega):
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
    dat = rsdf_builder._get_2c2e(cell, kpts, 0., omega)
    ref = _get_2c2e_slow(cell, kpts, omega)
    assert abs(dat - cp.asarray(ref)).max() < 1e-10

def test_sr_2c2e():
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
    dat = rsdf_builder._get_2c2e(cell, kpts, omega, 0.3)

    cell.omega = -omega
    dfbuilder = _RSGDFBuilder(cell, cell, kpts)
    dfbuilder.omega = omega
    dfbuilder.build()
    ref = dfbuilder.get_2c2e(kpts)
    for k in range(len(kpts)):
        assert abs(dat[k].get() - ref[k]).max() < 1e-10

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
               [0, [.035, 1.]]]
    )
    auxcell = df_cpu.make_auxcell(cell)
    nao = cell.nao
    kmesh = [2, 1, 1]
    kpts = cell.make_kpts(kmesh)
    with lib.temporary_env(rsdf_builder, PREFER_ED=True):
        dat, dat_neg, idx = rsdf_builder.compressed_cderi_kk(
            cell, auxcell, kpts=kpts, omega=0.15)
        ref = build_cderi(cell, auxcell, kpts, omega=0.15)[0]
    kk_conserv = k2gamma.double_translation_indices(kmesh)
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, kmesh, True)
    expLk = cp.exp(1j*cp.asarray(bvkmesh_Ls.dot(kpts.T)))
    for kp in sorted(dat):
        out = rsdf_builder._unpack_cderi_v2(
            dat[kp], idx[0], np.where(kk_conserv == kp)[1],
            conj_images_in_bvk_cell(kmesh), expLk, nao)
        ki_idx, kj_idx = np.where(kk_conserv == kp)
        for ki, kj in zip(ki_idx, kj_idx):
            if (ki, kj) in ref:
                _ref = ref[ki, kj]
            else:
                _ref = ref[kj, ki].conj().transpose(0,2,1)
            _ref = np.einsum('pij,plk->ijkl', _ref, _ref.conj(), optimize=True)
            _dat = np.einsum('pij,plk->ijkl', out[ki], out[ki].conj(), optimize=True)
            print(ki, kj)
            assert abs(_ref - _dat).max() < 1e-8

def test_diffuse_only():
    cell = pyscf.M(
        atom = 'He 1. .5 .5;C .1 1.3 2.1',
        basis = {'He': [[0, [0.12, 1]]],
                 'C' :[[0, [0.08, 1]]],},
        a = np.eye(3) * 2.5,
    )
    auxcell = cell.copy()
    auxcell.basis = [[0, [1., 1.]], [0, [.5, 1.]]]
    auxcell.build(False, False)

    opt = rsdf_builder.SRInt3c2eOpt(cell, auxcell, 0.3)
    opt.mesh = [7, 7, 7]
    opt.build(separate_dd=True)
    assert len(opt.img_idx) == 0

    def unexpected_sr(*args, **kwargs):
        raise AssertionError('An all-DD build must not evaluate SR 3c2e')

    with lib.temporary_env(rsdf_builder.SRInt3c2eOpt,
                           int3c2e_evaluator=unexpected_sr):
        full, _ = build_cderi(cell, auxcell, int3c2e_opt=opt)
        excluded, _ = build_cderi(cell, auxcell, exclude_dd=True, int3c2e_opt=opt)
    eri = cp.einsum('pij,pkl->ijkl', full[0,0], full[0,0])
    assert abs(lib.fp(eri.get()) - 0.002616152259096199) < 1e-10
    assert cp.all(excluded[0,0] == 0)

    kmesh = [3,2,1]
    kpts = cell.make_kpts(kmesh)
    opt = rsdf_builder.SRInt3c2eOpt(cell, auxcell, 0.3, kmesh)
    opt.mesh = [7, 7, 7]
    opt.build(separate_dd=True)
    assert len(opt.img_idx) == 0

    with lib.temporary_env(rsdf_builder.SRInt3c2eOpt,
                           int3c2e_evaluator=unexpected_sr):
        full, _ = build_cderi(cell, auxcell, kpts, kmesh=kmesh, int3c2e_opt=opt)
        excluded, _ = build_cderi(cell, auxcell, kpts, kmesh=kmesh,
                                  exclude_dd=True, int3c2e_opt=opt)
    eri = cp.einsum('pij,pkl->ijkl', full[0,0], full[0,0])
    assert abs(lib.fp(eri.get()) - 0.002616152259096199) < 1e-10
    assert cp.all(excluded[0,0] == 0)


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('mode', ['gamma', 'j_only', 'kk'])
@pytest.mark.parametrize('omega', [None, -.4])
def test_general_contraction_v2(cart, mode, omega):
    # Include d functions and diffuse primitives in general contractions to
    # exercise Cartesian/spherical conversion and compact/DD recontraction.
    cell = pyscf.M(
        atom='He .2 .3 .1; He 1.1 .8 1.4', unit='Bohr',
        a=np.eye(3)*6, cart=cart, precision=1e-10, verbose=0,
        basis=[[0, [2., .6, -.2], [.3, .4, .7], [.09, .2, .3]],
               [2, [1.8, .7, .2], [.6, .3, -.4]]])
    auxcell = cell.copy()
    auxcell.basis = [[0, [1., 1.]], [0, [.3, 1.]],
                     [1, [.7, 1.]], [2, [.8, 1.]]]
    auxcell.build()
    if omega is not None:
        cell.omega = auxcell.omega = omega
    kmesh = [1, 1, 1] if mode == 'gamma' else [3, 1, 1]
    kpts = cell.make_kpts(kmesh)
    opt = rsdf_builder.SRInt3c2eOpt(cell, auxcell, .4, kmesh).build(separate_dd=True)
    assert opt.cell.nao > cell.nao
    assert opt.dd_ft_opt is not None
    dat, negative = build_cderi(cell, auxcell, kpts, kmesh,
                                j_only=mode == 'j_only', omega=omega, int3c2e_opt=opt)
    assert negative is None
    cpu = _RSGDFBuilder(cell, auxcell, kpts)
    cpu.omega = .4
    cpu.j2c_eig_always = True
    with tempfile.NamedTemporaryFile() as tmpf:
        cpu.make_j3c(tmpf.name, aosym='s1', j_only=mode == 'j_only')
        for (ki, kj), value in dat.items():
            assert value.shape[1:] == (cell.nao, cell.nao)
            with _load3c(tmpf.name, 'j3c', kpts[[ki, kj]]) as cderi:
                ref = cderi[:].reshape(-1, cell.nao, cell.nao)
            # Metric eigenspaces may differ by unitary rotations.
            eri = np.einsum('Lij,Lkl->ijkl', value.get().conj(), value.get())
            expected = np.einsum('Lij,Lkl->ijkl', ref.conj(), ref)
            np.testing.assert_allclose(eri, expected, atol=2e-8, rtol=1e-8)


@pytest.mark.parametrize('gamma', [True, False])
def test_general_contraction_gdf_jk(gamma):
    from pyscf.pbc.df import GDF as CPU_GDF
    from gpu4pyscf.pbc.df.df import GDF
    cell = pyscf.M(
        atom='He .2 .3 .1; He 1.1 .8 1.4', unit='Bohr',
        a=np.eye(3)*6, precision=1e-10, verbose=0,
        basis=[[0, [2., .6, -.2], [.3, .4, .7], [.09, .2, .3]],
               [2, [1.8, .7, .2], [.6, .3, -.4]]])
    kpts = cell.make_kpts([1, 1, 1] if gamma else [3, 1, 1])
    auxbasis = [[0, [1., 1.]], [0, [.3, 1.]], [1, [.7, 1.]], [2, [.8, 1.]]]
    gpu = GDF(cell, kpts)
    gpu.auxbasis = auxbasis
    gpu.is_gamma_point = gamma
    cpu = CPU_GDF(cell, kpts)
    cpu.auxbasis = auxbasis
    rng = np.random.default_rng(41)
    factor = rng.random((len(kpts), cell.nao, 3)) * .1
    if not gamma:
        factor[2] = factor[1]
    dm = factor @ factor.transpose(0, 2, 1)
    if gamma:
        dm = dm[0]
    vj, vk = gpu.get_jk(cp.asarray(dm), kpts=kpts)
    refj, refk = cpu.get_jk(dm, kpts=kpts, exxdiv=None)
    np.testing.assert_allclose(vj.get(), refj, atol=2e-8, rtol=1e-8)
    np.testing.assert_allclose(vk.get(), refk, atol=2e-8, rtol=1e-8)
