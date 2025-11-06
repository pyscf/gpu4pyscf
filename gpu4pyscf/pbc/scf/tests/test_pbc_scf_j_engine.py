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

import numpy as np
import numpy as cp
import pyscf
from pyscf import lib, gto
from pyscf.pbc.scf.rsjk import RangeSeparationJKBuilder
from gpu4pyscf.pbc.df import fft
from gpu4pyscf.pbc.scf import j_engine
from gpu4pyscf.scf.j_engine import get_j

def test_j_engine():
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
    dm = np.random.rand(nao, nao)*.1 - .05
    dm = dm.dot(dm.T)
    vj = j_engine.PBCJMatrixOpt(cell).build()._get_j_sr(dm, hermi=1).get()
    cell.precision = 1e-10
    cell.build(0, 0)
    with_rsjk = RangeSeparationJKBuilder(cell)
    with_rsjk.exclude_dd_block = False
    with_rsjk.allow_drv_nodddd = False
    omega = j_engine.OMEGA
    ref = with_rsjk.build(omega)._get_jk_sr(
        dm, hermi=1, kpts=np.zeros((1,3)), with_k=False)[0,0]
    assert abs(vj - ref).max() < 1e-8

def test_sr_vj_hermi1_kpts_vs_cpu():
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        ''',
        a=np.eye(3)*7.,
        basis=('ccpvdz', [[3, [.5, 1]]]),
    )

    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts)) * .2
    vj = j_engine.PBCJMatrixOpt(cell).build()._get_j_sr(dm, hermi=1, kpts=kpts).get()
    cell.precision = 1e-10
    cell.build(0, 0)
    with_rsjk = RangeSeparationJKBuilder(cell, kpts=kpts)
    with_rsjk.exclude_dd_block = False
    with_rsjk.allow_drv_nodddd = False
    omega = j_engine.OMEGA
    ref = with_rsjk.build(omega)._get_jk_sr(
        dm, hermi=1, kpts=kpts, with_k=False)[0,0]
    # Small errors might be due to the rcut, Ecut estimation in the CPU
    # implementation
    assert abs(vj - ref).max() < 1e-8

def test_sr_vj_hermi1_gamma_point_vs_fft():
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
    kpt = np.zeros(3)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(2, nao, nao)*.5
    dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])
    vj = j_engine.PBCJMatrixOpt(cell).build()._get_j_sr(dm, hermi=1, kpts=kpt).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    omega = cell.omega = -j_engine.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, kpts=kpt, with_k=False)[0].get()
    s = cell.pbc_intor('int1e_ovlp')
    wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
    wcoulG_SR_at_G0 *= np.einsum('ij,nji->n', s, dm)
    ref += wcoulG_SR_at_G0[:,None,None] * s
    assert abs(vj - ref).max() < 1e-8

def test_sr_vj_hermi1_kpts_vs_fft():
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
    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    vj = j_engine.PBCJMatrixOpt(cell).build()._get_j_sr(dm, hermi=1, kpts=kpts).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    omega = cell.omega = -j_engine.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, with_k=False, kpts=kpts)[0].get()
    s = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
    wcoulG_SR_at_G0 *= np.einsum('kij,kji->', s, dm) / len(kpts)
    ref += wcoulG_SR_at_G0 * s
    assert abs(vj - ref).max() < 1e-8

def test_sr_vj_hermi0_gamma_point_vs_fft():
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
    dm = np.random.rand(nao, nao)*.2
    vj = j_engine.PBCJMatrixOpt(cell).build()._get_j_sr(dm, hermi=0).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    omega = cell.omega = -j_engine.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, with_k=False)[0].get()
    s = cell.pbc_intor('int1e_ovlp')
    wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
    wcoulG_SR_at_G0 *= np.einsum('ij,ji->', s, dm)
    ref += wcoulG_SR_at_G0 * s
    assert abs(vj - ref).max() < 1e-8

def test_sr_vj_hermi0_kpts_vs_fft():
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
    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    vj = j_engine.PBCJMatrixOpt(cell).build()._get_j_sr(dm, hermi=0, kpts=kpts).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    omega = cell.omega = -j_engine.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, kpts=kpts, with_k=False)[0].get()
    s = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    wcoulG_SR_at_G0 = np.pi / omega**2 / cell.vol
    wcoulG_SR_at_G0 *= np.einsum('kij,kji->', s, dm) / len(kpts)
    ref += wcoulG_SR_at_G0 * s
    assert abs(vj - ref).max() < 1e-8

def test_vj_kpts_band_vs_fft():
    pass

def test_vj_hermi1_gamma_point_vs_fft():
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
    kpt = np.zeros(3)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(2, nao, nao)*.5
    dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])
    vj = j_engine.get_j(cell, dm, hermi=1).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, kpts=kpt, with_k=False)[0].get()
    assert abs(vj - ref).max() < 1e-8

def test_vj_hermi1_kpts_vs_fft():
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
    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    vj = j_engine.get_j(cell, dm, hermi=1, kpts=kpts).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, hermi=1, with_k=False, kpts=kpts)[0].get()
    assert abs(vj - ref).max() < 1e-8

def test_vj_hermi0_gamma_point_vs_fft():
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
    kpt = np.zeros(3)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(2, nao, nao)*.5
    dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])
    vj = j_engine.get_j(cell, dm, hermi=0).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, kpts=kpt, with_k=False)[0].get()
    assert abs(vj - ref).max() < 1e-8

def test_vj_hermi0_kpts_vs_fft():
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
    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    vj = j_engine.get_j(cell, dm, hermi=0, kpts=kpts).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, kpts=kpts, with_k=False)[0].get()
    assert abs(vj - ref).max() < 1e-8
