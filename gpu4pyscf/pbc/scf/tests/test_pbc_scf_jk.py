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
from pyscf.pbc.df import fft as fft_cpu
from gpu4pyscf.pbc.df import fft
from gpu4pyscf.pbc.scf import rsjk

def test_sr_vk_hermi1_gamma_point_vs_cpu():
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        C   1.      1.    0.
        ''',
        a=np.eye(3)*6.,
        basis={'default': ('ccpvdz', [[3, [.5, 1]]]),
               'H': 'ccpvdz'}
    )

    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nao, nao)*.1 - .05
    dm = dm.dot(dm.T)
    vk = rsjk.PBCJKmatrixOpt(cell).build()._get_k_sr(dm, hermi=1, remove_G0=False).get()
    cell.precision = 1e-10
    cell.build(0, 0)
    with_rsjk = RangeSeparationJKBuilder(cell)
    with_rsjk.exclude_dd_block = False
    with_rsjk.allow_drv_nodddd = False
    omega = rsjk.OMEGA
    ref = with_rsjk.build(omega)._get_jk_sr(
        dm, hermi=1, kpts=np.zeros((1,3)), with_j=False)[0,0]
    assert abs(vk - ref).max() < 1e-8

def test_sr_vk_hermi1_kpts_vs_cpu():
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        ''',
        a=np.eye(3)*6.,
        basis={'O': ('ccpvdz', [[3, [.5, 1]]]),
               'H': 'ccpvdz'}
    )

    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts)) * .2
    vk = rsjk.PBCJKmatrixOpt(cell).build()._get_k_sr(
        dm, hermi=1, kpts=kpts, remove_G0=False).get()
    cell.precision = 1e-10
    cell.build(0, 0)
    with_rsjk = RangeSeparationJKBuilder(cell, kpts=kpts)
    with_rsjk.exclude_dd_block = False
    with_rsjk.allow_drv_nodddd = False
    omega = rsjk.OMEGA
    ref = with_rsjk.build(omega)._get_jk_sr(
        dm, hermi=1, kpts=kpts, with_j=False)[0,0]
    # Small errors might be due to the rcut, Ecut estimation in the CPU
    # implementation
    assert abs(vk - ref).max() < 1e-8

def test_sr_vk_hermi1_gamma_point_vs_fft():
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
    dm = np.random.rand(nao, nao)*.1 - .05
    dm = dm.dot(dm.T)
    vk = rsjk.PBCJKmatrixOpt(cell).build()._get_k_sr(dm, hermi=1, remove_G0=True).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    cell.omega = -rsjk.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, with_j=False)[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_sr_vk_hermi1_kpts_vs_fft():
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
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts)) * .2
    vk = rsjk.PBCJKmatrixOpt(cell).build()._get_k_sr(dm, hermi=1, kpts=kpts, remove_G0=True).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    cell.omega = -rsjk.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, with_j=False, kpts=kpts)[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_sr_vk_hermi0_gamma_point_vs_fft():
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
        basis=[[0, [.35, 1]], [1, [.3, 1]]],
    )
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nao, nao)*.2
    vk = rsjk.PBCJKmatrixOpt(cell).build()._get_k_sr(dm, hermi=0, remove_G0=True).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    cell.omega = -rsjk.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, with_j=False)[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_sr_vk_hermi0_kpts_vs_fft():
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
    nkpts = len(kpts)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nkpts, nao, nao)*.2
    dm[4:6] = dm[2:4].conj()
    vk = rsjk.PBCJKmatrixOpt(cell).build()._get_k_sr(dm, hermi=0, kpts=kpts, remove_G0=True).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    cell.omega = -rsjk.OMEGA
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, kpts=kpts, with_j=False)[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_vk_kpts_band_vs_fft():
    pass

def test_vk_hermi1_gamma_point_vs_fft():
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
    dm = np.random.rand(nao, nao)*.1 - .05
    dm = dm.dot(dm.T)
    vk = rsjk.get_k(cell, dm, hermi=1, exxdiv='ewald').get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, with_j=False, exxdiv='ewald')[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_vk_hermi1_kpts_vs_fft():
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
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts)) * .2
    vk = rsjk.get_k(cell, dm, hermi=1, kpts=kpts, exxdiv='ewald').get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, hermi=1, with_j=False, kpts=kpts, exxdiv='ewald')[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_vk_hermi0_gamma_point_vs_fft():
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
    vk = rsjk.get_k(cell, dm, hermi=0).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, with_j=False)[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_vk_hermi0_kpts_vs_fft():
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
    nkpts = len(kpts)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nkpts, nao, nao)*.2
    dm[4:6] = dm[2:4].conj()
    vk = rsjk.get_k(cell, dm, hermi=0, kpts=kpts).get()

    cell.precision = 1e-10
    cell.build(0, 0)
    ref = fft.FFTDF(cell).get_jk(dm, hermi=0, kpts=kpts, with_j=False)[1].get()
    assert abs(vk - ref).max() < 1e-8

def test_ejk_sr_ip1_per_atom_gamma_point():
    cell = pyscf.M(
        atom = '''
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        O   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        a=np.eye(3)*4.,
        basis={'H': [[0, [.25, 1]], [1, [.3, 1]]],
               'O': [[0, [.3,  1]], [2, [.2, 1]]]},
    )
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(nao, nao)
    dm = dm.dot(dm.T)
    ejk = rsjk.PBCJKmatrixOpt(cell).build()._get_ejk_sr_ip1(dm, remove_G0=True)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    cell.omega = -rsjk.OMEGA
    vj, vk = fft_cpu.FFTDF(cell).get_jk_e1(dm, exxdiv=None)
    vhf = vj - vk * .5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xpq,qp->x', vhf[:,p0:p1], dm[:,p0:p1])
    # Reduced accuracy because integral screening is set to cell.precision**.5 in rsjk
    assert abs(ejk - ref).max() < 1e-6

def test_ejk_sr_ip1_per_atom_kpts():
    cell = pyscf.M(
        atom = '''
        H   1.757    0.    0.4696
        H   0.757    0.    0.4696
        O   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        a=np.eye(3)*4.,
        basis={'H': [[0, [.35, 1]], [1, [.3, 1]]],
               'O': [[0, [.35, 1]], [2, [.3, 1]]]},
    )
    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    ejk = rsjk.PBCJKmatrixOpt(cell).build()._get_ejk_sr_ip1(
        dm, kpts=kpts, remove_G0=True)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    cell.omega = -rsjk.OMEGA
    vj, vk = fft_cpu.FFTDF(cell).get_jk_e1(dm, kpts=kpts, exxdiv=None)
    vhf = vj - vk * .5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xkpq,kqp->x', vhf[:,:,p0:p1], dm[:,:,p0:p1]).real
    ref /= len(kpts)
    # Reduced accuracy because integral screening is set to cell.precision**.5 in rsjk
    assert abs(ejk - ref).max() < 5e-6

def test_ejk_ip1_per_atom_gamma_point():
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        a=np.eye(3)*4.,
        basis=[[0, [.25, 1]], [1, [.3, 1]]],
    )
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(2, nao, nao) * .5
    dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])

    with_rsjk = rsjk.PBCJKmatrixOpt(cell).build()
    ejk = with_rsjk._get_ejk_sr_ip1(dm[0], remove_G0=True)
    ejk += with_rsjk._get_ejk_lr_ip1(dm[0])
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    with_fft = fft_cpu.FFTDF(cell)
    vj, vk = with_fft.get_jk_e1(dm[0], exxdiv=None)
    vhf = vj - vk*.5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xpq,qp->x', vhf[:,p0:p1], dm[0,:,p0:p1])
    assert abs(ejk - ref).max() < 1e-6

    ejk = with_rsjk._get_ejk_sr_ip1(dm, remove_G0=True)
    ejk += with_rsjk._get_ejk_lr_ip1(dm)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    vj, vk = with_fft.get_jk_e1(dm, exxdiv=None)
    vhf = vj[:,:1] + vj[:,1:] - vk
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xnpq,nqp->x', vhf[:,:,p0:p1], dm[:,:,p0:p1])
    assert abs(ejk - ref).max() < 1e-6

def test_ejk_ip1_per_atom_kpts():
    cell = pyscf.M(
        atom = '''
        O   0.000    0.    0.1174
        C   1.      1.    0.
        H   4.      0.    3.
        H   0.      1.    .6
        ''',
        a=np.eye(3)*4.,
        basis=[[0, [.25, 1]], [1, [.3, 1]]],
    )
    kpts = cell.make_kpts([1,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    with_rsjk = rsjk.PBCJKmatrixOpt(cell).build()
    ejk = with_rsjk._get_ejk_sr_ip1(dm, kpts=kpts, remove_G0=True)
    ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts=kpts)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    vj, vk = fft_cpu.FFTDF(cell).get_jk_e1(dm, kpts=kpts, exxdiv=None)
    vhf = vj - vk * .5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xkpq,kqp->x', vhf[:,:,p0:p1], dm[:,:,p0:p1]).real
    ref /= len(kpts)
    assert abs(ejk - ref).max() < 5e-6
