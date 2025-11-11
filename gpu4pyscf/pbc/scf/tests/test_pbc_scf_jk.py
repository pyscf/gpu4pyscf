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
from packaging.version import Version
import pyscf
from pyscf import lib, gto
from pyscf.pbc.scf.rsjk import RangeSeparationJKBuilder
from pyscf.pbc.df import fft as fft_cpu
from pyscf.pbc.tools import pbc as pbctools
from gpu4pyscf.pbc.df import fft
from gpu4pyscf.pbc.scf import rsjk
from gpu4pyscf.pbc.tools.pbc import probe_charge_sr_coulomb

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
    vk = rsjk.PBCJKMatrixOpt(cell).build()._get_k_sr(dm, hermi=1, exxdiv='ewald').get()
    s = cell.pbc_intor('int1e_ovlp', hermi=1)
    fac = probe_charge_sr_coulomb(cell, rsjk.OMEGA)
    vk += np.einsum('ij,jk,kl->il', s, dm, s) * fac

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
    vk = rsjk.PBCJKMatrixOpt(cell).build()._get_k_sr(
        dm, hermi=1, kpts=kpts, exxdiv='ewald').get()
    s = np.array(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
    fac = probe_charge_sr_coulomb(cell, rsjk.OMEGA, kpts) / len(kpts)
    vk += np.einsum('Kij,Kjk,Kkl->Kil', s, dm, s) * fac

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
    vk = rsjk.PBCJKMatrixOpt(cell).build()._get_k_sr(dm, hermi=1).get()

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
    vk = rsjk.PBCJKMatrixOpt(cell).build()._get_k_sr(dm, hermi=1, kpts=kpts).get()

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
    vk = rsjk.PBCJKMatrixOpt(cell).build()._get_k_sr(dm, hermi=0).get()

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
    vk = rsjk.PBCJKMatrixOpt(cell).build()._get_k_sr(dm, hermi=0, kpts=kpts).get()

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
    dm = cell.pbc_intor('int1e_ovlp')
    ejk = rsjk.PBCJKMatrixOpt(cell).build()._get_ejk_sr_ip1(dm, exxdiv=None)
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
    kpts = cell.make_kpts([3,1,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    if Version(pyscf.__version__) > Version('2.11'):
        exxdiv = 'ewald'
    else:
        exxdiv = None
    ejk = rsjk.PBCJKMatrixOpt(cell).build()._get_ejk_sr_ip1(dm, kpts=kpts, exxdiv=exxdiv)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    cell.omega = -rsjk.OMEGA
    vj, vk = fft_cpu.FFTDF(cell).get_jk_e1(dm, kpts=kpts, exxdiv=exxdiv)
    vhf = vj - vk * .5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xkpq,kqp->x', vhf[:,:,p0:p1], dm[:,:,p0:p1]).real
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
    kpt = np.zeros(3)
    np.random.seed(9)
    nao = cell.nao
    dm = np.random.rand(2, nao, nao) * .5
    dm = np.array([dm[0].dot(dm[0].T), dm[1].dot(dm[1].T)])

    with_rsjk = rsjk.PBCJKMatrixOpt(cell).build()
    ejk = with_rsjk._get_ejk_sr_ip1(dm[0], kpts=kpt)
    ejk += with_rsjk._get_ejk_lr_ip1(dm[0], kpts=kpt)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    with_fft = fft_cpu.FFTDF(cell)
    vj, vk = with_fft.get_jk_e1(dm[0])
    vhf = vj - vk*.5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xpq,qp->x', vhf[:,p0:p1], dm[0,:,p0:p1])
    assert abs(ejk - ref).max() < 1e-6

    if Version(pyscf.__version__) > Version('2.11'):
        ejk = with_rsjk._get_ejk_sr_ip1(dm, kpts=kpt, exxdiv='ewald')
        ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts=kpt, exxdiv='ewald')
        assert abs(ejk.sum(axis=0)).max() < 1e-8

        vj, vk = with_fft.get_jk_e1(dm, exxdiv='ewald')
        vhf = vj[:,:1] + vj[:,1:] - vk
        aoslices = cell.aoslice_by_atom()
        ref = np.empty((cell.natm, 3))
        for i in range(cell.natm):
            p0, p1 = aoslices[i, 2:]
            ref[i] = np.einsum('xnpq,nqp->x', vhf[:,:,p0:p1], dm[:,:,p0:p1])
        assert abs(ejk - ref).max() < 1e-6
    else:
        ejk = with_rsjk._get_ejk_sr_ip1(dm, kpts=kpt, exxdiv=None)
        ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts=kpt, exxdiv=None)
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
    kpts = cell.make_kpts([3,2,1])
    dm = np.asarray(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    with_rsjk = rsjk.PBCJKMatrixOpt(cell).build()
    ejk = with_rsjk._get_ejk_sr_ip1(dm, kpts=kpts, exxdiv=None)
    ejk += with_rsjk._get_ejk_lr_ip1(dm, kpts=kpts, exxdiv=None)
    assert abs(ejk.sum(axis=0)).max() < 1e-8

    vj, vk = fft_cpu.FFTDF(cell).get_jk_e1(dm, kpts=kpts, exxdiv=None)
    vhf = vj - vk * .5
    aoslices = cell.aoslice_by_atom()
    ref = np.empty((cell.natm, 3))
    for i in range(cell.natm):
        p0, p1 = aoslices[i, 2:]
        ref[i] = np.einsum('xkpq,kqp->x', vhf[:,:,p0:p1], dm[:,:,p0:p1]).real
    assert abs(ejk - ref).max() < 5e-6

def test_jk_sr_strain_deriv1():
    from gpu4pyscf.pbc.grad import rks, rks_stress
    from gpu4pyscf.pbc.grad import krks, krks_stress
    a = np.eye(3) * 6.
    np.random.seed(5)
    a += np.random.rand(3, 3) - .5
    cell = pyscf.M(
        atom='He 1 1 1; He 2 1.5 2.4',
        basis=[[0, [1.5, 1]]], a=a, unit='Bohr')
    # gamma point calculations
    nao = cell.nao
    dm = np.random.rand(nao, nao) - .5
    dm = dm.dot(dm.T)
    with_rsjk = rsjk.PBCJKMatrixOpt(cell).build()
    sigma = with_rsjk._get_jk_sr_strain_deriv(dm)
    #sigma_wo_exxdiv = with_rsjk._get_jk_sr_strain_deriv(dm, exxdiv=False)

    Ls = cell.get_lattice_Ls(rcut=cell.rcut+4)
    Ls = Ls[np.argsort(np.linalg.norm(Ls-.1, axis=1))]
    scell = cell.copy()
    scell = pbctools._build_supcell_(scell, cell, Ls)
    scell.omega = -.3
    nimgs = len(Ls)
    aoslices = cell.aoslice_by_atom()
    ao_repeats = aoslices[:,3] - aoslices[:,2]
    bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)
    sc_eri = scell.intor('int2e_ip1').reshape(3,nimgs,nao,nimgs,nao,nimgs,nao,nimgs,nao)
    eri1 = np.einsum('xokplinj,oky->xyijkl', sc_eri[:,:,:,:,:,0], bas_coords+Ls[:,None])
    eri1+= np.einsum('xplokinj,ply->xyijkl', sc_eri[:,:,:,:,:,0], bas_coords+Ls[:,None])
    eri1+= np.einsum('xinjokpl,iy->xyijkl', sc_eri[:,0], bas_coords)
    eri1+= np.einsum('xnjiokpl,njy->xyijkl', sc_eri[:,:,:,0], bas_coords+Ls[:,None])
    ej = -.5 * np.einsum('xyijkl,ji,lk->xy', eri1, dm, dm)
    ek = -.5 * np.einsum('xyijkl,jk,li->xy', eri1, dm, dm)
    #ref_wo_exxdiv = ej - ek*.5

    sc_s1 = scell.intor('int1e_ipovlp').reshape(3,nimgs,nao,nimgs,nao)
    s1 = np.einsum('xmij,miy->xyij', sc_s1[:,:,:,0], bas_coords+Ls[:,None])
    s1+= np.einsum('xinj,iy->xyij', sc_s1[:,0], bas_coords)
    s0 = cell.pbc_intor('int1e_ovlp')
    wcoulG_SR_at_G0 = np.pi/scell.omega**2/cell.vol
    j_dm = np.einsum('ij,ji->', s0, dm)
    ej += np.einsum('xyij,ji->xy', s1, dm) * wcoulG_SR_at_G0 * j_dm
    ej += .5 * j_dm * wcoulG_SR_at_G0 * j_dm * np.eye(3)
    ek += np.einsum('xyij,jk,kl,li->xy', s1, dm, s0, dm) * wcoulG_SR_at_G0
    ek += .5 *np.einsum('ij,jk,kl,li->', s0, dm, s0, dm) * wcoulG_SR_at_G0 * np.eye(3)
    ref = ej - ek*.5
    assert abs(ref - sigma).max() < 1e-6

    sigma = with_rsjk._get_jk_sr_strain_deriv(dm, k_factor=0)
    cell.omega = scell.omega
    xc = 'lda,'
    mf_grad = cell.RKS(xc=xc).to_gpu().Gradients()
    ref = rks_stress.get_vxc(mf_grad, cell, dm, with_j=True, with_nuc=False)
    ref -= rks_stress.get_vxc(mf_grad, cell, dm, with_j=False, with_nuc=False)
    assert abs(ref - sigma).max() < 1e-7

    # kpts calculations
    kpts = cell.make_kpts([1,1,1])
    dm = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    sigma = with_rsjk._get_jk_sr_strain_deriv(dm, kpts)
    #sigma_wo_exxdiv = with_rsjk._get_jk_sr_strain_deriv(dm, exxdiv=False)

    eri1 = np.einsum('xokplinj,oky->xyinjokpl', sc_eri[:,:,:,:,:,0], bas_coords+Ls[:,None])
    eri1+= np.einsum('xplokinj,ply->xyinjokpl', sc_eri[:,:,:,:,:,0], bas_coords+Ls[:,None])
    eri1+= np.einsum('xinjokpl,iy->xyinjokpl', sc_eri[:,0], bas_coords)
    eri1+= np.einsum('xnjiokpl,njy->xyinjokpl', sc_eri[:,:,:,0], bas_coords+Ls[:,None])
    expLk = np.exp(1j*Ls.dot(kpts.T))
    eri1 = np.einsum('xyiNjOkPl,Nn,Oo,Pp->xyinjokpl', eri1, expLk, expLk.conj(), expLk, optimize=True)
    nkpts = len(kpts)
    ej = -.5 * np.einsum('xyinjokpl,op,nji,plk->xy', eri1, np.eye(nkpts), dm, dm).real
    ek = -.5 * np.einsum('xyinjokpl,no,ojk,pli->xy', eri1, np.eye(nkpts), dm, dm).real
    #ref_wo_exxdiv = ej - ek*.5

    sc_s1 = scell.intor('int1e_ipovlp').reshape(3,nimgs,nao,nimgs,nao)
    s1 = np.einsum('xmij,miy,mk->xykij', sc_s1[:,:,:,0], bas_coords+Ls[:,None], expLk)
    s1+= np.einsum('xinj,iy,nk->xykij', sc_s1[:,0], bas_coords, expLk)
    s0 = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    wcoulG_SR_at_G0 = np.pi/scell.omega**2/cell.vol
    j_dm = np.einsum('kij,kji->', s0, dm).real
    ej += np.einsum('xykij,kji->xy', s1, dm).real * wcoulG_SR_at_G0 * j_dm
    ej += .5 * j_dm * wcoulG_SR_at_G0 * j_dm * np.eye(3)
    ek += np.einsum('xytij,tjk,tkl,tli->xy', s1, dm, s0, dm).real * wcoulG_SR_at_G0
    ek += .5 *np.einsum('tij,tjk,tkl,tli->', s0, dm, s0, dm).real * wcoulG_SR_at_G0 * np.eye(3)
    ref = ej - ek*.5
    assert abs(ref - sigma).max() < 2e-5

    sigma = with_rsjk._get_jk_sr_strain_deriv(dm, kpts, k_factor=0)
    cell.omega = scell.omega
    xc = 'lda,'
    mf_grad = cell.KRKS(xc=xc, kpts=kpts).to_gpu().Gradients()
    ref = krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_j=True, with_nuc=False)
    ref -= krks_stress.get_vxc(mf_grad, cell, dm, kpts=kpts, with_j=False, with_nuc=False)
    assert abs(ref - sigma).max() < 1e-7
