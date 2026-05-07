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

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.df.df import make_auxcell
from gpu4pyscf.lib.cupy_helper import tag_array, contract
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.grad import rhf, uhf
from gpu4pyscf.pbc.df.grad import krhf, kuhf
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.pbc.df.int2c2e import sr_int2c2e
from gpu4pyscf.pbc.df import rsdf_builder

def test_ej_ip1_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2)

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.3

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ej = rhf._jk_energy_per_atom(opt, dm, hermi=1, k_factor=0, omega=omega)
    assert abs(ej.sum(axis=0)).max() < 1e-11

    disp = 1e-3
    atom_coords = cell.atom_coords()
    dm_cart = opt.cell.apply_C_mat_CT(dm)
    def eval_j(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        opt = int3c2e.SRInt3c2eOpt(cell1, auxcell1, omega).build()
        jaux = opt.contract_dm(dm_cart)
        j2c = sr_int2c2e(auxcell1, omega)
        atom_coords[i,x] -= disp
        return float(cp.linalg.solve(j2c, jaux).dot(jaux).get()) * .5

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_j(i, x, disp)
        e2 = eval_j(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ej[i,x]) < 5e-6

    ej = rhf._jk_energy_per_atom(opt, dm, hermi=1, k_factor=0)
    assert abs(ej.sum(axis=0)).max() < 1e-11

    def eval_j(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        cderi = rsdf_builder.build_cderi(cell1, auxcell1)[0][0,0]
        jaux = cp.einsum('rpq,qp->r', cderi, dm)
        atom_coords[i,x] -= disp
        return float(jaux.dot(jaux).get()) * .5

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_j(i, x, disp)
        e2 = eval_j(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ej[i,x]) < 5e-6

def test_ejk_ip1_gamma_point_without_long_range():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2)

    auxcell = cell.copy()
    auxcell.basis = {
        'C':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.3

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ek = rhf._jk_energy_per_atom(opt, dm, hermi=1, j_factor=1, k_factor=1, omega=omega)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, omega=omega)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = .5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm, dm, optimize=True)
        ref -= .25 * cp.einsum('ijp,klp,jk,li->', cderi, cderi, dm, dm, optimize=True)
        atom_coords[i,x] -= disp
        return float(ref.get())

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 5e-6

def test_ejk_ip1_gamma_point_with_long_range():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2)

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.3

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    hermi = 1
    j_factor = 1
    k_factor = 1
    ek = rhf._jk_energy_per_atom(opt, dm, hermi, j_factor, k_factor)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        cderi = rsdf_builder.build_cderi(cell1, auxcell1)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = j_factor*.5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm, dm, optimize=True)
        ref -= k_factor*.25 * cp.einsum('ijp,klp,jk,li->', cderi, cderi, dm, dm, optimize=True)
        atom_coords[i,x] -= disp
        return float(ref.get())

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 5e-6

def test_ej_ip1_kpts_without_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.3

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    dm = cp.asarray(np.linalg.inv(cell.pbc_intor('int1e_ovlp', kpts=kpts))*.5)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    ej = krhf._j_energy_per_atom(opt, dm, kpts=kpts, omega=omega)
    assert abs(ej.sum(axis=0)).max() < 5e-12

    dm = SortedGTO.from_cell(cell).apply_C_mat_CT(dm)
    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_j(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        opt = int3c2e.SRInt3c2eOpt(cell1, auxcell1, omega, kmesh).build()
        jaux = opt.contract_dm(dm, kpts=kpts)
        j2c = sr_int2c2e(auxcell1, omega)
        ref = float(cp.linalg.solve(j2c, jaux).dot(jaux).get()) * .5
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_j(i, x, disp)
        e2 = eval_j(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ej[i,x]) < 5e-6

def test_ej_ip1_kpts_with_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.3

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    dm = cp.asarray(np.linalg.inv(cell.pbc_intor('int1e_ovlp', kpts=kpts))*.5)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    ej = krhf._j_energy_per_atom(opt, dm, kpts=kpts)
    assert abs(ej.sum(axis=0)).max() < 5e-12

    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_j(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts, kmesh, j_only=True)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = .5/nkpts**2 * jaux.dot(jaux).real.get()
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_j(i, x, disp)
        e2 = eval_j(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ej[i,x]) < 5e-6

def test_ejk_ip1_kpts_without_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [2, [.8, 1.]]]),
    }
    auxcell.build()
    omega = -0.3

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = np.linalg.eigh(cell.pbc_intor('int1e_ovlp', kpts=kpts))[1]
    nao = cell.nao
    nocc = 9
    mo_occ = np.zeros((nkpts, nao))
    mo_occ[:,:nocc] = 2
    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1

    ejk0 = krhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0.sum(axis=0)).max() < 2e-11

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk = krhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0 - ejk).max() < 1e-9

    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        nkpts = len(kpts)
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts=kpts,
                                         kmesh=kmesh, omega=omega)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,jk,li,pkl->', cderi_ij, dm[kj],
                                dm[ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

        j3c_kk = int3c2e.sr_aux_e2(cell1, auxcell1, omega, kpts, kmesh)
        j2c = sr_int2c2e(auxcell1, omega, kpts, kmesh)
        j2c_inv = cp.linalg.inv(j2c)
        jaux = cp.einsum('IIijp,Iji->p', j3c_kk, dm)
        ref = cp.einsum('p,pq,q->', jaux, j2c_inv[0], jaux).real.get()
        ref *= .5 / nkpts**2 * j_factor

        kk_conserv = krhf.double_translation_indices(kmesh)
        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                kp = kk_conserv[ki,kj]
                ek += cp.einsum('ijp,jk,li,qp,klq->', j3c_kk[ki,kj], dm[kj],
                                dm[ki], j2c_inv[kp], j3c_kk[kj,ki], optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 1e-6

def test_ejk_ip1_kpts_with_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [2, [.8, 1.]]]),
    }
    auxcell.build()
    omega = -0.3

    kmesh = [3,1,2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = np.linalg.eigh(cell.pbc_intor('int1e_ovlp', kpts=kpts))[1]
    nao = cell.nao
    nocc = 9
    mo_occ = np.zeros((nkpts, nao))
    mo_occ[:,:nocc] = 2
    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk = krhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 2e-9

    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        nkpts = len(kpts)
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts=kpts, kmesh=kmesh)[0]

        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,jk,li,pkl->', cderi_ij, dm[kj],
                                dm[ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 1e-6

def test_ejk_ip1_kpts_with_long_range1():
    cell = gto.Cell()
    cell.atom= [['H1', [0.0, 0.0, 0.0]], ['H2', [1.685,1.685,1.6]]]
    cell.a = '''
    0.00, 3.37, 3.37
    3.37, 0.00, 4.
    2.  , 3.37, 0.00'''
    cell.verbose = 0
    cell.basis = [[0, [3.3, 1]], [0, [1.1, 1]], [1, [0.8, 1]]]
    cell.unit = 'bohr'
    cell.build()
    auxcell = make_auxcell(cell)

    kmesh = [1,1,3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    mo_coeff = cp.array(np.linalg.eigh(s)[1])
    mo_occ = cp.zeros((nkpts, cell.nao))
    mo_occ[:,:3] = 2
    omega = -0.3

    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk = krhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 1e-11

    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr', inplace=False)
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr', inplace=False)
        nkpts = len(kpts)
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts=kpts, kmesh=kmesh)[0]

        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,jk,li,pkl->', cderi_ij, dm[kj], dm[ki], cderi_ji)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 1e-6

def test_uhf_ejk_ip1_gamma_point_without_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.3

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(2, nao, nao) - .5
    mo_occ = np.zeros((2, nao))
    mo_occ[:,:nocc] = 1
    dm = contract('spi,sqi->spq', mo_coeff*mo_occ[:,None], mo_coeff)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ek = uhf._jk_energy_per_atom(opt, dm, hermi=1, j_factor=1, k_factor=1, omega=omega)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    dm_sf = dm[0] + dm[1]
    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, omega=omega)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = .5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm_sf, dm_sf)
        ref -= .5 * cp.einsum('ijp,klp,sjk,sli->', cderi, cderi, dm, dm)
        atom_coords[i,x] -= disp
        return float(ref.get())

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 5e-6

def test_uhf_ejk_ip1_gamma_point_with_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]]),
    }
    auxcell.build()
    omega = -0.5

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(2, nao, nao) - .5
    mo_coeff[1]=mo_coeff[0]
    mo_occ = np.zeros((2, nao))
    mo_occ[:,:nocc] = 1
    dm = contract('spi,sqi->spq', mo_coeff*mo_occ[:,None], mo_coeff)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    hermi = 1
    j_factor = 1
    k_factor = 1
    ek = uhf._jk_energy_per_atom(opt, dm, hermi, j_factor, k_factor)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    dm_sf = dm[0] + dm[1]
    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        cderi = rsdf_builder.build_cderi(cell1, auxcell1)[0][0,0]
        cderi = cderi.transpose(1,2,0)
        ref = j_factor*.5 * cp.einsum('ijp,klp,ji,lk->', cderi, cderi, dm_sf, dm_sf)
        ref -= k_factor*.5 * cp.einsum('ijp,klp,sjk,sli->', cderi, cderi, dm, dm)
        atom_coords[i,x] -= disp
        return float(ref.get())

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 3e-6

def test_uhf_ejk_ip1_kpts_without_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [2, [.8, 1.]]]),
    }
    auxcell.build()
    omega = -0.3

    nao = cell.nao
    kmesh = [3,1,2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = cp.empty((2,nkpts,nao,nao), dtype=np.complex128)
    s = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    mo_coeff[0] = cp.asarray(np.linalg.eigh(s)[1])
    mo_coeff[1] = mo_coeff[0,:,:,::-1]
    nocc = 9
    mo_occ = cp.zeros((2, nkpts, nao))
    mo_occ[:,:,:nocc] = 1
    dm = cp.einsum('skpi,ski,skqi->skpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1
    ejk0 = kuhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0.sum(axis=0)).max() < 2e-11

    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk = kuhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor, omega=omega)
    assert abs(ejk0 - ejk).max() < 1e-9

    dm_sf = dm[0] + dm[1]
    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        nkpts = len(kpts)
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts=kpts,
                                         kmesh=kmesh, omega=omega)[0]
        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm_sf[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,sjk,sli,pkl->', cderi_ij, dm[:,kj],
                                dm[:,ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .5 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 3e-6

def test_uhf_ejk_ip1_kpts_with_long_range():
    np.random.seed(3)
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*2+np.random.rand(3,3))

    auxcell = cell.copy()
    auxcell.basis = {
        'C1':'''
C    S
      0.5000000000           1.0000000000
C    P
    102.9917624900           1.0000000000
     28.1325940100           1.0000000000
      9.8364318200           1.0000000000
C    P
      3.3490545000           1.0000000000
C    P
      1.4947618600           1.0000000000
C    P
      0.4000000000           1.0000000000
C    D
      0.1995412500           1.0000000000 ''',
        'C2': ('unc-weigend', [[0, [.5, 1.]], [2, [.8, 1.]]]),
    }
    auxcell.build()
    omega = -0.3

    nao = cell.nao
    kmesh = [3,1,2]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = cp.empty((2,nkpts,nao,nao), dtype=np.complex128)
    s = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    mo_coeff[0] = cp.asarray(np.linalg.eigh(s)[1])
    mo_coeff[1] = mo_coeff[0,:,:,::-1]
    nocc = 7
    mo_occ = cp.zeros((2, nkpts, nao))
    mo_occ[:,:,:nocc] = 1
    dm = cp.einsum('skpi,ski,skqi->skpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk = kuhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 3e-9

    dm_sf = dm[0] + dm[1]
    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        nkpts = len(kpts)
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts=kpts, kmesh=kmesh)[0]

        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm_sf[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,sjk,sli,pkl->', cderi_ij, dm[:,kj],
                                dm[:,ki], cderi_ji, optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .5 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 1e-6

def test_uhf_ejk_ip1_kpts_with_long_range1():
    cell = gto.Cell()
    cell.atom= [['H1', [0.0, 0.0, 0.0]], ['H2', [1.685,1.685,1.6]]]
    cell.a = '''
    0.00, 3.37, 3.37
    3.37, 0.00, 4.
    2.  , 3.37, 0.00'''
    cell.verbose = 0
    cell.basis = [[0, [3.3, 1]], [0, [1.1, 1]], [1, [0.8, 1]]]
    cell.unit = 'bohr'
    cell.build()
    auxcell = make_auxcell(cell)

    kmesh = [1,1,3]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    mo_coeff = cp.array([np.linalg.eigh(s)[1]]*2)
    mo_occ = cp.zeros((2, nkpts, cell.nao))
    mo_occ[:,:,:3] = 2
    omega = -0.3

    dm = cp.einsum('skpi,ski,skqi->skpq', mo_coeff, mo_occ, mo_coeff.conj())
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = .5
    k_factor = 1
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    ejk = kuhf._jk_energy_per_atom(
        opt, dm, kpts, hermi=1, j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 1e-11

    dm_sf = dm[0] + dm[1]
    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        nkpts = len(kpts)
        cderi = rsdf_builder.build_cderi(cell1, auxcell1, kpts=kpts, kmesh=kmesh)[0]

        jaux = 0
        for ki in range(nkpts):
            jaux += cp.einsum('pij,ji->p', cderi[ki,ki], dm_sf[ki])
        ref = j_factor * .5/nkpts**2 * jaux.dot(jaux).real.get()

        ek = 0
        for ki in range(nkpts):
            for kj in range(nkpts):
                if (ki, kj) in cderi:
                    cderi_ij = cderi[ki,kj]
                else:
                    cderi_ij = cderi[kj,ki].transpose(0,2,1).conj()
                if (kj, ki) in cderi:
                    cderi_ji = cderi[kj,ki]
                else:
                    cderi_ji = cderi[ki,kj].transpose(0,2,1).conj()
                ek += cp.einsum('pij,sjk,sli,pkl->', cderi_ij, dm[:,kj], dm[:,ki], cderi_ji)
        ek = float(ek.real.get())
        ref -= ek * .5 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 1e-6
