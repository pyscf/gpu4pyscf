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
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.df import int3c2e
from gpu4pyscf.pbc.df.grad import rhf
from gpu4pyscf.pbc.df.grad import krhf
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.pbc.df.int2c2e import sr_int2c2e

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
        a=np.diag([2.5, 1.9, 2.2])*3)

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
    omega = -0.2

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ej = rhf._jk_energy_per_atom(opt, dm, hermi=1, k_factor=0)
    assert abs(ej.sum(axis=0)).max() < 1e-11

    disp = 1e-3
    atom_coords = cell.atom_coords()
    dm = SortedGTO.from_cell(cell).apply_C_mat_CT(dm)
    def eval_j(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        opt = int3c2e.SRInt3c2eOpt(cell1, auxcell1, omega).build()
        jaux = opt.contract_dm(dm)
        j2c = sr_int2c2e(auxcell1, omega)
        atom_coords[i,x] -= disp
        return float(cp.linalg.solve(j2c, jaux).dot(jaux).get()) * .5

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_j(i, x, disp)
        e2 = eval_j(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ej[i,x]) < 5e-6

def test_ejk_ip1_gamma_point():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*3)

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
    omega = -0.2

    np.random.seed(8)
    nao = cell.nao
    nocc = 4
    mo_coeff = np.random.rand(nao, nao) - .5
    mo_occ = np.zeros(nao)
    mo_occ[:nocc] = 2
    dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega).build()
    ek = rhf._jk_energy_per_atom(opt, dm, hermi=1, j_factor=1, k_factor=1)
    assert abs(ek.sum(axis=0)).max() < 1e-11

    disp = 1e-3
    atom_coords = cell.atom_coords()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        j3c = int3c2e.sr_aux_e2(cell1, auxcell1, omega)
        j2c = sr_int2c2e(auxcell1, omega)
        j2c_inv = cp.linalg.inv(j2c)
        ref = .5 * cp.einsum('ijp,pq,klq,ji,lk->', j3c, j2c_inv, j3c, dm, dm, optimize=True)
        ref -= .25 * cp.einsum('ijp,pq,klq,jk,li->', j3c, j2c_inv, j3c, dm, dm, optimize=True)
        atom_coords[i,x] -= disp
        return float(ref.get())

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 5e-6

def test_ej_ip1_kpts():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*3)

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
    omega = -0.2

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    dm = cp.asarray(np.linalg.inv(cell.pbc_intor('int1e_ovlp', kpts=kpts))*.5)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    ej = krhf._j_energy_per_atom(opt, dm, kpts=kpts)
    assert abs(ej.sum(axis=0)).max() < 3e-12

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

def test_ejk_ip1_kpts():
    cell = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
               'C2': 'ccpvdz'},
        precision = 1e-10,
        a=np.diag([2.5, 1.9, 2.2])*3)

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
    omega = -0.2

    kmesh = [3,1,4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    mo_coeff = np.linalg.eigh(cell.pbc_intor('int1e_ovlp', kpts=kpts))[1]
    mo_coeff = mo_coeff[:,:,::-1]
    nao = cell.nao
    nocc = 4*nkpts
    mo_occ = np.zeros((nkpts, nao))
    mo_occ[:,:nocc] = 2
    dm = cp.einsum('kpi,ki,kqi->kpq', mo_coeff, mo_occ, mo_coeff.conj())
    dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
    opt = int3c2e.SRInt3c2eOpt(cell, auxcell, omega, kmesh).build()
    j_factor = 1
    k_factor = 1
    ejk = krhf._jk_energy_per_atom(opt, dm, kpts, hermi=1,
                                   j_factor=j_factor, k_factor=k_factor)
    assert abs(ejk.sum(axis=0)).max() < 2e-11

    disp = 1e-3
    atom_coords = cell.atom_coords().copy()
    def eval_jk(i, x, disp):
        atom_coords[i,x] += disp
        cell1 = cell.set_geom_(atom_coords, unit='Bohr')
        auxcell1 = auxcell.set_geom_(atom_coords, unit='Bohr')
        nkpts = len(kpts)

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
                ek += cp.einsum('ijp,jk,li,qp,lkq->', j3c_kk[ki,kj], dm[kj],
                                dm[ki], j2c_inv[kp], j3c_kk[kj,ki], optimize=True)
        ek = float(ek.real.get())
        ref -= ek * .25 / nkpts**2 * k_factor
        atom_coords[i,x] -= disp
        return ref

    for i, x in [(0, 0), (0, 1), (0, 2)]:
        e1 = eval_jk(i, x, disp)
        e2 = eval_jk(i, x, -disp)
        assert abs((e1 - e2)/(2*disp)- ejk[i,x]) < 3e-5
