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
import pyscf
from pyscf.gto import intor_cross
from pyscf.pbc.tools import pbc as pbctools
from gpu4pyscf.pbc.gto import int1e

def test_int1e_ovlp():
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*2,
        basis='def2-tzvpp',
    )
    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    pcell = cell.copy()
    pcell.precision = 1e-14
    pcell.build(0, 0)
    ref = pcell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)

    dat = int1e.int1e_ovlp(cell).get()[0]
    assert abs(dat - ref[0]).max() < 1e-10

    dat = int1e.int1e_ovlp(cell, kpts=kpts).get()
    assert abs(dat - ref).max() < 1e-10

def test_int1e_kin():
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*2,
        basis='def2-tzvpp',
    )
    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    pcell = cell.copy()
    pcell.precision = 1e-14
    pcell.build(0, 0)
    ref = pcell.pbc_intor('int1e_kin', hermi=1, kpts=kpts)

    dat = int1e.int1e_kin(cell).get()[0]
    assert abs(dat - ref[0]).max() < 1e-10

    dat = int1e.int1e_kin(cell, kpts=kpts).get()
    assert abs(dat - ref).max() < 1e-10

    mol = cell.to_mol()
    dat = int1e.int1e_kin(mol).get()
    ref = mol.intor('int1e_kin', hermi=1)
    assert abs(dat - ref).max() < 1e-12

def test_int1e_ipovlp():
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*2,
        basis='def2-tzvpp',
    )
    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    pcell = cell.copy()
    pcell.precision = 1e-14
    pcell.build(0, 0)
    ref = pcell.pbc_intor('int1e_ipovlp', hermi=0, kpts=kpts)

    dat = int1e.int1e_ipovlp(cell).get()[0]
    assert abs(dat - ref[0]).max() < 1e-10

    dat = int1e.int1e_ipovlp(cell, kpts=kpts).get()
    assert abs(dat - ref).max() < 1e-10

    mol = cell.to_mol()
    dat = int1e.int1e_ipovlp(mol).get()
    ref = mol.intor('int1e_ipovlp')
    assert abs(dat - ref).max() < 1e-12

def test_int1e_ipkin():
    cell = pyscf.M(
        atom='''C1  1.3    .2       .3
                C2  .19   .1      1.1
                C3  0.  0.  0.
        ''',
        precision = 1e-8,
        a=np.diag([2.5, 1.9, 2.2])*2,
        basis='def2-tzvpp',
    )
    kmesh = [6, 1, 1]
    kpts = cell.make_kpts(kmesh)
    pcell = cell.copy()
    pcell.precision = 1e-14
    pcell.build(0, 0)
    ref = pcell.pbc_intor('int1e_ipkin', hermi=0, kpts=kpts)

    dat = int1e.int1e_ipkin(cell).get()[0]
    assert abs(dat - ref[0]).max() < 1e-10

    dat = int1e.int1e_ipkin(cell, kpts=kpts).get()
    assert abs(dat - ref).max() < 1e-10

def test_int1e_ovlp1():
    L = 4
    n = 21
    cell = pyscf.M(unit = 'B',
               precision = 1e-10,
               a = ((L,0,0),(0,L,0),(0,0,L)),
               mesh = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.2, 1.0)]]})
    nk = [5, 4, 1]
    kpts = cell.make_kpts(nk, wrap_around=True)[[3, 8, 11]]
    s = int1e.int1e_ovlp(cell, kpts)
    ref = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    assert abs(s.get() - ref).max() < 1e-10
    k = int1e.int1e_kin(cell, kpts)
    ref = cell.pbc_intor('int1e_kin', kpts=kpts)
    assert abs(k.get() - ref).max() < 1e-10

def test_ovlp_stress_tensor():
    a = np.eye(3) * 6.
    np.random.seed(5)
    a += np.random.rand(3, 3) - .5
    cell = pyscf.M(
        atom='He 1 1 1; He 2 1.5 2.4',
        basis=[[0, [.5, 1]],
               [1, [1.5, 1], [.5, 1]],
               [2, [.8, 1]],
               [3, [.7, 1]]
              ], a=a, unit='Bohr')

    Ls = cell.get_lattice_Ls()
    Ls = Ls[np.argsort(np.linalg.norm(Ls-.1, axis=1))]
    scell = cell.copy()
    scell = pbctools._build_supcell_(scell, cell, Ls)

    aoslices = cell.aoslice_by_atom()
    ao_repeats = aoslices[:,3] - aoslices[:,2]
    bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)

    nao = cell.nao
    dm = np.random.rand(nao, nao) - .5
    dm = dm.dot(dm.T)
    ovlp10 = cell.pbc_intor('int1e_ipovlp')
    ovlp10 = np.einsum('xij,iy->xyij', ovlp10, bas_coords)
    sc_ovlp01 = intor_cross('int1e_ipovlp', scell, cell)
    ovlp01 = np.einsum('xnji,njy->xyij', sc_ovlp01.reshape(3,-1,nao,nao), bas_coords+Ls[:,None])
    ref = -(ovlp10 + ovlp01)
    ref = np.einsum('xyij,ji->xy', ref, dm)
    dat = int1e._Int1eOptV2(cell).get_ovlp_strain_deriv(dm)
    assert abs(dat - ref).max() < 1e-9

    nk = [5, 4, 1]
    kpts = cell.make_kpts(nk)
    dm = np.array(cell.pbc_intor('int1e_ovlp', kpts=kpts))
    ovlp10 = np.array(cell.pbc_intor('int1e_ipovlp', kpts=kpts))
    ovlp10 = np.einsum('kxij,iy->xykij', ovlp10, bas_coords)
    expLk = np.exp(1j*Ls.dot(kpts.T))
    ovlp01 = np.einsum('xnji,njy->xyinj', sc_ovlp01.reshape(3,-1,nao,nao), bas_coords+Ls[:,None])
    ovlp01 = np.einsum('xyiLj,Lk->xykij', ovlp01, expLk, optimize=True)
    ref = ovlp01 + ovlp10
    ref = -np.einsum('xykij,kji->xy', ref, dm).real / len(kpts)
    dat = int1e._Int1eOptV2(cell).get_ovlp_strain_deriv(dm, kpts=kpts)
    assert abs(dat - ref).max() < 1e-9
