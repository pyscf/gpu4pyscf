# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from pyscf.df.incore import aux_e2
from gpu4pyscf.df import int3c2e_bdiv as int3c2e
from gpu4pyscf.df.grad.rhf import _jk_energy_per_atom
from gpu4pyscf.lib.cupy_helper import tag_array

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas0 = 'def2-tzvpp'
auxbasis0 = 'def2-tzvpp-jkfit'

def setUpModule():
    global mol_cart, mol_sph, mol, auxmol
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=0,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
                O3  -.5   -.14     0.5
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]
                     ),
               'C2': 'ccpvdz',
               'O3': 'ccpvdz'}
    )
    auxmol = mol.copy()
    auxmol.basis = {
        'C1':'''
C    S
 50.0000000000           1.0000000000
C    S
 18.338091700            0.60189974570
C    S
  9.5470634000           0.19165883840
C    S
  5.1584143000           1.0000000
C    S
  2.8816701000           1.0000000
C    S
  1.6573522000           1.0000000
C    S
  0.97681020000          1.0000000
C    S
  0.35779270000          1.0000000
C    S
  0.21995500000          1.0000000
C    S
  0.13560770000          1.0000000
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
        'C2':'unc-weigend',
        'O3': [[0, [9.5, 1.]],
              [0, [3.5, 1.]],
              [0, [1.5, 1.]],
              [0, [.8, 1.]],
              [0, [.5, 1.]],
              [0, [.3, 1.]],
              [0, [.2, 1.]],
              [0, [.1, 1.]]
             ],
    }
    auxmol.build()

eps = 1e-3

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _check_grad(mol, grid_response=False, tol=1e-6):
    mol = mol.copy()
    mf = mol.RHF().to_gpu().density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-14
    mf.direct_scf_tol = 1e-20
    mf.verbose = 1
    mf.kernel()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.grid_response = grid_response

    g_scanner = g.as_scanner()
    g_analy = g_scanner(mol)[1]
    print('analytical gradient:')
    print(g_analy)

    f_scanner = mf.as_scanner()
    coords = mol.atom_coords()
    grad_fd = np.zeros_like(coords)
    for i in range(len(coords)):
        for j in range(3):
            coords = mol.atom_coords()
            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            e0 = f_scanner(mol)

            coords[i,j] -= 2.0 * eps
            mol.set_geom_(coords, unit='Bohr')
            e1 = f_scanner(mol)

            coords[i,j] += eps
            mol.set_geom_(coords, unit='Bohr')
            grad_fd[i,j] = (e0-e1)/2.0/eps
    grad_fd = np.array(grad_fd).reshape(-1,3)
    print('finite difference gradient:')
    print(grad_fd)
    print('difference between analytical and finite difference gradient:', np.linalg.norm(g_analy - grad_fd))
    assert(np.linalg.norm(g_analy - grad_fd) < tol)

def _vs_cpu(mol, grid_response=False, tol=1e-9):
    mf = mol.RHF().to_gpu().density_fit(auxbasis=auxbasis0)
    mf.verbose = 1
    mf.kernel()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.grid_response = grid_response
    g_analy = g.kernel()

    g_cpu = g.to_cpu()
    ref = g_cpu.kernel()
    print('CPU - GPU:', abs(g_analy - ref).max())
    assert abs(g_analy - ref).max() < tol

class KnownValues(unittest.TestCase):

    def test_grad_sph(self):
        _vs_cpu(mol_sph)

    def test_grad_cart(self):
        _vs_cpu(mol_cart, tol=1e-7)

    def test_j_energy_per_atom(self):
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = np.random.rand(nao, nao) - .5
        mo_occ = np.zeros(nao)
        mo_occ[:nocc] = 2
        dm = mo_coeff[:,:nocc].dot(mo_coeff[:,:nocc].T) * 2
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        ej = _jk_energy_per_atom(opt, dm, k_factor=0)
        assert abs(ej.sum(axis=0)).max() < 1e-12

        disp = 1e-3
        atom_coords = mol.atom_coords()
        mol0 = mol.copy()
        auxmol0 = auxmol.copy()
        def eval_j(i, x, disp):
            atom_coords[i,x] += disp
            mol1 = mol0.set_geom_(atom_coords, unit='Bohr')
            auxmol1 = auxmol0.set_geom_(atom_coords, unit='Bohr')
            j3c = aux_e2(mol1, auxmol1)
            j2c = auxmol1.intor('int2c2e')
            jaux = np.einsum('ijp,ji->p', j3c, dm)
            atom_coords[i,x] -= disp
            return np.linalg.solve(j2c, jaux).dot(jaux) * .5

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_j(i, x, disp)
            e2 = eval_j(i, x, -disp)
            assert abs((e1 - e2)/(2*disp) - ej[i,x]) < 3e-5

    def test_jk_energy_per_atom(self):
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = np.random.rand(nao, nao) - .5
        mo_occ = np.zeros(nao)
        mo_occ[:nocc] = 2
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
        ek = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1, hermi=1)
        assert abs(ek.sum(axis=0)).max() < 1e-12
        ek0 = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1, hermi=0)
        assert abs(ek - ek0).max() < 1e-9
        ek1 = _jk_energy_per_atom(opt, tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ),
                                  j_factor=1, k_factor=1, hermi=1)
        assert abs(ek - ek1).max() < 1e-9
        assert abs(lib.fp(ek) - -24.366562704166753) < 1e-9

        disp = 1e-3
        atom_coords = mol.atom_coords()
        mol0 = mol.copy()
        auxmol0 = auxmol.copy()
        def eval_jk(i, x, disp):
            atom_coords[i,x] += disp
            mol1 = mol0.set_geom_(atom_coords, unit='Bohr')
            auxmol1 = auxmol0.set_geom_(atom_coords, unit='Bohr')
            j3c = aux_e2(mol1, auxmol1)
            j2c = auxmol1.intor('int2c2e')
            eri = lib.einsum('ijp,pq,klq->ijkl', j3c, np.linalg.inv(j2c), j3c)
            ref = .5 * np.einsum('ijkl,ji,lk->', eri, dm, dm)
            ref -= .25 * np.einsum('ijkl,jk,li->', eri, dm, dm)
            atom_coords[i,x] -= disp
            return ref

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_jk(i, x, disp)
            e2 = eval_jk(i, x, -disp)
            assert abs((e1 - e2)/(2*disp)- ek1[i,x]) < 3e-5

        dm = np.random.rand(nao, nao)
        dm = dm - dm.T
        ek = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1, hermi=2)
        assert abs(ek.sum(axis=0)).max() < 1e-12
        ek0 = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1)
        assert abs(ek - ek0).max() < 3e-11
        assert abs(lib.fp(ek) - -2.4880988769692016) < 1e-9

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_jk(i, x, disp)
            e2 = eval_jk(i, x, -disp)
            assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 1e-6

    def test_uhf_jk_energy_per_atom(self):
        from gpu4pyscf.df.grad.uhf import _jk_energy_per_atom
        np.random.seed(8)
        nao = mol.nao
        nocc = 4
        mo_coeff = np.random.rand(2, nao, nao) - .5
        mo_occ = np.zeros((2, nao))
        mo_occ[0,:nocc+1] = 1
        mo_occ[1,:nocc] = 1
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        dm = np.einsum('spi,si,sqi->spq', mo_coeff, mo_occ, mo_coeff)
        ek = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1, hermi=1)
        assert abs(ek.sum(axis=0)).max() < 3e-11
        ek0 = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1, hermi=0)
        assert abs(ek - ek0).max() < 3e-10
        ek1 = _jk_energy_per_atom(opt, tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ),
                                  j_factor=1, k_factor=1, hermi=1)
        assert abs(ek - ek1).max() < 3e-10

        disp = 1e-3
        atom_coords = mol.atom_coords()
        mol0 = mol.copy()
        auxmol0 = auxmol.copy()
        def eval_jk(i, x, disp):
            atom_coords[i,x] += disp
            mol1 = mol0.set_geom_(atom_coords, unit='Bohr')
            auxmol1 = auxmol0.set_geom_(atom_coords, unit='Bohr')
            j3c = aux_e2(mol1, auxmol1)
            j2c = auxmol1.intor('int2c2e')
            eri = lib.einsum('ijp,pq,klq->ijkl', j3c, np.linalg.inv(j2c), j3c)
            ref = .5 * np.einsum('ijkl,mji,nlk->', eri, dm, dm)
            ref -= .5 * np.einsum('ijkl,sjk,sli->', eri, dm, dm)
            atom_coords[i,x] -= disp
            return ref

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_jk(i, x, disp)
            e2 = eval_jk(i, x, -disp)
            assert abs((e1 - e2)/(2*disp)- ek1[i,x]) < 2e-5

        disp = .5e-2
        mol0 = mol.copy()
        auxmol0 = mol.copy()
        mol0.omega = .15
        auxmol0.omega = .15
        opt = int3c2e.Int3c2eOpt(mol0, auxmol0).build()
        dm = np.einsum('spi,si,sqi->spq', mo_coeff, mo_occ, mo_coeff)
        ek = _jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1, hermi=1)
        assert abs(ek.sum(axis=0)).max() < 1e-12

        def inv(s):
            e, c = np.linalg.eigh(s)
            mask = e > 1e-7
            return (c[:,mask]/e[mask]).dot(c[:,mask].T)
        def eval_jk(i, x, disp):
            atom_coords[i,x] += disp
            mol1 = mol0.set_geom_(atom_coords, unit='Bohr')
            auxmol1 = auxmol0.set_geom_(atom_coords, unit='Bohr')
            j3c = aux_e2(mol1, auxmol1)
            j2c = auxmol1.intor('int2c2e')
            eri = lib.einsum('ijp,pq,klq->ijkl', j3c, inv(j2c), j3c)
            ref = .5 * np.einsum('ijkl,mji,nlk->', eri, dm, dm)
            ref -= .5 * np.einsum('ijkl,sjk,sli->', eri, dm, dm)
            atom_coords[i,x] -= disp
            return ref

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_jk(i, x, disp)
            e2 = eval_jk(i, x, -disp)
            assert abs((e1 - e2)/(2*disp)- ek[i,x]) < 2e-4

if __name__ == "__main__":
    print("Full Tests for DF RHF Gradient")
    unittest.main()
