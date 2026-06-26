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

import numpy as np
import cupy as cp
import unittest
import pyscf
from pyscf import lib
from gpu4pyscf import dft, scf
from gpu4pyscf.df.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e_bdiv as int3c2e
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.df.hessian import rhf_fast

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

xc0 = 'B3LYP'
bas0 = 'def2-tzvpp'
auxbasis0 = 'def2-tzvpp-jkfit'
disp0 = 'd3bj'
grids_level = 6
eps = 1e-3

def setUpModule():
    global mol_sph, mol_cart, mol, auxmol
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=0,
                      output='/dev/null', verbose=6)
    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=6)
    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C2   .19   .1      1.1
                C2   0.   .5      .5
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]
                     ),
               'C2': 'ccpvdz'}
    )
    auxmol = mol.copy(False)
    auxmol.basis = {
        'C1':'''
C    S
 50.0000000000           1.0000000000
C    S
  20.338091700            0.60189974570
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
  0.3995412500           1.0000000000 ''',
        'C2':[
              [0, [3.5, 1.]],
              [0, [1.5, 1.]],
              [0, [.5, 1.]],
              [0, [.2, 1.]],
              [0, [.1, 1.]],
              [1, [0.8, 1.]],
              [1, [0.5, 1.]],
              [2, [0.3, 1.]],
              [3, [0.6, 1.]],
             ],
    }
    auxmol.build()

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _make_rhf(mol, disp=None):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.kernel()
    return mf

def _make_uhf(mol, disp=None):
    mf = scf.UHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.kernel()
    return mf

def _make_rks(mol, xc, disp=None):
    mf = dft.rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.grids.level = grids_level
    mf.verbose = 1
    mf.kernel()
    return mf

def _make_uks(mol, xc, disp=None):
    mf = dft.uks.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis0)
    mf.conv_tol = 1e-12
    mf.disp = disp
    mf.grids.level = grids_level
    mf.verbose = 1
    mf.kernel()
    return mf

def _check_rhf_hessian(mf, h, ix=0, iy=0, tol=1e-3):
    pmol = mf.mol.copy()

    g = mf.nuc_grad_method()
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    _, g1 = g_scanner(pmol)
    h_fd = (g0 - g1)/2.0/eps

    print(f'Norm of analytical - finite difference Hessian H({ix},{iy})', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

def _check_dft_hessian(mf, h, ix=0, iy=0, tol=1e-3):
    pmol = mf.mol.copy()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    _, g1 = g_scanner(pmol)

    h_fd = (g0 - g1)/2.0/eps
    print(f'Norm of analytical - finite difference Hessian H({ix},{iy})', np.linalg.norm(h[ix,:,iy,:] - h_fd))
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

class KnownValues(unittest.TestCase):
    def test_hessian_rhf(self, disp=None):
        print('-----testing DF RHF Hessian----')
        mf = _make_rhf(mol_sph)
        mf.disp = disp
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_rhf_hessian(mf, h, ix=0, iy=0)
        _check_rhf_hessian(mf, h, ix=0, iy=1)
    
    def test_hessian_lda(self, disp=None):
        print('-----testing DF LDA Hessian----')
        mf = _make_rks(mol_sph, 'LDA')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_gga(self):
        print('-----testing DF PBE Hessian----')
        mf = _make_rks(mol_sph, 'PBE')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_hybrid(self):
        print('-----testing DF B3LYP Hessian----')
        mf = _make_rks(mol_sph, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_uks_hessian_hybrid(self):
        print('-----testing DF-UKS B3LYP Hessian----')
        mf = _make_uks(mol_sph, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_mgga(self):
        print('-----testing DF M06 Hessian----')
        mf = _make_rks(mol_sph, 'm06')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
    
    def test_hessian_rsh(self):
        print('-----testing DF wb97 Hessian----')
        mf = _make_rks(mol_sph, 'wb97')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
    
    def test_hessian_rhf_D3(self):
        print('----- testing DFRHF with D3BJ ------')
        mf = _make_rhf(mol_sph, disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rhf_D4(self):
        print('------ testing DFRKS with D4 --------')
        mf = _make_rhf(mol_sph, disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uhf_D3(self):
        print('-------- testing DFUHF with D3BJ ------')
        mf = _make_uhf(mol_sph, disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uhf_D4(self):
        print('--------- testing DFUHF with D4 -------')
        mf = _make_uhf(mol_sph, disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rks_D3(self):
        print('--------- testing DFRKS with D3BJ -------')
        mf = _make_rks(mol_sph, 'b3lyp', disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rks_D4(self):
        print('----------- testing DFRKS with D4 --------')
        mf = _make_rks(mol_sph, 'b3lyp', disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uks_D3(self):
        print('------------ testing DFUKS with D3BJ -------')
        mf = _make_uks(mol_sph, 'b3lyp', disp='d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uks_D4(self):
        print('------------- testing DFUKS with D4 ---------')
        mf = _make_uks(mol_sph, 'b3lyp', disp='d4')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_rks_wb97m_d3bj(self):
        print('----------- testing DFRKS, wb97m-d3bj --------')
        mf = _make_rks(mol_sph, 'wb97m-d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_uks_wb97m_d3bj(self):
        print('------------- testing DFUKS, wb97m-d3bj ---------')
        mf = _make_uks(mol_sph, 'wb97m-d3bj')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)

    def test_hessian_cart(self):
        print('-----testing DF Hessian (cartesian)----')
        mf = _make_rks(mol_cart, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_uks_cart(self):
        print('-----testing DF Hessian (cartesian)----')
        mf = _make_uks(mol_cart, 'b3lyp')
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)

    def test_hessian_qz(self):
        mol = pyscf.M(atom=atom, basis='def2-qzvpp', max_memory=32000, cart=0,
                      output='/dev/null', verbose=1)

        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-14
        mf.conv_tol_cpscf = 1e-7
        mf.kernel()

        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_dft_hessian(mf, h, ix=0,iy=0)
        _check_dft_hessian(mf, h, ix=0,iy=1)
        mol.stdout.close()

    def test_unstable_j2c(self):
        mol = pyscf.M(
            atom = """
                C     0.7765    0.7765    0.7765
                C    -0.7765    0.7765    0.7765
                C     0.7765   -0.7765    0.7765
                C     0.7765    0.7765   -0.7765
                C    -0.7765   -0.7765    0.7765
                C    -0.7765    0.7765   -0.7765
                C     0.7765   -0.7765   -0.7765
                C    -0.7765   -0.7765   -0.7765
                H     1.3926    1.3926    1.3926
                H    -1.3926    1.3926    1.3926
                H     1.3926   -1.3926    1.3926
                H     1.3926    1.3926   -1.3926
                H    -1.3926   -1.3926    1.3926
                H    -1.3926    1.3926   -1.3926
                H     1.3926   -1.3926   -1.3926
                H    -1.3926   -1.3926   -1.3926
            """,
            basis = "sto-3g",
            verbose = 4
        )

        mf = mol.RHF().to_gpu().density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-12
        mf.kernel()

        hobj = mf.Hessian()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        test_hessian_round1 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ)

        mf.kernel() # another cycle

        hobj = mf.Hessian()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        test_hessian_round2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ)

        assert np.max(np.abs(test_hessian_round1 - test_hessian_round2)) < 2e-7

    def test_jk_energy_per_atom(self):
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2
        mo_energy = cp.zeros_like(mo_occ)
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
        dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

        ref = rhf_fast._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1e-20)
        ej = rhf_fast._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=0)
        assert abs(ej-ref).max() < 1e-8

        ejk = rhf_fast._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1)
        assert abs(ejk.sum(axis=(0,1))).max() < 1e-10

        disp = .5e-3
        mol0 = mol.copy()
        auxmol0 = auxmol.copy()
        def eval_grad(i, x, disp):
            atom_coords = mol.atom_coords()
            atom_coords[i,x] += disp
            mol1 = mol0.set_geom_(atom_coords, unit='Bohr')
            atom_coords = auxmol.atom_coords()
            atom_coords[i,x] += disp
            auxmol1 = auxmol0.set_geom_(atom_coords, unit='Bohr')
            opt = int3c2e.Int3c2eOpt(mol1, auxmol1).build()
            return rhf_grad._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1)

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_grad(i, x, disp)
            e2 = eval_grad(i, x, -disp)
            assert abs((e1 - e2)/(2*disp) - ejk[i,:,x]).max() < 1e-5

    def test_jk_ip1(self):
        from gpu4pyscf.df.hessian.rhf import _get_jk_ip
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2

        obj = mol.RHF().to_gpu().density_fit(auxbasis=auxmol.basis).Hessian()
        obj.auxbasis_response=2
        vj, vk = _get_jk_ip(obj, mo_coeff, mo_occ)
        ref = vj - 0.5 * vk

        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        veff = rhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)
        assert abs(ref - veff).max() < 1e-8

    def test_jk_ip1_finite_diff(self):
        mol = mol + mol
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2

        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        veff = rhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=.5)

        from pyscf.df import incore
        disp = .5e-3
        mol0 = mol.copy()
        auxmol0 = auxmol.copy()
        dm = mo_coeff[:,:nocc].dot(mo_coeff[:,:nocc].T) * 2
        def eval_veff(i, x, disp):
            atom_coords = mol.atom_coords()
            atom_coords[i,x] += disp
            mol1 = mol0.set_geom_(atom_coords, unit='Bohr')
            atom_coords = auxmol.atom_coords()
            atom_coords[i,x] += disp
            auxmol1 = auxmol0.set_geom_(atom_coords, unit='Bohr')
            j3c = cp.array(incore.aux_e2(mol1, auxmol1))
            j2c = cp.array(auxmol1.intor('int2c2e'))
            eri = cp.einsum('ijp,pq,klq->ijkl', j3c, cp.linalg.inv(j2c), j3c)
            vj = cp.einsum('ijkl,ji->kl', eri, dm)
            vk = cp.einsum('ijkl,jk->il', eri, dm)
            return vj - .5 * vk * .5

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            v1 = eval_veff(i, x, disp)
            v2 = eval_veff(i, x, -disp)
            ref = mo_coeff.T.dot(v1 - v2).dot(mo_coeff[:,:nocc]) / (2*disp)
            assert abs(ref - veff[i,x]).max().get() < 1e-5

    def test_jk_ip1_limited_memory(self):
        mol = mol + mol
        mol = mol + mol
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2
        mo_energy = cp.zeros_like(mo_occ)
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()

        ref = rhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)

        with lib.temporary_env(rhf_fast, get_avail_mem=(lambda **kw:
                                                        nao**2*3*mol.natm*16)):
            veff = rhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)
            assert abs(ref - veff).max() < 1e-9

if __name__ == "__main__":
    print("Full Tests for DF Hessian")
    unittest.main()
