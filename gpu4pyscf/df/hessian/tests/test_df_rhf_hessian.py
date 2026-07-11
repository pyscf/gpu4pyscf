# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
#

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.df.hessian import rhf as df_rhf_cpu
from pyscf.hessian import rhf as rhf_cpu
from gpu4pyscf.df.hessian import rhf as df_rhf_hess
from gpu4pyscf.df.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e_bdiv as int3c2e
from gpu4pyscf.lib.cupy_helper import tag_array

def setUpModule():
    global mol1, mol, auxmol
    mol1 = pyscf.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''
        GHOST-O -1.  -.5      -.5
        O   0.   0.       0.
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''',
        basis = 'sto3g')

    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
                C3   1.      2.       3.
                C2   .19   .1      1.1
                C2   0.   .5      .5
        ''',
        basis={'C1': ('ccpvdz',
                      [[3, [1.1, 1.]],
                       [4, [2., 1.]]]),
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
    global mol1
    mol1.stdout.close()
    del mol1

class KnownValues(unittest.TestCase):
    def test_gen_vind(self):
        mf = mol1.RHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        nao, nmo = mo_coeff.shape
        mocc = mo_coeff[:,mo_occ>0]
        nocc = mocc.shape[1]

        fx_cpu = rhf_cpu.gen_vind(mf, mo_coeff, mo_occ)
        mo1 = np.random.rand(100, nmo*nocc)
        v1vo_cpu = fx_cpu(mo1).reshape(-1,nmo*nocc)

        mf = mf.to_gpu()
        hessobj = df_rhf_hess.Hessian(mf)
        fx_gpu = hessobj.gen_vind(mo_coeff, mo_occ)
        mo1 = cp.asarray(mo1)
        v1vo_gpu = fx_gpu(mo1)
        assert np.linalg.norm(v1vo_cpu - v1vo_gpu.get()) < 1e-8

    def test_partial_hess_elec(self):
        mf = mol1.RHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        e1_cpu, ej_cpu, ek_cpu = df_rhf_cpu._partial_hess_ejk(hobj)
        ref = e1_cpu + ej_cpu - ek_cpu
        dat = df_rhf_hess.Hessian(mf.to_gpu()).partial_hess_elec().get()
        assert abs(ref - dat).max() < 1e-8

    def test_make_h1(self):
        mf = mol1.RHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mocc = mo_coeff[:,mo_occ>0]
        hobj = mf.Hessian()
        h1_cpu = df_rhf_cpu.make_h1(hobj, mo_coeff, mo_occ)
        mo1_cpu, mo_e1_cpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1_cpu, verbose=1)
        h1_cpu = np.asarray(h1_cpu)
        h1_cpu = np.einsum('xypq,pi,qj->xyij', h1_cpu, mo_coeff, mocc)

        mf = mf.to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        hobj = df_rhf_hess.Hessian(mf)
        mo_occ = cp.asarray(mo_occ)
        h1_gpu = df_rhf_hess.make_h1(hobj, mo_coeff, mo_occ)
        h1_gpu = cp.asarray(h1_gpu)
        mo_energy = cp.asarray(mo_energy)
        mo_coeff = cp.asarray(mo_coeff)
        fx = hobj.gen_vind(mo_coeff, mo_occ)
        mo1_gpu, mo_e1_gpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1_gpu, fx, verbose=1)
        assert np.linalg.norm(h1_cpu - h1_gpu.get()) < 1e-5
        assert np.linalg.norm((mo_e1_cpu - mo_e1_gpu)) < 1e-4

    def test_df_rhf_hess_elec(self):
        mf = mol1.RHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.hess_elec()

        mf = mf.to_gpu()
        hobj = mf.Hessian()
        hess_gpu = hobj.hess_elec()
        assert np.linalg.norm(hess_cpu - hess_gpu.get()) < 1e-5

    def test_df_rhf_hessian(self):
        mf = mol1.RHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.kernel()
        mf = mf.to_gpu()
        hobj = df_rhf_hess.Hessian(mf)
        hess_gpu = hobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_jk_energy_per_atom(self):
        np.random.seed(8)
        nao = mol.nao
        nocc = nao - 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5) * .2
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        dm = (mo_coeff*mo_occ).dot(mo_coeff.T)
        dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

        with lib.temporary_env(df_rhf_hess, get_avail_mem=(lambda **kw: nao**2*auxmol.nao*60)):
            ref = df_rhf_hess._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1e-20)
            ej = df_rhf_hess._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=0)
            assert abs(ej-ref).max().get() < 5e-8

        ejk = df_rhf_hess._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1).get()
        assert abs(ejk.sum(axis=(0,1))).max() < 1e-9

        disp = .5e-3
        mol0 = mol.copy(deep=True)
        auxmol0 = auxmol.copy(deep=True)
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

    def test_jk_ip1_finite_diff(self):
        from pyscf.df import incore
        mol2 = mol + mol
        np.random.seed(9)
        nao = mol2.nao
        nocc = nao - 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5) * .2
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2

        opt = int3c2e.Int3c2eOpt(mol2, auxmol).build()
        veff = df_rhf_hess._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)

        disp = .5e-3
        mol0 = mol2.copy(deep=True)
        auxmol0 = auxmol.copy(deep=True)
        dm = mo_coeff[:,:nocc].dot(mo_coeff[:,:nocc].T) * 2
        def eval_veff(i, x, disp):
            atom_coords = mol2.atom_coords()
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
            return vj - .5 * vk

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            v1 = eval_veff(i, x, disp)
            v2 = eval_veff(i, x, -disp)
            ref = mo_coeff.T.dot(v1 - v2).dot(mo_coeff[:,:nocc]) / (2*disp)
            assert abs(ref - veff[i,x]).max().get() < 1e-5

    def test_jk_ip1_limited_memory(self):
        mol1 = mol + mol
        mol2 = mol1 + mol1
        np.random.seed(8)
        nao = mol2.nao
        nocc = 20
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_occ = cp.zeros(nao)
        mo_occ[:nocc] = 2
        opt = int3c2e.Int3c2eOpt(mol2, auxmol).build()

        ref = df_rhf_hess._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)

        with lib.temporary_env(df_rhf_hess, get_avail_mem=(lambda **kw: nao**2*3*mol2.natm*20)):
            veff = df_rhf_hess._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)
            assert abs(ref - veff).max() < 1e-9

if __name__ == "__main__":
    print("Full Tests for DF RHF Hessian")
    unittest.main()
