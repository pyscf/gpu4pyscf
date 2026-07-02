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

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from pyscf.df.hessian import uhf as df_uhf_cpu
from pyscf.hessian import uhf as uhf_cpu
from gpu4pyscf.df.hessian import rhf_fast
from gpu4pyscf.df.hessian import uhf_fast
from gpu4pyscf.df.grad import uhf as uhf_grad
from gpu4pyscf.df import int3c2e_bdiv as int3c2e
from gpu4pyscf.lib.cupy_helper import tag_array

def setUpModule():
    global mol1, mol, auxmol
    mol1 = pyscf.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''
        O   0.   0.       0.
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''',
        basis = 'sto3g')

    mol = pyscf.M(
        atom='''C1   1.3    .2       .3
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
        mf = mol1.UHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        nao, nmoa = mo_coeff[0].shape
        nao, nmob = mo_coeff[1].shape
        mocca = mo_coeff[0][:,mo_occ[0]>0]
        moccb = mo_coeff[1][:,mo_occ[1]>0]
        nocca = mocca.shape[1]
        noccb = moccb.shape[1]

        fx_cpu = uhf_cpu.gen_vind(mf, mo_coeff, mo_occ)
        mo1 = np.random.rand(100, nmoa*nocca+nmob*noccb)
        v1vo_cpu = fx_cpu(mo1)

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        fx_gpu = hessobj.gen_vind(mo_coeff, mo_occ)
        mo1 = cp.asarray(mo1)
        v1vo_gpu = fx_gpu(mo1)
        assert np.linalg.norm(v1vo_cpu - v1vo_gpu.get()) < 1e-8

    def test_partial_hess_elec(self):
        mf = mol1.UHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        e1_cpu, ej_cpu, ek_cpu = df_uhf_cpu._partial_hess_ejk(hobj)
        ref = e1_cpu + ej_cpu - ek_cpu
        dat = uhf_fast.Hessian(mf.to_gpu()).partial_hess_elec().get()
        assert abs(ref - dat).max() < 1e-8

    def test_make_h1(self):
        mf = mol1.UHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mocca = mo_coeff[0][:,mo_occ[0]>0]
        moccb = mo_coeff[1][:,mo_occ[1]>0]
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        h1a_cpu, h1b_cpu = df_uhf_cpu.make_h1(hobj, mo_coeff, mo_occ)
        mo1_cpu, mo_e1_cpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, (h1a_cpu, h1b_cpu), verbose=1)
        h1a_cpu = np.asarray(h1a_cpu)
        h1b_cpu = np.asarray(h1b_cpu)
        h1a_cpu = np.einsum('xypq,pi,qj->xyij', h1a_cpu, mo_coeff[0], mocca)
        h1b_cpu = np.einsum('xypq,pi,qj->xyij', h1b_cpu, mo_coeff[1], moccb)

        mf = mf.to_gpu()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        hobj = uhf_fast.Hessian(mf)
        h1a_gpu, h1b_gpu = uhf_fast.make_h1(hobj, mo_coeff, mo_occ)
        h1a_gpu = cp.asarray(h1a_gpu)
        h1b_gpu = cp.asarray(h1b_gpu)
        fx = hobj.gen_vind(mo_coeff, mo_occ)
        mo1_gpu, mo_e1_gpu = hobj.solve_mo1(mo_energy, mo_coeff, mo_occ, (h1a_gpu, h1b_gpu), fx, verbose=1)
        assert np.linalg.norm(h1a_cpu - h1a_gpu.get()) < 1e-5
        assert np.linalg.norm(h1b_cpu - h1b_gpu.get()) < 1e-5
        mo1_cpu = (np.asarray(mo1_cpu[0]), np.asarray(mo1_cpu[1]))
        mo1_gpu = (cp.asarray(mo1_gpu[0]).get(), cp.asarray(mo1_gpu[1]).get())
        mo_e1_cpu = (np.asarray(mo_e1_cpu[0]), np.asarray(mo_e1_cpu[1]))
        mo_e1_gpu = (cp.asarray(mo_e1_gpu[0]).get(), cp.asarray(mo_e1_gpu[1]).get())

        # mo1 is not consistent in PySCF and GPU4PySCF
        #assert np.linalg.norm((mo1_cpu[0] - mo1_gpu[0])) < 1e-4
        assert np.linalg.norm((mo_e1_cpu[0] - mo_e1_gpu[0])) < 1e-4
        #assert np.linalg.norm((mo1_cpu[1] - mo1_gpu[1])) < 1e-4
        assert np.linalg.norm((mo_e1_cpu[1] - mo_e1_gpu[1])) < 1e-4

    def test_df_uhf_hess_elec(self):
        mf = mol1.UHF().density_fit()
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

    def test_df_uhf_hessian(self):
        mf = mol1.UHF().density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.kernel()
        mf = mf.to_gpu()
        hobj = uhf_fast.Hessian(mf)
        hess_gpu = hobj.kernel()
        assert np.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_jk_energy_per_atom(self):
        np.random.seed(8)
        nao = mol.nao
        nocc = 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_coeff = cp.repeat(mo_coeff[None], 2, axis=0)
        mo_occ = cp.zeros((2,nao))
        mo_occ[:,:nocc] = 1
        dm = cp.einsum('npi,nqi->npq', mo_coeff[:,:,:nocc], mo_coeff[:,:,:nocc])
        dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        dm_rhf = tag_array(dm[0]+dm[1], mo_coeff=mo_coeff[0], mo_occ=mo_occ[0]*2)
        ref = rhf_fast._jk_energy_per_atom(opt, dm_rhf, j_factor=1, k_factor=1)
        with lib.temporary_env(uhf_fast, get_avail_mem=(lambda **kw: 5000000)):
            ejk = uhf_fast._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1)
        assert abs(ejk.sum(axis=(0,1))).max().get() < 1e-9
        assert abs(ejk-ref).max().get() < 1e-8

        mo_coeff = cp.array(np.random.rand(2, nao, nao) - .5)
        mo_occ[1,5] = 0
        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        dm = cp.einsum('npi,ni,nqi->npq', mo_coeff, mo_occ, mo_coeff)
        dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        ejk = uhf_fast._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1).get()

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
            return uhf_grad._jk_energy_per_atom(opt, dm, j_factor=1, k_factor=1)

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            e1 = eval_grad(i, x, disp)
            e2 = eval_grad(i, x, -disp)
            assert abs((e1 - e2)/(2*disp) - ejk[i,:,x]).max() < 1e-5

    def test_jk_ip1(self):
        from gpu4pyscf.df.hessian.uhf import _get_jk_ip
        np.random.seed(8)
        nao = mol.nao
        mo_coeff = cp.array(np.random.rand(2, nao, nao) - .5)
        mo_occ = cp.zeros((2, nao))
        mo_occ[0,:5] = 1
        mo_occ[1,:4] = 1

        obj = mol.RHF().to_gpu().density_fit(auxbasis=auxmol.basis).Hessian()
        obj.auxbasis_response = 2
        vj, vk = _get_jk_ip(obj, mo_coeff, mo_occ)
        refa = vj[0] - vk[0]
        refb = vj[1] - vk[1]

        opt = int3c2e.Int3c2eOpt(mol, auxmol).build()
        veff = uhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)
        assert abs(refa - veff[0]).max() < 1e-8
        assert abs(refb - veff[1]).max() < 1e-8

    def test_jk_ip1_finite_diff(self):
        mol2 = mol + mol
        np.random.seed(9)
        nao = mol2.nao
        mo_coeff = cp.array(np.random.rand(2, nao, nao) - .5) * .2
        mo_occ = cp.zeros((2, nao))
        mo_occ[0,:nao-5] = 1
        mo_occ[1,:nao-6] = 1

        opt = int3c2e.Int3c2eOpt(mol2, auxmol).build()
        veff = uhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)

        from pyscf.df import incore
        disp = .5e-3
        mol0 = mol2.copy(deep=True)
        auxmol0 = auxmol.copy(deep=True)
        dm = cp.einsum('npi,ni,nqi->npq', mo_coeff, mo_occ, mo_coeff)
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
            vj = cp.einsum('ijkl,nji->kl', eri, dm)
            vk = cp.einsum('ijkl,njk->nil', eri, dm)
            return vj - vk

        for i, x in [(0, 0), (0, 1), (0, 2)]:
            v1 = eval_veff(i, x, disp)
            v2 = eval_veff(i, x, -disp)
            ref = mo_coeff[0].T.dot(v1[0] - v2[0]).dot(mo_coeff[0,:,:nao-5]) / (2*disp)
            assert abs(ref - veff[0][i,x]).max().get() < 5e-5
            ref = mo_coeff[1].T.dot(v1[1] - v2[1]).dot(mo_coeff[1,:,:nao-6]) / (2*disp)
            assert abs(ref - veff[1][i,x]).max().get() < 5e-5

    def test_jk_ip1_limited_memory(self):
        mol1 = mol + mol
        mol1 = mol1 + mol1
        np.random.seed(8)
        nao = mol1.nao
        nocc = 5
        mo_coeff = cp.array(np.random.rand(nao, nao) - .5)
        mo_coeff = cp.repeat(mo_coeff[None], 2, axis=0)
        mo_occ = cp.zeros((2,nao))
        mo_occ[:,:nocc] = 1
        dm = cp.einsum('npi,nqi->npq', mo_coeff[:,:,:nocc], mo_coeff[:,:,:nocc])
        dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

        opt = int3c2e.Int3c2eOpt(mol1, auxmol).build()
        ref = rhf_fast._get_veff(opt, mo_coeff[0], mo_occ[0]*2, j_factor=1, k_factor=1)

        with lib.temporary_env(uhf_fast, get_avail_mem=(lambda **kw: nao**2*3*mol1.natm*20)):
            veff = uhf_fast._get_veff(opt, mo_coeff, mo_occ, j_factor=1, k_factor=1)
            assert abs(ref - veff[0]).max() < 1e-9
            assert abs(ref - veff[1]).max() < 1e-9

if __name__ == "__main__":
    print("Full Tests for DF UHF Hessian")
    unittest.main()
