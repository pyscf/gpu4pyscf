# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
from pyscf import gto, scf, lib
from pyscf import grad, hessian
from pyscf.hessian import rhf as rhf_cpu
from gpu4pyscf.hessian import rhf as rhf_gpu

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_hessian_rhf(self):
        mf = mol.RHF().run()
        mf.conv_tol_cpscf = 1e-8
        hobj = mf.Hessian()
        ref = hobj.kernel()
        e2_gpu = hobj.to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-8

    def test_partial_hess_elec(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e1_cpu, ej_cpu, ek_cpu = rhf_cpu._partial_hess_ejk(hobj)

        mf = mf.to_gpu()
        mf.kernel()
        hobj = mf.Hessian()
        e1_gpu, ej_gpu, ek_gpu = rhf_gpu._partial_hess_ejk(hobj)

        assert abs(e1_cpu - e1_gpu.get()).max() < 1e-5
        assert abs(ej_cpu - ej_gpu.get()).max() < 1e-5
        assert abs(ek_cpu - ek_gpu.get()).max() < 1e-5

    def test_ejk_ip2(self):
        mol = gto.M(
            atom = '''
            O       0.0000000000    -0.0000000000     0.1174000000
                C 1. 1. 0
                H1 3.1 0.12 4.35
                H2 2.1 1.31 6
            ''',
            basis='6-31g**', unit='B')
        np.random.seed(9)
        nao = mol.nao
        mo_coeff = np.random.rand(nao, nao)
        dm = mo_coeff.dot(mo_coeff.T) * 2
        mo_occ = np.ones(nao) * 2
        mo_energy = np.random.rand(nao)

        ej, ek = rhf_gpu._partial_ejk_ip2(mol, dm)
        assert abs(lib.fp(ej.get()) - -792951.4785844693) < 1e-7
        assert abs(lib.fp(ek.get()) - -352265.0466989743) < 1e-7
        mf = mol.RHF()
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
        mf.mo_energy = mo_energy
        h = rhf_cpu.Hessian(mf)
        e1, refj, refk = rhf_cpu._partial_hess_ejk(h, mo_energy, mo_coeff, mo_occ)
        assert abs(ej.get() - refj).max() < 1e-6
        assert abs(ek.get() - refk).max() < 1e-6

    def test_get_jk(self):
        mol = gto.M(
            atom = '''
            O       0.0000000000    -0.0000000000     0.1174000000
            H      -0.7570000000     4.0000000000    -0.4696000000
            H       0.7570000000     4.0000000000    -0.4696000000
                C 1. 1. 0
                H1 3.1 0.12 4.35
                H2 2.1 1.31 6
            ''',
            basis='def2-tzvpp', unit='B')
        np.random.seed(9)
        nao = mol.nao
        mo_coeff = np.random.rand(nao, nao)
        dm = mo_coeff.dot(mo_coeff.T) * 2

        vj, vk = rhf_gpu._get_jk(mol, dm)
        assert abs(lib.fp(vj.get()) -  87674.69061160382) < 1e-7
        assert abs(lib.fp(vk.get()) - -9.317650662101629) < 1e-7

        h1ao = [None] * mol.natm
        aoslices = mol.aoslice_by_atom()
        for ia in range(mol.natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, mol.nbas)*3
            vj1, vj2, vk1, vk2 = rhf_cpu._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['ji->s2kl', -dm[:,p0:p1],  # vj1
                                          'lk->s1ij', -dm         ,  # vj2
                                          'li->s1kj', -dm[:,p0:p1],  # vk1
                                          'jk->s1il', -dm         ], # vk2
                                         shls_slice=shls_slice)
            vj1[:,p0:p1] += vj2
            vk1[:,p0:p1] += vk2
            vj1 = vj1 + vj1.transpose(0,2,1)
            vk1 = vk1 + vk1.transpose(0,2,1)
            h1ao[ia] = vj1, vk1
        h1ao = np.array(h1ao)
        refj = h1ao[:,0]
        refk = h1ao[:,1]
        assert abs(vj.get() - refj).max() < 1e-8
        assert abs(vk.get() - refk).max() < 1e-8

    def test_hessian_rhf_D3(self):
        print('----- testing RHF with D3BJ ------')
        mf = mol.RHF()
        mf.disp = 'd3bj'
        mf.run()
        mf.conv_tol_cpscf = 1e-8
        ref = mf.Hessian().kernel()
        e2_gpu = mf.Hessian().to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-8

if __name__ == "__main__":
    print("Full Tests for RHF Hessian")
    unittest.main()
