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
import numpy
from pyscf import gto, scf, lib
from pyscf import grad, hessian
from pyscf.hessian import uhf as uhf_cpu
from gpu4pyscf.hessian import uhf as uhf_gpu

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
    mol.spin = 1
    mol.charge = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_hessian_uhf(self):
        mf = mol.UHF().run()
        mf.conv_tol_cpscf = 1e-8
        hobj = mf.Hessian()
        ref = hobj.kernel()
        e2_gpu = hobj.to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-8

    def test_partial_hess_elec(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e1_cpu, ej_cpu, ek_cpu = uhf_cpu._partial_hess_ejk(hobj)

        mf = mf.to_gpu()
        mf.kernel()
        hobj = mf.Hessian()
        e1_gpu, ej_gpu, ek_gpu = uhf_gpu._partial_hess_ejk(hobj)

        assert numpy.linalg.norm(e1_cpu - e1_gpu.get()) < 1e-5
        assert numpy.linalg.norm(ej_cpu - ej_gpu.get()) < 1e-5
        assert numpy.linalg.norm(ek_cpu - ek_gpu.get()) < 1e-5

    def test_hessian_uhf_D3(self):
        print('----- testing UHF with D3BJ ------')
        mf = mol.UHF()
        mf.disp = 'd3bj'
        mf.run()
        mf.conv_tol_cpscf = 1e-8
        ref = mf.Hessian().kernel()
        e2_gpu = mf.Hessian().to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-8

if __name__ == "__main__":
    print("Full Tests for UHF Hessian")
    unittest.main()
