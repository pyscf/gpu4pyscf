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
        assert abs(ref - e2_gpu).max() < 1e-6

    def test_partial_hess_elec(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e1_cpu, ej_cpu, ek_cpu = uhf_cpu._partial_hess_ejk(hobj)
        e2_cpu = ej_cpu - ek_cpu

        mf = mf.to_gpu()
        mf.kernel()
        hobj = mf.Hessian()
        e1_gpu, e2_gpu = uhf_gpu._partial_hess_ejk(hobj)

        assert numpy.linalg.norm(e1_cpu - e1_gpu.get()) < 1e-5
        assert numpy.linalg.norm(e2_cpu - e2_gpu.get()) < 1e-5

    def test_hessian_uhf_D3(self):
        print('----- testing UHF with D3BJ ------')
        mf = mol.UHF()
        mf.disp = 'd3bj'
        mf.run()
        mf.conv_tol_cpscf = 1e-8
        ref = mf.Hessian().kernel()
        e2_gpu = mf.Hessian().to_gpu().kernel()
        assert abs(ref - e2_gpu).max() < 1e-6

if __name__ == "__main__":
    print("Full Tests for UHF Hessian")
    unittest.main()
