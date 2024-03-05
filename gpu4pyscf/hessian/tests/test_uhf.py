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
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.spin = 2
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_partial_hess_elec(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e1_cpu, ej_cpu, ek_cpu = uhf_cpu.partial_hess_elec(hobj)

        mf = mf.to_gpu()
        mf.kernel()
        hobj = mf.Hessian()
        e1_gpu, ej_gpu, ek_gpu = uhf_gpu.partial_hess_elec(hobj)

        assert numpy.linalg.norm(e1_cpu - e1_gpu) < 1e-5
        assert numpy.linalg.norm(ej_cpu - ej_gpu) < 1e-5
        assert numpy.linalg.norm(ek_cpu - ek_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for UHF Hessian")
    unittest.main()