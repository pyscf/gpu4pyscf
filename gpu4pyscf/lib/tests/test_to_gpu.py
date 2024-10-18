# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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
import cupy
import pyscf
import pytest
from pyscf import scf, lib
from pyscf.dft import rks
from packaging import version

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='sto3g'
grids_level = 1
pyscf_24 = version.parse(pyscf.__version__) <= version.parse('2.4.0')

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    @pytest.mark.skipif(pyscf_24, reason='requires pyscf 2.5 or higher')
    def test_rhf(self):
        mf = scf.RHF(mol).to_gpu()
        e_tot = mf.to_gpu().kernel()
        assert numpy.abs(e_tot - -74.96306312971964) < 1e-7

        mf = scf.RHF(mol).run()
        gobj = mf.nuc_grad_method().to_gpu()
        g = gobj.kernel()
        assert numpy.abs(lib.fp(g) - -0.016405814100962965) < 1e-7
        # RHF Hessian is not supported yet
        # mf = scf.RHF(mol).run()
        # h = mf.Hessian().to_gpu()
        # h.kernel()

    @pytest.mark.skipif(pyscf_24, reason='requires pyscf 2.5 or higher')
    def test_rks(self):
        mf = rks.RKS(mol).to_gpu()
        e_tot = mf.to_gpu().kernel()
        assert numpy.abs(e_tot - -74.73210527989748) < 1e-6

        mf = rks.RKS(mol).run()
        gobj = mf.nuc_grad_method().to_gpu()
        g = gobj.kernel()
        assert numpy.abs(lib.fp(g) - -0.04340162663176693) < 1e-6

        # RKS Hessian it not supported yet
        # mf = rks.RKS(mol).run()
        # h = mf.Hessian().to_gpu()
        # h.kernel()

    @pytest.mark.skipif(pyscf_24, reason='requires pyscf 2.5 or higher')
    def test_df_RHF(self):
        mf = scf.RHF(mol).density_fit().to_gpu()
        e_tot = mf.to_gpu().kernel()
        assert numpy.abs(e_tot - -74.96314991764658) < 1e-7

        mf = scf.RHF(mol).density_fit().run()
        gobj = mf.nuc_grad_method().to_gpu()
        g = gobj.kernel()
        assert numpy.abs(lib.fp(g) - -0.01641213202225146) < 1e-7

        mf = scf.RHF(mol).density_fit().run()
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian().to_gpu()
        h = hobj.kernel()
        assert numpy.abs(lib.fp(h) - 2.198079352288524) < 1e-4

    @pytest.mark.skipif(pyscf_24, reason='requires pyscf 2.5 or higher')
    def test_df_b3lyp(self):
        mf = rks.RKS(mol, xc='b3lyp').density_fit().to_gpu()
        e_tot = mf.to_gpu().kernel()
        assert numpy.abs(e_tot - -75.31295618175646) < 1e-7

        mf = rks.RKS(mol, xc='b3lyp').density_fit().run()
        gobj = mf.nuc_grad_method().to_gpu()
        g = gobj.kernel()
        assert numpy.abs(lib.fp(g) - -0.04079319540057938) < 1e-5

        mf = rks.RKS(mol, xc='b3lyp').density_fit().run()
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian().to_gpu()
        h = hobj.kernel()
        assert numpy.abs(lib.fp(h) - 2.1527804103141848) < 1e-4

    @pytest.mark.skipif(pyscf_24, reason='requires pyscf 2.5 or higher')
    def test_df_RKS(self):
        mf = rks.RKS(mol, xc='wb97x').density_fit().to_gpu()
        e_tot = mf.to_gpu().kernel()
        assert numpy.abs(e_tot - -75.30717654021076) < 1e-6

        mf = rks.RKS(mol, xc='wb97x').density_fit().run()
        gobj = mf.nuc_grad_method().to_gpu()
        g = gobj.kernel()
        assert numpy.abs(lib.fp(g) - -0.03432881437746617) < 1e-5

        mf = rks.RKS(mol, xc='wb97x').density_fit().run()
        mf.conv_tol_cpscf = 1e-7
        hobj = mf.Hessian().to_gpu()
        h = hobj.kernel()
        assert numpy.abs(lib.fp(h) - 2.1858589608638384) < 1e-4

if __name__ == "__main__":
    print("Full tests for to_gpu module")
    unittest.main()
    