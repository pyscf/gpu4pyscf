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
    
