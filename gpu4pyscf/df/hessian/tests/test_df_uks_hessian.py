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
import pyscf
from pyscf import lib
from pyscf import gto, dft
from packaging import version

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.charge = 1
    mol.spin = 1
    mol.basis = 'sto3g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):

    def test_df_uks_hess_elec(self):
        mf = dft.UKS(mol, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.grids.level = 1
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_cpu = hobj.partial_hess_elec()

        mf = mf.to_gpu()
        mf.grids.level = 1
        mf.kernel()
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hess_gpu = hobj.partial_hess_elec()
        assert numpy.linalg.norm(hess_cpu - hess_gpu.get()) < 1e-5

    def test_df_lda(self):
        mf = dft.UKS(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        mf_gpu = mf.to_gpu()
        hessobj = mf_gpu.Hessian()
        hess_gpu = hessobj.kernel()
        assert abs(lib.fp(hess_gpu) - -1.2463973314907781) < 1e-7

        if version.parse(pyscf.__version__) >= version.parse('2.14.0'):
            hessobj = mf.Hessian()
            hessobj.auxbasis_response = 2
            hess_cpu = hessobj.kernel()
            assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_df_gga(self):
        mf = dft.UKS(mol, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        mf_gpu = mf.to_gpu()
        hessobj = mf_gpu.Hessian()
        hessobj.base.cphf_grids = hessobj.base.grids
        hess_gpu = hessobj.kernel()
        assert abs(lib.fp(hess_gpu) - -1.2334161112529511) < 1e-7

        if version.parse(pyscf.__version__) >= version.parse('2.14.0'):
            hessobj = mf.Hessian()
            hessobj.auxbasis_response = 2
            hess_cpu = hessobj.kernel()
            assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_df_mgga(self):
        mf = dft.UKS(mol, xc='tpss').density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        mf_gpu = mf.to_gpu()
        hessobj = mf_gpu.Hessian()
        hessobj.base.cphf_grids = hessobj.base.grids
        hess_gpu = hessobj.kernel()
        assert abs(lib.fp(hess_gpu) - -1.1875258141432055) < 1e-7

        if version.parse(pyscf.__version__) >= version.parse('2.14.0'):
            hessobj = mf.Hessian()
            hessobj.auxbasis_response = 2
            hess_cpu = hessobj.kernel()
            assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for DF UKS Hessian")
    unittest.main()
