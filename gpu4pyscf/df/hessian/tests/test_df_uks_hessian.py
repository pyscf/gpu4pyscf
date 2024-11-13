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
from pyscf import gto, dft

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 1
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

        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_df_gga(self):
        mf = dft.UKS(mol, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hessobj.base.cphf_grids = hessobj.base.grids
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_df_mgga(self):
        mf = dft.UKS(mol, xc='tpss').density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        hessobj = mf.Hessian()
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hessobj.base.cphf_grids = hessobj.base.grids
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

if __name__ == "__main__":
    print("Full Tests for DF UKS Hessian")
    unittest.main()
    