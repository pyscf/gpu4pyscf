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
#

import unittest
import numpy
import os
from pathlib import Path

from pyscf import gto, dft
from gpu4pyscf.dft.rks import RKS as RKS_gpu

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'sto3g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):

    def test_df_rks_hess_elec(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit()
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
        mf = dft.RKS(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        hessobj = mf.Hessian()
        hessobj.auxbasis_response = 2
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_df_gga(self):
        mf = dft.RKS(mol, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        hessobj = mf.Hessian()
        hessobj.auxbasis_response = 2
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hessobj.base.cphf_grids = hessobj.base.grids
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_df_mgga(self):
        mf = dft.RKS(mol, xc='tpss').density_fit()
        mf.conv_tol = 1e-10
        mf.grids.level = 1
        mf.conv_tol_cpscf = 1e-8
        mf.kernel()

        hessobj = mf.Hessian()
        hessobj.auxbasis_response = 2
        hess_cpu = hessobj.kernel()

        mf = mf.to_gpu()
        hessobj = mf.Hessian()
        hessobj.base.cphf_grids = hessobj.base.grids
        hess_gpu = hessobj.kernel()
        assert numpy.linalg.norm(hess_cpu - hess_gpu) < 1e-5

    def test_bugfix_Pb_hessian_large_difference(self):
        mol = gto.M(
            atom = """
                Pb      -1.19671400       3.62001400      -1.53322400
                C       -2.47357700       2.87507600      -3.73923000
                C       -1.81973600       1.82251600      -3.02001100
                C       -0.64462300       1.21389800      -3.56493600
                C       -0.15044700       0.07880100      -3.01227100
                C       -0.66986800      -0.46343700      -1.77201700
                C       -1.69585300       0.28202100      -1.14699800
                C       -2.30115600       1.38204200      -1.73736700
                H       -2.24910900       2.86292100      -4.81842400
                H       -3.26665300       1.71853000      -1.35858900
                H        0.67506800      -0.45383700      -3.47323000
                H       -2.00357100      -0.01277900      -0.13639500
                H       -0.23961500       1.61363300      -4.51628300
                H     -0.261327   -1.587183   -1.205135
                C       -3.45090200       3.69969500      -3.23780400
                H       -3.56660900       3.91571300      -2.14979900
                H       -4.45809100       4.18707600      -4.16425900
            """,
            basis = 'def2-tzvp',
            ecp = 'def2-tzvp',
            charge = 4,
            spin = 0,
            verbose = 0,
        )

        mf = RKS_gpu(mol, xc = "wB97X").density_fit(auxbasis = 'def2-tzvp-jkfit')
        mf.grids.atom_grid = (99, 590)
        mf.max_cycle = 50
        mf.conv_tol = 1e-11
        mf.conv_tol_cpscf = 1e-10

        mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        hobj = mf.Hessian()
        hobj.grid_response = True
        test_hessian = hobj.kernel()
        test_hessian = test_hessian.transpose(0,2,1,3).reshape(mol.natm * 3, mol.natm * 3)

        # Numerical gradient, dx = 1e-4
        ref_gradient_path = Path(os.path.realpath(__file__)).parent / "reference_data/test_bugfix_Pb_hessian_large_difference_numerical_gradient_1em4.txt"
        ref_gradient = numpy.loadtxt(ref_gradient_path)
        assert numpy.max(numpy.abs(test_gradient - ref_gradient)) < 3e-6

        # Numerical hessian, dx = 1e-3
        ref_hessian_path = Path(os.path.realpath(__file__)).parent / "reference_data/test_bugfix_Pb_hessian_large_difference_numerical_hessian_1em3.txt"
        ref_hessian = numpy.loadtxt(ref_hessian_path)
        assert numpy.max(numpy.abs(test_hessian - ref_hessian)) < 2e-4


if __name__ == "__main__":
    print("Full Tests for DF RKS Hessian")
    unittest.main()
