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
import pyscf
import pytest
import cupy
from gpu4pyscf.dft import rks, uks

def setUpModule():
    global mol
    atom = '''
C                 -0.07551087    1.68127663   -0.10745193
O                  1.33621755    1.87147409   -0.39326987
C                  1.67074668    2.95729545    0.49387976
C                  0.41740763    3.77281969    0.78495878
C                 -0.60481480    3.07572636    0.28906224
H                 -0.19316298    1.01922455    0.72486113
O                  0.35092043    5.03413298    1.45545728
H                  0.42961487    5.74279041    0.81264173
O                 -1.95331750    3.53349874    0.15912025
H                 -2.55333895    2.78846397    0.23972698
O                  2.81976302    3.20110148    0.94542226
C                 -0.81772499    1.09230218   -1.32146482
H                 -0.70955636    1.74951833   -2.15888136
C                 -2.31163857    0.93420736   -0.98260166
H                 -2.72575463    1.89080093   -0.74107186
H                 -2.41980721    0.27699120   -0.14518512
O                 -0.26428017   -0.18613595   -1.64425697
H                 -0.72695910   -0.55328886   -2.40104423
O                 -3.00083741    0.38730252   -2.10989934
H                 -3.93210821    0.28874990   -1.89865997
'''

    mol = pyscf.M(atom=atom, basis='def2-tzvpp', max_memory=32000, cart=0)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    @pytest.mark.smoke
    def test_b3lyp_with_d3bj(self):
        print('-------- DFRKS with D3(BJ) -------')
        mf = rks.RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.disp = 'd3bj'
        e_dft = mf.kernel()
        assert np.abs(e_dft - -685.0326965348272) < 1e-7

        g = mf.nuc_grad_method().kernel()
        assert np.abs(cupy.linalg.norm(g) - 0.17498362161082373) < 1e-5

        h = mf.Hessian().kernel()
        assert np.abs(cupy.linalg.norm(h) - 3.7684319231335377) < 1e-4

    @pytest.mark.smoke
    def test_b3lyp_d3bj(self):
        print('-------- DFRKS with D3(BJ) -------')
        mf = rks.RKS(mol, xc='b3lyp-d3bj').density_fit(auxbasis='def2-tzvpp-jkfit')
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        e_dft = mf.kernel()
        assert np.abs(e_dft - -685.0326965348272) < 1e-7

        g = mf.nuc_grad_method().kernel()
        assert np.abs(cupy.linalg.norm(g) - 0.17498362161082373) < 1e-5

        h = mf.Hessian().kernel()
        assert np.abs(cupy.linalg.norm(h) - 3.7684319231335377) < 1e-4

    @pytest.mark.smoke
    def test_DFUKS(self):
        print('------- DFUKS with D3(BJ) -------')
        mf = uks.UKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.disp = 'd3bj'
        e_dft = mf.kernel()
        assert np.abs(e_dft - -685.0326965349493) < 1e-7

        g = mf.nuc_grad_method().kernel()
        assert np.abs(cupy.linalg.norm(g) - 0.17498264516108836) < 1e-5

        h = mf.Hessian().kernel()
        assert np.abs(cupy.linalg.norm(h) - 3.768429871470736) < 1e-4

    @pytest.mark.smoke
    def test_RKS(self):
        print('-------- RKS with D3(BJ) -------')
        mf = rks.RKS(mol, xc='b3lyp')
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-12
        mf.disp = 'd3bj'
        e_dft = mf.kernel()
        assert np.abs(e_dft - -685.0325611822375) < 1e-7

        g = mf.nuc_grad_method().kernel()
        assert np.abs(cupy.linalg.norm(g) - 0.1750368231223345) < 1e-6

    @pytest.mark.smoke
    def test_DFRKS_with_SMD(self):
        print('----- DFRKS with SMD -----')
        mf = rks.RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
        mf = mf.SMD()
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.disp = 'd3bj'
        e_dft = mf.kernel()
        assert np.abs(e_dft - -685.0578838805443) < 1e-7

        g = mf.nuc_grad_method().kernel()
        assert np.abs(cupy.linalg.norm(g) - 0.16905807654571403) < 1e-5

        h = mf.Hessian().kernel()
        assert np.abs(cupy.linalg.norm(h) - 3.743840896534178) < 1e-4

    @pytest.mark.smoke
    def test_DFUKS_with_SMD(self):
        print('------- DFUKS with SMD ---------')
        mf = uks.UKS(mol, xc='b3lyp').density_fit(auxbasis='def2-tzvpp-jkfit')
        mf = mf.SMD()
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf.conv_tol_cpscf = 1e-8
        mf.disp = 'd3bj'
        e_dft = mf.kernel()
        assert np.abs(e_dft - -685.05788388063) < 1e-7

        g = mf.nuc_grad_method().kernel()
        assert np.abs(cupy.linalg.norm(g) - 0.1690582751813457) < 1e-5

        h = mf.Hessian().kernel()
        assert np.abs(cupy.linalg.norm(h) - 3.743858482519822) < 1e-4

if __name__ == "__main__":
    print("Full Smoke Tests")
    unittest.main()
