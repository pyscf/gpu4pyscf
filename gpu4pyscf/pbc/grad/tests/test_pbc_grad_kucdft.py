#!/usr/bin/env python
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
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf.pbc.dft import kucdft
from gpu4pyscf.pbc.grad import kucdft as kucdft_grad
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt


def setUpModule():
    global cell
    cell = pyscf.M(
            atom='''
            C 0.000000000000   0.000000000000   0.000000000000
            C 1.687            1.687            1.5
            ''',
            basis='gth-szv',
            pseudo='gth-pbe',
            a='''
            0.000000000, 3.370137329, 3.370137329
            3.370137329, 0.000000000, 3.370137329
            3.370137329, 3.370137329, 0.000000000''',
            unit='B',
            verbose=0,
            output = '/dev/null',
        )


def tearDownModule():
    global cell
    cell.stdout.close()
    del cell


class KnownValues(unittest.TestCase):
    def test_finite_diff_cdft_grad(self):
        kpts = cell.make_kpts([2, 2, 2])
        charge_constraints = ([[0]], [4.1])
        mf = kucdft.CDFT_KUKS(cell, kpts, charge_constraints=charge_constraints)
        mf.xc = 'pbe'
        mf.minao_ref = 'gth-szv'
        mf.conv_tol = 1e-12
        mf._numint = MultiGridNumInt(cell)
        mf.kernel()
        
        dm0 = mf.make_rdm1()
        v_lagrange0 = mf.v_lagrange.copy()

        g_obj = kucdft_grad.Gradients(mf)
        g_ana = g_obj.kernel()
        
        atom_idx = 1
        coord_idx = 0
        h = 1e-4
        
        base_coords = cell.atom_coords()
        coords_plus = base_coords.copy()
        coords_plus[atom_idx, coord_idx] += h
        cell_plus = cell.copy()
        cell_plus.set_geom_(coords_plus, unit='Bohr')
        
        mf_plus = kucdft.CDFT_KUKS(cell_plus, kpts, charge_constraints=charge_constraints)
        mf_plus.verbose = 0
        mf_plus.xc = 'pbe'
        mf_plus.minao_ref = 'gth-szv'
        mf_plus.conv_tol = 1e-12
        mf_plus.v_lagrange = v_lagrange0
        mf_plus._numint = MultiGridNumInt(cell_plus)
        mf_plus.kernel(dm0=dm0)
        e_plus = mf_plus.e_tot
        
        coords_minus = base_coords.copy()
        coords_minus[atom_idx, coord_idx] -= h
        cell_minus = cell.copy()
        cell_minus.set_geom_(coords_minus, unit='Bohr')
        
        mf_minus = kucdft.CDFT_KUKS(cell_minus, kpts, charge_constraints=charge_constraints)
        mf_minus.verbose = 0
        mf_minus.xc = 'pbe'
        mf_minus.minao_ref = 'gth-szv'
        mf_minus.conv_tol = 1e-12
        mf_minus.v_lagrange = v_lagrange0
        mf_minus._numint = MultiGridNumInt(cell_minus)
        mf_minus.kernel(dm0=dm0)
        e_minus = mf_minus.e_tot
        
        g_fd = (e_plus - e_minus) / (2.0 * h)
        self.assertAlmostEqual(g_ana[atom_idx, coord_idx], g_fd, 5)

    def test_ref_cdft_grad(self):
        kpts = cell.make_kpts([2, 2, 2])
        charge_constraints = ([[0]], [4.1])
        mf = kucdft.CDFT_KUKS(cell, kpts, charge_constraints=charge_constraints)
        mf.xc = 'pbe'
        mf.minao_ref = 'gth-szv'
        mf.conv_tol = 1e-12
        mf.kernel()

        g_obj = kucdft_grad.Gradients(mf)
        g_ana = g_obj.kernel()
        ref_num = np.array([[-0.00178033, -0.00178033,  0.13308021],
                            [ 0.00177992,  0.00177992, -0.13304387],]) # step = 1e-4
        ref_ana = np.array([[-0.00178033, -0.00178033,  0.1330802 ],
                            [ 0.00177992,  0.00177992, -0.13304387],]) 
        self.assertTrue(np.allclose(g_ana, ref_ana, atol=1e-5))
        self.assertTrue(np.allclose(g_ana, ref_num, atol=1e-5))


if __name__ == '__main__':
    print("Full Tests for PBC UCDFT Gradients")
    unittest.main()
