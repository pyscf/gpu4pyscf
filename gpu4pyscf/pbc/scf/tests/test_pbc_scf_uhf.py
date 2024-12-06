#!/usr/bin/env python
#
# Copyright 2024 The GPU4PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import scf as pscf
from gpu4pyscf.pbc.scf import kuhf

def setUpModule():
    global cell
    L = 4
    n = 21
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               verbose = 7,
               output = '/dev/null',
               a = ((L,0,0),(0,L,0),(0,0,L)),
               mesh = [n,n,n],
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]]},
               spin = 2)

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_kuhf_bands(self):
        nk = [2, 2, 1]
        kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pscf.KUHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kmf_cpu = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, kmf_cpu.e_tot, 8)
        self.assertAlmostEqual(kmf.e_tot, -4.021029656152094, 8)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = kmf.get_bands(kpts_bands)[0]
        e_ref = kmf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, 6)

    def test_uhf_bands(self):
        mf = pscf.UHF(cell).run(conv_tol=1e-9)
        mf_cpu = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_cpu.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -3.9546467710639632, 8)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = mf.get_bands(kpts_bands)[0]
        e_ref = mf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, 6)

    def test_small_system(self):
        mol = pbcgto.Cell(
            atom='H 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            spin=1,
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -0.10439957735616917, 8)

        mol = pbcgto.Cell(
            atom='He 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -2.2719576422665635, 8)


if __name__ == '__main__':
    print("Tests for PBC UHF and PBC KUHF")
    unittest.main()
