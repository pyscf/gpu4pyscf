#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
from pyscf import lib, gto, scf
from gpu4pyscf import tdscf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol.basis = '631g'
        # FIXME: mo_coeff of uhf_symm.SymAdaptedUHF not converted to cupy arrays
        mol.symmetry = True
        cls.mol = mol.build()
        cls.mf = scf.UHF(mol).density_fit().run(conv_tol=1e-10).to_gpu()

        mol1 = gto.Mole()
        mol1.verbose = 7
        mol1.output = '/dev/null'
        mol1.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol1.basis = '631g'
        mol1.spin = 2
        cls.mol1 = mol1.build()
        cls.mf1 = scf.UHF(mol1).run(conv_tol=1e-10).to_gpu()

    @classmethod
    def tearDownClass(cls):
        cls.mol1.stdout.close()

    def test_tda(self):
        mf = self.mf
        td = mf.TDA()
        assert td.device == 'gpu'
        td.nstates = 5
        e = td.kernel()[0]
        ref = [11.0179839, 11.0179839, 11.9031214, 11.9031214, 13.1701375]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tdhf(self):
        mf = self.mf
        td = mf.TDHF()
        assert td.device == 'gpu'
        td.nstates = 5
        td.conv_tol = 1e-5
        e = td.kernel()[0]
        ref = [10.8924334, 10.8924334, 11.8352278, 11.8352278, 12.6350840]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tda1(self):
        mf1 = self.mf1
        td = mf1.TDA()
        assert td.device == 'gpu'
        td.nstates = 5
        e = td.kernel()[0]
        ref = [ 3.3211349, 18.5597821, 21.0147390, 21.6150240, 25.0938938]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tdhf1(self):
        mf1 = self.mf1
        td = mf1.TDHF()
        assert td.device == 'gpu'
        td.nstates = 4
        e = td.kernel()[0]
        ref = [ 3.3126683, 18.4954862, 20.8493515, 21.5480882,]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tda_vind(self):
        mf = self.mf1
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDA().set().gen_vind()[0](zs)
        dat = mf.TDA().set().gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tdhf_vind(self):
        mf = self.mf1
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,2,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDHF().set().gen_vind()[0](zs)
        dat = mf.TDHF().set().gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for uhf-TDA and uhf-TDHF")
    unittest.main()
