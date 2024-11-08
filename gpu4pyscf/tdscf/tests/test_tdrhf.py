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
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol.basis = '631g'
        mol.symmetry = True
        cls.mol = mol.build()
        cls.mf = mf = scf.RHF(mol).to_gpu().run()
        cls.df_mf = mf.density_fit().run()
        cls.nstates = 5 # make sure first 3 states are converged

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_tda_singlet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDA().set(nstates=nstates)
        assert td.device == 'gpu'
        e = td.kernel()[0]
        ref = [11.9027511, 11.9027511, 16.8603101]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.65616659, 5)

        df_mf = self.df_mf
        td = df_mf.TDA().set(nstates=nstates)
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.65618093, 5)

    def test_tda_triplet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDA().set(nstates=nstates)
        assert td.device == 'gpu'
        td.singlet = False
        e = td.kernel()[0]
        ref = [11.0174650, 11.0174650, 13.1694960]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

        df_mf = self.df_mf
        td = df_mf.TDA().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

    def test_tdhf_singlet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDHF().set(nstates=nstates)
        assert td.device == 'gpu'
        e = td.kernel()[0]
        ref = [11.8348584, 11.8348584, 16.6630381]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.64009191, 5)

        df_mf = self.df_mf
        td = df_mf.TDHF().set(nstates=nstates)
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.64011895, 5)

    def test_tdhf_triplet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDHF().set(nstates=nstates)
        assert td.device == 'gpu'
        td.singlet = False
        e = td.kernel()[0]
        ref = [10.8919091, 10.8919091, 12.6343507]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 5)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

        df_mf = self.df_mf
        td = df_mf.TDHF().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

    def test_tda_vind(self):
        mf = self.mf
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,nocc,nvir)
        ref = mf.to_cpu().TDA().set(singlet=False).gen_vind()[0](zs)
        dat = mf.TDA().set(singlet=False).gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

        df_mf = self.df_mf
        ref = df_mf.to_cpu().TDA().set(singlet=True).gen_vind()[0](zs)
        dat = df_mf.TDA().set(singlet=True).gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tdhf_vind(self):
        mf = self.mf
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,2,nocc,nvir)
        ref = mf.to_cpu().TDHF().set(singlet=True).gen_vind()[0](zs)
        dat = mf.TDHF().set(singlet=True).gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

        df_mf = self.df_mf
        ref = df_mf.to_cpu().TDHF().set(singlet=False).gen_vind()[0](zs)
        dat = df_mf.TDHF().set(singlet=False).gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for rhf-TDA and rhf-TDHF")
    unittest.main()
