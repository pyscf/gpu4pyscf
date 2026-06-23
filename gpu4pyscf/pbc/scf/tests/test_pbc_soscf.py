#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf.pbc import scf
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt

def setUpModule():
    global cell
    cell = pyscf.M(
        unit = 'B',
        atom = '''
C  0.          0.          0.
C  1.68506879  1.68506879  1.68506879
''',
        a = '''
0.          3.37013758  3.37013758
3.37013758  0.          3.37013758
3.37013758  3.37013758  0.
''',
        basis = '''
C S
        4.3362376436   0.1490797872
        1.2881838513  -0.0292640031
        0.4037767149  -0.6882040510
C S
        0.2187877657   1.0000000000
C P
        1.2881838513  -0.2775560300
        0.4037767149  -0.4712295093
''',
        pseudo = 'gth-pade',
        precision = 1e-7,
        mesh = [35]*3,
        verbose = 5,
        output = '/dev/null')


def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = scf.RHF(cell)
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -9.870755717258616, 8)

    def test_nr_uhf(self):
        mf = scf.UHF(cell).density_fit()
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -9.87737235242746, 8)

    def test_nr_rks_lda(self):
        mf = dft.RKS(cell, xc='lda,')
        mf = mf.multigrid_numint()
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -9.454224868598352, 8)

    def test_nr_uks_lda(self):
        mf = dft.RKS(cell, xc='lda,')
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -9.454224868598352, 7)

    def test_nr_rks_gga(self):
        mf = dft.RKS(cell, xc='b88,')
        mf = mf.multigrid_numint()
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -9.646435172665008, 8)

    def test_nr_uks_gga(self):
        mf = dft.UKS(cell, xc='pbe0')
        mf = mf.multigrid_numint()
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.021211009946274, 8)

    def test_nr_uks_rsh(self):
        mf = dft.UKS(cell, xc='camb3lyp')
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf = mf.multigrid_numint()
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.04741991140345, 8)

    def test_nr_krhf(self):
        mf = scf.KRHF(cell, cell.make_kpts([2,1,1]))
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.293741115059207, 8)

    def test_nr_kuhf(self):
        mf = scf.KUHF(cell, cell.make_kpts([2,1,1]))
        mf.rsjk = PBCJKMatrixOpt(cell)
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.293741115059207, 8)

    def test_nr_krks_lda(self):
        mf = dft.KRKS(cell, kpts=cell.make_kpts([2,1,1]), xc='lda,')
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.027594965662892, 8)

    def test_nr_kuks_lda(self):
        mf = dft.KUKS(cell, kpts=cell.make_kpts([2,1,1]), xc='lda,')
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.027594965662892, 8)

    def test_nr_krks_gga(self):
        mf = dft.KRKS(cell, kpts=cell.make_kpts([2,1,1]), xc='b88,')
        mf = mf.multigrid_numint()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.202133062075863, 8)

    def test_nr_kuks_gga(self):
        mf = dft.KUKS(cell, kpts=cell.make_kpts([2,1,1]), xc='pbe0').density_fit()
        mf = mf.multigrid_numint()
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.547948373602484, 8)

    def test_nr_krks_rsh(self):
        mf = dft.KRKS(cell, kpts=cell.make_kpts([2,1,1]), xc='camb3lyp')
        ref = mf.copy().run()
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(mf.e_tot, -10.550541291043928, 8)

    def test_rks_gen_g_hop(self):
        mf = dft.KRKS(cell, kpts=cell.make_kpts([2,1,1]))
        mf.xc = 'b3lyp5'
        nao = cell.nao_nr()
        cp.random.seed(1)
        mo = cp.random.random((2,nao,nao)) + 0j
        mo_occ = cp.zeros((2,nao))
        mo_occ[:,:5] = 2
        nocc, nvir = 5, nao-5
        dm1 = cp.random.random(2*nvir*nocc)
        dm1 = dm1 + cp.random.random(dm1.shape) * .1j
        mf = mf.newton()
        mf.grids.build()
        hcore = mf.get_hcore()
        g, hop, hdiag = mf.gen_g_hop(mo, mo_occ, hcore)
        dat = hop(dm1)
        self.assertAlmostEqual(lib.fp(dat.get()), -2.580202335681725-0.28028319954452735j, 9)

        mf_ref = cell.KRKS(kpts=cell.make_kpts([2,1,1]))
        mf_ref.xc = 'b3lyp5'
        mf_ref = mf_ref.newton()
        mf_ref.grids.build()
        g_ref, hop_ref, hdiag_ref = mf_ref.gen_g_hop(mo.get(), mo_occ.get(), hcore.get())
        self.assertAlmostEqual(abs(dat.get() - hop_ref(dm1.get())).max(), 0, 9)
        self.assertAlmostEqual(abs(g.get() - g_ref).max(), 0, 9)
        self.assertAlmostEqual(abs(hdiag.get() - hdiag_ref).max(), 0, 9)

        # test un-aligned occupancy
        mo_occ[1,4] = 0
        dm1 = cp.append(dm1, 0.5)
        g, hop, hdiag = mf.gen_g_hop(mo, mo_occ, hcore)
        dat = hop(dm1)
        self.assertAlmostEqual(lib.fp(dat.get()), 7.474044381526779+0.0640933714545938j, 9)

        g_ref, hop_ref, hdiag_ref = mf_ref.gen_g_hop(mo.get(), mo_occ.get(), hcore.get())
        self.assertAlmostEqual(abs(dat.get() - hop_ref(dm1.get())).max(), 0, 9)
        self.assertAlmostEqual(abs(g.get() - g_ref).max(), 0, 9)
        self.assertAlmostEqual(abs(hdiag.get() - hdiag_ref).max(), 0, 9)

    def test_uks_gen_g_hop(self):
        mf = dft.KUKS(cell, kpts=cell.make_kpts([2,1,1]))
        mf.xc = 'b3lyp5'
        nao = cell.nao_nr()
        cp.random.seed(1)
        mo = cp.random.random((2,2,nao,nao)) + 0j
        mo_occ = cp.zeros((2,2,nao))
        mo_occ[:,:,:5] = 1
        nocc, nvir = 5, nao-5
        dm1 = cp.random.random(4*nvir*nocc)
        dm1 = dm1 + cp.random.random(dm1.shape) * .1j
        mf = mf.newton()
        mf.grids.build()
        hcore = cp.array([mf.get_hcore()]*2)
        g, hop, hdiag = mf.gen_g_hop(mo, mo_occ, hcore)
        dat = hop(dm1)
        self.assertAlmostEqual(lib.fp(dat.get()), -10.115070233160678-0.818925838878515j, 9)

        mf_ref = cell.KUKS(kpts=cell.make_kpts([2,1,1]))
        mf_ref.xc = 'b3lyp5'
        mf_ref = mf_ref.newton()
        mf_ref.grids.build()
        g_ref, hop_ref, hdiag_ref = mf_ref.gen_g_hop(mo.get(), mo_occ.get(), hcore.get())
        self.assertAlmostEqual(abs(dat.get() - hop_ref(dm1.get())).max(), 0, 8)
        self.assertAlmostEqual(abs(g.get() - g_ref).max(), 0, 9)
        self.assertAlmostEqual(abs(hdiag.get() - hdiag_ref).max(), 0, 9)

        # test un-aligned occupancy
        mo_occ[1,1,4] = 0
        dm1 = cp.append(dm1, 0.5)
        g, hop, hdiag = mf.gen_g_hop(mo, mo_occ, hcore)
        dat = hop(dm1)
        self.assertAlmostEqual(lib.fp(dat.get()), 0.21794953957922497-0.08227815483834572j, 9)

        g_ref, hop_ref, hdiag_ref = mf_ref.gen_g_hop(mo.get(), mo_occ.get(), hcore.get())
        self.assertAlmostEqual(abs(dat.get() - hop_ref(dm1.get())).max(), 0, 8)
        self.assertAlmostEqual(abs(g.get() - g_ref).max(), 0, 9)
        self.assertAlmostEqual(abs(hdiag.get() - hdiag_ref).max(), 0, 9)

    def test_exxdiv_treatment_newton(self):
        mf_fo = scf.KRHF(cell, cell.make_kpts([2,1,1]))
        mf_fo.exxdiv = None
        mf_fo.conv_tol_grad = 1e-4
        mf_fo.kernel()
        mf_so = scf.KRHF(cell, cell.make_kpts([2,1,1]), exxdiv=None).newton()
        mf_so.exxdiv = None
        mf_so.conv_tol_grad = 1e-4
        mf_so.kernel()
        self.assertAlmostEqual(mf_so.e_tot, mf_fo.e_tot, 8)

if __name__ == "__main__":
    print("Full Tests for PBC Newton solver")
    unittest.main()
