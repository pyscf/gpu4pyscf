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
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf import dft

def setUpModule():
    global h2o_z0, h2o_z1
    h2o_z0 = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
        basis = '6-31g')

    h2o_z1 = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
        basis = '6-31g',
        charge = 1,
        spin = 1,)

def tearDownModule():
    global h2o_z0, h2o_z1
    h2o_z0.stdout.close()
    h2o_z1.stdout.close()
    del h2o_z0, h2o_z1

class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = scf.RHF(h2o_z0)
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 2
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.98394849812, 9)

    def test_nr_rohf(self):
        mf = scf.ROHF(h2o_z1)
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 20
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.5783963795897, 9)

    def test_nr_uhf(self):
        mf = scf.UHF(h2o_z1)
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 2
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), -75.58051984397145, 9)

    def test_nr_rks_lda(self):
        mf = dft.RKS(h2o_z0)
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_rks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.RKS(h2o_z0)
        mf.xc = 'wb97x'
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_rks(self):
        mf = dft.RKS(h2o_z0)
        mf.xc = 'b3lyp'
        eref = mf.kernel()
        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_rks_gen_g_hop(self):
        mf = dft.RKS(h2o_z0)
        mf.grids.build()
        mf.xc = 'b3lyp'
        nao = h2o_z0.nao_nr()
        mo = cp.random.random((nao,nao))
        mo_occ = cp.zeros(nao)
        mo_occ[:5] = 2
        nocc, nvir = 5, nao-5
        dm1 = cp.random.random(nvir*nocc)
        nr = mf.newton()
        g, hop, hdiag = nr.gen_g_hop(mo, mo_occ)
        mf_cpu = mf.to_cpu().newton()
        hop_ref = mf_cpu.gen_g_hop(mo.get(), mo_occ.get())[1]
        self.assertAlmostEqual(abs(hop(dm1).get() - hop_ref(dm1.get())).max(), 0, 9)

    def test_nr_roks(self):
        mf = dft.ROKS(h2o_z1)
        mf.xc = 'b3lyp'
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_uks_lda(self):
        mf = dft.UKS(h2o_z1)
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 2
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_uks_rsh(self):
        '''test range-separated Coulomb'''
        mf = dft.UKS(h2o_z1)
        mf.xc = 'wb97x'
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_nr_uks(self):
        mf = dft.UKS(h2o_z1)
        mf.xc = 'b3lyp'
        eref = mf.kernel()

        mf.max_cycle = 1
        mf.conv_check = False
        mf.kernel()
        nr = mf.newton()
        nr.max_cycle = 3
        nr.conv_tol_grad = 1e-5
        self.assertAlmostEqual(nr.kernel(), eref, 9)

    def test_uks_gen_g_hop(self):
        mf = dft.UKS(h2o_z0)
        mf.grids.build()
        mf.xc = 'hse06'
        nao = h2o_z0.nao_nr()
        mo = cp.random.random((2, nao,nao))
        mo_occ = cp.zeros((2,nao))
        mo_occ[:,:5] = 1
        nocc, nvir = 5, nao-5
        dm1 = cp.random.random(nvir*nocc*2)
        nr = mf.newton()
        g, hop, hdiag = nr.gen_g_hop(mo, mo_occ)
        mf_cpu = mf.to_cpu().newton()
        hop_ref = mf_cpu.gen_g_hop(mo.get(), mo_occ.get())[1]
        self.assertAlmostEqual(abs(hop(dm1).get() - hop_ref(dm1.get())).max(), 0, 9)

    def test_with_df(self):
        mf = scf.RHF(h2o_z0).density_fit().newton().run()
        self.assertTrue(mf._eri is None)
        self.assertAlmostEqual(mf.e_tot, -75.983944727996, 9)
        self.assertEqual(mf.__class__.__name__, 'SecondOrderDFRHF')

        mf = scf.RHF(h2o_z0).newton().density_fit().run()
        self.assertTrue(mf._eri is None)
        self.assertAlmostEqual(mf.e_tot, -75.9839484980661, 9)
        mf = mf.undo_newton()
        self.assertEqual(mf.__class__.__name__, 'RHF')

    def test_secondary_auxbasis(self):
        mf_ref = scf.UHF(h2o_z0).run()
        mf = scf.UHF(h2o_z0).newton().density_fit(auxbasis=[[0, [1., 1.]]]).run()
        self.assertAlmostEqual(mf_ref.e_tot, mf.e_tot, 8)

        mf_ref = scf.UHF(h2o_z0).density_fit().run()
        mf = scf.UHF(h2o_z0).density_fit().newton().density_fit(auxbasis=[[0, [1., 1.]]]).run()
        self.assertAlmostEqual(mf_ref.e_tot, mf.e_tot, 8)

if __name__ == "__main__":
    print("Full Tests for Newton solver")
    unittest.main()
