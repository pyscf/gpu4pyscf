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
import scipy.linalg
import numpy as np
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
        mf = dft.UKS(h2o_z1)
        mf.grids.build()
        mf.xc = 'hse06'
        nao = h2o_z1.nao_nr()
        cp.random.seed(2)
        mo = cp.random.random((2, nao,nao)) - 0.4
        mo_occ = cp.zeros((2,nao))
        mo_occ[0,:5] = 1
        mo_occ[1,:4] = 1
        nr = mf.newton()
        g, hop, hdiag = nr.gen_g_hop(mo, mo_occ)
        dm1 = cp.random.random(g.size)
        mf_cpu = mf.to_cpu().newton()
        hop_ref = mf_cpu.gen_g_hop(mo.get(), mo_occ.get())[1]
        dat = hop(dm1).get()
        ref = hop_ref(dm1.get())
        self.assertAlmostEqual(abs(dat - ref).max(), 0, 9)

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



class SmallNewtonContracts(unittest.TestCase):
    def _check_accumulated_complex_rotations(self, unrestricted):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
        reference = mol.RHF().run(conv_tol=1e-12)
        base = scf.UHF(mol) if unrestricted else scf.RHF(mol)
        obj = base.newton()
        coeff = cp.asarray(reference.mo_coeff)
        occ = cp.asarray(reference.mo_occ)
        if unrestricted:
            coeff = cp.stack([coeff, coeff])
            occ = cp.stack([occ*.5, occ*.5])
        occupation = occ[0] if unrestricted else occ
        mask = (occupation[:, None] == 0) & (occupation[None, :] > 0)
        nvar = int(mask.sum())
        dx = cp.linspace(.002, .009, nvar) * (1+.3j)
        dx = cp.concatenate([dx, -.7*dx]) if unrestricted else dx
        first = obj.update_rotate_matrix(dx, occ)
        actual = obj.update_rotate_matrix(-.4*dx, occ, u0=first)
        for spin in range(2 if unrestricted else 1):
            step = dx[spin*nvar:(spin+1)*nvar] if unrestricted else dx
            generator = np.zeros((mol.nao, mol.nao), dtype=complex)
            generator[mask.get()] = step.get()
            generator -= generator.conj().T
            expected = scipy.linalg.expm(generator) @ scipy.linalg.expm(-.4*generator)
            rotation = actual[spin].get() if unrestricted else actual.get()
            np.testing.assert_allclose(rotation, expected, atol=1e-13)
            np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(mol.nao), atol=1e-13)
        moved = obj.rotate_mo(coeff, actual)
        s = mol.intor('int1e_ovlp')
        for c in (moved if unrestricted else [moved]):
            np.testing.assert_allclose(c.get().conj().T @ s @ c.get(), np.eye(mol.nao), atol=1e-12)

    def test_accumulated_complex_rotations_restricted(self):
        self._check_accumulated_complex_rotations(False)

    def test_accumulated_complex_rotations_unrestricted(self):
        self._check_accumulated_complex_rotations(True)

    def test_uhf_newton_restricted_initial_orbitals(self):
        mol = gto.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
        initial = mol.ROHF().run(conv_tol=1e-12)
        expected = mol.UHF().run(conv_tol=1e-12)
        obj = scf.UHF(mol).newton()
        obj.conv_tol = 1e-11
        obj.max_cycle = 20
        value = obj.kernel(cp.asarray(initial.mo_coeff), cp.asarray(initial.mo_occ))
        assert obj.converged
        np.testing.assert_allclose(value, expected.e_tot, atol=1e-10)
        np.testing.assert_array_equal(obj.mo_occ.sum(axis=1).get(), mol.nelec)
        cpu = obj.to_cpu()
        np.testing.assert_allclose(cpu.make_rdm1(), obj.make_rdm1().get(), atol=1e-12)
        np.testing.assert_allclose(cpu.energy_tot(), obj.e_tot, atol=1e-10)


if __name__ == "__main__":
    print("Full Tests for Newton solver")
    unittest.main()
