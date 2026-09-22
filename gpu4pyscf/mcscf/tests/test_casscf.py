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
import tempfile
from unittest import mock

import cupy
import pyscf
from pyscf import dft, lib, mcscf as cpu_mcscf, scf
from pyscf.fci import direct_spin1

from gpu4pyscf import mcscf
from gpu4pyscf.fci.direct_spin1 import FCISolver
from gpu4pyscf.mcscf import df as gpu_mcscf_df


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.M(
            atom='N 0 0 -0.7; N 0 0 0.7', basis='sto-3g',
            verbose=0, output='/dev/null')
        cls.mf_cpu = scf.RHF(cls.mol).density_fit(auxbasis='weigend')
        cls.mf_cpu.kernel()
        cls.mf_gpu = cls.mf_cpu.to_gpu()
        cls.mf_rks_cpu = dft.RKS(cls.mol).density_fit(auxbasis='weigend')
        cls.mf_rks_cpu.xc = 'pbe'
        cls.mf_rks_cpu.kernel()
        cls.mf_rks_gpu = cls.mf_rks_cpu.to_gpu()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_df_casscf(self):
        mc = mcscf.DFCASSCF(self.mf_gpu, 4, 4)
        mc.max_cycle_macro = 20
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-5
        e_tot = mc.kernel()[0]

        self.assertIsInstance(mc, gpu_mcscf_df.DFCASSCF)
        self.assertIsInstance(mc.fcisolver, FCISolver)
        self.assertIsInstance(mc.ci, cupy.ndarray)
        self.assertIsInstance(mc.mo_coeff, cupy.ndarray)
        self.assertTrue(mc.converged)
        self.assertLess(abs(e_tot - -107.5445420582518), 2e-8)

    def test_rks_reference(self):
        ref = cpu_mcscf.DFCASCI(
            self.mf_rks_cpu, 4, 4, auxbasis='weigend')
        ref.canonicalization = False
        mc = mcscf.DFCASSCF(self.mf_rks_gpu, 4, 4)
        mc.max_cycle_macro = 1
        e_tot = mc.kernel()[0]

        self.assertIsInstance(mc, gpu_mcscf_df.DFCASSCF)
        self.assertIs(mc._scf, self.mf_rks_gpu)
        self.assertIs(mc.with_df, self.mf_rks_gpu.with_df)
        self.assertLess(abs(e_tot - ref.kernel()[0]), 1e-8)

    def test_unconverged_fci(self):
        mol = pyscf.M(
            atom='H 0 0 0; H 0 0 1; H 0 1.2 0; H 0 1.2 1.3',
            basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis='weigend').run()
        mc = mcscf.DFCASSCF(mf.to_gpu(), 4, 4)
        mc.chkfile = None
        mc.max_cycle_macro = 3
        mc.fcisolver.max_cycle = 1
        mc.kernel()
        self.assertFalse(mc.fcisolver.converged)
        self.assertFalse(mc.converged)

        mc.fcisolver.max_cycle = 50
        e_tot = mc.kernel()[0]
        ref = cpu_mcscf.DFCASCI(mf, 4, 4)
        ref.canonicalization = False
        self.assertTrue(mc.fcisolver.converged)
        self.assertTrue(mc.converged)
        self.assertLess(abs(e_tot - ref.kernel()[0]), 1e-8)

    def test_checkpoint_consistency(self):
        mc = mcscf.DFCASSCF(self.mf_gpu, 4, 4)
        mc.max_cycle_macro = 2
        mc.chk_ci = True
        ref = cpu_mcscf.DFCASCI(self.mf_cpu, 4, 4, auxbasis='weigend')
        dump_chk = mc.dump_chk

        def check_checkpoint(env):
            dump_chk(env)
            data = lib.chkfile.load(mc.chkfile, 'mcscf')
            h1e, ecore = ref.get_h1eff(data['mo_coeff'])
            eri = ref.get_h2eff(data['mo_coeff'])
            energy = direct_spin1.energy(h1e, eri, data['ci'], 4, (2, 2)) + ecore
            self.assertLess(abs(energy - data['e_tot']), 1e-8)
            dm1 = direct_spin1.make_rdm1(data['ci'], 4, (2, 2))
            self.assertLess(abs(dm1 - data['casdm1']).max(), 1e-10)

        with tempfile.TemporaryDirectory() as tmpdir:
            mc.chkfile = tmpdir + '/casscf.chk'
            with mock.patch.object(mc, 'dump_chk',
                                   side_effect=check_checkpoint) as save:
                mc.kernel()
                self.assertGreaterEqual(save.call_count, 2)


if __name__ == '__main__':
    unittest.main()
