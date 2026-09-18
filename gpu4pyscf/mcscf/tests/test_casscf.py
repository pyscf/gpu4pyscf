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

import cupy
import pyscf
from pyscf import dft, mcscf as cpu_mcscf, scf

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
        self.assertGreater(mc.timing['macro_cycles'], 1)
        self.assertLessEqual(mc.timing['macro_cycles'], mc.max_cycle_macro)
        self.assertGreater(mc.timing['ao2mo_wall'], 0.)
        self.assertGreater(mc.timing['fci_wall'], 0.)
        self.assertGreater(mc.timing['orbital_derivatives_wall'], 0.)
        self.assertGreater(mc.timing['total_wall'], 0.)

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


if __name__ == '__main__':
    unittest.main()
