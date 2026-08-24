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

import pyscf
from pyscf import ao2mo
from pyscf import dft as cpu_dft
from pyscf import mcscf as cpu_mcscf
from pyscf import scf as cpu_scf

from gpu4pyscf import mcscf
from gpu4pyscf.mcscf import df as gpu_mcscf_df


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.M(
            atom='N 0 0 -0.7; N 0 0 0.7', basis='sto-3g',
            verbose=0, output='/dev/null')
        cls.mf_cpu = cpu_scf.RHF(cls.mol).density_fit(auxbasis='weigend')
        cls.mf_cpu.conv_tol = 1e-12
        cls.mf_cpu.kernel()
        cls.mf_gpu = cls.mf_cpu.to_gpu()
        cls.mf_rks_cpu = cpu_dft.RKS(cls.mol).density_fit(auxbasis='weigend')
        cls.mf_rks_cpu.xc = 'pbe'
        cls.mf_rks_cpu.conv_tol = 1e-12
        cls.mf_rks_cpu.kernel()
        cls.mf_rks_gpu = cls.mf_rks_cpu.to_gpu()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_df_casci(self):
        mc_cpu = cpu_mcscf.DFCASCI(self.mf_cpu, 4, 4, auxbasis='weigend')
        mc_cpu.canonicalization = False
        mc_gpu = mcscf.DFCASCI(self.mf_gpu, 4, 4)

        self.assertIsInstance(mc_gpu, gpu_mcscf_df.DFCASCI)
        h1_cpu, ecore_cpu = mc_cpu.get_h1eff()
        h1_gpu, ecore_gpu = mc_gpu.get_h1eff()
        self.assertLess(abs(h1_gpu - h1_cpu).max(), 1e-8)
        self.assertLess(abs(ecore_gpu - ecore_cpu), 1e-8)

        eri_cpu = ao2mo.restore(1, mc_cpu.get_h2eff(), mc_cpu.ncas)
        eri_gpu = ao2mo.restore(1, mc_gpu.get_h2eff(), mc_gpu.ncas)
        self.assertLess(abs(eri_gpu - eri_cpu).max(), 1e-8)

        e_cpu = mc_cpu.kernel()[0]
        e_gpu = mc_gpu.kernel()[0]
        self.assertLess(abs(e_gpu - e_cpu), 1e-8)
        self.assertGreater(mc_gpu.timing['ao2mo_wall'], 0)
        self.assertGreater(mc_gpu.timing['h1e_wall'], 0)
        self.assertGreater(mc_gpu.timing['fci']['contract_2e_calls'], 0)

    def test_rks_reference(self):
        mc_cpu = cpu_mcscf.DFCASCI(
            self.mf_rks_cpu, 4, 4, auxbasis='weigend')
        mc_cpu.canonicalization = False
        mc_gpu = mcscf.DFCASCI(self.mf_rks_gpu, 4, 4)

        self.assertIsInstance(mc_gpu, gpu_mcscf_df.DFCASCI)
        self.assertIs(mc_gpu._scf, self.mf_rks_gpu)
        self.assertIs(mc_gpu.with_df, self.mf_rks_gpu.with_df)
        self.assertLess(abs(mc_gpu.kernel()[0] - mc_cpu.kernel()[0]), 1e-8)

    def test_conversions(self):
        ref = mcscf.DFCASCI(self.mf_gpu, 4, 4).kernel()[0]

        mc_gpu = mcscf.DFCASCI(self.mf_cpu, 4, 4, auxbasis='weigend')
        self.assertLess(abs(mc_gpu.kernel()[0] - ref), 1e-9)

        mc_cpu = cpu_mcscf.DFCASCI(
            self.mf_cpu, 4, 4, auxbasis='weigend')
        mc_cpu.canonicalization = False
        mc_from_cpu = mc_cpu.to_gpu()
        self.assertIsInstance(mc_from_cpu, gpu_mcscf_df.DFCASCI)
        self.assertLess(abs(mc_from_cpu.kernel()[0] - ref), 1e-8)

        mc_to_cpu = mc_gpu.to_cpu()
        self.assertIsInstance(mc_to_cpu, cpu_mcscf.casci.CASCI)
        self.assertIsNotNone(mc_to_cpu.with_df)
        mc_to_cpu.canonicalization = False
        self.assertLess(abs(mc_to_cpu.kernel()[0] - ref), 1e-8)


if __name__ == '__main__':
    print('Full tests for GPU DFCASCI')
    unittest.main()
