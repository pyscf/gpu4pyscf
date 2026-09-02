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
from pyscf import df as cpu_df
from pyscf import scf

from gpu4pyscf import df


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.M(
            atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='cc-pvdz', verbose=0, output='/dev/null')
        cls.mo = scf.RHF(cls.mol).run(conv_tol=1e-12).mo_coeff
        cls.cpu_df = cpu_df.DF(cls.mol, auxbasis='weigend').build()
        cls.gpu_df = df.DF(cls.mol, auxbasis='weigend').build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_compact(self):
        ref = self.cpu_df.ao2mo(self.mo)
        out = self.gpu_df.ao2mo(self.mo).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertLess(abs(out - ref).max(), 1e-8)

    def test_general(self):
        coeffs = (self.mo[:, :3], self.mo[:, 1:5],
                  self.mo[:, 2:6], self.mo[:, :2])
        ref = self.cpu_df.ao2mo(coeffs, compact=False)
        out = self.gpu_df.ao2mo(coeffs, compact=False).get()
        self.assertEqual(out.shape, ref.shape)
        self.assertLess(abs(out - ref).max(), 1e-8)

    def test_host_cderi(self):
        with_df = df.DF(self.mol, auxbasis='weigend')
        with_df.use_gpu_memory = False
        with_df.build()
        ref = self.cpu_df.ao2mo(self.mo[:, 2:9])
        out = with_df.ao2mo(self.mo[:, 2:9]).get()
        self.assertLess(abs(out - ref).max(), 1e-8)


if __name__ == '__main__':
    print('Full tests for GPU DF AO2MO')
    unittest.main()
