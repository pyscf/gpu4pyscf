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
from pyscf import gto
from pyscf import lib
from gpu4pyscf.dft import gks
from pyscf.dft import gks as gks_cpu
try:
    import mcfun
except ImportError:
    mcfun = None


def setUpModule():
    global mol, mol1
    mol = gto.Mole()
    mol.atom = '''
        O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587'''
    mol.spin = None
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build()

    mol1 = gto.M(
        atom = '''
            O    0    0    0
            H    0.   -0.757   0.587
            H    0.   0.757    0.587''',
        charge = 1,
        spin = 1,
        basis = 'sto3g',
        output = '/dev/null'
        )


def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1


class KnownValues(unittest.TestCase):
    def test_mcol_gks_lda(self):
        
        mf_gpu = gks.GKS(mol)
        mf_gpu.xc = 'lda,'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 6
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -74.0600297733097, 6)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -26.421986983504258, 5)

        mf_gpu = gks.GKS(mol1)
        mf_gpu.xc = 'lda,vwn'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 50
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -74.3741809222222, 6)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -27.63368769213053, 5)

    def test_mcol_gks_gga(self):

        mf_gpu = gks.GKS(mol)
        mf_gpu.xc = 'pbe'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 6
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -75.2256398121708, 6)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -26.81184613393452, 5)

        mf_gpu = gks.GKS(mol1)
        mf_gpu.xc = 'pbe'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 50
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -74.869954771937, 6) # pyscf result
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -27.92164954706164, 5) # pyscf result

    def test_mcol_gks_hyb(self):
        mf_gpu = gks.GKS(mol)
        mf_gpu.xc = 'b3lyp'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 6
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -75.312587317089, 6) # pyscf result
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -27.2469582128507, 5) # pyscf result

        mf_gpu = gks.GKS(mol1)
        mf_gpu.xc = 'b3lyp'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 50
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -74.9528036305753, 6) # pyscf result
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -28.49145406025193, 5) # pyscf result

    def test_mcol_gks_mgga(self):
        mf_gpu = gks.GKS(mol)
        mf_gpu.xc = 'm06l'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 6
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -75.3053691716776, 6) # pyscf result
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -27.03099891671804, 5) # pyscf result

        mf_gpu = gks.GKS(mol1)
        mf_gpu.xc = 'm06l'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 50
        eks4_gpu = mf_gpu.kernel()
        self.assertAlmostEqual(eks4_gpu, -74.9468853267496, 6) # pyscf result
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -28.188215296679516, 5) # pyscf result

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    def test_to_cpu(self):
        mf_gpu = gks.GKS(mol1)
        mf_gpu.xc = 'lda,vwn'
        mf_gpu.collinear = 'mcol'
        mf_gpu._numint.spin_samples = 50
        eks4_gpu = mf_gpu.kernel()

        mf_cpu = mf_gpu.to_cpu()
        eks4_cpu = mf_cpu.kernel()

        self.assertAlmostEqual(eks4_gpu, eks4_cpu, 6)
        self.assertAlmostEqual(eks4_gpu, -74.3741809222222, 6)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), lib.fp(mf_cpu.mo_energy), 5)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -27.63368769213053, 5)

    @unittest.skip("NumInt2C in PySCF has no to_gpu method.")
    def test_to_gpu(self):
        mf_cpu = gks_cpu.GKS(mol1)
        mf_cpu.xc = 'lda,vwn'
        mf_cpu.collinear = 'mcol'
        mf_cpu._numint.spin_samples = 50
        eks4_cpu = mf_cpu.kernel()

        mf_gpu = mf_cpu.to_gpu()
        eks4_gpu = mf_cpu.kernel()

        self.assertAlmostEqual(eks4_gpu, eks4_cpu, 6)
        self.assertAlmostEqual(eks4_gpu, -74.3741809222222, 6)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), lib.fp(mf_cpu.mo_energy), 5)
        self.assertAlmostEqual(lib.fp(mf_gpu.mo_energy.get()), -27.63368769213053, 5)


if __name__ == "__main__":
    print("Test GKS")
    unittest.main()
