# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
import numpy as np
import cupy as cp
from packaging.version import Version
import pyscf
from pyscf.pbc.scf import addons as cpu_addons
from gpu4pyscf.pbc.scf import smearing


def setUpModule():
    global cell
    cell = pyscf.M(
        atom = 'He 0 0 1; He 1 0 1',
        basis = [[0, [1., 1.]], [0, [0.5, 1]]],
        a = np.eye(3) * 3,
        verbose = 7,
        output = '/dev/null')

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_krhf_smearing(self):
        nao = cell.nao
        mf = cell.KRHF(kpts=cell.make_kpts([2,1,1])).to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        nkpts = len(mf.kpts)
        mo_energy_kpts = cp.array([cp.arange(nao)*.2+cp.cos(i+.5)*.1 for i in range(nkpts)])
        mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1656394960533021/2, 9)

        mf.smearing_method = 'gauss'
        mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 0.94924016074521311/2, 9)

        mf.kernel(dm0=np.array([np.eye(nao)]*nkpts))
        self.assertAlmostEqual(mf.entropy, 0, 15)

    def test_kuhf_smearing(self):
        nao = cell.nao
        mf = cell.KUHF(kpts=cell.make_kpts([2,1,1])).to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        nkpts = len(mf.kpts)
        mo_energy_kpts = cp.array([cp.arange(nao)*.2+cp.cos(i+.5)*.1 for i in range(nkpts)])
        mo_energy_kpts = cp.array([mo_energy_kpts, mo_energy_kpts+cp.cos(mo_energy_kpts)*.02])
        mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1803390081500869/2, 9)

        mf.smearing_method = 'gauss'
        mf.mu0 = 0.3
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 0.5066105772152231, 9)

        mf = mf.to_cpu()
        ref = np.array(mf.get_occ(mo_energy_kpts.get()))
        self.assertAlmostEqual(abs(occ.get() - ref).max(), 0, 9)

    def test_rhf_smearing(self):
        nao = cell.nao
        mf = cell.RHF().to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        mo_energy = cp.arange(nao)*.2+cp.cos(.5)*.1
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.0922723199786408, 9)

        mf.smearing_method = 'gauss'
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 0.4152467504725415, 9)

        mf.kernel()
        self.assertAlmostEqual(mf.entropy, 0, 15)

    def test_uhf_smearing(self):
        nao = cell.nao
        mf = cell.UHF().to_gpu()
        mf = mf.smearing(0.1, 'fermi')
        mo_energy = cp.arange(nao)*.2+cp.cos(.5)*.1
        mo_energy = cp.array([mo_energy, mo_energy+cp.cos(mo_energy)*.02])
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.1007387905421022, 9)

        mf.smearing_method = 'gauss'
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 0.42189309944541731, 9)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.12'),
                     'Require new interface developed in pyscf-2.12')
    def test_to_gpu(self):
        mf = cpu_addons.smearing(cell.RHF(), sigma=0.1)
        gpu_mf = mf.to_gpu()
        assert isinstance(gpu_mf, smearing._SmearingKSCF)
        assert gpu_mf.sigma == 0.1

        mf = gpu_mf.to_cpu()
        assert isinstance(mf, cpu_addons._SmearingKSCF)
        assert mf.sigma == 0.1

if __name__ == "__main__":
    print("Basic Tests for GPU PBC-SCF Smearing")
    unittest.main()
