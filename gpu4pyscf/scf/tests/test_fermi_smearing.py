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
from packaging.version import Version
import pyscf
from pyscf.scf import addons as cpu_addons
from pyscf.scf import hf as cpu_hf
from gpu4pyscf.scf import hf, smearing


def setUpModule():
    global mol
    atom = """
    O       0.0000000000    -0.0000000000     0.1174000000
    H      -0.7570000000    -0.0000000000    -0.4696000000
    H       0.7570000000     0.0000000000    -0.4696000000
    """

    mol = pyscf.M(
        atom=atom,  # water molecule
        basis="6-31g",  # basis set
        verbose=7,
        output="/dev/null",
    )


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):

    def test_gradient(self):
        gpu_mf = hf.RHF(mol).smearing(sigma=0.1).run()
        gpu_gradient = gpu_mf.nuc_grad_method().kernel()
        cpu_mf = cpu_addons.smearing(cpu_hf.RHF(mol), sigma=0.1).run()
        cpu_gradient = cpu_mf.nuc_grad_method().kernel()
        assert np.allclose(gpu_mf.e_tot, cpu_mf.e_tot, atol=1e-9)
        assert np.allclose(gpu_gradient, cpu_gradient, atol=1e-7)

    def test_df_uhf_gradient(self):
        gpu_mf = mol.UHF().to_gpu().density_fit().smearing(sigma=0.1).run()
        gpu_gradient = gpu_mf.nuc_grad_method().kernel()
        cpu_mf = cpu_addons.smearing(mol.UHF().density_fit(), sigma=0.1).run()
        cpu_gradient = cpu_mf.nuc_grad_method().kernel()
        assert np.allclose(gpu_mf.e_tot, cpu_mf.e_tot, atol=1e-9)
        assert np.allclose(gpu_gradient, cpu_gradient, atol=1e-7)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.12'),
                     'Require new interface developed in pyscf-2.12')
    def test_to_gpu(self):
        mf = cpu_addons.smearing(mol.RHF(), sigma=0.1)
        gpu_mf = mf.to_gpu()
        assert isinstance(gpu_mf, smearing._SmearingSCF)
        assert gpu_mf.sigma == 0.1

        mf = gpu_mf.to_cpu()
        assert isinstance(mf, cpu_addons._SmearingSCF)
        assert mf.sigma == 0.1

if __name__ == "__main__":
    print("Basic Tests for GPU Fermi Smearing")
    unittest.main()
