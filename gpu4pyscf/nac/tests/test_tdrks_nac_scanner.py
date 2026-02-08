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
import numpy as np
import cupy as cp
import scipy.linalg
import pyscf
from pyscf import lib, gto, scf, dft
from gpu4pyscf import tdscf, nac
from gpu4pyscf.nac import tdrhf as tdrhf_nac
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""
atom1 = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

bas0 = "cc-pvdz"

def setUpModule():
    global mol, mol1
    mol = pyscf.M(
        atom=atom, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)
    mol1 = pyscf.M(
        atom=atom1, basis=bas0, max_memory=32000, output="/dev/null", verbose=1)


def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1


class KnownValues(unittest.TestCase):
    @unittest.skipIf(num_devices > 1, '')
    def test_nac_scanner_ge(self):
        mf = dft.RKS(mol, xc="b3lyp").to_gpu().density_fit()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(0,1)
        nac_benchmark = nac1.kernel()
        nac_benchmark_de = nac_benchmark[0]

        nac_scanner = nac1.as_scanner()
        new_nac = nac_scanner(mol1)
        assert (new_nac[1]*nac_benchmark_de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac_benchmark_de) > 0.99
        new_nac = nac_scanner(mol)
        assert (new_nac[1]*nac_benchmark_de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac_benchmark_de) > 0.99
        new_nac = nac_scanner(mol1)
        assert (new_nac[1]*nac_benchmark_de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac_benchmark_de) > 0.99

    @unittest.skipIf(num_devices > 1, '')
    def test_nac_scanner_ee(self):
        mf = dft.RKS(mol, xc="b3lyp").to_gpu().density_fit()
        mf.kernel()
        td = mf.TDA().set(nstates=5)
        td.kernel()
        nac1 = td.nac_method()
        nac1.states=(1,2)
        nac_benchmark = nac1.kernel()
        nac_benchmark_de = nac_benchmark[0]

        nac_scanner = nac1.as_scanner()
        new_nac = nac_scanner(mol1)
        assert (new_nac[1]*nac_benchmark_de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac_benchmark_de) > 0.99
        new_nac = nac_scanner(mol)
        assert (new_nac[1]*nac_benchmark_de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac_benchmark_de) > 0.99
        new_nac = nac_scanner(mol1)
        assert (new_nac[1]*nac_benchmark_de).sum()/np.linalg.norm(new_nac[1])/np.linalg.norm(nac_benchmark_de) > 0.99

    def test_sign(self):
        mol = pyscf.M(atom='H 0 1 0; H 0 0 1; H .2 .5, .8; H 1. 0. .2', basis=[[0, [1,1]]])
        s = mol.intor('int1e_ovlp')
        h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        mo1 = scipy.linalg.eigh(h, s)[1]
        mo2 = mo1[:,[1,0,3,2]]
        mo2[:,2] *= -1
        nocc = 2
        nvir = mo1.shape[1] - nocc
        x0 = np.zeros((nocc, nvir))
        x0[0,1] = .5**.5
        x1 = np.zeros((nocc, nvir))
        x1[1,0] = -.5**.5
        assert tdrhf_nac._wfn_overlap(mo1, mo2, x0, x1, s)


if __name__ == "__main__":
    print("Full Tests for TD-RKS nonadiabatic coupling vectors scanner.")
    unittest.main()
