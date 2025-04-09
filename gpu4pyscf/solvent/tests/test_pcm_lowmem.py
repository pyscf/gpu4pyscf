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
import numpy
import pyscf
import pytest
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.dft import rks_lowmem
from gpu4pyscf.scf import hf_lowmem
from gpu4pyscf.solvent import pcm

def setUpModule():
    global mol, epsilon, lebedev_order, e_tolerance
    mol = gto.Mole()
    mol.atom = '''
O  0.0000   0.7375  -0.0528
O  0.0000  -0.7375  -0.1528
H  0.8190   0.8170   0.4220
H  -0.8190 -0.8170   0.4220
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 35.9
    lebedev_order = 17
    e_tolerance = 1e-8

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _energy_with_solvent(mf, method):
    cm = pcm.PCM(mol)
    cm.eps = epsilon
    cm.verbose = 0
    cm.lebedev_order = lebedev_order
    cm.method = method
    mf = mf.PCM(cm)
    e_tot = mf.kernel()
    return e_tot

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_lowmem_RHF_CPCM(self):
        # e_reference = _energy_with_solvent(scf.RHF(mol), 'COSMO')
        e_reference = -148.75667517564528
        e_test = _energy_with_solvent(hf_lowmem.RHF(mol), 'COSMO')
        print(f"Energy error in lowmem RHF with COSMO: {numpy.abs(e_test - e_reference)}")
        assert numpy.abs(e_test - e_reference) < e_tolerance

    def test_lowmem_RHF_IEFPCM(self):
        # e_reference = _energy_with_solvent(scf.RHF(mol), 'IEF-PCM')
        e_reference = -148.75667903183302
        e_test = _energy_with_solvent(hf_lowmem.RHF(mol), 'IEF-PCM')
        print(f"Energy error in lowmem RHF with IEF-PCM: {numpy.abs(e_test - e_reference)}")
        assert numpy.abs(e_test - e_reference) < e_tolerance

    def test_lowmem_RHF_SSVPE(self):
        # e_reference = _energy_with_solvent(scf.RHF(mol), 'SS(V)PE')
        e_reference = -148.7566491351642
        e_test = _energy_with_solvent(hf_lowmem.RHF(mol), 'SS(V)PE')
        print(f"Energy error in lowmem RHF with SS(V)PE: {numpy.abs(e_test - e_reference)}")
        assert numpy.abs(e_test - e_reference) < e_tolerance

    def test_lowmem_RKS_CPCM(self):
        # e_reference = _energy_with_solvent(dft.RKS(mol, xc='b3lyp'), 'C-PCM')
        e_reference = -149.44440516844412
        e_test = _energy_with_solvent(rks_lowmem.RKS(mol, xc='b3lyp'), 'C-PCM')
        print(f"Energy error in lowmem RKS with C-PCM: {numpy.abs(e_test - e_reference)}")
        assert numpy.abs(e_test - e_reference) < e_tolerance

    def test_lowmem_RKS_IEFPCM(self):
        # e_reference = _energy_with_solvent(dft.RKS(mol, xc='wb97m-v'), 'IEF-PCM')
        e_reference = -149.4518330789009
        e_test = _energy_with_solvent(rks_lowmem.RKS(mol, xc='wb97m-v'), 'IEF-PCM')
        print(f"Energy error in lowmem RKS with IEF-PCM: {numpy.abs(e_test - e_reference)}")
        assert numpy.abs(e_test - e_reference) < e_tolerance

    def test_lowmem_RKS_SSVPE(self):
        # e_reference = _energy_with_solvent(dft.RKS(mol, xc='pbe'), 'SS(V)PE')
        e_reference = -149.2918698409403
        e_test = _energy_with_solvent(rks_lowmem.RKS(mol, xc='pbe'), 'SS(V)PE')
        print(f"Energy error in lowmem RKS with SS(V)PE: {numpy.abs(e_test - e_reference)}")
        assert numpy.abs(e_test - e_reference) < e_tolerance

    # TODO: Missing functionalities and tests for unrestricted HF and DFT

if __name__ == "__main__":
    print("Tests for PCM with lowmem SCF and DFT modules")
    unittest.main()
