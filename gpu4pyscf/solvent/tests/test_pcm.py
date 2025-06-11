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
import cupy
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.solvent import pcm
from packaging import version
try:
    # Some PCM methods are registered when importing the CPU version.
    # However, pyscf-2.7 does note automatically import this module.
    from pyscf.solvent import pcm as pcm_on_cpu
except ImportError:
    pass

pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')

def setUpModule():
    global mol, epsilon, lebedev_order
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    mol.nelectron = mol.nao * 2
    epsilon = 35.9
    lebedev_order = 3

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _energy_with_solvent(mf, method):
    cm = pcm.PCM(mol)
    cm.eps = epsilon
    cm.verbose = 0
    cm.lebedev_order = 29
    cm.method = method
    mf = mf.PCM(cm)
    e_tot = mf.kernel()
    return e_tot

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_D_S(self):
        cm = pcm.PCM(mol)
        cm.lebedev_order = 3
        cm.method = 'IEF-PCM'
        cm.build()

        D0, S0 = pcm.get_D_S_slow(cm.surface, with_S=True, with_D=True)
        D1, S1 = pcm.get_D_S(cm.surface, with_S=True, with_D=True)

        assert cupy.linalg.norm(D0 - D1) < 1e-8
        assert cupy.linalg.norm(S0 - S1) < 1e-8

    def test_CPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'C-PCM')
        print(f"Energy error in RHF with C-PCM: {numpy.abs(e_tot - -71.19244927767662)}")
        assert numpy.abs(e_tot - -71.19244927767662) < 1e-5

    def test_COSMO(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'COSMO')
        print(f"Energy error in RHF with COSMO: {numpy.abs(e_tot - -71.16259314943571)}")
        assert numpy.abs(e_tot - -71.16259314943571) < 1e-5

    def test_IEFPCM(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'IEF-PCM')
        print(f"Energy error in RHF with IEF-PCM: {numpy.abs(e_tot - -71.19244457024647)}")
        assert numpy.abs(e_tot - -71.19244457024647) < 1e-5

    def test_SSVPE(self):
        e_tot = _energy_with_solvent(scf.RHF(mol), 'SS(V)PE')
        print(f"Energy error in RHF with SS(V)PE: {numpy.abs(e_tot - -71.13576912425178)}")
        assert numpy.abs(e_tot - -71.13576912425178) < 1e-5

    def test_uhf(self):
        e_tot = _energy_with_solvent(scf.UHF(mol), 'IEF-PCM')
        print(f"Energy error in UHF with IEF-PCM: {numpy.abs(e_tot - -71.19244457024645)}")
        assert numpy.abs(e_tot - -71.19244457024645) < 1e-5

    def test_rks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in RKS with IEF-PCM: {numpy.abs(e_tot - -71.67007402042326)}")
        assert numpy.abs(e_tot - -71.67007402042326) < 1e-5

    def test_uks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp'), 'IEF-PCM')
        print(f"Energy error in UKS with IEF-PCM: {numpy.abs(e_tot - -71.67007402042326)}")
        assert numpy.abs(e_tot - -71.67007402042326) < 1e-5

    def test_dfrks(self):
        e_tot = _energy_with_solvent(dft.RKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFRKS with IEF-PCM: {numpy.abs(e_tot - -71.67135250643568)}")
        assert numpy.abs(e_tot - -71.67135250643568) < 1e-5

    def test_dfuks(self):
        e_tot = _energy_with_solvent(dft.UKS(mol, xc='b3lyp').density_fit(), 'IEF-PCM')
        print(f"Energy error in DFUKS with IEF-PCM: {numpy.abs(e_tot - -71.67135250643567)}")
        assert numpy.abs(e_tot - -71.67135250643567) < 1e-5
        
    def test_to_cpu(self):
        mf = dft.RKS(mol, xc='b3lyp')
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

        mf = dft.RKS(mol, xc='b3lyp').density_fit()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        import pyscf
        mf = pyscf.dft.RKS(mol, xc='b3lyp').PCM()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

        mf = pyscf.dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        e_cpu = mf.kernel()
        mf = mf.to_gpu()
        e_gpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_cpu_1(self):
        mf = dft.RKS(mol, xc='b3lyp').PCM()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

        mf = dft.RKS(mol, xc='b3lyp').density_fit().PCM()
        e_gpu = mf.kernel()
        mf = mf.to_cpu()
        e_cpu = mf.kernel()
        assert abs(e_cpu - e_gpu) < 1e-8

if __name__ == "__main__":
    print("Full Tests for PCMs")
    unittest.main()
