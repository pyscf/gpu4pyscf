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
    global mol, epsilon, lebedev_order, g_tolerance
    mol = gto.Mole()
    mol.atom = '''
O  0.0000   0.7375  -0.0528
O  0.0000  -0.7375  -0.1528
H  0.8190   0.8170   0.4220
H  -0.8190 -0.8170   0.4220
    '''
    mol.basis = '6-31g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 60.0
    lebedev_order = 11
    g_tolerance = 1e-7

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _gradient_with_solvent(mf, method, lowmem_pcm = True):
    cm = pcm.PCM(mol)
    cm.eps = epsilon
    cm.verbose = 0
    cm.lebedev_order = lebedev_order
    cm.method = method
    cm.lowmem_intermediate_storage = lowmem_pcm
    mf = mf.PCM(cm)
    mf.conv_tol = 1e-12
    mf.kernel()

    g_tot = mf.Gradients().kernel()
    return g_tot

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    @pytest.mark.slow
    def test_lowmem_gradient_RHF_CPCM(self):
        # g_reference = _gradient_with_solvent(scf.RHF(mol), 'COSMO', False)
        g_reference = numpy.array([
            [-0.0163175067980038,  0.0071812687357845,  0.007727516476731 ],
            [ 0.0559012485168635, -0.011353691832632 , -0.0294711367871   ],
            [-0.0071729382647378, -0.0177750521871014, -0.0040458964935528],
            [-0.0324108034542936,  0.0219474752840269,  0.025789516802629 ],
        ])
        g_test = _gradient_with_solvent(hf_lowmem.RHF(mol), 'COSMO')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RHF with COSMO: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RHF_IEFPCM(self):
        # g_reference = _gradient_with_solvent(scf.RHF(mol), 'IEF-PCM', False)
        g_reference = numpy.array([
            [-0.0163091640644741,  0.0071854047204253,  0.0077257383274488],
            [ 0.0558924741414254, -0.0113579898790315, -0.0294541518164558],
            [-0.0071898644361561, -0.0177786593955563, -0.004052144958452 ],
            [-0.0323934456409908,  0.0219512445542212,  0.0257805584461862],
        ])
        g_test = _gradient_with_solvent(hf_lowmem.RHF(mol), 'IEF-PCM')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RHF with IEF-PCM: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RHF_SSVPE(self):
        # g_reference = _gradient_with_solvent(scf.RHF(mol), 'SS(V)PE', False)
        g_reference = numpy.array([
            [-0.0162824856323564,  0.0071927008916414,  0.0078360528313351],
            [ 0.0558764125359919, -0.0113435908548165, -0.0295307062332608],
            [-0.0072341291910355, -0.0178094153295815, -0.0041153768893911],
            [-0.0323597977127988,  0.0219603052928713,  0.0258100302900295],
        ])
        g_test = _gradient_with_solvent(hf_lowmem.RHF(mol), 'SS(V)PE')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RHF with SS(V)PE: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RKS_CPCM(self):
        # g_reference = _gradient_with_solvent(dft.RKS(mol, xc='b3lyp'), 'C-PCM', False)
        g_reference = numpy.array([
            [ 0.0107095697534382, -0.0239253590983287,  0.0181754716966574],
            [ 0.0304517151611092,  0.0203233799157037, -0.0119343636494438],
            [-0.0295443021479662, -0.0159663515299563, -0.0169380713186013],
            [-0.0116148919620014,  0.019574656096594 ,  0.0107002954608816],
        ])
        g_test = _gradient_with_solvent(rks_lowmem.RKS(mol, xc='b3lyp'), 'C-PCM')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RKS with C-PCM: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RKS_IEFPCM(self):
        # g_reference = _gradient_with_solvent(dft.RKS(mol, xc='wb97m-v'), 'IEF-PCM', False)
        g_reference = numpy.array([
            [ 0.0092344759118757, -0.0157357350651268,  0.0179434519747163],
            [ 0.0319343630951492,  0.0122162239055141, -0.0133994229373763],
            [-0.028147669061958 , -0.0159401762636575, -0.0161802707141887],
            [-0.0130159955568279,  0.0194631436530942,  0.0116404316908084],
        ])
        g_test = _gradient_with_solvent(rks_lowmem.RKS(mol, xc='wb97m-v'), 'IEF-PCM')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RKS with IEF-PCM: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RKS_SSVPE(self):
        # g_reference = _gradient_with_solvent(dft.RKS(mol, xc='pbe'), 'SS(V)PE', False)
        g_reference = numpy.array([
            [ 0.0197558534156823, -0.031520015730371 ,  0.0220234927827623],
            [ 0.0221629691155982,  0.0281576122655399, -0.0063872890269219],
            [-0.0371864317077676, -0.015528121099998 , -0.0213824873378206],
            [-0.0047281642564727,  0.0188991382028949,  0.0057502451040964],
        ])
        g_test = _gradient_with_solvent(rks_lowmem.RKS(mol, xc='pbe'), 'SS(V)PE')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RKS with SS(V)PE: {diff}")
        assert diff < g_tolerance

    # TODO: Missing functionalities and tests for unrestricted HF and DFT

if __name__ == "__main__":
    print("Tests for PCM with lowmem SCF and DFT modules")
    unittest.main()
