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

def _gradient_with_solvent(mf, method):
    cm = pcm.PCM(mol)
    cm.eps = epsilon
    cm.verbose = 0
    cm.lebedev_order = lebedev_order
    cm.method = method
    mf = mf.PCM(cm)
    mf.conv_tol = 1e-10
    mf.kernel()

    g_tot = mf.Gradients().kernel()
    return g_tot

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):
    def test_lowmem_gradient_RHF_CPCM(self):
        # g_reference = _gradient_with_solvent(scf.RHF(mol), 'COSMO')
        g_reference = numpy.array([
            [-0.0163174947894124,  0.0071813633451422,  0.0077272830289788],
            [ 0.055901325985305 , -0.0113535254296157, -0.0294709948305932],
            [-0.0071729768409255, -0.0177751943530026, -0.0040459320222851],
            [-0.0324108543551454,  0.0219473564375562,  0.0257896438220912],
        ])
        g_test = _gradient_with_solvent(hf_lowmem.RHF(mol), 'COSMO')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RHF with COSMO: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RHF_IEFPCM(self):
        # g_reference = _gradient_with_solvent(scf.RHF(mol), 'IEF-PCM')
        g_reference = numpy.array([
            [-0.0163091544133066,  0.0071854944214561,  0.0077255306119172],
            [ 0.0558925557229586, -0.0113578352019582, -0.0294540386690879],
            [-0.0071899040819504, -0.0177787858574423, -0.004052172393671 ],
            [-0.0323934972278823,  0.0219511266380314,  0.0257806804490408],
        ])
        g_test = _gradient_with_solvent(hf_lowmem.RHF(mol), 'IEF-PCM')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RHF with IEF-PCM: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RHF_SSVPE(self):
        # g_reference = _gradient_with_solvent(scf.RHF(mol), 'SS(V)PE')
        g_reference = numpy.array([
            [-0.0162824724891702,  0.0071927956010561,  0.0078358202316499],
            [ 0.0558764894925583, -0.0113434242569014, -0.0295305650377395],
            [-0.0072341681793649, -0.0178095577692442, -0.0041154124459146],
            [-0.0323598488242109,  0.0219601864251614,  0.0258101572502076],
        ])
        g_test = _gradient_with_solvent(hf_lowmem.RHF(mol), 'SS(V)PE')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RHF with SS(V)PE: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RKS_CPCM(self):
        # g_reference = _gradient_with_solvent(dft.RKS(mol, xc='b3lyp'), 'C-PCM')
        g_reference = numpy.array([
            [ 0.0107094975860696, -0.0239253315377735,  0.0181754675482018],
            [ 0.0304517629380441,  0.0203233534121517, -0.0119343709048924],
            [-0.0295443405231933, -0.0159663759731676, -0.0169380599890039],
            [-0.0116148291827615,  0.0195746794891644,  0.0107002955042028],
        ])
        g_test = _gradient_with_solvent(rks_lowmem.RKS(mol, xc='b3lyp'), 'C-PCM')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RKS with C-PCM: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RKS_IEFPCM(self):
        # g_reference = _gradient_with_solvent(dft.RKS(mol, xc='wb97m-v'), 'IEF-PCM')
        g_reference = numpy.array([
            [ 0.0092344065127647, -0.015735748870309 ,  0.0179434350777853],
            [ 0.0319344700130117,  0.0122162799479534, -0.0133993962512032],
            [-0.0281476667354037, -0.0159401990506446, -0.0161802722099711],
            [-0.013016035388716 ,  0.019463124206746 ,  0.0116404233769247],
        ])
        g_test = _gradient_with_solvent(rks_lowmem.RKS(mol, xc='wb97m-v'), 'IEF-PCM')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RKS with IEF-PCM: {diff}")
        assert diff < g_tolerance

    def test_lowmem_gradient_RKS_SSVPE(self):
        # g_reference = _gradient_with_solvent(dft.RKS(mol, xc='pbe'), 'SS(V)PE')
        g_reference = numpy.array([
            [ 0.0197558602164731, -0.0315200201481581,  0.0220234876552322],
            [ 0.0221629676914258,  0.0281576145012744, -0.0063872949461893],
            [-0.0371864324361556, -0.0155281188695062, -0.0213824833655754],
            [-0.0047281688870682,  0.0188991381631425,  0.0057502521572663],
        ])
        g_test = _gradient_with_solvent(rks_lowmem.RKS(mol, xc='pbe'), 'SS(V)PE')

        diff = numpy.linalg.norm(g_test - g_reference)
        print(f"Gradient error norm in lowmem RKS with SS(V)PE: {diff}")
        assert diff < g_tolerance

    # TODO: Missing functionalities and tests for unrestricted HF and DFT

if __name__ == "__main__":
    print("Tests for PCM with lowmem SCF and DFT modules")
    unittest.main()
