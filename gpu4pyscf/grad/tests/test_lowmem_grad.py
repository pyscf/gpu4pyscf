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

import numpy as np
import cupy as cp
import unittest
import pytest
import pyscf
from gpu4pyscf import scf, dft
from gpu4pyscf.dft import rks_lowmem
from gpu4pyscf.scf import hf_lowmem

atom = '''
O  0.0000   0.7375  -0.0528
O  0.0000  -0.7375  -0.1528
H  0.8190   0.8170   0.4220
H  -0.8190 -0.8170   0.4220
'''

bas0='cc-pvtz'

def setUpModule():
    global mol_sph, mol_cart, g_tolerance
    mol_sph = pyscf.M(atom=atom, basis=bas0, max_memory=32000,
                      output='/dev/null', verbose=1)

    mol_cart = pyscf.M(atom=atom, basis=bas0, max_memory=32000, cart=1,
                       output='/dev/null', verbose=1)

    g_tolerance = 1e-6

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def _compute_gradient(mf):
    mf.direct_scf_tol = 1e-14
    mf.conv_tol = 1e-10
    mf.kernel()

    gradient = mf.Gradients().kernel()
    return gradient

class KnownValues(unittest.TestCase):
    def test_lowmem_grad_rhf_sph(self):
        # reference_gradient = _compute_gradient(scf.RHF(mol_sph))
        reference_gradient = np.array([
            [-0.0231124185697982,  0.0477463908153393,  0.0017847846861955],
            [ 0.0628962970989975, -0.0523678930180633, -0.0423486883580151],
            [ 0.0076605925301827, -0.0089288203680652,  0.0049101450721163],
            [-0.0474444710589199,  0.013550322571044 ,  0.0356537585995285],
        ])
        test_gradient = _compute_gradient(hf_lowmem.RHF(mol_sph))
        diff = np.linalg.norm(reference_gradient - test_gradient)
        print('|| normal - lowmem || = ', diff)
        assert(diff < g_tolerance)

    def test_lowmem_grad_rhf_cart(self):
        # reference_gradient = _compute_gradient(scf.RHF(mol_cart))
        reference_gradient = np.array([
            [-0.0231709866271921,  0.0479429271566953,  0.00176540098277  ],
            [ 0.0629254153292695, -0.052543078070185 , -0.0423631051648066],
            [ 0.0076440958355062, -0.009010117906703 ,  0.0049338036815316],
            [-0.0473985245372481,  0.0136102688204638,  0.0356639005003621],
        ])
        test_gradient = _compute_gradient(hf_lowmem.RHF(mol_cart))
        diff = np.linalg.norm(reference_gradient - test_gradient)
        print('|| normal - lowmem || = ', diff)
        assert(diff < g_tolerance)

    def test_lowmem_grad_rks_sph(self):
        # reference_gradient = _compute_gradient(dft.RKS(mol_sph, xc='hse06'))
        reference_gradient = np.array([
            [-0.0004866290992456,  0.0191253340123403,  0.0105671081232781],
            [ 0.0408226714263633, -0.0230644399530089, -0.0265373902924315],
            [-0.0112112571535956, -0.007686673482683 , -0.006196507041498 ],
            [-0.0291218915636033,  0.0116432036827433,  0.022169968876786 ],
        ])
        test_gradient = _compute_gradient(rks_lowmem.RKS(mol_sph, xc='hse06'))
        diff = np.linalg.norm(reference_gradient - test_gradient)
        print('|| normal - lowmem || = ', diff)
        assert(diff < g_tolerance)

    def test_lowmem_grad_rks_cart(self):
        # reference_gradient = _compute_gradient(dft.RKS(mol_cart, xc='wb97x-d3bj'))
        reference_gradient = np.array([
            [-0.0001341153195139,  0.01996248589164  ,  0.0112309450215729],
            [ 0.0408547038470197, -0.0238988659015736, -0.0262240779373344],
            [-0.0122785779247628, -0.0084235722236089, -0.0068005041139765],
            [-0.028441210971953 ,  0.0123647678259413,  0.0217944221902671],
        ])
        test_gradient = _compute_gradient(rks_lowmem.RKS(mol_cart, xc='wb97x-d3bj'))
        diff = np.linalg.norm(reference_gradient - test_gradient)
        print('|| normal - lowmem || = ', diff)
        assert(diff < g_tolerance)

if __name__ == "__main__":
    print("Full Tests for hf_lowmem Gradient")
    unittest.main()
