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
from unittest.mock import patch
import numpy as np
import cupy as cp
from pyscf.lib import fp
from pyscf.data.nist import BOHR
from gpu4pyscf.sem.integral.fock import (get_hcore)
from gpu4pyscf.sem.integral import eri_2c2e
from gpu4pyscf.sem.gto.params import load_sem_params
from gpu4pyscf.sem.gto.mole import Mole
from gpu4pyscf.sem.integral.fock import get_jk_debug, get_jk


def compute_single_multipole_angular_factors():
    ch = cp.zeros((45, 3, 5), dtype=cp.float64)
    
    def set_ch(i, l, m, v):
        ch[i, l, m+2] = v
    set_ch(0,0,0, 1.0)
    set_ch(1,1,0, 1.0)
    set_ch(2,1,1, 1.0)
    set_ch(3,1,-1,1.0)
    set_ch(4,2,0, 1.15470054)
    set_ch(5,2,1, 1.0)
    set_ch(6,2,-1,1.0)
    set_ch(7,2,2, 1.0)
    set_ch(8,2,-2,1.0)
    set_ch(9,0,0,1.0)
    set_ch(9,2,0,1.33333333)
    set_ch(10,2,1,1.0)
    set_ch(11,2,-1,1.0)
    set_ch(12,1,0,1.15470054)
    set_ch(13,1,1,1.0)
    set_ch(14,1,-1,1.0)
    set_ch(17,0,0,1.0)
    set_ch(17,2,0,-0.66666667)
    set_ch(17,2,2,1.0)
    set_ch(18,2,-2,1.0)
    set_ch(19,1,1,-0.57735027)
    set_ch(20,1,0,1.0)
    set_ch(22,1,1,1.0)
    set_ch(23,1,-1,1.0)
    set_ch(24,0,0,1.0)
    set_ch(24,2,0,-0.66666667)
    set_ch(24,2,2,-1.0)
    set_ch(25,1,-1,-0.57735027)
    set_ch(27,1,0,1.0)
    set_ch(28,1,-1,-1.0)
    set_ch(29,1,1,1.0)
    set_ch(30,0,0,1.0)
    set_ch(30,2,0,1.33333333)
    set_ch(31,2,1,0.57735027)
    set_ch(32,2,-1,0.57735027)
    set_ch(33,2,2,-1.15470054)
    set_ch(34,2,-2,-1.15470054)
    set_ch(35,0,0,1.0)
    set_ch(35,2,0,0.66666667)
    set_ch(35,2,2,1.0)
    set_ch(36,2,-2,1.0)
    set_ch(37,2,1,1.0)
    set_ch(38,2,-1,1.0)
    set_ch(39,0,0,1.0)
    set_ch(39,2,0,0.66666667)
    set_ch(39,2,2,-1.0)
    set_ch(40,2,-1,-1.0)
    set_ch(41,2,1,1.0)
    set_ch(42,0,0,1.0)
    set_ch(42,2,0,-1.33333333)
    set_ch(44,0,0,1.0)
    set_ch(44,2,0,-1.33333333)
    
    return ch


class KnownValues(unittest.TestCase):
    def test_hcore_mopac(self):
        hcore_data = []
        for i in range(9,19):
            for j in range(i,19):
                spin = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', spin=spin, verbose=0)
                mol.build()

                original_get_parameter = mol.params.get_parameter
                ch_matrix = compute_single_multipole_angular_factors() 

                def fake_get_parameter(name, *args, **kwargs):
                    if name == 'multipole_angular_factors':
                        if kwargs.get('to_gpu', False):
                            return cp.asarray(ch_matrix)
                        return ch_matrix
                    
                    return original_get_parameter(name, *args, **kwargs)

                with patch.object(mol.params, 'get_parameter', side_effect=fake_get_parameter):
                    mol._compute_integrals()

                hcore_data.append(fp(mol.get_hcore().get()))

        ref_fp = np.array([-186.25073663439207 , -194.52074499636456 , -101.36576343587237 ,
            -115.19434127849564 ,  -27.900618504002495,  -17.59864657510129 ,
            -30.34164746465325 ,  -20.367491809642495,  -28.51863200143069 ,
            -189.00768098693908 ,  -62.007667542135515,   29.034876504403574,
            16.307441330087897,   88.56527186890173 ,  104.20413276168838 ,
            86.74897422346433 ,  102.67273575907151 ,   90.53606113719528 ,
            -57.08854318623475 ,   -6.904819248617284,  -14.57042305553767 ,
            2.788250747809499,   -6.493832240061437,   -5.203105571437746,
            -12.987195472604036,  -17.55903626381888 ,  -72.89726451074392 ,
            -25.1681069072309  ,    2.318545881795475,   -5.502872431957957,
            -5.157482478188103,  -11.036471862184253,  -18.527248552992894,
            -87.01456965056335 , -159.57953217323345 , -112.25276405761778 ,
            -233.16724628532836 , -169.17760244784088 , -270.9458989120248  ,
            82.06048884743232 , -159.90992039889957 , -246.80835508013405 ,
            -168.11160761117014 , -247.94072737464853 ,   73.19761206086434 ,
            -307.3411046477649  , -241.00571164743553 , -342.6831932838425  ,
            73.51550790782134 , -325.14345238278514 , -426.3334375026003  ,
            64.97960289110503 , -454.62583836751173 ,   75.34752607162743 ,
            -61.45212453952013 ])
        assert np.abs(np.array(hcore_data) - ref_fp).max() < 1.0E-12

    def test_hcore(self):
        hcore_data = []
        for i in range(9,19):
            for j in range(i,19):
                spin = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', spin=spin, verbose=0)
                mol.build()

                hcore_data.append(fp(mol.get_hcore().get()))

        ref_fp = np.array([-186.25073663439213, -194.5207449963646, -101.36576343587237, 
            -115.1943412784957, -27.900618500002366, -17.598646575847443, 
            -30.341647455178865, -20.367491809195123, -28.518631995132953, 
            -189.00768098693914, -62.00766754213554, 29.034876504403517, 
            16.30744133008784, 88.56527187644929, 104.20413276155236, 
            86.74897423704357, 102.6727357599022, 90.53606114665266, 
            -57.0885431862348, -6.904819248617283, -14.570423055537674, 
            2.7882507476294074, -6.493832240192666, -5.203105571440557, 
            -12.987195472671992, -17.55903626386964, -72.89726451074394, 
            -25.168106907230914, 2.3185458817537015, -5.502872432284795, 
            -5.157482477322264, -11.036471862255306, -18.52724855252533, 
            -87.01456965056332, -159.5795321757449, -112.25276405439249, 
            -233.16724627163086, -169.17760241879006, -270.94589887366067, 
            82.06048888650248, -159.90992040003965, -246.80835508204078, 
            -168.1116076044457, -247.94072736929797, 73.19761207159344, 
            -307.34110465036935, -241.0057116207191, -342.6831932593243, 
            73.5155079451761, -325.14345238441035, -426.3334375261449, 
            64.97960289885444, -454.6258383714598, 75.34752609929531, 
            -61.45212453952014])
        assert np.abs(np.array(hcore_data) - ref_fp).max() < 1.0E-12

    def test_jk(self):
        for i in range(9,19):
            for j in range(i,19):
                spin = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', spin=spin, verbose=0)
                mol.build()
                nao = mol.nao
                np.random.seed(42)
                dm = np.random.rand(nao, nao)
                dm = dm + dm.T
                fock_debug = get_jk_debug(mol, dm)
                fock = get_jk(mol, dm)
                J_diff = np.abs(fock[0] - fock_debug[0]).max()
                K_diff = np.abs(fock[1] - fock_debug[1]).max()
                assert J_diff < 1.0E-12
                assert K_diff < 1.0E-12


if __name__ == "__main__":
    print("Running tests for fock...")
    unittest.main()
