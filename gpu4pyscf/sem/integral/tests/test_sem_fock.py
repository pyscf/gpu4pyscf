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
from pyscf.lib import fp
from pyscf.data.nist import BOHR
from gpu4pyscf.sem.integral.fock import (get_hcore)
from gpu4pyscf.sem.integral import eri_2c2e
from gpu4pyscf.sem.gto.params import load_sem_params
from gpu4pyscf.sem.gto.mole import Mole
from gpu4pyscf.sem.integral.fock import get_jk_debug, get_jk

class KnownValues(unittest.TestCase):
    def test_hcore(self):
        hcore_data = []
        for i in range(9,19):
            for j in range(i,19):
                spin = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', spin=spin, verbose=0)
                mol.build()
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
