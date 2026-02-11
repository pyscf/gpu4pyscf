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
from pyscf.data.nist import BOHR
from gpu4pyscf.sem.integral.eri_2c2e import multipole_eval
from gpu4pyscf.sem.gto.mole import Mole

class KnownValues(unittest.TestCase):
    def test_multipole_eval(self):
        l1_list = [0, 1, 1, 1]
        l2_list = [1, 0, 1, 1]
        m_list = [0, 0, 0, 1]
        # some random parameters
        da_list = [0.16678948276690012,
                   0.06575738359734129,
                   0.09292992608352953,
                   0.1628868106871415]
        db_list = [0.16574387336703944,
                   0.11076175062613176,
                   0.13357315035066467,
                   0.05215569517479601]
        r_list = [1.545666036411078,
                  1.011783470209541,
                  1.0929373308184624,
                  1.6228779406065064]
        add_list =  [0.025601841313958207,
                     0.13134806030115942,
                     0.16621676747468783,
                     0.3311702013807096]
        # benchmark from yunze qiu's code
        output_list = [-0.06903952720901207,
                        0.053739381853752544,
                        -0.013052013372655213,
                        0.0016397774382511665]
        output = multipole_eval(
                        cp.array(r_list),
                        cp.array(l1_list),
                        cp.array(l2_list),
                        cp.array(m_list),
                        cp.array(da_list),
                        cp.array(db_list),
                        cp.array(add_list),
                    )
        assert np.abs(output.get() - np.array(output_list)).max() < 1e-13

        
if __name__ == "__main__":
    print("Running tests for hcore2c1e...")
    unittest.main()
