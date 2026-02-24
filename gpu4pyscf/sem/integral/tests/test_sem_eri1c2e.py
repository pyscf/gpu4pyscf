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
from gpu4pyscf.sem.integral.eri_1c2e import rsc
from gpu4pyscf.sem.gto.mole import Mole

class KnownValues(unittest.TestCase):
    def test_rsc(self):
        k_list = [1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1]
        na_list = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        nb_list = [1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2]
        nc_list = [2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
        nd_list = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        ea_list = [2.2429841169968476,
            2.8498933339211323,
            2.3755157836489795,
            1.1296652635836515,
            2.0551690000606504,
            1.0501141705752264,
            2.7589667323079743,
            1.3509515044179379,
            1.4156474032621769,
            1.3573668308940625,
            1.3089111804548499,
            1.929143152150822,
            2.6676474435367923]
        eb_list = [1.8898398798708869,
            2.642099618141959,
            1.250604254143524,
            1.910253021293723,
            1.4292028032281119,
            2.5820254633696194,
            1.1693041970406362,
            2.478045519743538,
            2.2367429686754314,
            1.964775289881262,
            1.5952551274459805,
            1.067773835130554,
            1.797001383289766]
        ec_list = [1.9901352554643983,
            2.657102206318137,
            2.481068978553926,
            1.2202618010144353,
            1.1846822426522576,
            1.4923827918440853,
            1.889510651390524,
            1.6909016168231226,
            2.5188215328461703,
            2.985023846102097,
            2.152375090980905,
            2.419501623650163,
            1.6557316015854482]
        ed_list = [1.971113563731713,
            1.4883178765397587,
            1.0870619873783316,
            1.2611576445767372,
            1.0739431423941115,
            1.4782662146587722,
            2.2321458020170164,
            2.503158957105369,
            2.2792171797656726,
            1.4619381877662316,
            2.433570933256709,
            1.4749184579330503,
            1.5195079237051563]
        ref = [13.280093960946706,
            5.672320326325945,
            3.0150503845812806,
            6.347248348490742,
            5.287055250749571,
            2.033307934510677,
            13.773894676819967,
            3.974689188863744,
            6.710448866646316,
            8.15168949800282,
            5.955861166056645,
            5.831755903292418,
            10.431167544741871]
        output = rsc(cp.array(k_list),
            cp.array(na_list),
            cp.array(ea_list),
            cp.array(nb_list),
            cp.array(eb_list),
            cp.array(nc_list),
            cp.array(ec_list),
            cp.array(nd_list),
            cp.array(ed_list),
        )
        assert (output.get() - np.array(ref)).max() < 1e-12


if __name__ == "__main__":
    print("Running tests for eri1c2e...")
    unittest.main()
