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
from pyscf import gto
from pyscf.dispersion import dftd3

class KnownValues(unittest.TestCase):
    def test_energy_r2scan_d3(self):
        mol = gto.M(
            atom='''
             C   -0.755422531  -0.796459123  -1.023590391
             C    0.634274834  -0.880017014  -1.075233285
             C    1.406955202   0.199695367  -0.653144334
             C    0.798863737   1.361204515  -0.180597909
             C   -0.593166787   1.434312023  -0.133597923
             C   -1.376239198   0.359205222  -0.553258516
             I   -1.514344238   3.173268101   0.573601106
             H    1.110906949  -1.778801728  -1.440619836
             H    1.399172302   2.197767355   0.147412751
             H    2.486417780   0.142466525  -0.689380574
             H   -2.454252250   0.422581120  -0.512807958
             H   -1.362353593  -1.630564523  -1.348743149
             S   -3.112683203   6.289227834   1.226984439
             H   -4.328789697   5.797771251   0.973373089
             C   -2.689135032   6.703163830  -0.489062886
             H   -1.684433029   7.115457372  -0.460265708
             H   -2.683867206   5.816530502  -1.115183775
             H   -3.365330613   7.451201412  -0.890098894
            ''')

        dftd3_model = dftd3.DFTD3Dispersion(mol, "r2SCAN", atm=True)
        res = dftd3_model.get_dispersion()
        assert np.allclose(res['energy'], -0.005790963570050724)

        dftd3_model = dftd3.DFTD3Dispersion(mol, "r2SCAN")
        res = dftd3_model.get_dispersion()
        assert np.allclose(res['energy'], -0.005784012374055654)

    def test_gradient_r2scan_d3(self):
        mol = gto.M(
            atom='''
             H    0.002144194   0.361043475   0.029799709
             C    0.015020592   0.274789738   1.107648016
             C    1.227632658   0.296655040   1.794629427
             C    1.243958826   0.183702791   3.183703934
             C    0.047958213   0.048915002   3.886484583
             C   -1.165135654   0.026954348   3.200213281
             C   -1.181832083   0.139828643   1.810376587
             H    2.155807907   0.399177037   1.249441585
             H    2.184979344   0.198598553   3.716170761
             H    0.060934662  -0.040672756   4.964014252
             H   -2.093220602  -0.078628959   3.745125056
             H   -2.122845437   0.123257119   1.277645797
             Br  -0.268325907  -3.194209024   1.994458950
             C    0.049999933  -5.089197474   1.929391171
             F    0.078949601  -5.512441335   0.671851563
             F    1.211983937  -5.383996300   2.498664481
             F   -0.909987405  -5.743747328   2.570721738
            ''')

        ref = np.array([
            [+7.13721248e-07, +2.19571763e-05, -3.77372946e-05],
            [+9.19838860e-07, +3.53459763e-05, -1.43306994e-06],
            [+7.43860881e-06, +3.78237447e-05, +8.46031238e-07],
            [+8.06120927e-06, +3.79834948e-05, +8.58427570e-06],
            [+1.16592466e-06, +3.62585085e-05, +1.16326308e-05],
            [-3.69381337e-06, +3.39047971e-05, +6.92483428e-06],
            [-3.05404225e-06, +3.29484247e-05, +1.80766271e-06],
            [+3.51228183e-05, +2.08136972e-05, -1.76546837e-05],
            [+3.49762054e-05, +1.66544908e-05, +2.14435772e-05],
            [+1.57516340e-06, +1.41373959e-05, +4.21574793e-05],
            [-3.35392428e-05, +1.49030766e-05, +2.29976305e-05],
            [-3.38817253e-05, +1.82002569e-05, -1.72487448e-05],
            [-2.15610724e-05, -1.87935101e-04, -3.02815495e-05],
            [+1.27580963e-06, -5.96841724e-05, -5.99713166e-06],
            [+9.01173808e-07, -2.23010304e-05, -7.96228701e-06],
            [+7.42062176e-06, -2.79631452e-05, +7.03703317e-07],
            [-3.84119900e-06, -2.30475903e-05, +1.21693625e-06],]
        )
        dftd3_model = dftd3.DFTD3Dispersion(mol, "r2SCAN", atm=False)
        res = dftd3_model.get_dispersion(grad=True)
        assert np.linalg.norm(ref - res['gradient']) < 1e-10

if __name__ == "__main__":
    print("Full tests for DFTD3 module")
    unittest.main()
