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
from gpu4pyscf.sem.integral.hcore2c1e import bfn

class TestBfnGPU(unittest.TestCase):
    
    def test_numerical_accuracy(self):
        test_points = np.array([
            0.0, 1e-8,              # Tiny region
            0.1, 0.4, 0.6, 1.5, 2.8, # Small region (merged masks)
            3.000001, 5.0, 10.0     # Large region
        ], dtype=np.float64)
        
        # calculated from yunze qiu's code
        ref_results = [np.array([2.               , 0.               , 0.666666666666667,
                    0.               , 0.4              , 0.               ,
                    0.285714285714286, 0.               , 0.222222222222222,
                    0.               , 0.181818181818182, 0.               ,
                    0.153846153846154]),
            np.array([2.               , 0.               , 0.666666666666667,
                    0.               , 0.4              , 0.               ,
                    0.285714285714286, 0.               , 0.222222222222222,
                    0.               , 0.181818181818182, 0.               ,
                    0.153846153846154]),
            np.array([ 2.003335000396825, -0.066733357142857,  0.668667857451499,
                    -0.040047637566138,  0.401429497607023, -0.028608480759981,
                    0.28682615461483 , -0.022252538073038,  0.223131954342139,
                    -0.01820747031857 ,  0.182587968306367, -0.015406847410759,
                    0.154513310855098]),
            np.array([ 2.053761625396826, -0.270957714285714,  0.698972692768959,
                    -0.163066582010582,  0.423095214237614, -0.1166715998076  ,
                    0.303686878099678, -0.090841411033411,  0.236932537850298,
                    -0.074379676146076,  0.194268765629142, -0.06297072297637 ,
                    0.164638909539308]),
            np.array([ 2.122178514285714, -0.414586377142857,  0.740223923809524,
                    -0.250430724155844,  0.452640353246753, -0.179547244115884,
                    0.326706073126873, -0.139979220757909,  0.255788904180264,
                    -0.114716424074749,  0.210238113039901, -0.097184512263464,
                    0.178488269016432]),
            np.array([ 2.839039273009553, -1.243853300477599,  1.180568200149815,
                    -0.775409748851002,  0.771279932935026, -0.565613039367213,
                    0.576587102381108, -0.445806338038797,  0.461405454260907,
                    -0.368113423585194,  0.384949764323314, -0.313581210779661,
                    0.330389566769686]),
            np.array([ 5.851370170892627, -3.805030909600268,  3.133490949749577,
                    -2.53749419048313 ,  2.226378470202441, -1.919130043630157,
                    1.738948648828005, -1.547434228923628,  1.430129516825118,
                    -1.297960948365997,  1.21579535529978 , -1.118466904953941,
                    1.057940578232879]),
            np.array([ 6.678587770522015, -4.485583923736506,  3.688199484827106,
                    -3.023580849751096,  2.647147981333816, -2.299867273593033,
                    2.078854756580286, -1.861119623377944,  1.715603762508833,
                    -1.564969533255477,  1.462024398524883, -1.351024764172327,
                    1.27449051519846 ]),
            np.array([ 29.6812842311155  , -23.747722563692037,  20.18219520563869 ,
                    -17.574662286531925,  15.621554401889963, -14.062425008025173,
                    12.806374221485296, -11.755055499835724,  10.873195431378345,
                    -10.112227633434115,   9.456828964247272,  -8.878955688571141,
                    8.371790578544763]),
            np.array([ 2202.646574940679 , -1982.381926526597 ,  1806.1701896353593,
                    -1660.7955271300568,  1538.3283640886561, -1433.4824019763366,
                    1342.5571337548768, -1262.856590392251 ,  1192.3613026268781,
                    -1129.5214116564744,  1073.1251632842045, -1022.2089044080398,
                    975.9958896510312])]
        ref_results = np.array(ref_results) 

        x_gpu = cp.asarray(test_points)
        bf_gpu = bfn(x_gpu)
        res_gpu = cp.asnumpy(bf_gpu)

        np.testing.assert_allclose(res_gpu, ref_results, rtol=1e-11, atol=1e-10,
                                   err_msg="GPU results deviate from CPU reference")

    def test_shape_consistency(self):
        x1 = cp.random.random(10)
        res1 = bfn(x1)
        self.assertEqual(res1.shape, (10, 13))
        
        x2 = cp.random.random((5, 5))
        res2 = bfn(x2)
        self.assertEqual(res2.shape, (5, 5, 13))


if __name__ == '__main__':
    print("Running tests for hcore2c1e...")
    unittest.main()