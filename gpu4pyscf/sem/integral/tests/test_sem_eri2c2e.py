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
from gpu4pyscf.sem.integral.eri_2c2e import (multipole_eval, a_function_ijl, solve_poij,
    calc_aij_tensor)
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

    def test_a_function_ijl(self):
        n1_list = [1, 1, 1, 1, 1, 1, 2, 2, 2]
        n2_list = [1, 1, 1, 2, 2, 2, 2, 2, 2]
        l_list = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        # some random parameters
        z1_list = [0.50316167805243,
            0.2648333731980974,
            1.4617661768387178,
            0.2039547597225122,
            0.7282823302772807,
            0.20825337078144468,
            1.0911134739715618,
            1.3533593095506586,
            0.12105243449211955]
        z2_list = [0.3161324883247225,
            1.6983452300444837,
            0.5238296050633506,
            0.2633908412014694,
            0.3833326286330341,
            1.159032159106229,
            1.1388522711858338,
            0.8459058868428406,
            0.37827697340525224]
        # benchmark from yunze qiu's code
        ref_list = [0.9228591640770297,
            0.9747652117708852,
            8.33640756573915,
            0.9525777184247998,
            3.6928172611002816,
            23.31911534529494,
            0.9988546524810269,
            3.9657220706423044,
            222.62800619496616]
        output = a_function_ijl(
                        cp.array(z1_list),
                        cp.array(z2_list),
                        cp.array(n1_list),
                        cp.array(n2_list),
                        cp.array(l_list),
                    )
        assert np.abs(output.get() - np.array(ref_list)).max() < 1e-13

    def test_solve_poij(self):
        l_list = [2, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 2, 1, 2, 1]
        d_list = [3.8312328759258705,
            3.437273418275381,
            1.0661212836661842,
            2.723758533172069,
            3.388662688413195,
            2.197307168083692,
            2.8626443300792266,
            1.050320041477397,
            4.574440176553458,
            2.8491613410138052,
            3.087169616355614,
            2.9527633331446355,
            1.601832774575922,
            3.9908196171085084,
            4.407493066382512,
            2.9704154938065592,
            3.1512666336639343,
            3.4407758699132343]
        fg_list = [14.448686,
            9.445299,
            11.035907,
            7.552804,
            8.179341,
            13.335519,
            12.357026,
            11.304042,
            12.446818,
            19.999574,
            4.059972,
            7.115328,
            6.652155,
            5.194805,
            8.758856,
            9.17035,
            11.142654,
            17.858776]
        ref_list =  [0.21171946342044537,
            0.5969898028355257,
            0.3999246378197412,
            0.6819475012266928,
            1.6634216770023402,
            1.0202597381469742,
            0.46268420653455816,
            1.2036131078594716,
            0.4885153690252852,
            0.3040546899943632,
            1.109076822393453,
            1.9121666805794475,
            2.0453060884771928,
            2.619096024392446,
            0.3347972867897028,
            0.5959145164202259,
            0.25966346705584986,
            0.34312504763831003]
        output = solve_poij(
                        cp.array(l_list),
                        cp.array(d_list),
                        cp.array(fg_list),
                    )
        assert np.abs(output.get() - np.array(ref_list)).max() < 1e-13

    def test_cal_aij_tensor(self):
        # test for atom He, F, Cl, As
        zs_list = [3.313204, 6.043849, 2.63705, 2.926171]
        zp_list = [3.657133, 2.906722, 2.118146, 1.765191]
        zd_list = [0.0, 0.0, 1.324033, 1.392142]
        nsp_list = [1, 2, 3, 4]
        nd_list = [3, 3, 3, 4]
        dorbs_list = [False, False, True, True]
        elemet_id_list = [1, 8, 16, 32] # 0-based !
        ref_8 = np.array([[0.              , 0.80507860289616, 0.              ],
            [0.80507860289616, 3.55070228301778, 0.              ],
            [0.              , 0.              , 0.              ]])
        ref_16 = np.array([[ 0.              ,  2.82325771086925,  9.4993220754193 ],
            [ 2.82325771086925, 12.48177193978136,  3.35862761583999],
            [ 9.4993220754193 ,  3.35862761583999, 31.94408170082052]])
        ref_32 = np.array([[ 0.              ,  2.88711669010284, 10.52062977872613],
            [ 2.88711669010284, 28.8841173295192 ,  5.35152009301035],
            [10.52062977872613,  5.35152009301035, 46.43820638684803]])
        output = calc_aij_tensor(
                        cp.array(zs_list),
                        cp.array(zp_list),
                        cp.array(zd_list),
                        cp.array(nsp_list),
                        cp.array(nd_list),
                        cp.array(dorbs_list),
                        cp.array(elemet_id_list),
                    )
        assert np.abs(output[:,:,0].get() - 0).max() < 1e-13
        assert np.abs(output[:,:,1].get() - ref_8).max() < 1e-13
        assert np.abs(output[:,:,2].get() - ref_16).max() < 1e-13
        assert np.abs(output[:,:,3].get() - ref_32).max() < 1e-13



        
if __name__ == "__main__":
    print("Running tests for eri2c2e...")
    unittest.main()
