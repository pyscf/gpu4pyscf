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
from pyscf.lib import fp
from unittest.mock import Mock
from gpu4pyscf.sem.integral.eri_1c2e import (rsc, calc_sp_two_electron, 
    calc_scprm, calc_repd_and_eiscor, get_eri1c2e)
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
        ref_fp = 17.576340526338015
        assert np.abs(fp(output.get()) - ref_fp) < 1.0E-12

    def test_calc_sp_two_electron(self):
        ns_list = [1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
        es_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.742341,8.388111,6.042706,
            0.479722,0.956297,0.0,0.0,0.0,0.848418,1.045904,1.094426,1.619853,1.13245,1.459152,
            0.519518,0.74647,1.899598,0.0,0.0,0.0,2.006543,0.0,3.094777,0.0]
        ep_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.669626,1.843048,2.376473,
            1.015507,2.464067,0.0,0.0,0.0,2.451729,1.076844,0.755378,0.848266,1.39074,1.392614,
            1.0,0.753327,3.0,0.0,0.0,0.0,3.316832,0.0,3.065764,0.0]
        gss = cp.array([14.448686  ,  9.445299  , 11.035907  ,  7.552804  ,  8.179341  ,
            13.335519  , 12.357026  , 11.304042  , 12.446818  , 19.999574  ,
            4.059972  ,  7.115328  ,  6.652155  ,  5.194805  ,  8.758856  ,
            9.17035   , 11.142654  , 17.858776  ,  3.369251  ,  5.725773  ,
            4.63821583,  5.71785192,  5.98311681,  8.85557334,  6.19099019,
            7.97703715,  2.84015263,  4.08087637, 10.38491111,  8.707424  ,
            10.354885  ,  7.518301  ,  6.66503   ,  5.522356  ,  7.616791  ,
            19.999857  ])
        gsp = cp.array([ 0.        , 11.201419  , 19.998647  , 10.203146  ,  7.294021  ,
            11.528134  ,  9.63619   , 15.807424  , 18.496082  , 16.896951  ,
            7.061183  ,  3.253024  ,  7.459435  ,  5.090534  ,  8.483679  ,
            5.944296  ,  7.487881  ,  4.168451  ,  6.129351  ,  4.781065  ,
            5.73916422,  5.80001559,  4.73676999,  5.58863124,  6.75742772,
            7.78686802,  3.42593356,  4.09945211, 12.14536185,  3.436116  ,
            7.993674  ,  6.594443  ,  6.213867  ,  2.907562  ,  5.010425  ,
            1.175304  ])
        hsp = cp.array([0.        , 0.299954  , 1.641886  , 1.501452  , 1.252845  ,
            0.717322  , 2.871545  , 5.010801  , 2.604382  , 1.77928   ,
            0.640715  , 0.877379  , 0.43506   , 1.425012  , 0.871681  ,
            5.005404  , 5.004267  , 4.574549  , 0.300325  , 1.240572  ,
            0.19383458, 1.40373178, 0.90110528, 0.64803943, 1.52051839,
            1.88018948, 0.3900875 , 0.99349784, 2.03739422, 0.662036  ,
            1.295974  , 0.290742  , 0.280662  , 3.095789  , 4.996553  ,
            0.299867  ])
        gpp = cp.array([ 0.        ,  9.214548  , 11.54365   , 12.862153  ,  7.829395  ,
            10.778326  , 12.570756  , 13.618205  ,  8.417366  ,  8.96356   ,
            9.28354   ,  4.737311  ,  7.668857  ,  5.18515   ,  8.662754  ,
            8.165473  ,  9.551886  , 11.8525    ,  0.999505  ,  7.172103  ,
            14.60487391,  6.41472644,  4.49976341,  5.05309436,  8.28459522,
            8.29575858,  5.95696911,  4.48754567, 17.87090732, 20.000041  ,
            6.090184  ,  6.066801  ,  9.310836  ,  8.042391  ,  9.649216  ,
            9.174784  ])
        gp2 = cp.array([ 0.        , 13.046115  ,  9.059036  , 13.602858  ,  6.401072  ,
            9.486212  , 10.576425  , 10.332765  , 12.179816  , 16.027799  ,
            17.034978  ,  8.428485  ,  6.673299  ,  4.769775  ,  7.734264  ,
            7.301878  ,  8.128436  , 15.669543  , 18.999148  ,  7.431876  ,
            12.80259663,  5.62313345,  3.94448156,  4.42953011,  7.26225583,
            7.27204161,  5.2218645 ,  3.93377152, 15.6655935 ,  6.782785  ,
            6.299226  ,  5.305947  ,  8.712542  ,  6.735106  ,  8.343792  ,
            14.926948  ])
        main_group_list = [True,True,True,True,True,True,True,True,True,True,True,True,True,
            True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,
            False,True,True,True,True,True,True,True]
        topology = Mock()
        topology.principal_quantum_number_s = cp.array(ns_list, dtype=cp.int32)
        topology.eta_2e = cp.array([es_list, ep_list], dtype=cp.float64).T
        topology.is_main_group = cp.array(main_group_list, dtype=cp.bool_)

        one_center_integrals = Mock()
        one_center_integrals.gss = gss
        one_center_integrals.gsp = gsp
        one_center_integrals.hsp = hsp
        one_center_integrals.gpp = gpp
        one_center_integrals.gp2 = gp2

        output = calc_sp_two_electron(topology, one_center_integrals)

        gss_ref_fp = -28.091611825685217
        gsp_ref_fp = -12.79212598084127
        hsp_ref_fp = -11.045284154366275
        gpp_ref_fp = -32.90158196383739
        gp2_ref_fp = -30.462069291485648
        
        assert np.abs(gss_ref_fp - fp(output[0].get())) < 1.0E-12
        assert np.abs(gsp_ref_fp - fp(output[1].get())) < 1.0E-12
        assert np.abs(hsp_ref_fp - fp(output[2].get())) < 1.0E-12
        assert np.abs(gpp_ref_fp - fp(output[3].get())) < 1.0E-12
        assert np.abs(gp2_ref_fp - fp(output[4].get())) < 1.0E-12

    def test_calc_scprm(self):
        es_list = [8.388111, 6.042706, 0.479722]
        ep_list = [1.843048, 2.376473, 1.015507]
        ed_list = [0.7086, 7.14775, 4.31747]
        dorbs_list = [True, True, True]
        topology = Mock()
        topology.principal_quantum_number_s = cp.array([3, 3, 3], dtype=cp.int32)
        topology.principal_quantum_number_d = cp.array([3, 3, 3], dtype=cp.int32)
        topology.eta_2e = cp.array([es_list, ep_list, ed_list], dtype=cp.float64).T
        topology.has_d_orbitals = cp.array(dorbs_list, dtype=cp.bool_)

        output = calc_scprm(topology)
        r016 = cp.array([6.427188952734019, 45.761841997886826, 4.350910283451273])
        r036 = cp.array([6.303058793902619, 21.30707301198425, 9.18401049707871])
        r066 = cp.array([4.977414295920683, 50.20789307601905, 30.327175981101643])
        r155 = cp.array([1.3444126119112705, 3.101482906187396, 0.4450588613510365])
        r125 = cp.array([0.2858363783001884, 3.7976216883694534, 0.2877284819329573])
        r244 = cp.array([2.7257038866197025e-03, 2.3278062576395492e+01, 6.9697798805561989e-03])
        r236 = cp.array([1.7849117670040142, 4.971693623027796,  1.2509185992453404])
        r266 = cp.array([2.627421972600759, 26.503182902423344, 16.008771583466864])
        r234 = cp.array([0.0627082853794884, 5.392859366023948,  0.0798020119657711])
        r246 = cp.array([5.3149714854392330e-03, 2.4675965798530605e+01, 2.3750249045898286e-01])
        r355 = cp.array([0.8025838514432496, 1.851514984298315,  0.2656900507309903])
        r466 = cp.array([1.7135360690878458, 17.28468450157974,   10.440503206612684 ])

        assert ((output[0] - r016)/output[0]).max() < 5e-13
        assert ((output[1] - r036)/output[1]).max() < 5e-13
        assert ((output[2] - r066)/output[2]).max() < 5e-13
        assert ((output[3] - r155)/output[3]).max() < 5e-13
        assert ((output[4] - r125)/output[4]).max() < 5e-13
        assert ((output[5] - r244)/output[5]).max() < 5e-13
        assert ((output[6] - r236)/output[6]).max() < 5e-13
        assert ((output[7] - r266)/output[7]).max() < 5e-13
        assert ((output[8] - r234)/output[8]).max() < 5e-13
        assert ((output[9] - r246)/output[9]).max() < 5e-13
        assert ((output[10] - r355)/output[10]).max() < 5e-13
        assert ((output[11] - r466)/output[11]).max() < 5e-13

    def test_calc_repd_and_eiscor(self):
        atomic_numbers_list = [14, 15, 16] # 1-based
        f0sd_list = [0.0, 0.0, 0.0]
        g2sd_list = [0.0, 0.0, 0.0]
        dorbs_list = [True, True, True]
        integrals_tuple = (cp.array([ 6.427188952733883, 45.761841997886734,  4.350910283451282]),
            cp.array([ 6.303058793902614, 21.307073011984233,  9.18401049707871 ]),
            cp.array([ 4.977414295920683, 50.20789307601914 , 30.327175981101703]),
            cp.array([1.344412611911269, 3.101482906187396, 0.445058861351037]),
            cp.array([0.285836378300252, 3.797621688369437, 0.287728481932957]),
            cp.array([2.725703886619664e-03, 2.327806257639549e+01,
                    6.969779880556212e-03]),
            cp.array([1.784911767004013, 4.971693623027796, 1.250918599245339]),
            cp.array([ 2.627421972600759, 26.503182902423905, 16.00877158346675 ]),
            cp.array([0.062708285379488, 5.39285936602396 , 0.079802011965771]),
            cp.array([5.314971485439233e-03, 2.467596579853038e+01,
                    2.375024904589828e-01]),
            cp.array([0.802583851443231, 1.851514984298315, 0.265690050730991]),
            cp.array([ 1.713536069087379, 17.284684501584913, 10.44050320660976 ])) # from calc_scprm

        mock_topology = Mock()
        mock_topology.atom_ids_0based = cp.array(atomic_numbers_list, dtype=cp.int32) - 1
        mock_topology.has_d_orbitals = cp.array(dorbs_list, dtype=cp.bool_)

        mock_one_center_integrals = Mock()
        mock_one_center_integrals.f0_sd = cp.array(f0sd_list, dtype=cp.float64)
        mock_one_center_integrals.g2_sd = cp.array(g2sd_list, dtype=cp.float64)
        
        output = calc_repd_and_eiscor(
            topology=mock_topology,
            one_center_integrals=mock_one_center_integrals,
            integrals_tuple=integrals_tuple
            )

        ref_fp = 126.45779262444755
        ref_f0dd = cp.array([ 4.977414295920683, 50.20789307601905 , 30.327175981101643])
        ref_f2dd = cp.array([ 2.627421972600759, 26.503182902423344, 16.008771583466864])
        ref_f4dd = cp.array([ 1.713536069087846, 17.28468450157974 , 10.440503206612684])
        ref_f0sd = cp.array([ 6.427188952734019, 45.761841997886826,  4.350910283451273])
        ref_g2sd6 = cp.array([2.725703886619702e-03, 2.327806257639549e+01, 6.969779880556199e-03])
        ref_f0pd = cp.array([ 6.303058793902619, 21.30707301198425 ,  9.18401049707871 ])
        ref_f2pd = cp.array([1.784911767004014, 4.971693623027796, 1.25091859924534 ])
        ref_g1pd = cp.array([1.34441261191127 , 3.101482906187396, 0.445058861351037])
        ref_g3pd = cp.array([0.80258385144325 , 1.851514984298315, 0.26569005073099 ])
        
        assert np.abs(ref_fp-fp(output[0].get())) < 5.0e-12
        assert np.abs((output[2]['f0dd'] - ref_f0dd)/ref_f0dd).max() < 5.0e-13
        assert np.abs((output[2]['f2dd'] - ref_f2dd)/ref_f2dd).max() < 5.0e-13
        assert np.abs((output[2]['f4dd'] - ref_f4dd)/ref_f4dd).max() < 5.0e-13
        assert np.abs((output[2]['f0sd'] - ref_f0sd)/ref_f0sd).max() < 5.0e-13
        assert np.abs((output[2]['g2sd'] - ref_g2sd6)/ref_g2sd6).max() < 5.0e-13
        assert np.abs((output[2]['f0pd'] - ref_f0pd)/ref_f0pd).max() < 5.0e-13
        assert np.abs((output[2]['f2pd'] - ref_f2pd)/ref_f2pd).max() < 5.0e-13
        assert np.abs((output[2]['g1pd'] - ref_g1pd)/ref_g1pd).max() < 5.0e-13
        assert np.abs((output[2]['g3pd'] - ref_g3pd)/ref_g3pd).max() < 5.0e-13

        ref_eiscor = 1.4 # test for La 57(1-based)
        integrals_tuple = (
            cp.array([1.0]),
            cp.array([0.0]),
            cp.array([2.0]),
            cp.array([0.0]),
            cp.array([0.0]),
            cp.array([3.0]),
            cp.array([0.0]),
            cp.array([4.0]),
            cp.array([0.0]),
            cp.array([0.0]),
            cp.array([0.0]),
            cp.array([5.0])
        )
        mock_topology = Mock()
        mock_topology.atom_ids_0based = cp.array([57], dtype=cp.int32) - 1
        mock_topology.has_d_orbitals = cp.array([True], dtype=cp.bool_)

        mock_one_center_integrals = Mock()
        mock_one_center_integrals.f0_sd = cp.array([0.0], dtype=cp.float64)
        mock_one_center_integrals.g2_sd = cp.array([0.0], dtype=cp.float64)

        output2 = calc_repd_and_eiscor(topology=mock_topology,
            one_center_integrals=mock_one_center_integrals,
            integrals_tuple=integrals_tuple)
        assert np.abs((output2[1].get() - ref_eiscor)/ref_eiscor).max() < 1.0e-13

    def test_get_eri1c2e(self):
        from pyscf.data import elements

        max_z = 54

        gss_list = []
        gsp_list = []
        hsp_list = []
        gpp_list = []
        gp2_list = []
        repd_list = []
        eisol_corr_list = []
        params_dict_list = []

        for z in range(1, max_z + 1):
            symb = elements._symbol(z)
            spin = z % 2

            mol = Mole(f"{symb} 0 0 0", charge=0, spin=spin, verbose=0)
            mol.build()

            gss, gsp, hsp, gpp, gp2, repd, eisol_corr, params_dict = get_eri1c2e(mol)
            gss_list.append(gss.item())
            gsp_list.append(gsp.item())
            hsp_list.append(hsp.item())
            gpp_list.append(gpp.item())
            gp2_list.append(gp2.item())
            repd_list.append(repd)
            eisol_corr_list.append(eisol_corr.item())
            params_dict_list.append(params_dict)

        repd_array = np.zeros((52, max_z))
        for i in range(max_z):
            repd_array[:,i] = repd_list[i][:,0].get()
        gss_array = np.array(gss_list)
        gsp_array = np.array(gsp_list)
        hsp_array = np.array(hsp_list)
        gpp_array = np.array(gpp_list)
        gp2_array = np.array(gp2_list)
        eisol_corr_array = np.array(eisol_corr_list)

        ref_gss_fp = -41.22489165175284
        ref_gsp_fp = -6.702623477446204
        ref_hsp_fp = -15.521527334821878
        ref_gpp_fp = 11.884473942638197
        ref_gp2_fp = 12.727034135811378
        ref_eiscor_fp = -237.28223185801244
        ref_repd_fp = 176.3394963033965

        assert np.abs(ref_gss_fp-fp(gss_array)) < 1.0E-12
        assert np.abs(ref_gsp_fp-fp(gsp_array)) < 1.0E-12
        assert np.abs(ref_hsp_fp-fp(hsp_array)) < 1.0E-12
        assert np.abs(ref_gpp_fp-fp(gpp_array)) < 1.0E-12
        assert np.abs(ref_gp2_fp-fp(gp2_array)) < 1.0E-12
        assert np.abs(ref_eiscor_fp-fp(eisol_corr_array)) < 1.0E-12
        assert np.abs(ref_repd_fp-fp(repd_array)) < 1.0E-10
        

if __name__ == "__main__":
    print("Running tests for eri1c2e...")
    unittest.main()
