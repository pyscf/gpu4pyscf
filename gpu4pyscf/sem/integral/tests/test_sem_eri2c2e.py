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
from pyscf.data.nist import BOHR
from pyscf.lib import fp
from gpu4pyscf.sem.integral.eri_2c2e import (multipole_eval, a_function_ijl, solve_poij,
    calc_aij_tensor, debug_rijkl, calc_multipole_scaling_params, calc_local_rep_core,
    global_transform_gpu, build_eri2c2e)
from gpu4pyscf.sem.integral import eri_2c2e
from gpu4pyscf.sem.gto.params import load_sem_params
from gpu4pyscf.sem.gto.mole import Mole


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
        mol = Mole("He 0 0 0; F 0 0 1; Cl 0 0 2; As 0 0 3;", charge=1)
        mol.build()
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
                        mol.topology,
                        mol.atom_ids_0based
                    )
        assert np.abs(output[:,:,0].get() - 0).max() < 1e-13
        assert np.abs(output[:,:,1].get() - ref_8).max() < 1e-13
        assert np.abs(output[:,:,2].get() - ref_16).max() < 1e-13
        assert np.abs(output[:,:,3].get() - ref_32).max() < 1e-13

    def test_sem_rijkl(self):
        import random
        random.seed(42)
        np.random.seed(42)

        n_atom = 50

        po_ss = np.random.random(n_atom)
        po_sp = np.random.random(n_atom)
        po_pp = np.random.random(n_atom)
        po_sd = np.random.random(n_atom)
        po_pd = np.random.random(n_atom)
        po_dd = np.random.random(n_atom)
        po_pp_mono = np.random.random(n_atom)
        po_dd_mono = np.random.random(n_atom)
        core_rho = np.random.random(n_atom)

        ddp_sp = np.random.random(n_atom)
        ddp_pp = np.random.random(n_atom)
        ddp_sd = np.random.random(n_atom)
        ddp_pd = np.random.random(n_atom)
        ddp_dd = np.random.random(n_atom)

        po_tensor = np.zeros((3, 3, 3, n_atom))
        ddp_tensor = np.zeros((3, 3, n_atom))

        po_tensor[0, 0, 0, :] = po_ss
        po_tensor[0, 1, 1, :] = po_tensor[0, 1, 2, :] = po_sp
        ddp_tensor[0, 1, :] = ddp_sp

        po_tensor[1, 1, 0, :] = po_pp_mono
        po_tensor[1, 1, 1, :] = po_tensor[1, 1, 2, :] = po_pp
        ddp_tensor[1, 1, :] = ddp_pp

        po_tensor[0, 2, 1, :] = po_tensor[0, 2, 2, :] = po_sd
        ddp_tensor[0, 2, :] = ddp_sd

        po_tensor[1, 2, 1, :] = po_tensor[1, 2, 2, :] = po_pd
        ddp_tensor[1, 2, :] = ddp_pd

        po_tensor[2, 2, 0, :] = po_dd_mono
        po_tensor[2, 2, 1, :] = po_tensor[2, 2, 2, :] = po_dd
        ddp_tensor[2, 2, :] = ddp_dd

        for i in range(3):
            for j in range(i + 1, 3):
                po_tensor[j, i, :, :] = po_tensor[i, j, :, :]
                ddp_tensor[j, i, :] = ddp_tensor[i, j, :]

        lorb = [0, 1, 1, 1, 2, 2, 2, 2, 2] # s, px, py, pz, d...
        ij_to_lilj = {}
        idx = 0
        for i in range(9):
            for j in range(i + 1):
                ij_to_lilj[idx] = (lorb[i], lorb[j])
                idx += 1
        
        ni_list = []
        nj_list = []
        ij_list = []
        kl_list = []
        li_list = []
        lj_list = []
        lk_list = []
        ll_list = []
        ic_list = []
        r_list = []
        for _ in range(50):
            ni = random.randint(0, n_atom - 1)
            nj = random.randint(0, n_atom - 1)
            
            ij = random.randint(0, 44)
            ij_list.append(ij)
            li, lj = ij_to_lilj[ij]
            
            kl = random.randint(0, 44)
            kl_list.append(kl)
            lk, ll = ij_to_lilj[kl]
            
            ic = random.randint(0, 2)
            r = random.uniform(0.5, 2.0)
            ni_list.append(ni)
            nj_list.append(nj)
            li_list.append(li)
            lj_list.append(lj)
            lk_list.append(lk)
            ll_list.append(ll)
            ic_list.append(ic)
            r_list.append(r)
        d_ch = compute_single_multipole_angular_factors()
        val_new = debug_rijkl(
                cp.array(ni_list), cp.array(nj_list), 
                cp.array(ij_list), cp.array(kl_list), 
                cp.array(li_list), cp.array(lj_list), 
                cp.array(lk_list), cp.array(ll_list), 
                cp.array(ic_list), cp.array(r_list), 
                cp.asarray(po_tensor), cp.asarray(ddp_tensor), cp.asarray(core_rho), d_ch
            )
        ref = np.array([ 1.994855709852433e-05,  0.000000000000000e+00,
            1.134353137393831e-02,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            -4.504124508308797e-04,  1.436381846400173e-03,
            0.000000000000000e+00,  0.000000000000000e+00,
            5.788165620091654e-02,  0.000000000000000e+00,
            -5.908478927452386e-06,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            4.317660617199628e-02, -7.540034891197262e-06,
            2.170010753122021e-02,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  2.331397168949274e-01,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            -5.809689192389939e-02,  0.000000000000000e+00,
            -5.467598605319420e-04,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00,
            0.000000000000000e+00,  0.000000000000000e+00])
        assert np.abs(val_new.get() - ref).max() < 1e-13

    def test_calc_multipole_scaling_params(self):
        # test a pseudo molecule.
        mol = Mole("H 0 0 0; He 0 0 1; Li 0 0 2; Be 0 0 3; B 0 0 4; B 0 0 5")
        mol.build()
        am, ad, aq, dd, qq = calc_multipole_scaling_params(
                mol.one_center_integrals, mol.topology, mol.atom_ids_0based)
        ref_am = cp.array([0.5309794168288758, 0.3471083359963919, 0.4055621018435662,
            0.2775604275255757, 0.3005852375935441, 0.3005852375935441])
        ref_ad = cp.array([0.5309794168288758, 0.5756121484302844, 0.844494230338215 ,
            0.4062418667876224, 0.4282415831723232, 0.4282415831723232])
        ref_aq = cp.array([0.5309794168288758, 0.976860735176923 , 1.176211613778141 ,
            0.3118239061299627, 0.6170022380603419, 0.6170022380603419])
        ref_dd = cp.array([0.                , 0.2475819099414987, 0.3558536654003256,
            1.1578786254010267, 0.9214782547949236, 0.9214782547949236])
        ref_qq = cp.array([0.                , 0.2118043476246238, 0.4146834870436352,
            0.9594652130351418, 0.8279806728602983, 0.8279806728602983])
        assert np.abs(am - ref_am).max() < 1e-14
        assert np.abs(ad - ref_ad).max() < 1e-14
        assert np.abs(aq - ref_aq).max() < 1e-14
        assert np.abs(dd - ref_dd).max() < 1e-14
        assert np.abs(qq - ref_qq).max() < 1e-14

    def test_calc_local_rep_core(self):
        idx0 = 5
        idx1 = 7
        pair_i = cp.array([0], dtype=np.int32)
        pair_j = cp.array([1], dtype=np.int32)
        ele_id = cp.array([idx0, idx1], dtype=np.int32)
        r_vec  = cp.array([2.05], dtype=np.float64)
        mol_am = cp.array([0.490071284110568, 0.415415881345135])
        mol_ad = cp.array([0.38704355457748 , 1.684325809335958])
        mol_aq = cp.array([0.655585969543187, 1.093542208483153])
        mol_dd = cp.array([0.753564251039701, 0.237113065251533])
        mol_qq = cp.array([0.719236189046182, 0.539307108619962])
        po_tensor = cp.zeros((3, 3, 3, 2))
        v1 = [1.020259738146974, 1.203613107859472]
        v2 = [1.291844274595479, 0.296854680506929]
        v3 = [0.762676480627553, 0.457229722018273]
        po_tensor[0, 0, 0] = cp.array(v1)
        po_tensor[0, 1, 1] = cp.array(v2)
        po_tensor[1, 0, 1] = cp.array(v2)
        po_tensor[1, 1, 0] = cp.array(v1)
        po_tensor[1, 1, 2] = cp.array(v3)
        ddp_tensor = cp.array([[[0.               , 0.               ],
            [0.753564251039701, 0.237113065251533],
            [0.               , 0.               ]],

            [[0.753564251039701, 0.237113065251533],
            [1.017153573098649, 0.76269542729457 ],
            [0.               , 0.               ]],

            [[0.               , 0.               ],
            [0.               , 0.               ],
            [0.               , 0.               ]]])
        mol_core_rho = cp.array([1.020259738146974, 1.203613107859472])
        tore = cp.array([4, 6], dtype=np.int32)
        natorb = cp.array([4, 4], dtype=np.int32)
        ch_gpu = cp.zeros((45, 3, 5))
        non_zero_elements = [
            ((0, 0, 2), 1.0),
            ((1, 1, 2), 1.0),
            ((2, 1, 3), 1.0),
            ((3, 1, 1), 1.0),
            ((4, 2, 2), 1.15470054),
            ((5, 2, 3), 1.0),
            ((6, 2, 1), 1.0),
            ((7, 2, 4), 1.0),
            ((8, 2, 0), 1.0),
            ((9, 0, 2), 1.0),
            ((9, 2, 2), 1.33333333),
            ((10, 2, 3), 1.0),
            ((11, 2, 1), 1.0),
            ((12, 1, 2), 1.15470054),
            ((13, 1, 3), 1.0),
            ((14, 1, 1), 1.0),
            ((17, 0, 2), 1.0),
            ((17, 2, 2), -0.66666667),
            ((17, 2, 4), 1.0),
            ((18, 2, 0), 1.0),
            ((19, 1, 3), -0.57735027),
            ((20, 1, 2), 1.0),
            ((22, 1, 3), 1.0),
            ((23, 1, 1), 1.0),
            ((24, 0, 2), 1.0),
            ((24, 2, 2), -0.66666667),
            ((24, 2, 4), -1.0),
            ((25, 1, 1), -0.57735027),
            ((27, 1, 2), 1.0),
            ((28, 1, 1), -1.0),
            ((29, 1, 3), 1.0),
            ((30, 0, 2), 1.0),
            ((30, 2, 2), 1.33333333),
            ((31, 2, 3), 0.57735027),
            ((32, 2, 1), 0.57735027),
            ((33, 2, 4), -1.15470054),
            ((34, 2, 0), -1.15470054),
            ((35, 0, 2), 1.0),
            ((35, 2, 2), 0.66666667),
            ((35, 2, 4), 1.0),
            ((36, 2, 0), 1.0),
            ((37, 2, 3), 1.0),
            ((38, 2, 1), 1.0),
            ((39, 0, 2), 1.0),
            ((39, 2, 2), 0.66666667),
            ((39, 2, 4), -1.0),
            ((40, 2, 1), -1.0),
            ((41, 2, 3), 1.0),
            ((42, 0, 2), 1.0),
            ((42, 2, 2), -1.33333333),
            ((44, 0, 2), 1.0),
            ((44, 2, 2), -1.33333333)
            ]
        for indices, value in non_zero_elements:
            ch_gpu[indices] = value
        dorbs = cp.array([False, False])
        task_arrays = (eri_2c2e.TASK_ACTION_GPU, eri_2c2e.TASK_TARGET_GPU, 
                eri_2c2e.TASK_IJ_GPU, eri_2c2e.TASK_KL_GPU, 
                eri_2c2e.TASK_LI_GPU, eri_2c2e.TASK_LJ_GPU, 
                eri_2c2e.TASK_LK_GPU, eri_2c2e.TASK_LL_GPU)
        rep_out, core_out, gab_out = eri_2c2e.calc_local_rep_core(pair_i, pair_j, ele_id, r_vec, 
            mol_am, mol_ad, mol_aq, mol_dd, mol_qq, 
            po_tensor, ddp_tensor, mol_core_rho, ch_gpu, 
            tore, natorb, dorbs, 
            task_arrays)
        
        gab_ref = 8.996735808633392
        rep_ref = np.array([ 8.996735808633392, -0.916579168516647,  9.425981045491742,
            8.564484909535242,  8.564484909535242,  1.213230611010226,
            -0.210941470258991,  1.046806925017705,  0.99008128869159 ,
            0.99008128869159 ,  9.209415694893828, -1.013429619998259,
            8.990548840068326,  8.456467096843134,  8.456467096843134,
            0.24433704373884 , -0.201756546960697,  0.28058061704789 ,
            -0.280605797362601,  8.48006230380868 , -0.704413518013866,
            8.659161605659385,  8.249728058559436,  8.136327566319833,
            0.24433704373884 , -0.201756546960697,  0.28058061704789 ,
            -0.280605797362601,  0.056700246119801,  8.48006230380868 ,
            -0.704413518013866,  8.659161605659385,  8.136327566319833,
            8.249728058559436])

        assert np.abs(core_out[0, 4:].get()).max() < 1.0E-14
        assert np.abs(gab_out.get() - gab_ref).max() < 1.0E-14
        assert np.abs(rep_out.get()[0,:34] - rep_ref).max() < 1.0E-14
    
    def test_global_transform(self):
        mol = Mole("S 0 0 0; Cl 1.10 0.5 0.2", spin=1)
        mol.build()
        ch_matrix = compute_single_multipole_angular_factors() 
        original_get_parameter = mol.params.get_parameter

        def fake_get_parameter(name, *args, **kwargs):
            if name == 'multipole_angular_factors':
                if kwargs.get('to_gpu', False):
                    return cp.asarray(ch_matrix)
                return ch_matrix
            
            return original_get_parameter(name, *args, **kwargs)

        with patch.object(mol.params, 'get_parameter', side_effect=fake_get_parameter):

            w, e1b, e2a, enuc = build_eri2c2e(mol)

            w_ref_sum = 765.5732131377281
            enuc_ref = 348.9729772632329
            e1b_fp_ref = -5.294466310759472
            e2a_fp_ref = -37.9762130220567

            assert np.abs(np.sum(w.get()) - w_ref_sum) < 1.0E-12
            assert cp.abs(fp(e1b.get()) - e1b_fp_ref) < 1.0E-12
            assert cp.abs(fp(e2a.get()) - e2a_fp_ref) < 1.0E-12
            assert cp.abs(enuc - enuc_ref).max() < 1.0E-12

    def test_w_mopac(self):
        w_data = []
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
                w_data.append(fp(mol.two_center_integrals.w.get()))

        ref_fp = np.array([ -6.43980217011038 ,  -7.051317374611991,  -3.671268184840752,
            -5.2502169345725  ,  -4.9620780994001  ,  -3.45179642606731 ,
            -5.803616674288176,  -5.994724394119057,  -6.031992329755209,
            -6.935813458820498,  -7.82604401281199 ,  -3.960028030057424,
            -5.776609643102629,  -5.071832139840405,  -3.683895599189428,
            -5.923470141194815,  -6.458473083151411,  -6.247543162915174,
            -7.688833729273631,  -3.712686678027008,  -5.194549386208714,
            -9.795409118712247,  -5.408740901501991, -10.710925765137244,
            -10.592475519919835, -11.54656438448019 ,  -7.107382479622645,
            -6.346574381181104, -11.587880067996998,  -5.88398462988605 ,
            -12.898519610428334, -12.574849349994793, -13.750963002939391,
            -9.310989139764041, -61.158010965816466, -34.853291736792045,
            -71.99967871940949 , -66.80600167685198 , -64.24595064359323 ,
            -1.638841218083287, -32.30146488277617 , -51.862745565820184,
            -51.025838985787686, -50.029638310386964,  -3.768118288827674,
            -73.20402460818245 , -67.46885064262953 , -66.62888105398254 ,
            -2.554963411717122, -66.41253120220155 , -63.587515344625885,
            -5.317056695734674, -68.89495034975631 ,  -4.684237265405856,
            -8.437068078977768])
        assert np.abs(np.array(w_data) - ref_fp).max() < 1.0E-12

    def test_w(self):
        w_data = []
        for i in range(9,19):
            for j in range(i,19):
                spin = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', spin=spin, verbose=0)
                mol.build()
                w_data.append(fp(mol.two_center_integrals.w.get()))
        ref_fp = np.array([-6.439802170110388, -7.051317374611998, -3.671268184840747, 
            -5.250216934572502, -4.962078095666421, -3.451796425119889, 
            -5.8036166709585615, -5.994724394371016, -6.031992327034467, 
            -6.935813458820508, -7.826044012811992, -3.96002803005742, 
            -5.776609643102634, -5.071832135270871, -3.6838955981420556, 
            -5.923470136946263, -6.458473082876013, -6.247543159490855, 
            -7.688833729273631, -3.7126866780270054, -5.194549386208721,
            -9.79540912331636, -5.408740902429464, -10.710925769703561, 
            -10.592475523346703, -11.54656438883279, -7.107382479622651, 
            -6.346574381181103, -11.587880071257405, -5.88398463031342, 
            -12.898519614145721, -12.574849353219376, -13.750963006514203, 
            -9.310989139764043, -61.15801088642186, -34.85329172503441, 
            -71.99967861862422, -66.80600163171182, -64.24595056213516, 
            -1.6388412212822823, -32.30146488081963, -51.86274554663286, 
            -51.0258389734551, -50.029638295218, -3.7681182889082523, 
            -73.2040245139079, -67.46885059914581, -66.6288809774995, 
            -2.554963414618908, -66.41253119383835, -63.587515311101214, 
            -5.317056696461119, -68.89495027850253, -4.684237266703826, 
            -8.43706807897778])
        assert np.abs(np.array(w_data) - ref_fp).max() < 1.0E-12

    def test_enuc_corr(self):
        mol_list = [
            'O 0.0000 0.0000 0.0000; H 0.7570 0.5860 0.0000',
            'C 0.0000 0.0000 0.0000; H 0.7570 0.5860 0.0000',
            'N 0.0000 0.0000 0.0000; H 0.7570 0.5860 0.0000',
            'C 0.0000 0.0000 0.0000; C 0.7570 0.5860 0.0000',
            'O 0.0000 0.0000 0.0000; Si 0.7570 0.5860 0.0000',
        ]
        charge_list = [
            1, 1, 0, 0, 0
        ]
        ref_enuc_list = [ # eV
            65.2464832302607,
            47.832267184627206,
            57.078102938548284,
            186.9787103939841,
            195.90025256296212
        ]
        for i in range(len(mol_list)):
            mol = Mole(mol_list[i], charge=charge_list[i], verbose=0)
            mol.build()
            assert cp.abs(mol.enuc - ref_enuc_list[i]).max() < 1.0E-13


if __name__ == "__main__":
    print("Running tests for eri2c2e...")
    unittest.main()
