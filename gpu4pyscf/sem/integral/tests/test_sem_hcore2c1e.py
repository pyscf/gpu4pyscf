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
from gpu4pyscf.sem.integral.hcore2c1e import (bfn, afn, 
    ovlp_in_2c1e, get_direction_cosines, calc_local_overlap, rotation_transform)
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

    def test_bfn(self):
        test_points = np.array([
            0.0, 1e-8,              # Tiny region
            0.1, 0.4, 0.6, 1.5, 2.8, # Small region (merged masks)
            3.000001, 5.0, 10.0     # Large region
        ], dtype=np.float64)

        x_gpu = cp.asarray(test_points)
        bf_gpu = bfn(x_gpu)
        fp_ref = -1707.2743320877785
        assert abs(fp(bf_gpu.get()) - fp_ref) < 1e-10

    def test_afn(self):

        def afn_cpu(p):
            from math import exp
            quo = 1.0/p
            af = np.empty(20, dtype=float)
            af[0] = quo*exp(-p)
            for n in range(1,20):
                af[n] = n*quo*af[n-1] + af[0]
            return af

        test_points = np.array([
            0.1, 0.4, 0.6, 1.5, 2.8, # Small region (merged masks)
            3.000001, 5.0, 10.0     # Large region
        ], dtype=np.float64)

        output_list = []
        for i in test_points:
            output_list.append(afn_cpu(i))
        afn_ref = np.array(output_list)
        afn_gpu = afn(cp.asarray(test_points))
        np.testing.assert_allclose(
            afn_gpu.get(),
            afn_ref,
            rtol=1e-11,
            atol=1e-10,
            err_msg="GPU results deviate from CPU reference",
        )

    def test_s1e(self):
        na_list = []
        nb_list = []
        la_list = []
        lb_list = []
        m_list = []
        ua_list = []
        ub_list = []
        r_list = []

        for ina in range(1,7):
            for inb in range(1,7):
                for ila in range(0, 3):
                    for ilb in range(0, 3):
                        for m in range(abs(ila-ilb), max(ila,ilb)+1):
                            na_list.append(ina)
                            nb_list.append(inb)
                            la_list.append(ila)
                            lb_list.append(ilb)
                            m_list.append(m)
                            ua_list.append(1.3)
                            ub_list.append(2.3)
                            r_list.append(1.5) # 1.5 A
        na_array = np.array(na_list, dtype=np.int32)
        nb_array = np.array(nb_list, dtype=np.int32)
        la_array = np.array(la_list, dtype=np.int32)
        lb_array = np.array(lb_list, dtype=np.int32)
        m_array = np.array(m_list, dtype=np.int32)
        ua_array = np.array(ua_list)
        ub_array = np.array(ub_list)
        r_array = np.array(r_list)
        # reference from yunze qiu's code
        ref_fp = 0.3655446289482449

        h2e = ovlp_in_2c1e(cp.asarray(na_array), cp.asarray(nb_array),
            cp.asarray(la_array), cp.asarray(lb_array), cp.asarray(m_array),
            cp.asarray(ua_array), cp.asarray(ub_array), cp.asarray(r_array/BOHR)) # INPUT is BOHR
        gpu_fp = fp(h2e.get())

        assert abs(ref_fp - gpu_fp) < 1e-12

    def test_get_direction_cosines(self):
        rij_vec = np.array([[1.2,2.3,3.4]])
        output = get_direction_cosines(rij_vec)

        ref_fp = 2.0543587916601833
        output_fp = fp(output[0].get())

        assert np.abs(ref_fp - output_fp).max() < 1.0E-12
        
    def test_local_ovlp(self):
        xj = np.array([[1.2, 1.3, 1.4], [1.5, 1.6, 1.7]]) # in A
        na_mat = cp.array([[2, 2, 0], [2, 2, 0]])
        nb_mat = cp.array([[1, 1, 0], [2, 2, 0]])
        za_exps = ((5.421751, 2.27096, 0.0), (2.047558, 1.702841, 0.0))
        zb_exps = ((1.268641, 0.0, 0.0), (2.047558, 1.702841, 0.0))
        r0 = cp.sqrt(xj[0, 0]**2 + xj[0, 1]**2 + xj[0, 2]**2)
        r1 = cp.sqrt(xj[1, 0]**2 + xj[1, 1]**2 + xj[1, 2]**2)
        r_dict = cp.array([r0, r1])/0.529177210903
        output = calc_local_overlap(na_mat, nb_mat, za_exps, zb_exps, r_dict)
        output_cpu = output.get()
        # ref from yunze qiu's code
        # It should be noted that, the BOHR is 0.529177210903 in original yunze's code
        assert np.abs(output_cpu[0, 0, 0, 0] - 0.00786591020414) < 1.0E-13
        assert np.abs(output_cpu[0, 1, 0, 0] - 0.04810704122904) < 1.0E-13
        assert np.abs(output_cpu[1, 0, 0, 0] - 0.01077318687617) < 1.0E-13
        assert np.abs(output_cpu[1, 0, 1, 0] - 0.03131644351964) < 1.0E-13
        assert np.abs(output_cpu[1, 0, 2, 0] - 0.0) < 1.0E-13
        assert np.abs(output_cpu[1, 1, 0, 0] - 0.03131644351964) < 1.0E-13
        assert np.abs(output_cpu[1, 1, 1, 0] - 0.06528009727285) < 1.0E-13
        assert np.abs(output_cpu[1, 1, 1, 1] - 0.01182581751136) < 1.0E-13

    def test_rotation_transform(self):
        xj = np.array([[1.2, 1.3, 1.4], [1.5, 1.6, 1.7]]) # in A
        na_mat = cp.array([[2, 2, 0], [2, 2, 0]])
        nb_mat = cp.array([[1, 1, 0], [2, 2, 0]])
        za_exps = ((5.421751, 2.27096, 0.0), (2.047558, 1.702841, 0.0))
        zb_exps = ((1.268641, 0.0, 0.0), (2.047558, 1.702841, 0.0))
        r0 = cp.sqrt(xj[0, 0]**2 + xj[0, 1]**2 + xj[0, 2]**2)
        r1 = cp.sqrt(xj[1, 0]**2 + xj[1, 1]**2 + xj[1, 2]**2)
        r_dict = cp.array([r0, r1])/0.529177210903
        S_local = calc_local_overlap(na_mat, nb_mat, za_exps, zb_exps, r_dict)
        C_tensor = get_direction_cosines(xj)
        output = rotation_transform(S_local, C_tensor)
        output_cpu = output.get()

        ref0_fp = -0.019398144170323862
        ref1_fp = 0.004737573186912597
        output0_fp = fp(output_cpu[0,:4,0])
        output1_fp = fp(output_cpu[1,:4,:4])

        assert np.abs(ref0_fp - output0_fp).max() < 1.0E-12
        assert np.abs(ref1_fp - output1_fp).max() < 1.0E-12

    def test_hcore2c1e_mopac(self):
        # benchmark from yunze qiu's code
        output_data = []
        for i in range(9,19):
            for j in range(i,19):
                charge = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', charge=charge)
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

                gpu_fp = fp(mol.get_hcore().get())
                output_data.append(gpu_fp)
        total_fp = fp(np.array(output_data))
        ref_fp = -63.26958713097162
        assert np.abs(ref_fp - total_fp).max() < 1.0E-12

    def test_hcore2c1e(self):
        output_data = []
        for i in range(9,19):
            for j in range(i,19):
                charge = (i + j) % 2
                mol = Mole(f'{i} 0 0 1; {j} 0 1 0', charge=charge)
                mol.build()

                gpu_fp = fp(mol.get_hcore().get())
                output_data.append(gpu_fp)
        total_fp = fp(np.array(output_data))
        ref_fp = -63.26958715423671
        assert np.abs(ref_fp - total_fp).max() < 1.0E-12

        
if __name__ == "__main__":
    print("Running tests for hcore2c1e...")
    unittest.main()
