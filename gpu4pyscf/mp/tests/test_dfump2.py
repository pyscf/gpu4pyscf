# Copyright 2026 The PySCF Developers. All Rights Reserved.
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


import pytest
import unittest
import pyscf
import gpu4pyscf
import cupy as cp

# IntelliSense hinting for imported modules
import pyscf.df
import gpu4pyscf.df.int3c2e_bdiv
import gpu4pyscf.mp.dfump2


def setUpModule():
    global mol, aux, mf, mp, with_df, intopt, vhfopt
    token = """
    O    0.    0.    0.  
    H    0.94  0.    0.  
    H   -0.24  0.    0.91
    """
    mol = pyscf.gto.Mole(atom=token, basis='def2-TZVPP', max_memory=32000, spin=2, output='/dev/null', cart=False).build()
    aux = pyscf.gto.Mole(atom=token, basis='def2-TZVPP-ri', max_memory=32000, spin=2, output='/dev/null', cart=False).build()
    mol.output = aux.output = '/dev/null'
    mol.incore_anyway = True
    mf = pyscf.scf.UHF(mol).density_fit().run(conv_tol=1e-13)

    with_df = pyscf.df.DF(mol, auxbasis='def2-TZVPP-ri').build()
    mf._eri = with_df.get_ao_eri()
    mp = pyscf.mp.ump2.UMP2(mf)
    mp.kernel(with_t2=True)
    intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)


def tearDownModule():
    global mol, aux, mf, mp, intopt
    mol.stdout.close()
    aux.stdout.close()
    del mol, aux, mf, mp, intopt


class KnownValues(unittest.TestCase):
    def test_dfump2(self):
        mf_gpu = mf.to_gpu()
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run()
        print(mp.e_corr_os, mp.e_corr_ss, mp.e_corr)
        e_corr_ref = -0.2276193004374617
        self.assertAlmostEqual(mp_gpu.e_corr_os, -0.17417641954815585, 8)
        self.assertAlmostEqual(mp_gpu.e_corr_ss, -0.05344288088930613, 8)
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 8)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(j2c_decomp_alg='eig')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 8)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(j3c_backend='vhfopt')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 8)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(fp_type='FP32')
        self.assertAlmostEqual(mp_gpu.e_corr, e_corr_ref, 5)

        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(with_t2=True)
        for t2_gpu, t2_cpu in zip(mp_gpu.t2, mp.t2):
            self.assertTrue(cp.allclose(t2_gpu, t2_cpu, atol=1e-6))

    def test_dfump2_frozen(self):
        mf_gpu = mf.to_gpu()
        mp_frz = pyscf.mp.ump2.UMP2(mf).run(frozen=[1, 2])
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(frozen=mp_frz.frozen)
        self.assertAlmostEqual(mp_gpu.e_corr, -0.0637475985546614, 8)
        self.assertAlmostEqual(mp_gpu.e_corr, mp_frz.e_corr, 8)

        mp_frz = pyscf.mp.ump2.UMP2(mf).run(frozen=[[0, 1], [1, 2]])
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run(frozen=mp_frz.frozen)
        self.assertAlmostEqual(mp_gpu.e_corr, -0.08095767866785737, 8)
        self.assertAlmostEqual(mp_gpu.e_corr, mp_frz.e_corr, 8)

    def test_scf_from_gpu(self):
        mf_gpu = gpu4pyscf.scf.UHF(mol).density_fit().run(conv_tol=1e-13)
        mp_gpu = gpu4pyscf.mp.dfump2.DFUMP2(mf_gpu, auxbasis='def2-TZVPP-ri').run()
        self.assertAlmostEqual(mp_gpu.e_corr, -0.2276193004374617, 8)

    def test_to_gpu(self):
        mp_gpu = pyscf.mp.dfump2.DFUMP2(mf).to_gpu()
        self.assertTrue(isinstance(mp_gpu, gpu4pyscf.mp.dfump2.DFUMP2))
        e_gpu, _ = mp_gpu.kernel()
        e_corr_ref = -0.2276193004374617
        self.assertAlmostEqual(e_gpu, e_corr_ref, 8)
