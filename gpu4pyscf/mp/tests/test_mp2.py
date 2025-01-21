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
from functools import reduce
import numpy
import pytest
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp as mp_cpu
from gpu4pyscf import mp as mp_gpu

def setUpModule():
    global mol, mf, mf1
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mol.incore_anyway = True
    mol.max_memory = 32000
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

import pyscf
from packaging import version
pyscf_25 = version.parse(pyscf.__version__) <= version.parse('2.5.0')
class KnownValues(unittest.TestCase):
    def test_mp2(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        co = mf.mo_coeff[:,:nocc]
        cv = mf.mo_coeff[:,nocc:]
        g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)

        pt = mp_gpu.MP2(mf.to_gpu())

        emp2, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(emp2, -0.204019967288338, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.05153088565639835, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.15248908163191538, 8)
        self.assertAlmostEqual(abs(t2.get() - t2ref0).max(), 0, 8)

        pt.max_memory = 1
        pt.frozen = None
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.204019967288338, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.05153088565639835, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.15248908163191538, 8)
        self.assertAlmostEqual(abs(t2.get() - t2ref0).max(), 0, 8)

    def test_mp2_frozen(self):
        pt = mp_gpu.mp2.MP2(mf.to_gpu())
        pt.frozen = [0]
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.20168270592254167, 8)
        pt.set_frozen()
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.20168270592254167, 8)

    def test_mp2_with_df(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        mf_df = mf.density_fit('weigend')
        pt = mp_gpu.dfmp2.DFMP2(mf_df.to_gpu())
        e, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)

        pt_cpu = mp_cpu.dfmp2.DFMP2(mf_df)
        eris = mp_cpu.mp2._make_eris(pt_cpu, mo_coeff=mf.mo_coeff, ao2mofn=mf_df.with_df.ao2mo)
        g = eris.ovov.ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        e, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(e, -0.20425449198334983, 8)
        self.assertAlmostEqual(abs(t2.get() - t2ref0).max(), 0, 8)

        pt = mp_gpu.MP2(mf.density_fit('weigend').to_gpu())
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

        pt = mp_gpu.dfmp2.DFMP2(mf.density_fit('weigend').to_gpu())
        e = pt.kernel(mf.mo_energy, mf.mo_coeff)[0]
        self.assertAlmostEqual(e, -0.20425449198334983, 8)

        pt.frozen = [1]
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

        pt = mp_gpu.dfmp2.DFMP2(mf.to_gpu())
        pt.frozen = [1]
        pt.with_df = mf.to_gpu().density_fit('weigend').with_df
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

    def test_to_cpu(self):
        pt = mp_gpu.mp2.MP2(mf.to_gpu())
        e_gpu = pt.kernel()[0]
        pt = pt.to_cpu()
        e_cpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

        pt = mp_gpu.dfmp2.DFMP2(mf.to_gpu())
        e_gpu = pt.kernel()[0]
        pt = pt.to_cpu()
        e_cpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

    @pytest.mark.skipif(pyscf_25, reason='requires pyscf 2.6 or higher')
    def test_to_gpu(self):
        pt = mp_cpu.mp2.MP2(mf)
        e_cpu = pt.kernel()[0]
        pt = pt.to_gpu()
        e_gpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

        pt = mp_cpu.dfmp2.DFMP2(mf)
        e_cpu = pt.kernel()[0]
        pt = pt.to_gpu()
        e_gpu = pt.kernel()[0]
        assert abs(e_cpu - e_gpu) < 1e-6

if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()
