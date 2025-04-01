# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib, gto
from gpu4pyscf.tdscf import rhf, rks
from gpu4pyscf import tdscf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = [
            ['O', (0. , 0., 0.)],
            ['H', (0. , -0.757, 0.587)],
            ['H', (0. , 0.757, 0.587)], ]
        mol.basis = 'def2svp'
        cls.mol = mol.build()

        cls.mf = mf = mol.RHF().PCM().to_gpu()
        cls.mf.with_solvent.method = 'C-PCM'
        cls.mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        cls.mf.with_solvent.eps = 78
        cls.mf.with_solvent.eps_optical = 1.78
        cls.mf = mf.run(conv_tol=1e-10)

        cls.mfu = mfu = mol.UHF().PCM().to_gpu()
        cls.mfu.with_solvent.method = 'C-PCM'
        cls.mfu.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        cls.mfu.with_solvent.eps = 78
        cls.mfu.with_solvent.eps_optical = 1.78
        cls.mfu = mfu.run(conv_tol=1e-10)

        mf_b3lyp_nodf = mol.RKS().PCM().to_gpu()
        mf_b3lyp_nodf.xc = 'b3lyp'
        mf_b3lyp_nodf.grids.atom_grid = (99,590)
        mf_b3lyp_nodf.with_solvent.method = 'C-PCM'
        mf_b3lyp_nodf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf.with_solvent.eps = 78
        mf_b3lyp_nodf.with_solvent.eps_optical = 1.78
        mf_b3lyp_nodf.cphf_grids = mf_b3lyp_nodf.grids
        cls.mf_b3lyp_nodf = mf_b3lyp_nodf.run(conv_tol=1e-10)

        mf_b3lyp_nodf_u = mol.UKS().PCM().to_gpu()
        mf_b3lyp_nodf_u.xc = 'b3lyp'
        mf_b3lyp_nodf_u.grids.atom_grid = (99,590)
        mf_b3lyp_nodf_u.with_solvent.method = 'C-PCM'
        mf_b3lyp_nodf_u.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf_u.with_solvent.eps = 78
        mf_b3lyp_nodf_u.with_solvent.eps_optical = 1.78
        mf_b3lyp_nodf_u.cphf_grids = mf_b3lyp_nodf_u.grids
        cls.mf_b3lyp_nodf_u = mf_b3lyp_nodf_u.run(conv_tol=1e-10)

        mf_b3lyp_nodf_iefpcm = mol.RKS().PCM().to_gpu()
        mf_b3lyp_nodf_iefpcm.xc = 'b3lyp'
        mf_b3lyp_nodf_iefpcm.grids.atom_grid = (99,590)
        mf_b3lyp_nodf_iefpcm.with_solvent.method = 'IEF-PCM'
        mf_b3lyp_nodf_iefpcm.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf_iefpcm.with_solvent.eps = 78
        mf_b3lyp_nodf_iefpcm.with_solvent.eps_optical = 1.78
        mf_b3lyp_nodf_iefpcm.cphf_grids = mf_b3lyp_nodf_iefpcm.grids
        cls.mf_b3lyp_nodf_iefpcm = mf_b3lyp_nodf_iefpcm.run(conv_tol=1e-10)

        mf_b3lyp_nodf_iefpcm_u = mol.RKS().PCM().to_gpu()
        mf_b3lyp_nodf_iefpcm_u.xc = 'b3lyp'
        mf_b3lyp_nodf_iefpcm_u.grids.atom_grid = (99,590)
        mf_b3lyp_nodf_iefpcm_u.with_solvent.method = 'IEF-PCM'
        mf_b3lyp_nodf_iefpcm_u.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf_iefpcm_u.with_solvent.eps = 78
        mf_b3lyp_nodf_iefpcm_u.with_solvent.eps_optical = 1.78
        mf_b3lyp_nodf_iefpcm_u.cphf_grids = mf_b3lyp_nodf_iefpcm_u.grids
        cls.mf_b3lyp_nodf_iefpcm_u = mf_b3lyp_nodf_iefpcm.run(conv_tol=1e-10)

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_CPCM(self):
        """
        $rem
        JOBTYPE              sp          
        METHOD               hf     
        BASIS                def2-svp             
        CIS_N_ROOTS          5       
        CIS_SINGLETS         TRUE        
        CIS_TRIPLETS         FALSE       
        SYMMETRY             FALSE       
        SYM_IGNORE           TRUE   
        ! RPA 2               # whether tddft or tda
        BASIS_LIN_DEP_THRESH 12
        SOLVENT_METHOD PCM
        $end

        $PCM
        Theory CPCM
        HeavyPoints 302
        HPoints 302
        $end

        $solvent
        dielectric 78
        $end
        """
        mf = self.mf
        td = mf.TDHF()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-75.61072291, -75.54419399, -75.51949191, -75.45219025, -75.40975027])
        assert np.allclose(es_gound, ref)

        td = mf.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-75.60864828, -75.54169327, -75.51738767, -75.44915784, -75.40839714])
        assert np.allclose(es_gound, ref)

    def test_b3lyp_CPCM(self):
        mf = self.mf_b3lyp_nodf
        td = mf.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06898428, -75.99630982, -75.98765186, -75.91045133, -75.84783748])
        assert np.allclose(es_gound, ref)

        td = mf.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06789176, -75.99609709, -75.98589720, -75.90894600, -75.84699115])
        assert np.allclose(es_gound, ref)

    def test_b3lyp_IEFPCM(self):
        mf = self.mf_b3lyp_nodf_iefpcm
        td = mf.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06881645, -75.99631929, -75.98713725, -75.91015704, -75.84668800])
        assert np.allclose(es_gound, ref)

        td = mf.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06773319, -75.99610928, -75.98534912, -75.90861455, -75.84576041])
        assert np.allclose(es_gound, ref)

    def test_unrestricted_hf_CPCM(self):
        mf = self.mfu
        td = mf.TDHF()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-75.64482315, -75.61072291, -75.57156784, -75.56769949, -75.54419399])
        assert np.allclose(es_gound, ref)

    def test_unrestricted_b3lyp_CPCM(self):
        mf = self.mf_b3lyp_nodf_u
        td = mf.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.09301571, -76.06898428, -76.01822101, -76.01369024, -75.99630982])
        assert np.allclose(es_gound, ref)


if __name__ == "__main__":
    print("Full Tests for PCM TDDFT")
    unittest.main()