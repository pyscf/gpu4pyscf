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
from gpu4pyscf.solvent.tdscf import pcm as pcm_td
from gpu4pyscf.grad.tests.test_tdrhf_grad import diagonalize, diagonalize_tda
from gpu4pyscf.grad.tests.test_tduhf_grad import diagonalize as diagonalize_u


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

        mf = self.mol.RHF().PCM().to_gpu()
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf.with_solvent.eps = 78
        mf = mf.run(conv_tol=1e-10)
        td = mf.TDHF(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-75.61072291, -75.54419399, -75.51949191, -75.45219025, -75.40975027])
        assert np.allclose(es_gound, ref)

        a, b = td.get_ab()
        es_get_ab = diagonalize(a, b)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

        td = mf.TDA(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-75.60864828, -75.54169327, -75.51738767, -75.44915784, -75.40839714])
        assert np.allclose(es_gound, ref)

        es_get_ab = diagonalize_tda(a)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

    def test_b3lyp_CPCM(self):
        mf_b3lyp_nodf = self.mol.RKS().PCM().to_gpu()
        mf_b3lyp_nodf.xc = 'b3lyp'
        mf_b3lyp_nodf.grids.atom_grid = (99,590)
        mf_b3lyp_nodf.with_solvent.method = 'C-PCM'
        mf_b3lyp_nodf.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf.with_solvent.eps = 78
        mf_b3lyp_nodf.cphf_grids = mf_b3lyp_nodf.grids
        mf = mf_b3lyp_nodf.run(conv_tol=1e-10)
        td = mf.TDDFT(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06898428, -75.99630982, -75.98765186, -75.91045133, -75.84783748])
        assert np.allclose(es_gound, ref)

        a, b = td.get_ab()
        es_get_ab = diagonalize(a, b)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

        td = mf.TDA(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06789176, -75.99609709, -75.98589720, -75.90894600, -75.84699115])
        assert np.allclose(es_gound, ref)

        es_get_ab = diagonalize_tda(a)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

    def test_b3lyp_IEFPCM(self):
        mf_b3lyp_nodf_iefpcm = self.mol.RKS().PCM().to_gpu()
        mf_b3lyp_nodf_iefpcm.xc = 'b3lyp'
        mf_b3lyp_nodf_iefpcm.grids.atom_grid = (99,590)
        mf_b3lyp_nodf_iefpcm.with_solvent.method = 'IEF-PCM'
        mf_b3lyp_nodf_iefpcm.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf_iefpcm.with_solvent.eps = 78
        mf_b3lyp_nodf_iefpcm.cphf_grids = mf_b3lyp_nodf_iefpcm.grids
        mf = mf_b3lyp_nodf_iefpcm.run(conv_tol=1e-10)
        td = mf.TDDFT(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06881645, -75.99631929, -75.98713725, -75.91015704, -75.84668800])
        assert np.allclose(es_gound, ref)

        a, b = td.get_ab()
        es_get_ab = diagonalize(a, b)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

        td = mf.TDA(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.06773319, -75.99610928, -75.98534912, -75.90861455, -75.84576041])
        assert np.allclose(es_gound, ref)

        es_get_ab = diagonalize_tda(a)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

    def test_unrestricted_hf_CPCM(self):
        mfu = self.mol.UHF().PCM().to_gpu()
        mfu.with_solvent.method = 'C-PCM'
        mfu.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mfu.with_solvent.eps = 78
        mf = mfu.run(conv_tol=1e-10)
        td = mf.TDHF(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-75.64482315, -75.61072291, -75.57156784, -75.56769949, -75.54419399])
        assert np.allclose(es_gound, ref)

        a, b = td.get_ab()
        es_get_ab = diagonalize_u(a, b)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

    def test_unrestricted_b3lyp_CPCM(self):
        mf_b3lyp_nodf_u = self.mol.UKS().PCM().to_gpu()
        mf_b3lyp_nodf_u.xc = 'b3lyp'
        mf_b3lyp_nodf_u.grids.atom_grid = (99,590)
        mf_b3lyp_nodf_u.with_solvent.method = 'C-PCM'
        mf_b3lyp_nodf_u.with_solvent.lebedev_order = 29 # 302 Lebedev grids
        mf_b3lyp_nodf_u.with_solvent.eps = 78
        mf_b3lyp_nodf_u.cphf_grids = mf_b3lyp_nodf_u.grids
        mf = mf_b3lyp_nodf_u.run(conv_tol=1e-10)
        td = mf.TDDFT(equilibrium_solvation=False)
        es = td.kernel(nstates=5)[0]
        es_gound = es + mf.e_tot
        ref = np.array([-76.09301571, -76.06898428, -76.01822101, -76.01369024, -75.99630982])
        assert np.allclose(es_gound, ref)

        a, b = td.get_ab()
        es_get_ab = diagonalize_u(a, b)[0]
        assert np.linalg.norm(es_get_ab - es) < 1e-8

    def test_eps_optical_from_solvent_name(self):
        # The solvent name determines eps and eps_optical (issue #867).
        mf = self.mol.RHF().to_gpu().PCM('toluene')
        self.assertAlmostEqual(mf.with_solvent.get_eps(), 2.3741, 12)
        td = mf.TDA(equilibrium_solvation=False)
        self.assertAlmostEqual(td.with_solvent.eps, 1.4961**2, 12)


class SolventTDResponse_high_cost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        molecule = gto.M(atom='He 0 0 0; H .13 .07 .76', charge=1,
                         basis='6-31g', verbose=0)
        cpu_mf = molecule.RHF().PCM()
        cpu_mf.with_solvent.method = 'C-PCM'
        cpu_mf.with_solvent.eps = 10
        cpu_mf.with_solvent.lebedev_order = 7
        cpu_mf.conv_tol = 1e-12
        cpu_mf.conv_tol_grad = 1e-10
        cpu_mf.kernel()
        assert cpu_mf.converged
        cpu_td = cpu_mf.TDA(equilibrium_solvation=True).set(
            nstates=3, conv_tol=1e-12, lindep=1e-24)
        cpu_td.kernel()
        assert np.all(cpu_td.converged)
        # Current local PySCF lacks the old PCM-TD derivative conversion classes.
        # Use the supported SCF conversion and native GPU TD factory instead.
        gpu_mf = cpu_mf.to_gpu()
        np.testing.assert_allclose(gpu_mf.mo_coeff.get(), cpu_mf.mo_coeff, atol=0, rtol=0)
        np.testing.assert_allclose(gpu_mf.energy_tot(dm=gpu_mf.make_rdm1()),
                                   cpu_mf.e_tot, atol=1e-11, rtol=0)
        gpu_td = gpu_mf.TDA(equilibrium_solvation=True).set(
            nstates=3, conv_tol=1e-12, lindep=1e-24)
        gpu_td.kernel()
        assert isinstance(gpu_td, pcm_td.WithSolventTDSCF)
        assert np.all(gpu_td.converged)
        np.testing.assert_allclose(gpu_td.e, cpu_td.e, atol=1e-10, rtol=0)
        cls.small_pcm_td_reference = gpu_td

    @classmethod
    def tearDownClass(cls):
        del cls.small_pcm_td_reference

    def test_pcm_derivative_undo_and_reapply(self):
        for method in ['Gradients', 'nac_method']:
            with self.subTest(method=method):
                small_pcm_td_reference = self.small_pcm_td_reference
                td = small_pcm_td_reference
                original = getattr(td, method)()
                original.cphf_conv_tol = 1e-12
                arguments = {'state': 1} if method == 'Gradients' else {'states': (0, 1)}
                values = original.kernel(**arguments)
                if method == 'Gradients':
                    values = (values,)
                reference = [cp.asnumpy(value).copy() for value in values]
                assert all(value.shape == (td.mol.natm, 3) for value in reference)
                assert np.linalg.norm(reference[0]) > 1e-5
                assert all(np.isfinite(value).all() for value in reference)
                original_class = type(original)
                original_keys = set(original.__dict__)
                original_pcm = td.with_solvent
                original_xy = [(x.copy(), y) for x, y in td.xy]
                assert not hasattr(original, 'with_solvent')

                bare = original.undo_solvent()
                plain = getattr(td.undo_solvent(), method)()
                assert type(bare) is type(plain)
                assert bare is not original
                assert bare.base is td
                assert type(original) is original_class
                assert set(original.__dict__) == original_keys
                assert td.with_solvent is original_pcm
                density = td._scf.make_rdm1()
                assert bare.solvent_response(density) == 0
                assert float(cp.linalg.norm(original.solvent_response(density))) > 1e-5

                reapplied = getattr(bare.base, method)()
                reapplied.cphf_conv_tol = original.cphf_conv_tol
                assert type(reapplied) is original_class
                result = reapplied.kernel(**arguments)
                if method == 'Gradients':
                    result = (result,)
                for actual, expected in zip(result, reference):
                    np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-10, rtol=1e-10)
                for (x, _), (old_x, _) in zip(td.xy, original_xy):
                    np.testing.assert_allclose(x.get(), old_x.get(), atol=0, rtol=0)
                assert original.base is td and td.with_solvent is original_pcm

    def test_pcm_triplet_response_matches_same_orbital_vacuum(self):
        small_pcm_td_reference = self.small_pcm_td_reference
        mf = small_pcm_td_reference._scf
        triplet = mf.TDA(equilibrium_solvation=True).set(
            singlet=False, nstates=3, conv_tol=1e-12, lindep=1e-24)
        vacuum = mf.undo_solvent().TDA().set(
            singlet=False, nstates=3, conv_tol=1e-12, lindep=1e-24)
        np.testing.assert_allclose(vacuum._scf.mo_coeff.get(), mf.mo_coeff.get(), atol=0, rtol=0)
        co = mf.mo_coeff[:, mf.mo_occ > 0]
        cv = mf.mo_coeff[:, mf.mo_occ == 0]
        x = small_pcm_td_reference.xy[0][0]
        density = co @ x @ cv.T
        density = density+density.T
        pcm_response = triplet.gen_response(singlet=False, hermi=0)(density)
        gas_response = vacuum.gen_response(singlet=False, hermi=0)(density)
        np.testing.assert_allclose(pcm_response.get(), gas_response.get(), atol=1e-11, rtol=0)
        triplet.kernel()
        vacuum.kernel()
        assert np.all(triplet.converged) and np.all(vacuum.converged)
        np.testing.assert_allclose(triplet.e, vacuum.e, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(triplet.oscillator_strength(), np.zeros(3))


if __name__ == "__main__":
    print("Full Tests for PCM TDDFT")
    unittest.main()
