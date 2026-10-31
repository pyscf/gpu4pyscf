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
from pyscf import lib, gto, scf
from gpu4pyscf import tdscf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol.basis = '631g'
        mol.symmetry = True
        cls.mol = mol.build()
        cls.mf = scf.RHF(mol).to_gpu().run()
        cls.df_mf = scf.RHF(mol).to_gpu().density_fit().run()
        cls.nstates = 5 # make sure first 3 states are converged

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_tda_singlet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDA().set(nstates=nstates)
        assert td.device == 'gpu'
        e = td.kernel()[0]
        ref = [11.9027511, 11.9027511, 16.8603101]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 4)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.65616659, 5)

        df_mf = self.df_mf
        td = df_mf.TDA().set(nstates=nstates)
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.65618093, 5)

    def test_tda_triplet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDA().set(nstates=nstates)
        assert td.device == 'gpu'
        td.singlet = False
        e = td.kernel()[0]
        ref = [11.0174650, 11.0174650, 13.1694960]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 4)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

        df_mf = self.df_mf
        td = df_mf.TDA().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

    def test_tdhf_singlet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDHF().set(nstates=nstates)
        assert td.device == 'gpu'
        e = td.kernel()[0]
        ref = [11.8348584, 11.8348584, 16.6630381]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 4)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.64009191, 5)

        df_mf = self.df_mf
        td = df_mf.TDHF().set(nstates=nstates)
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(lib.fp(np.linalg.norm(dip, axis=1)), -0.64011895, 5)

    def test_tdhf_triplet(self):
        mf = self.mf
        nstates = self.nstates
        td = mf.TDHF().set(nstates=nstates)
        assert td.device == 'gpu'
        td.singlet = False
        e = td.kernel()[0]
        ref = [10.8919091, 10.8919091, 12.6343507]
        self.assertAlmostEqual(abs(e[:len(ref)] * 27.2114 - ref).max(), 0, 4)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

        df_mf = self.df_mf
        td = df_mf.TDHF().set(nstates=nstates)
        td.singlet = False
        e = td.kernel()[0]
        ref = td.to_cpu().kernel()[0][:3]
        self.assertAlmostEqual(abs(e[:len(ref)] - ref).max(), 0, 7)
        dip = td.transition_dipole()
        self.assertAlmostEqual(abs(dip).max(), 0, 8)

    def test_tda_vind(self):
        mf = self.mf
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,nocc,nvir)
        ref = mf.to_cpu().TDA().set(singlet=False).gen_vind()[0](zs)
        dat = mf.TDA().set(singlet=False).gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

        df_mf = self.df_mf
        ref = df_mf.to_cpu().TDA().set(singlet=True).gen_vind()[0](zs)
        dat = df_mf.TDA().set(singlet=True).gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tdhf_vind(self):
        mf = self.mf
        nocc = self.mol.nelectron // 2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc
        zs = np.random.rand(3,2,nocc,nvir)
        ref = mf.to_cpu().TDHF().set(singlet=True).gen_vind()[0](zs)
        dat = mf.TDHF().set(singlet=True).gen_vind()[0](zs).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

        df_mf = self.df_mf
        ref = df_mf.to_cpu().TDHF().set(singlet=False).gen_vind()[0](zs)
        dat = df_mf.TDHF().set(singlet=False).gen_vind()[0](zs).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tda_scanner(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
              H  0.2  0.   .8
              F  0.   0.2  0.''',
            basis = '631g')
        mf = mol.RHF().to_gpu().density_fit().run()
        td = mf.TDA()
        td.nstates = 5
        ref = td.kernel()[0]
        td_scan = td.as_scanner()
        td_scan.max_cycle = 1
        td_scan(mol)
        self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, delta=2e-6)

    def test_tdhf_scanner(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
              H  0.2  0.   .8
              F  0.   0.2  0.''',
            basis = '631g')
        mf = mol.RHF().to_gpu().density_fit().run()
        td = mf.TDHF()
        td.nstates = 5
        ref = td.kernel()[0]
        td_scan = td.as_scanner()
        td_scan.max_cycle = 1
        td_scan(mol)
        self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, delta=2e-6)


class RHFRestart(unittest.TestCase):
    def test_rhf_lazy_scf_restart_and_transition_density(self):
        for method in ['TDA', 'TDHF']:
            with self.subTest(method=method):
                mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .721 .633',
                            basis='sto-3g', verbose=0)
                mf = mol.RHF().to_gpu()
                mf.conv_tol = 1e-12
                assert mf.mo_energy is None
                td = getattr(mf, method)().set(nstates=2, conv_tol=1e-10, lindep=1e-24)
                td.kernel()
                assert mf.converged and np.all(td.converged)
                a, b = td.get_ab()
                size = a.shape[0]*a.shape[1]
                a, b = a.reshape(size, size), b.reshape(size, size)
                dense = a if method == 'TDA' else np.block([[a, b], [-b, -a]])
                expected = np.sort(np.linalg.eigvals(dense).real)
                expected = expected[expected > td.positive_eig_threshold][:2]
                np.testing.assert_allclose(cp.asnumpy(td.e), expected, atol=1e-10, rtol=0)

                # Loading genuine orbitals from host storage must not change the response.
                host_mf = mf.copy()
                for name in ('mo_coeff', 'mo_energy', 'mo_occ'):
                    setattr(host_mf, name, cp.asnumpy(getattr(mf, name)))
                host_td = getattr(host_mf, method)()
                vind, _ = host_td.gen_vind()
                np.testing.assert_allclose(vind(cp.eye(len(dense))).get(), dense.T,
                                           atol=1e-10, rtol=1e-10)

                # Reuse the public list-of-(X,Y) state, rather than replacing the solver.
                reference_e = cp.asnumpy(td.e).copy()
                td.max_cycle = 1
                td.kernel()
                assert np.all(td.converged)
                np.testing.assert_allclose(cp.asnumpy(td.e), reference_e, atol=1e-10, rtol=0)
                co = mf.mo_coeff[:, mf.mo_occ == 2].get()
                cv = mf.mo_coeff[:, mf.mo_occ == 0].get()
                r = mol.intor('int1e_r', comp=3)
                dipoles = []
                for energy, (x, y) in zip(cp.asnumpy(td.e), td.xy):
                    x = cp.asnumpy(x)
                    y = cp.asnumpy(y) if isinstance(y, (np.ndarray, cp.ndarray)) else np.zeros_like(x)
                    vector = x.ravel() if method == 'TDA' else np.r_[x.ravel(), y.ravel()]
                    assert np.linalg.norm(dense @ vector - energy*vector) < 1e-10
                    density = 2*(co @ x @ cv.T + cv @ y.T @ co.T)
                    dipoles.append(np.einsum('kpq,pq->k', r, density))
                dipoles = np.asarray(dipoles)
                np.testing.assert_allclose(td.transition_dipole(), dipoles, atol=1e-11)
                oscillator = (2/3)*cp.asnumpy(td.e)*np.sum(dipoles**2, axis=1)
                np.testing.assert_allclose(td.oscillator_strength(), oscillator, atol=1e-12)
                np.testing.assert_allclose(td.transition_dipole(), td.to_cpu().transition_dipole(), atol=1e-11)

                x = td.xy[0][0].get().copy()
                weights, nto = td.get_nto(state=1, verbose=0)
                singular = np.linalg.svd(x/np.linalg.norm(x), compute_uv=False)
                np.testing.assert_allclose(weights, singular**2, atol=1e-11)
                np.testing.assert_allclose(nto.T @ mf.get_ovlp().get() @ nto, np.eye(mol.nao), atol=1e-10)
                np.testing.assert_allclose(td.xy[0][0].get(), x, atol=0, rtol=0)

    def test_rhf_scanner_geometry_and_changed_basis(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = mol.RHF().to_gpu().run(conv_tol=1e-12)
        td = mf.TDA().set(nstates=1, conv_tol=1e-10, lindep=1e-24).run()
        scanner = td.as_scanner()
        assert scanner.as_scanner() is scanner
        coords = mol.atom_coords(unit='Angstrom')
        coords[1, 2] += .03
        scanned = scanner(coords)
        fresh = scanner._scf.TDA().set(nstates=1, conv_tol=1e-10, lindep=1e-24).run()
        np.testing.assert_allclose(scanned, fresh.e_tot, atol=1e-10, rtol=0)
        larger = scanner.mol.copy().build(basis='6-31g')
        scanned = scanner(larger)
        fresh = scanner._scf.TDA().set(nstates=1, conv_tol=1e-10, lindep=1e-24).run()
        assert np.all(scanner.converged)
        assert scanner.xy[0][0].shape == (1, larger.nao-1)
        np.testing.assert_allclose(scanned, fresh.e_tot, atol=1e-10, rtol=0)


if __name__ == "__main__":
    print("Full Tests for rhf-TDA and rhf-TDHF")
    unittest.main()
