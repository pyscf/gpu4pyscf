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
        mol.verbose = 0
        mol.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol.basis = '631g'
        # FIXME: mo_coeff of uhf_symm.SymAdaptedUHF not converted to cupy arrays
        mol.symmetry = True
        cls.mol = mol.build()
        cls.mf = scf.UHF(mol).density_fit().run(conv_tol=1e-10).to_gpu()

        mol1 = gto.Mole()
        mol1.verbose = 7
        mol1.output = '/dev/null'
        mol1.atom = [
            ['H' , (0. , 0. , .917)],
            ['F' , (0. , 0. , 0.)], ]
        mol1.basis = '631g'
        mol1.spin = 2
        cls.mol1 = mol1.build()
        cls.mf1 = scf.UHF(mol1).run(conv_tol=1e-10).to_gpu()

    @classmethod
    def tearDownClass(cls):
        cls.mol1.stdout.close()

    def check_multipoles(self, td, expect_y):
        # Keep the converged amplitudes/phases fixed for the CPU comparison.
        cpu_td = td.to_cpu()
        if expect_y:
            self.assertGreater(sum(float(cp.linalg.norm(y)) for _, ys in td.xy for y in ys), 1e-6)
        for name in ('transition_dipole', 'transition_velocity_dipole',
                     'transition_magnetic_dipole', 'transition_quadrupole',
                     'transition_velocity_quadrupole', 'transition_magnetic_quadrupole',
                     'transition_octupole', 'transition_velocity_octupole'):
            with self.subTest(observable=name, solver=type(td).__name__):
                np.testing.assert_allclose(getattr(td, name)(), getattr(cpu_td, name)(),
                                           atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(td.transition_dipole(xy=[td.xy[1]]),
                                   cpu_td.transition_dipole(xy=[cpu_td.xy[1]]), atol=1e-10)
        dipole = td.transition_dipole()
        expected = (2./3.) * td.e * np.einsum('nx,nx->n', dipole.conj(), dipole).real
        np.testing.assert_allclose(td.oscillator_strength(), expected, atol=1e-12)
        np.testing.assert_allclose(td.oscillator_strength(gauge='velocity'),
                                   cpu_td.oscillator_strength(gauge='velocity'), atol=1e-10)

    def test_tda(self):
        mf = self.mf
        td = mf.TDA()
        assert td.device == 'gpu'
        td.nstates = 5
        e = td.kernel()[0]
        ref = [11.0179839, 11.0179839, 11.9031214, 11.9031214, 13.1701375]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tdhf(self):
        mf = self.mf
        td = mf.TDHF()
        assert td.device == 'gpu'
        td.nstates = 5
        td.conv_tol = 1e-5
        e = td.kernel()[0]
        ref = [10.8924334, 10.8924334, 11.8352278, 11.8352278, 12.6350840]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tda1(self):
        mf1 = self.mf1
        td = mf1.TDA()
        assert td.device == 'gpu'
        td.nstates = 5
        e = td.kernel()[0]
        ref = [ 3.3211349, 18.5597821, 21.0147390, 21.6150240, 25.0938938]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        self.check_multipoles(td, expect_y=False)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tdhf1(self):
        mf1 = self.mf1
        td = mf1.TDHF()
        assert td.device == 'gpu'
        td.nstates = 4
        e = td.kernel()[0]
        ref = [ 3.3126683, 18.4954862, 20.8493515, 21.5480882,]
        self.assertAlmostEqual(abs(e * 27.2114 - ref).max(), 0, 4)
        self.check_multipoles(td, expect_y=True)
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(e - ref).max(), 0, 4)

    def test_tda_vind(self):
        mf = self.mf1
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDA().set().gen_vind()[0](zs)
        dat = mf.TDA().set().gen_vind()[0](cp.asarray(zs)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tdhf_vind(self):
        mf = self.mf1
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,2,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDHF().set().gen_vind()[0](zs)
        dat = mf.TDHF().set().gen_vind()[0](zs).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tda_scanner(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
              H  0.2  0.   .8
              F  0.   0.2  0.''',
            basis = '631g')
        mf = mol.UHF().to_gpu().density_fit().run(conv_tol=1e-10)
        td = mf.TDA()
        td.nstates = 5
        ref = td.kernel()[0]
        td_scan = td.as_scanner()
        td_scan.max_cycle = 1
        td_scan(mol)
        self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, 6)

    def test_tdhf_scanner(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
              H  0.2  0.   .8
              F  0.   0.2  0.''',
            basis = '631g')
        mf = mol.UHF().to_gpu().density_fit().run(conv_tol=1e-10)
        td = mf.TDHF()
        td.nstates = 5
        ref = td.kernel()[0]
        td_scan = td.as_scanner()
        td_scan.max_cycle = 1
        td_scan(mol)
        self.assertAlmostEqual(abs(td_scan.e - ref).max(), 0, 6)


class UHFRestart(unittest.TestCase):
    def test_uhf_lazy_scf_restart_and_transition_density(self):
        for method in ['TDA', 'TDHF']:
            with self.subTest(method=method):
                mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', charge=1, spin=1,
                            basis='sto-3g', verbose=0)
                mf = mol.UHF().to_gpu()
                mf.conv_tol = 1e-12
                assert mf.mo_energy is None
                td = getattr(mf, method)().set(nstates=2, conv_tol=1e-10, lindep=1e-24)
                td.kernel()
                assert mf.converged and np.all(td.converged)
                aa, bb = td.get_ab()
                na = aa[0].shape[0]*aa[0].shape[1]
                nb = aa[2].shape[0]*aa[2].shape[1]
                a = np.block([[aa[0].reshape(na, na), aa[1].reshape(na, nb)],
                              [aa[1].reshape(na, nb).T, aa[2].reshape(nb, nb)]])
                b = np.block([[bb[0].reshape(na, na), bb[1].reshape(na, nb)],
                              [bb[1].reshape(na, nb).T, bb[2].reshape(nb, nb)]])
                dense = a if method == 'TDA' else np.block([[a, b], [-b, -a]])
                roots = np.linalg.eigvals(dense)
                assert np.max(np.abs(roots.imag)) < 1e-10
                expected = np.sort(roots.real[roots.real > td.positive_eig_threshold])[:2]
                np.testing.assert_allclose(cp.asnumpy(td.e), expected, atol=1e-10, rtol=0)
                reference_e = cp.asnumpy(td.e).copy()
                td.max_cycle = 1
                td.kernel()
                assert np.all(td.converged)
                np.testing.assert_allclose(cp.asnumpy(td.e), reference_e, atol=1e-10, rtol=0)

                r = mol.intor('int1e_r', comp=3)
                dipoles = []
                for energy, (xs, ys) in zip(cp.asnumpy(td.e), td.xy):
                    xs = [cp.asnumpy(x) for x in xs]
                    ys = [cp.asnumpy(y) if isinstance(y, (np.ndarray, cp.ndarray)) else np.zeros_like(x)
                          for x, y in zip(xs, ys)]
                    vector = np.concatenate([x.ravel() for x in xs])
                    if method == 'TDHF':
                        vector = np.r_[vector, np.concatenate([y.ravel() for y in ys])]
                    assert np.linalg.norm(dense @ vector - energy*vector) < 1e-10
                    density = np.zeros((mol.nao, mol.nao))
                    for spin in range(2):
                        co = mf.mo_coeff[spin][:, mf.mo_occ[spin] > 0].get()
                        cv = mf.mo_coeff[spin][:, mf.mo_occ[spin] == 0].get()
                        density += co @ xs[spin] @ cv.T + cv @ ys[spin].T @ co.T
                    dipoles.append(np.einsum('kpq,pq->k', r, density))
                dipoles = np.asarray(dipoles)
                np.testing.assert_allclose(td.transition_dipole(), dipoles, atol=1e-11)
                oscillator = (2/3)*cp.asnumpy(td.e)*np.sum(dipoles**2, axis=1)
                np.testing.assert_allclose(td.oscillator_strength(), oscillator, atol=1e-12)
                np.testing.assert_allclose(td.transition_dipole(), td.to_cpu().transition_dipole(), atol=1e-11)


if __name__ == "__main__":
    print("Full Tests for uhf-TDA and uhf-TDHF")
    unittest.main()
