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
import pyscf
from packaging.version import Version
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc import scf as pscf
from gpu4pyscf.pbc.scf import kuhf
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt

def setUpModule():
    global cell
    L = 4
    cell = pbcgto.Cell()
    cell.build(unit = 'B',
               verbose = 7,
               output = '/dev/null',
               precision = 1e-9,
               a = ((L,0,0),(0,L,0),(0,0,L)),
               atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                       ['He', (L/2.   ,L/2.,L/2.+.5)]],
               basis = { 'He': [[0, (0.8, 1.0)],
                                [0, (1.0, 1.0)],
                                [0, (1.2, 1.0)]]},
               spin = 2)

def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnownValues(unittest.TestCase):
    def test_kuhf_bands(self):
        nk = [2, 2, 1]
        kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = pscf.KUHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kmf_cpu = kmf.to_cpu().run()
        self.assertAlmostEqual(kmf.e_tot, kmf_cpu.e_tot, delta=1e-8)
        self.assertAlmostEqual(kmf.e_tot, -4.021029656152094, delta=1e-8)
        pop = kmf.analyze()[0][0]
        self.assertAlmostEqual(lib.fp(pop), 0.02897067698093582, 5)

        np.random.seed(1)
        kpts_bands = np.random.random((1,3))
        e = kmf.get_bands(kpts_bands)[0]
        e_ref = kmf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, delta=5e-6)

    def test_uhf_bands(self):
        mf = pscf.UHF(cell).run(conv_tol=1e-9)
        mf_cpu = mf.to_cpu().run()
        self.assertAlmostEqual(mf.e_tot, mf_cpu.e_tot, delta=1e-8)
        self.assertAlmostEqual(mf.e_tot, -3.9546467710639632, 7)
        for with_meta_lowdin in (True, False):
            populations, charges = mf.analyze(
                verbose=0, with_meta_lowdin=with_meta_lowdin)[0]
            self.assertEqual(np.asarray(populations).shape, (2, cell.nao_nr()))
            self.assertEqual(np.asarray(charges).shape, (cell.natm,))
            self.assertAlmostEqual(np.sum(populations[0]), mf.nelec[0], 7)
            self.assertAlmostEqual(np.sum(populations[1]), mf.nelec[1], 7)
            if with_meta_lowdin:
                self.assertAlmostEqual(
                    lib.fp(populations[0] + populations[1]),
                    -0.04691820429296646, 5)

        np.random.seed(1)
        kpts_bands = np.random.random((4,3))
        e = mf.get_bands(kpts_bands)[0]
        e_ref = mf_cpu.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(abs(e.get()-e_ref).max(), 0, delta=5e-6)

    def test_small_system(self):
        mol = pbcgto.Cell(
            atom='H 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            spin=1,
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -0.10439957735616917, 8)

        mol = pbcgto.Cell(
            atom='He 0 0 0;',
            a=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            basis=[[0, [1, 1]]],
            verbose=7,
            output='/dev/null'
        )
        mf = pscf.KUHF(mol,kpts=[[0., 0., 0.]]).run()
        self.assertAlmostEqual(mf.e_tot, -2.2719576422665635, 8)

    def test_density_fit(self):
        from gpu4pyscf.pbc.df.df import GDF
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]
        cell.spin = 2
        cell.build()

        ref = cell.UHF().density_fit().run()
        mf = ref.to_gpu().run(conv_tol=1e-8)
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(ref.e_tot, -0.11995733902879813, 8)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

        ref = cell.UHF().density_fit().run()
        mf = ref.to_gpu().run(conv_tol=1e-8)
        self.assertTrue(isinstance(mf.with_df, GDF))
        self.assertAlmostEqual(ref.e_tot, -0.11995733902879813, 8)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_rsjk(self):
        L = 4.
        cell = pbcgto.Cell()
        cell.a = np.eye(3)*L
        cell.atom =[['H' , ( L/2+0., L/2+0. ,   L/2+1.)],
                    ['H' , ( L/2+1., L/2+0. ,   L/2+1.)]]
        cell.basis = [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]
        cell.build()

        ref = -0.36989524966775006
        mf = cell.UHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run(conv_tol=1e-8)
        self.assertAlmostEqual(mf.e_tot, ref, 8)

        mf = cell.KUHF().to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run(conv_tol=1e-8)
        self.assertAlmostEqual(mf.e_tot, ref, 8)

        mf = cell.KUHF(kpts=cell.make_kpts([2,1,1])).to_gpu()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run(conv_tol=1e-8)
        ref = -0.35369830482164666
        self.assertAlmostEqual(mf.e_tot, ref, 8)

    def test_rsjk_with_df(self):
        ref = cell.UHF(exxdiv='ewald').to_gpu().run()
        mf = cell.UHF(exxdiv='ewald').to_gpu().density_fit()
        mf.rsjk = PBCJKMatrixOpt(cell)
        mf.j_engine = PBCJMatrixOpt(cell)
        mf.run()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 6)
        self.assertAlmostEqual(mf.e_tot, -3.954646833686388, 6)

        kmf = cell.KUHF(exxdiv='ewald', kpts=cell.make_kpts([2,1,1])).to_gpu()
        kmf.rsjk = PBCJKMatrixOpt(cell)
        kmf.j_engine = PBCJMatrixOpt(cell)
        kmf.run()
        self.assertAlmostEqual(kmf.e_tot, -3.994410799375493, 6)

    def test_initial_guess_tag(self):
        mf = cell.UHF().to_gpu()
        s = mf.get_ovlp()

        dm = mf.get_init_guess(key='minao', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='hcore', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='atom', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        dm = mf.get_init_guess(key='mod_huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 3
        assert abs(cp.einsum('nij,ji->n', dm, s).get() - (3, 1)).max() < 1e-6

        kmesh = [2, 2, 1]
        kpts = cell.make_kpts(kmesh)
        mf = cell.KUHF(kpts=kpts).to_gpu()
        s = mf.get_ovlp()

        dm = mf.get_init_guess(key='minao', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='hcore', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='atom', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

        dm = mf.get_init_guess(key='mod_huckel', s1e=s)
        assert hasattr(dm, 'mo_coeff') and dm.mo_coeff.ndim == 4
        assert abs(cp.einsum('nkij,kji->n', dm, s).real.get() - (9, 7)).max() < 1e-6

    @unittest.skipIf(Version(pyscf.__version__) <= Version('2.14'),
                     'Temporarily disabled for PySCF <= 2.14: CPU KUHF empty-beta occupations')
    def test_fully_polarized_kpoint_occupations(self):
        from pyscf.pbc.scf import KUHF as CpuKUHF
        cell = pbcgto.M(atom='H .2 .3 .4', spin=1, unit='Bohr',
                        a=np.diag([5., 5.2, 5.4]), mesh=[25, 27, 29],
                        basis=[[0, [.8, 1.]], [1, [.4, 1.]]],
                        pseudo='gth-pade', precision=1e-10, verbose=0)
        kpts = cell.make_kpts([2, 1, 1])
        mf = pscf.KUHF(cell, kpts=kpts)
        # KUHF.nelec counts electrons over all sampled k-points.
        mf.nelec = (len(kpts), 0)
        hcore, overlap = mf.get_hcore(), mf.get_ovlp()
        energy, coeff = mf.eig((hcore, hcore), overlap)
        cpu = CpuKUHF(cell, kpts=kpts)
        cpu.nelec = mf.nelec
        reference = np.asarray(cpu.get_occ(energy.get()))
        np.testing.assert_array_equal(reference.sum(axis=(1, 2)), cpu.nelec)
        self.assertEqual(np.count_nonzero(reference[1]), 0)
        occupation = mf.get_occ(energy, coeff)
        np.testing.assert_array_equal(occupation.get(), reference)
        density = mf.init_guess_by_1e()
        counts = cp.einsum('skij,kji->s', density, overlap).real.get()
        np.testing.assert_allclose(counts, mf.nelec, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(density[1].get(), 0.)
        mf.conv_tol = cpu.conv_tol = 1e-11
        mf.kernel(dm0=density)
        cpu.kernel(dm0=density.get())
        self.assertTrue(mf.converged and cpu.converged)
        np.testing.assert_allclose(mf.e_tot, cpu.e_tot, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(mf.mo_occ.get(), np.asarray(cpu.mo_occ))
        np.testing.assert_array_equal(mf.make_rdm1()[1].get(), 0.)


class UnrestrictedSCFProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyscf.pbc import scf as cpu_scf
        cls.cell = pbcgto.M(
            atom='He .2 .3 .4', unit='Bohr', a=np.diag([5., 5.2, 5.4]),
            basis=[[0, [.8, 1.]], [1, [.4, 1.]]], pseudo='gth-pade',
            mesh=[25, 27, 29], precision=1e-10, verbose=0)
        cls.kpts = cls.cell.make_kpts([2, 1, 1])
        cls.models = {}
        for name in ('UHF', 'KUHF'):
            kwargs = {} if name == 'UHF' else {'kpts': cls.kpts}
            gpu = getattr(pscf, name)(cls.cell, **kwargs).run(conv_tol=1e-11)
            cpu = getattr(cpu_scf, name)(cls.cell, **kwargs).run(conv_tol=1e-11)
            assert gpu.converged and cpu.converged
            cls.models[name] = gpu, cpu

    def _check_density_dispatch(self, name):
        from gpu4pyscf.pbc.dft import UniformGrids
        from gpu4pyscf.pbc.dft.multigrid_v3 import MultiGridNumInt
        from pyscf.pbc.dft.numint import KNumInt as CpuNumInt
        sampled = name.startswith('K')
        source = self.models[name][0]
        kwargs = {'kpts': self.kpts} if sampled else {}
        mf = getattr(pscf, name)(self.cell, **kwargs)
        mf.mo_coeff = source.mo_coeff.copy()
        mf.mo_occ = source.mo_occ.copy()
        grids = UniformGrids(self.cell)
        mf.build()
        self.assertIsInstance(mf._numint, MultiGridNumInt)
        original_mesh = np.asarray(mf._numint.mesh).copy()
        grids.mesh = np.asarray(self.cell.mesh) + 2
        density = mf.make_rdm1()
        before = density.copy()
        actual = mf.get_rho(grids=grids)
        explicit = mf.get_rho(dm=density, grids=grids)
        np.testing.assert_allclose(actual.get(), explicit.get(), atol=1e-12, rtol=0)
        coordinates = grids.coords.get()
        kpts = self.kpts if sampled else np.zeros((1, 3))
        ao = self.cell.pbc_eval_gto('GTOval_sph', coordinates, kpts=kpts)
        charge = density.sum(axis=0)
        charge = charge.get().reshape(len(kpts), self.cell.nao, self.cell.nao)
        reference = CpuNumInt().eval_rho(self.cell, ao, charge, hermi=1)
        np.testing.assert_allclose(actual.get(), reference, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(density.get(), before.get())
        np.testing.assert_array_equal(mf._numint.mesh, original_mesh)

    def test_get_rho_uhf_built(self):
        self._check_density_dispatch('UHF')

    def test_get_rho_kuhf_built(self):
        self._check_density_dispatch('KUHF')


    def test_default_fock_energy_gradient_and_canonicalization_kuhf(self):
        rotation = np.eye(self.cell.nao)
        angle = .13
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]]
        mf, cpu = self.models['KUHF']
        coeff = cp.matmul(mf.mo_coeff, cp.asarray(rotation))
        occupation = mf.mo_occ
        density = mf.make_rdm1(coeff, occupation)
        hcore = mf.get_hcore()
        energy = mf.energy_elec(density, hcore)
        reference = cpu.energy_elec(density.get(), hcore.get())
        np.testing.assert_allclose(energy, reference, atol=1e-9, rtol=0)
        gradient = mf.get_grad(coeff, occupation)
        ref_grad = cpu.get_grad(coeff.get(), occupation.get())
        self.assertGreater(float(cp.linalg.norm(gradient)), 1e-3)
        np.testing.assert_allclose(gradient.get(), ref_grad, atol=1e-9, rtol=0)
        eigenvalues, canonical = mf.canonicalize(coeff, occupation)
        ref_values = cpu.canonicalize(coeff.get(), occupation.get())[0]
        np.testing.assert_allclose(eigenvalues.get(), np.asarray(ref_values), atol=1e-9, rtol=0)
        np.testing.assert_allclose(mf.make_rdm1(canonical, occupation).get(),
                                   density.get(), atol=1e-11, rtol=0)

    def test_single_band_shape_and_normalization(self):
        mf, cpu = self.models['KUHF']
        energy, coeff = mf.get_bands(np.zeros(3))
        batched = mf.get_bands(np.zeros((1, 3)))[0]
        reference = cpu.get_bands(np.zeros(3))[0]
        expected_shape = (2, self.cell.nao)
        self.assertEqual(energy.shape, expected_shape)
        unbatched = batched[:, 0]
        np.testing.assert_allclose(energy.get(), unbatched.get(), atol=1e-11, rtol=0)
        np.testing.assert_allclose(energy.get(), np.asarray(reference), atol=1e-9, rtol=0)
        overlap = cp.asarray(self.cell.pbc_intor('int1e_ovlp'))
        for c in coeff:
            np.testing.assert_allclose((c.conj().T @ overlap @ c).get(),
                                       np.eye(self.cell.nao), atol=1e-10, rtol=0)

    def test_kpts_without_gamma(self):
        cell = self.cell
        kpts = cell.make_kpts([2, 2, 1], with_gamma_point=False)
        for mf_cls in (pscf.KUHF, dft.KUKS):
            with self.subTest(mf_cls=mf_cls.__name__):
                with self.assertRaisesRegex(NotImplementedError, 'include the Gamma point'):
                    mf_cls(cell, kpts=kpts).kernel()


if __name__ == '__main__':
    print("Tests for PBC UHF and PBC KUHF")
    unittest.main()
