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

import numpy as np
import cupy as cp
from packaging.version import Version
import unittest
import pyscf
from pyscf import gto
from pyscf import lib
from pyscf.data import nist
from gpu4pyscf import scf
import scipy.linalg
from pyscf.x2c import x2c as cpu_x2c
from gpu4pyscf.x2c import x2c, sfx2c1e
from pyscf.x2c import sfx2c1e as cpu_sfx2c

def setUpModule():
    global mol, mol1
    mol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )
    mol1 = gto.M(
        verbose = 0,
        atom = '''
            Ne     0.    0.    0.
            ''',
        basis = 'cc-pvdz',
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_sfx2c1e(self):
        hcore_ref = mol.RHF().sfx2c1e().get_hcore()

        myx2c = mol.RHF().to_gpu().sfx2c1e()
        self.assertAlmostEqual(abs(myx2c.get_hcore().get() - hcore_ref).max(), 0, 10)

        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075429077955874, 9)

        myx2c.with_x2c.approx = 'ATOM1E'
        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.075429682026396, 9)

        if Version(pyscf.__version__) > Version('2.8'):
            mf = myx2c.to_cpu().run()
            self.assertAlmostEqual(mf.e_tot, -76.075429682026396, 9)

        mf = myx2c.undo_x2c().run()
        self.assertAlmostEqual(mf.e_tot, mol.RHF().kernel(), 9)

    def test_sfx2c1e_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        hcore_ref = pmol.RHF().sfx2c1e().get_hcore()

        myx2c = pmol.RHF().to_gpu().sfx2c1e()
        self.assertAlmostEqual(abs(myx2c.get_hcore().get() - hcore_ref).max(), 0, 10)

        e = myx2c.kernel()
        self.assertAlmostEqual(e, -76.07574079913672, 9)

    def test_sfx2c1e_picture_change(self):
        c = lib.param.LIGHT_SPEED
        myx2c = mol.RHF().sfx2c1e()
        href = myx2c.with_x2c.get_hcore()

        def tv(with_x2c):
            xmol = with_x2c.get_xmol()
            with lib.temporary_env(xmol, cart=mol.cart):
                t = cp.asarray(xmol.intor_symmetric('int1e_kin'))
                w = cp.asarray(xmol.intor_symmetric('int1e_pnucp'))
            return t, 'int1e_nuc', w

        myx2c = mol.RHF().to_gpu().sfx2c1e()
        t, v, w = tv(myx2c.with_x2c)
        h1 = myx2c.with_x2c.picture_change((v, w*(.5/c)**2-t), t)
        self.assertAlmostEqual(abs(href - h1.get()).max(), 0, 9)

    def test_ghf(self):
        with lib.temporary_env(lib.param, LIGHT_SPEED=15):
            ref = mol.GHF().x2c1e().run()
            mf = mol.GHF().to_gpu().x2c1e().run()
            self.assertAlmostEqual(mf.e_tot, ref.e_tot, 9)
            self.assertAlmostEqual(abs(ref.mo_energy - mf.mo_energy.get()).max(), 0, 5)

        myx2c = scf.GHF(mol).x2c1e()
        e_gpu = myx2c.kernel()
        myx2c_cpu = mol.GHF().x2c1e()
        e_cpu = myx2c_cpu.kernel()
        self.assertAlmostEqual(e_gpu, -76.075431226329414, 9)
        self.assertAlmostEqual(e_cpu, e_gpu, 9)
        self.assertAlmostEqual(lib.fp(myx2c.mo_energy.get()), -31.811713632863754, 5)
        self.assertAlmostEqual(lib.fp(myx2c.mo_energy.get()), lib.fp(myx2c_cpu.mo_energy), 5)

    @unittest.skip('pyscf has bugs in ghf atomix-X approximation')
    def test_ghf_atomX(self):
        with lib.temporary_env(lib.param, LIGHT_SPEED=15):
            ref = mol.GHF().x2c1e()
            ref.with_x2c.approx = 'ATOM1E'
            ref.run()
            mf = mol.GHF().to_gpu().x2c1e()
            mf.with_x2c.approx = 'ATOM1E'
            mf.run()
            self.assertAlmostEqual(mf.e_tot, ref.e_tot, 9)
            self.assertAlmostEqual(abs(ref.mo_energy - mf.mo_energy.get()).max(), 0, 5)

    def test_undo_x2c(self):
        mf = mol.RHF().to_gpu().x2c().density_fit()
        self.assertEqual(mf.__class__.__name__, 'DFsfX2C1eRHF')
        mf = mf.undo_x2c()
        self.assertEqual(mf.__class__.__name__, 'DFRHF')

        mf = mol.GHF().to_gpu().x2c().density_fit()
        self.assertEqual(mf.__class__.__name__, 'DFX2C1eGHF')
        mf = mf.undo_x2c()
        self.assertEqual(mf.__class__.__name__, 'DFGHF')

        mf = mol.GHF().x2c()
        self.assertEqual(mf.__class__.__name__, 'X2C1eGHF')
        mf = mf.undo_x2c()
        self.assertEqual(mf.__class__.__name__, 'GHF')

    def test_recontract_matrix(self):
        mol = gto.M(
            atom='C 0 0 0; C 1.685 1.685 1.685',
            basis='ccpvtz'
        )
        x2cobj = x2c.SpinOrbitalX2CHelper(mol)
        ref = mol.intor('int1e_ovlp')

        xmol = x2cobj.get_xmol()
        dat = xmol.intor('int1e_ovlp')
        dat = x2c._orbital_pair_cart2sph(xmol, dat)
        dat = x2c._recontract_matrix(xmol, dat)
        assert abs(ref - dat.get()).max() < 1e-8

    def test_1e_vs_atom1e(self):
        myx2c = scf.GHF(mol1).x2c1e()
        # Orbital energies need tighter convergence than the total energy.
        myx2c.conv_tol = 1e-13
        myx2c.conv_tol_grad = 1e-9
        e_gpu = myx2c.kernel()

        myx2c_atom = scf.GHF(mol1).x2c1e()
        myx2c_atom.with_x2c.approx = 'ATOM1E'
        myx2c_atom.conv_tol = 1e-13
        myx2c_atom.conv_tol_grad = 1e-9
        e_gpu_atom = myx2c_atom.kernel()
        ref = mol1.GHF().x2c1e().run(conv_tol=1e-13, conv_tol_grad=1e-9)
        self.assertTrue(myx2c.converged and myx2c_atom.converged and ref.converged)
        self.assertAlmostEqual(e_gpu_atom, -128.615723692333, 9)
        self.assertAlmostEqual(e_gpu, e_gpu_atom, 9)
        self.assertAlmostEqual(float(abs(myx2c.mo_energy.get() - ref.mo_energy).max()), 0, 8)
        self.assertAlmostEqual(float(abs(myx2c.mo_energy - myx2c_atom.mo_energy).max().get()), 0, 8)

    @unittest.skipIf(Version(pyscf.__version__) < Version('2.9'),
                     'pyscf x2c does not support not reset function')
    def test_to_cpu(self):
        myx2c = scf.GHF(mol).x2c1e()
        e_gpu = myx2c.kernel()

        mfx2c_cpu = myx2c.to_cpu()
        e_cpu = mfx2c_cpu.kernel()
        self.assertAlmostEqual(e_cpu, e_gpu, 9)

class SpinOrbitalPictureChange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        molecule = gto.M(atom='H .1 -.2 .3; F .1 -.2 2.0', unit='Bohr',
                         basis='def2-svp', cart=False, verbose=0)
        helper = x2c.SpinOrbitalX2CHelper(molecule)
        primitive = helper.get_xmol()
        # Compare the exact same primitive AO order, not independently decontracted
        # bases. Only the CPU reference disables its additional decontraction.
        cpu_mol = primitive.copy()
        cpu_mol.cart = molecule.cart
        reference = cpu_x2c.SpinOrbitalX2CHelper(cpu_mol)
        reference.xuncontract = False
        s = cpu_mol.intor_symmetric('int1e_ovlp')
        overlap = gto.intor_cross('int1e_ovlp', cpu_mol, molecule)
        contraction = scipy.linalg.solve(s, overlap, assume_a='pos')
        np.testing.assert_allclose(contraction.T @ s @ contraction,
                                   molecule.intor_symmetric('int1e_ovlp'),
                                   atol=2e-12, rtol=2e-12)
        cls.spin_orbital_helper = helper, reference, scipy.linalg.block_diag(contraction, contraction)

    @classmethod
    def tearDownClass(cls):
        del cls.spin_orbital_helper

    def test_spin_orbital_x_and_metric(self):
        helper, reference, _ = self.spin_orbital_helper
        x = helper.get_xmat()
        np.testing.assert_allclose(x.get(), reference.get_xmat(), atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(helper.get_xmat(helper.get_xmol()).get(), x.get(),
                                   atol=1e-12, rtol=1e-12)
        r = helper._get_rmat(x)
        np.testing.assert_allclose(r.get(), reference._get_rmat(), atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(helper._get_rmat().get(), r.get(), atol=1e-12, rtol=1e-12)
        s = scipy.linalg.block_diag(*([reference.mol.intor_symmetric('int1e_ovlp')]*2))
        t = scipy.linalg.block_diag(*([reference.mol.intor_symmetric('int1e_kin')]*2))
        metric = s + x.get().conj().T @ t @ x.get() / (2*lib.param.LIGHT_SPEED**2)
        np.testing.assert_allclose(r.get().conj().T @ metric @ r.get(), s,
                                   atol=2e-11, rtol=2e-11)
        root = x2c._sqrt(cp.asarray(s))
        inverse_root = x2c._invsqrt(cp.asarray(s))
        np.testing.assert_allclose((root @ root).get(), s, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose((inverse_root @ root).get(), np.eye(len(s)),
                                   atol=2e-10, rtol=2e-10)

    def test_spin_orbital_picture_change_odd(self):
        helper, reference, contraction = self.spin_orbital_helper
        primitive = reference.mol
        p0 = primitive.intor_symmetric('int1e_kin')
        p = scipy.linalg.block_diag(p0, p0)
        expected = contraction.T @ reference.picture_change((None, None), p) @ contraction
        actual = helper.picture_change((None, None), p).get()
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
        np.testing.assert_allclose(actual, actual.conj().swapaxes(-1, -2),
                                   atol=2e-11, rtol=2e-11)

    def test_spin_orbital_picture_change_large(self):
        helper, reference, contraction = self.spin_orbital_helper
        primitive = reference.mol
        v = np.array([scipy.linalg.block_diag(a, a) for a in primitive.intor('int1e_r')])
        expected = contraction.T @ reference.picture_change((v, None), None) @ contraction
        actual = helper.picture_change((cp.asarray(v), None), None).get()
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
        np.testing.assert_allclose(actual, actual.conj().swapaxes(-1, -2),
                                   atol=2e-11, rtol=2e-11)

    def test_spin_orbital_picture_change_small(self):
        helper, reference, contraction = self.spin_orbital_helper
        primitive = reference.mol
        c1 = .5/lib.param.LIGHT_SPEED
        w = np.array([cpu_x2c._sigma_dot(a)*c1**2 for a in
                      primitive.intor('int1e_sprsp').reshape(3, 4, primitive.nao, primitive.nao)])
        expected = contraction.T @ reference.picture_change((None, w), None) @ contraction
        actual = helper.picture_change((None, cp.asarray(w)), None).get()
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
        np.testing.assert_allclose(actual, actual.conj().swapaxes(-1, -2),
                                   atol=2e-11, rtol=2e-11)

class SpinOrbitalSCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        molecule = gto.M(atom='H 0 0 0; F 0 0 1.7', unit='Bohr', basis='def2-svp',
                         cart=False, verbose=0)
        reference = molecule.GHF().x2c1e().run(conv_tol=1e-11)
        assert reference.converged
        gpu = scf.GHF(molecule).x2c1e()
        gpu.mo_coeff = cp.asarray(reference.mo_coeff)
        gpu.mo_occ = cp.asarray(reference.mo_occ)
        gpu.mo_energy = cp.asarray(reference.mo_energy)
        cls.x2c_scf = gpu, reference

    @classmethod
    def tearDownClass(cls):
        del cls.x2c_scf

    def test_spin_orbital_dipole_units_and_density(self):
        for picture_change in [False, True]:
            with self.subTest(picture_change=picture_change):
                gpu, reference = self.x2c_scf
                dm = reference.make_rdm1()
                expected = reference.dip_moment(dm=dm, unit='AU', verbose=0,
                                                picture_change=picture_change)
                actual = gpu.dip_moment(dm=cp.asarray(dm), unit='AU', verbose=0,
                                        picture_change=picture_change)
                assert np.linalg.norm(expected) > .1
                np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
                np.testing.assert_allclose(gpu.dip_moment(unit='AU', verbose=0,
                                           picture_change=picture_change), actual, atol=1e-12)
                debye = gpu.dip_moment(dm=dm, unit='Debye', verbose=0, picture_change=picture_change)
                np.testing.assert_allclose(debye, actual*nist.AU2DEBYE, atol=2e-12, rtol=2e-12)
                # A zero electronic density leaves the nuclear contribution unchanged by
                # picture change, guarding explicit-density handling and sign conventions.
                nuclear = gpu.mol.atom_charges() @ gpu.mol.atom_coords()
                np.testing.assert_allclose(gpu.dip_moment(dm=cp.zeros_like(cp.asarray(dm)), unit='AU',
                                           verbose=0, picture_change=picture_change), nuclear, atol=1e-12)


class X2COperators(unittest.TestCase):
    def test_geometry_reset(self):
        molecule = gto.M(atom='H 0 0 0; F 0 0 1.7', unit='Bohr', basis='sto-3g', verbose=0)
        mf = scf.GHF(molecule).x2c1e()
        helper = mf.with_x2c
        assert mf.x2c1e() is mf
        assert mf.with_x2c is helper
        mf.with_x2c = None
        assert mf.x2c1e() is mf
        assert isinstance(mf.with_x2c, x2c.SpinOrbitalX2CHelper)
        assert mf.with_x2c.mol is molecule
        h_before = mf.get_hcore().get()
        # Populate genuine integral caches before changing geometry.
        mf.get_jk(dm=mf.get_init_guess())
        original_coords = molecule.atom_coords().copy()
        new_coords = original_coords.copy()
        new_coords[1, 2] += .3
        moved = molecule.set_geom_(new_coords, unit='Bohr', inplace=False)
        assert mf.reset(moved) is mf
        assert mf.mol is moved and mf.with_x2c.mol is moved
        fresh = scf.GHF(moved).x2c1e()
        h_after = mf.get_hcore().get()
        assert np.linalg.norm(h_after-h_before) > 1e-3
        np.testing.assert_allclose(h_after, fresh.get_hcore().get(), atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(mf.with_x2c.get_xmat().get(), fresh.with_x2c.get_xmat().get(),
                                   atol=2e-11, rtol=2e-11)
        dm = fresh.get_init_guess()
        j, k = mf.get_jk(dm=dm)
        j_ref, k_ref = fresh.get_jk(dm=dm)
        np.testing.assert_allclose(j.get(), j_ref.get(), atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(k.get(), k_ref.get(), atol=2e-11, rtol=2e-11)
        np.testing.assert_array_equal(molecule.atom_coords(), original_coords)

    def test_sqrt_and_invsqrt_linear_dependency(self):
        for phase in [1., 1j]:
            with self.subTest(phase=phase):
                molecule = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g', verbose=0)
                overlap = molecule.intor_symmetric('int1e_ovlp')
                # Three actual AO combinations in a two-dimensional span have one known
                # redundant direction: psi3 = phi1 + phase*phi2.
                coefficients = np.array([[1., 0., 1.], [0., 1., phase]])
                gram = coefficients.conj().T @ overlap @ coefficients
                null = np.array([-1., -phase, 1.])
                null /= np.linalg.norm(null)
                projector = np.eye(3) - np.outer(null, null.conj())
                root = x2c._sqrt(cp.asarray(gram)).get()
                inverse_root = x2c._invsqrt(cp.asarray(gram)).get()
                np.testing.assert_allclose(root, root.conj().T, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(inverse_root, inverse_root.conj().T, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(root @ root, gram, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(root @ inverse_root, projector, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(inverse_root @ gram @ inverse_root, projector,
                                           atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(root @ null, 0., atol=2e-13)
                np.testing.assert_allclose(inverse_root @ null, 0., atol=2e-13)

    def test_sfx2c_to_ks(self):
        molecule = gto.M(atom='H 0 0 0; F 0 0 1.7', unit='Bohr', basis='sto-3g', verbose=0)
        mf = scf.RHF(molecule).sfx2c1e()
        mf.conv_tol = 3e-10
        target = mf.to_ks(xc='pbe')
        assert isinstance(target, sfx2c1e.SFX2C1E_SCF)
        assert target.with_x2c is mf.with_x2c
        assert target.xc == 'pbe' and target.conv_tol == mf.conv_tol
        assert not target.converged
        np.testing.assert_allclose(target.get_hcore().get(), mf.get_hcore().get(), atol=1e-12, rtol=0)


class ScalarX2C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        molecule = gto.M(atom='H .2 -.1 .3; F .2 -.1 2.0', unit='Bohr',
                         basis='def2-svp', cart=False, charge=0, verbose=0)
        reference = molecule.RHF().sfx2c1e().run(conv_tol=1e-11)
        assert reference.converged
        gpu = scf.RHF(molecule).sfx2c1e()
        gpu.mo_coeff = cp.asarray(reference.mo_coeff)
        gpu.mo_occ = cp.asarray(reference.mo_occ)
        gpu.mo_energy = cp.asarray(reference.mo_energy)
        cls.scalar_x2c = gpu, reference

    @classmethod
    def tearDownClass(cls):
        del cls.scalar_x2c

    def test_scalar_dipole_density_and_units(self):
        for picture_change in [False, True]:
            with self.subTest(picture_change=picture_change):
                gpu, cpu = self.scalar_x2c
                dm = cpu.make_rdm1()
                original = dm.copy()
                original_env = gpu.mol._env.copy()
                expected = cpu.dip_moment(dm=dm, unit='AU', picture_change=picture_change, verbose=0)
                actual = gpu.dip_moment(dm=cp.asarray(dm), unit='AU',
                                        picture_change=picture_change, verbose=0)
                assert np.linalg.norm(expected) > .1
                np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)
                np.testing.assert_allclose(gpu.dip_moment(unit='AU', picture_change=picture_change,
                                           verbose=0), actual, atol=1e-12, rtol=0)
                np.testing.assert_allclose(gpu.dip_moment(dm=dm, unit='Debye',
                                           picture_change=picture_change, verbose=0),
                                           actual*nist.AU2DEBYE, atol=2e-12, rtol=2e-12)
                nuclear = gpu.mol.atom_charges() @ gpu.mol.atom_coords()
                np.testing.assert_allclose(gpu.dip_moment(dm=cp.zeros_like(cp.asarray(dm)), unit='AU',
                                           picture_change=picture_change, verbose=0), nuclear, atol=1e-12)
                np.testing.assert_array_equal(dm, original)
                np.testing.assert_array_equal(gpu.mol._env, original_env)

    def test_scalar_dipole_translation_charge(self):
        for picture_change in [False, True]:
            with self.subTest(picture_change=picture_change):
                gpu, cpu = self.scalar_x2c
                dm = cpu.make_rdm1()
                mol = gpu.mol
                shift = np.array([.19, -.13, .27])
                moved = mol.set_geom_(mol.atom_coords()+shift, unit='Bohr', inplace=False)
                translated = scf.RHF(moved).sfx2c1e()
                before = gpu.dip_moment(dm=dm, unit='AU', picture_change=picture_change, verbose=0)
                after = translated.dip_moment(dm=dm, unit='AU', picture_change=picture_change, verbose=0)
                # Translating nuclei and their AO density changes the dipole by Q*shift.
                electrons = np.einsum('ij,ji', mol.intor_symmetric('int1e_ovlp'), dm)
                np.testing.assert_allclose(electrons, mol.nelectron, atol=1e-11, rtol=0)
                np.testing.assert_allclose(after-before, mol.charge*shift, atol=2e-8, rtol=0)

    def test_scalar_metric_and_explicit_primitive(self):
        mf, _ = self.scalar_x2c
        helper = mf.with_x2c
        primitive = helper.get_xmol()
        with lib.temporary_env(primitive, cart=mf.mol.cart):
            s = primitive.intor_symmetric('int1e_ovlp')
            t = primitive.intor_symmetric('int1e_kin')
        x = helper.get_xmat()
        np.testing.assert_allclose(helper.get_xmat(primitive).get(), x.get(), atol=1e-12, rtol=0)
        r = helper._get_rmat(x)
        np.testing.assert_allclose(helper._get_rmat().get(), r.get(), atol=1e-12, rtol=0)
        metric = s + x.get().conj().T @ t @ x.get()/(2*lib.param.LIGHT_SPEED**2)
        np.testing.assert_allclose(r.get().conj().T @ metric @ r.get(), s, atol=2e-11, rtol=2e-11)


if __name__ == "__main__":
    print("Full Tests for x2c")
    unittest.main()
