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
import unittest
from pyscf import gto
from pyscf import lib
from gpu4pyscf import scf
import scipy.linalg
from gpu4pyscf.x2c import x2c

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
        e_gpu = myx2c.kernel()

        myx2c_atom = scf.GHF(mol1).x2c1e()
        myx2c_atom.with_x2c.approx = 'ATOM1E'
        e_gpu_atom = myx2c_atom.kernel()
        self.assertAlmostEqual(e_gpu_atom, -128.615723692333, 9)
        self.assertAlmostEqual(e_gpu, e_gpu_atom, 9)
        self.assertAlmostEqual(lib.fp(myx2c.mo_energy.get()), -41.15250349727189, 9)
        self.assertAlmostEqual(lib.fp(myx2c.mo_energy.get()), lib.fp(myx2c_atom.mo_energy.get()), 9)

    def test_to_cpu(self):
        myx2c = scf.GHF(mol).x2c1e()
        e_gpu = myx2c.kernel()

        mfx2c_cpu = myx2c.to_cpu()
        e_cpu = mfx2c_cpu.kernel()
        self.assertAlmostEqual(e_cpu, e_gpu, 9)

if __name__ == "__main__":
    print("Full Tests for x2c")
    unittest.main()
