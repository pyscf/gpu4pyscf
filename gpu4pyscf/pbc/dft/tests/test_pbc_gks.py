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

import unittest
import numpy as np
import cupy as cp
from pyscf.pbc import gto, dft
from gpu4pyscf.pbc import dft as gpu_dft
from gpu4pyscf.pbc import scf as gpu_scf
try:
    import mcfun
except ImportError:
    mcfun = None


def setUpModule():
    global cell
    cell = gto.Cell()
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
    cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
    cell.spin = 1
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()

class KnownValues(unittest.TestCase):
    def test_collinear_gks_gga(self):
        mf = gpu_dft.GKS(cell)
        mf.xc = 'pbe'
        mf.collinear = 'col'
        mf.run()

        ref = dft.GKS(cell, xc=mf.xc)
        ref.collinear = 'col'
        self.assertAlmostEqual(
            mf.e_tot, ref.energy_tot(dm=mf.make_rdm1().get()), delta=1e-8)

    @unittest.skip('Multicollinear periodic XC is not implemented on GPU.')
    def test_mcol_gks_gga(self):
        mf = gpu_dft.GKS(cell).density_fit()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.7331713336996268, 7)

    @unittest.skip('Spin-orbit periodic X2C is not implemented on GPU.')
    def test_ncol_x2c_gks_lda(self):
        mf = gpu_dft.GKS(cell).x2c()
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.6093309656369732, 7)

    @unittest.skip('Spin-orbit periodic X2C is not implemented on GPU.')
    def test_mcol_x2c_gks_lda(self):
        mf = gpu_dft.GKS(cell).x2c()
        mf.xc = 'lda,vwn'
        mf._numint.spin_samples = 50
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.6093309656369732, 6)

    def test_KGKS(self):
        cell = gto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''

        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.verbose = 0
        cell.mesh = [29] * 3
        cell.precision = 1e-10
        cell.build()
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
        # In the absence of off diagonal blocks in the spin space, dft.KGKS should reproduce the dft.KRKS results
        # Reference from dft.KRKS
        # mf = dft.KRKS(cell, kpts)
        # mf.xc = 'lda'
        # mf.conv_tol = 1e-10
        # e_ref = mf.kernel() # -10.38125412115097
        # self.assertAlmostEqual(e_ref, -10.38125412115097, 8)
        mf = gpu_dft.KGKS(cell, kpts)
        mf.xc = 'lda'
        mf.conv_tol = 1e-10
        e_kgks = mf.kernel()
        self.assertAlmostEqual(e_kgks, -10.38125412115097, 8)

    def test_veff_vs_cpu(self):
        cell = gto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''

        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.verbose = 0
        cell.mesh = [29] * 3
        cell.precision = 1e-10
        cell.build()
        kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
        mf = gpu_dft.KGKS(cell, kpts)
        n2c = cell.nao * 2
        rng = np.random.default_rng(1)
        c = (rng.random((len(kpts), n2c, cell.nelectron))
             + 1j*rng.random((len(kpts), n2c, cell.nelectron)))
        # Positive-semidefinite densities keep the spin densities physical.
        dm = (c @ c.swapaxes(-1, -2).conj()) / n2c
        mf.xc = 'pbe'
        mf.collinear = 'col'
        ref = mf.to_cpu().get_veff(cell, dm)
        dat = mf.get_veff(cell, cp.asarray(dm)).get()
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_collinear_kgks_lda(self):
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = gpu_dft.KGKS(cell, kpts=cell.make_kpts([3,1,1]))
        mf.xc = 'lda,vwn'
        mf.collinear = 'col'
        mf.run()

        ref = dft.KGKS(cell, mf.kpts, xc=mf.xc)
        ref.collinear = 'col'
        self.assertAlmostEqual(
            mf.e_tot, ref.energy_tot(dm=mf.make_rdm1().get()), delta=1e-8)

    def test_collinear_kgks_gga(self):
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = gpu_dft.KGKS(cell, kpts=cell.make_kpts([2,1,1]))
        mf.xc = 'pbe'
        mf.collinear = 'col'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.6373456924395708, 7)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    @unittest.skip('Multicollinear periodic XC is not implemented on GPU.')
    def test_mcol_kgks_gga(self):
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = gpu_dft.KGKS(cell, kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.xc = 'pbe'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 6
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.6312141891350893, 5)

    @unittest.skip('Spin-orbit periodic X2C is not implemented on GPU.')
    def test_ncol_x2c_kgks_lda(self):
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = gpu_dft.KGKS(cell, kpts=cell.make_kpts([2,1,1])).x2c()
        mf.xc = 'lda,vwn'
        mf.collinear = 'ncol'
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.533809531603591, 6)

    @unittest.skipIf(mcfun is None, "mcfun library not found.")
    @unittest.skip('Spin-orbit periodic X2C is not implemented on GPU.')
    def test_mcol_x2c_kgks_lda(self):
        cell = gto.Cell()
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.atom = 'He 0.,  0.,  0.; H 0.8917,  0.8917,  0.8917'
        cell.basis = [[0, [2, 1]], [1, [.5, 1]]]
        cell.spin = 1
        cell.build()
        mf = gpu_dft.KGKS(cell, kpts=cell.make_kpts([2,1,1])).x2c()
        mf.xc = 'lda,vwn'
        mf.collinear = 'mcol'
        mf._numint.spin_samples = 50
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -1.533809531603591, 6)

    def test_to_hf(self):
        mf = gpu_dft.KGKS(cell).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, gpu_scf.kghf.KGHF))

        mf = gpu_dft.KGKS(cell, kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(not a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, gpu_scf.kghf.KGHF))

if __name__ == '__main__':
    print("Full Tests for GKS")
    unittest.main()
