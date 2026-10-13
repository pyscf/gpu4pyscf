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
from pyscf.pbc import gto, dft
from gpu4pyscf.pbc import dft as gpu_dft


class KnownValues(unittest.TestCase):
    def test_collinear_gks_gga(self):
        # Adapted from pyscf/pbc/dft/test/test_gks.py. Use matching uniform
        # grids and a live CPU reference instead of its atom-grid energy.
        cell = gto.M(
            a=[[0, 1.7834, 1.7834], [1.7834, 0, 1.7834],
               [1.7834, 1.7834, 0]],
            atom='He 0 0 0; H .8917 .8917 .8917',
            basis=[[0, [2, 1]], [1, [.5, 1]]], spin=1,
            mesh=[21]*3, precision=1e-10, verbose=0)
        ref = dft.GKS(cell, xc='pbe')
        ref.grids = dft.gen_grid.UniformGrids(cell)
        ref.collinear = 'col'
        ref.run(conv_tol=1e-10)
        mf = gpu_dft.GKS(cell, xc='pbe')
        mf.collinear = 'col'
        mf.run(dm0=ref.make_rdm1(), conv_tol=1e-10)
        self.assertTrue(ref.converged and mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, delta=1e-7)
        dm = mf.make_rdm1()
        vref = ref.get_veff(dm=dm.get())
        v = mf.get_veff(dm=dm)
        np.testing.assert_allclose(v.get(), vref, atol=1e-7, rtol=0)
        self.assertAlmostEqual(float(v.exc), float(vref.exc), delta=1e-7)

    def test_kgks_matches_krks(self):
        # Adapted from pyscf/pbc/dft/test/test_kgks.py: test_KGKS.
        # A smaller closed-shell He cell keeps the k-mesh regression cheap.
        cell = gto.M(
            unit='B', a=np.eye(3)*4, mesh=[21]*3, precision=1e-10,
            atom='He 1.5 2 1.5; He 2 2 2.5',
            basis={'He': [[0, (.8, 1.)], [0, (1., 1.)], [0, (1.2, 1.)]]},
            verbose=0)
        kpts = cell.make_kpts([2, 1, 1])
        ref = dft.KRKS(cell, kpts, xc='lda,vwn').run(
            init_guess='hcore', conv_tol=1e-10)
        mf = gpu_dft.KGKS(cell, kpts, xc='lda,vwn')
        mf.collinear = 'col'
        mf.run(init_guess='hcore', conv_tol=1e-10)
        self.assertTrue(ref.converged and mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, delta=1e-7)
        # Also check the complex k-point effective potential against CPU KGKS.
        gref = dft.KGKS(cell, kpts, xc=mf.xc)
        gref.collinear = 'col'
        dm = mf.make_rdm1()
        vref = gref.get_veff(dm=dm.get())
        v = mf.get_veff(dm=dm)
        np.testing.assert_allclose(v.get(), vref, atol=1e-7, rtol=0)
        self.assertAlmostEqual(float(v.exc), float(vref.exc), delta=1e-7)


if __name__ == '__main__':
    unittest.main()
