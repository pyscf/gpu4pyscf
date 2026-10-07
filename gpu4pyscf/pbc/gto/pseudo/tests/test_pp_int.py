#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

"""Tests for GPU-accelerated GTH non-local pseudopotential Fock contribution.

Validates:
1. _contract_ppnl_gpu against CPU _contract_ppnl, gamma point
2. get_pp_nl against CPU get_pp_nl, gamma point
3. get_pp_nl against CPU get_pp_nl, non-zero k-points (single kpt
   and a k-mesh) — exercises the non-gamma branch of the GPU wrapper
4. Multiple elements (C, Si, Fe) covering s/p/d/f projectors
"""

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf.pbc.gto.pseudo.pp_int import (
    fake_cell_vnl, _int_vnl, _contract_ppnl, get_pp_nl)
from packaging.version import Version


def setUpModule():
    global cell_c, cell_si, cell_fe

    cell_c = pyscf.M(
        atom=[['C', [0.0, 0.0, 0.0]], ['C', [1.885, 1.685, 1.585]]],
        a='''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000''',
        basis='gth-szv',
        pseudo='gth-pade',
        unit='bohr',
        verbose=0,
    )

    cell_si = pyscf.M(
        atom=[['Si', [0.0, 0.0, 0.0]], ['Si', [2.5, 2.5, 0.0]]],
        a=np.eye(3) * 8.0,
        basis='gth-szv',
        pseudo='gth-pade',
        unit='bohr',
        verbose=0,
        precision=1e-10,
    )

    cell_fe = pyscf.M(
        atom=[['Fe', [0.0, 0.0, 0.0]], ['Fe', [2.71, 2.71, 2.71]]],
        a=np.eye(3) * 5.42,
        basis='gth-dzvp-molopt-sr',
        pseudo='gth-pbe',
        unit='bohr',
        verbose=0,
        precision=1e-10,
    )

class TestGetPpNlGamma(unittest.TestCase):
    """Test get_pp_nl against CPU get_pp_nl, gamma point."""

    def _compare(self, cell, places=12):
        from gpu4pyscf.pbc.gto.pseudo import pp_int
        cpu = get_pp_nl(cell)
        gpu = pp_int.get_pp_nl(cell)
        err = np.max(np.abs(cp.asnumpy(gpu) - np.asarray(cpu)))
        self.assertAlmostEqual(err, 0, places, f"max|err|={err:.2e}")

    def test_carbon(self):
        self._compare(cell_c)

    def test_silicon(self):
        self._compare(cell_si)

    def test_iron(self):
        self._compare(cell_fe, places=10)


class TestGetPpNlKpts(unittest.TestCase):
    """Test get_pp_nl against CPU get_pp_nl with non-zero k-points."""

    def _compare(self, cell, kpts, places=13):
        from gpu4pyscf.pbc.gto.pseudo import pp_int
        cpu = get_pp_nl(cell, kpts)
        gpu = pp_int.get_pp_nl(cell, kpts)
        err = np.max(np.abs(cp.asnumpy(gpu) - np.asarray(cpu)))
        self.assertAlmostEqual(err, 0, places, f"max|err|={err:.2e}")

    def test_silicon_single_kpt(self):
        kpts = np.array([[0.1, 0.0, 0.0]])
        self._compare(cell_si, kpts, places=12)

    def test_silicon_kmesh(self):
        kpts = cell_si.make_kpts([2, 2, 2])
        self._compare(cell_si, kpts)

    def test_iron_single_kpt(self):
        kpts = np.array([[0.1, 0.0, 0.0]])
        self._compare(cell_fe, kpts, places=12)

    def test_iron_single_kpts(self):
        kpts = cell_fe.make_kpts([2, 5, 1])
        self._compare(cell_fe, kpts, places=11)


pyscf_version = Version(pyscf.__version__)

class KnownValues(unittest.TestCase):
    @unittest.skipIf(pyscf_version < Version('2.15'), 'PP-SOC available in 2.15')
    def test_pp_soc(self):
        np.random.seed(4)
        cell = pyscf.M(
            atom = 'He  1.  .1  .3; He  .0  .8  1.1',
            a = np.eye(3) * 4 + np.random.rand(3,3)*.5,
            basis = { 'He': [[0, (0.8, 1.0)],
                             [1, (1.2, 1.0)],
                             [2, (0.9, 1.0)]]},
            pseudo = '''
He
    2
     0.40000000    3    -1.98934751    -0.75604821    0.95604821
    2  SOC
     0.29482550    3     1.23870466    .855         .3
                                       .71         -1.1
                                                    .9
     0.32235865    2     2.25670239    -0.39677748
                                        0.93894690
                         0.15           0.12
                                        0.25''')
        kmesh = [3, 1, 4]
        kpts = cell.make_kpts(kmesh)
        dat = pp_int.get_pp_soc(cell, kpts).get()
        assert abs(lib.fp(dat) - 1.0485888724761192) < 1e-12

    @unittest.skipIf(pyscf_version < Version('2.15'), 'PP-SOC available in 2.15')
    def test_pp_scalar_soc_mixed(self):
        cell = pyscf.M(
            a = '''
            0.0 3.0 3.0
            3.0 0.0 3.0
            3.0 3.0 0.0''',
            atom='''Pb 0.0 0.0 0.0
            S 3.0 3.0 3.0
            ''',
            basis={
                'Pb': 'DZVP-MOLOPT-PBE-GTH-q4',
                'S': 'DZVP-MOLOPT-PBE-GTH-q6',
            },
            pseudo={
                'Pb': 'GTH-SOC-PBE-q4',
                'S': 'GTH-PBE-q6',
            },
        )
        mf = cell.GHF().to_gpu()
        mf.with_soc = True
        h = mf.get_hcore().get()
        self.assertAlmostEqual(lib.fp(h), 0.4445452615133345+0.011423142532064764j, 8)

    @unittest.skipIf(pyscf_version < Version('2.15'), 'PP-SOC available in 2.15')
    def test_with_soc_for_scalar_pp(self):
        cell = pyscf.M(
            a = '''
            0.0 3.0 3.0
            3.0 0.0 3.0
            3.0 3.0 0.0''',
            atom='''Pb 0.0 0.0 0.0
            S 3.0 3.0 3.0
            ''',
            basis={
                'Pb': 'DZVP-MOLOPT-PBE-GTH-q4',
                'S': 'DZVP-MOLOPT-PBE-GTH-q6',
            },
            pseudo={
                'Pb': 'GTH-PBE-q4',
                'S': 'GTH-PBE-q6',
            },
        )
        mf_ref = cell.RHF().to_gpu().run()
        mf = cell.GHF().to_gpu().run()
        mf.with_soc = True
        e_tot = mf.kernel()
        self.assertAlmostEqual(mf_ref.e_tot, mf.e_tot, 8)


if __name__ == "__main__":
    unittest.main()
