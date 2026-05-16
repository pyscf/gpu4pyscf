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
2. get_pp_nl_gpu against CPU get_pp_nl, gamma point
3. get_pp_nl_gpu against CPU get_pp_nl, non-zero k-points (single kpt
   and a k-mesh) — exercises the non-gamma branch of the GPU wrapper
4. Multiple elements (C, Si, Fe) covering s/p/d/f projectors
"""

import unittest
import numpy as np
import pyscf
from pyscf.pbc.gto.pseudo.pp_int import (
    fake_cell_vnl, _int_vnl, _contract_ppnl, get_pp_nl)


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
    )

    cell_fe = pyscf.M(
        atom=[['Fe', [0.0, 0.0, 0.0]], ['Fe', [2.71, 2.71, 2.71]]],
        a=np.eye(3) * 5.42,
        basis='gth-dzvp-molopt-sr',
        pseudo='gth-pbe',
        unit='bohr',
        verbose=0,
    )


class TestContractPpnl(unittest.TestCase):
    """Test GPU _contract_ppnl_gpu against CPU _contract_ppnl, gamma point."""

    def _compare(self, cell, places=13):
        from gpu4pyscf.pbc.gto.pseudo.pp_int import _contract_ppnl_gpu
        kpts = np.zeros((1, 3))
        fakecell, hl_blocks = fake_cell_vnl(cell)
        ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts)

        cpu = _contract_ppnl(cell, fakecell, hl_blocks, ppnl_half, kpts=kpts)
        gpu = _contract_ppnl_gpu(cell, fakecell, hl_blocks, ppnl_half, kpts=kpts)

        err = np.max(np.abs(np.asarray(gpu) - np.asarray(cpu)))
        self.assertAlmostEqual(err, 0, places, f"max|err|={err:.2e}")

    def test_carbon(self):
        self._compare(cell_c)

    def test_silicon(self):
        self._compare(cell_si)

    def test_iron(self):
        self._compare(cell_fe, places=12)


class TestGetPpNlGamma(unittest.TestCase):
    """Test get_pp_nl_gpu against CPU get_pp_nl, gamma point."""

    def _compare(self, cell, places=12):
        from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
        cpu = get_pp_nl(cell)
        gpu = get_pp_nl_gpu(cell)
        err = np.max(np.abs(np.asarray(gpu) - np.asarray(cpu)))
        self.assertAlmostEqual(err, 0, places, f"max|err|={err:.2e}")

    def test_carbon(self):
        self._compare(cell_c)

    def test_silicon(self):
        self._compare(cell_si)

    def test_iron(self):
        self._compare(cell_fe, places=10)


class TestGetPpNlKpts(unittest.TestCase):
    """Test get_pp_nl_gpu against CPU get_pp_nl with non-zero k-points."""

    def _compare(self, cell, kpts, places=13):
        from gpu4pyscf.pbc.gto.pseudo.pp_int import get_pp_nl_gpu
        cpu = get_pp_nl(cell, kpts)
        gpu = get_pp_nl_gpu(cell, kpts)
        err = np.max(np.abs(np.asarray(gpu) - np.asarray(cpu)))
        self.assertAlmostEqual(err, 0, places, f"max|err|={err:.2e}")

    def test_silicon_single_kpt(self):
        kpts = np.array([[0.1, 0.0, 0.0]])
        self._compare(cell_si, kpts)

    def test_silicon_kmesh(self):
        kpts = cell_si.make_kpts([2, 2, 2])
        self._compare(cell_si, kpts)

    def test_iron_single_kpt(self):
        kpts = np.array([[0.1, 0.0, 0.0]])
        self._compare(cell_fe, kpts, places=12)


if __name__ == "__main__":
    unittest.main()
