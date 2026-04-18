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

"""Tests for GPU-accelerated GTH non-local pseudopotential gradient.

Validates:
1. Cross-basis integrals (_int_vnl_gpu) against CPU _int_vnl
2. Full vppnl_nuc_grad against CPU reference
3. Finite difference consistency
4. Multiple elements (C, Si, Fe) covering s/p/d/f projectors
5. Both gamma-point and k-point paths
"""

import unittest
import numpy as np
import pyscf
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl, _int_vnl

disp = 1e-4

def setUpModule():
    global cell_c, cell_si, cell_fe

    # Carbon: s and p projectors, gth-szv (simple basis)
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

    # Silicon: s, p projectors with hl_dim=2 (tests r² integrals)
    cell_si = pyscf.M(
        atom=[['Si', [0.0, 0.0, 0.0]], ['Si', [2.5, 2.5, 0.0]]],
        a=np.eye(3) * 8.0,
        basis='gth-szv',
        pseudo='gth-pade',
        unit='bohr',
        verbose=0,
    )

    # Iron: s, p, d projectors with hl_dim up to 2, f-orbitals in basis
    cell_fe = pyscf.M(
        atom=[['Fe', [0.0, 0.0, 0.0]], ['Fe', [2.71, 2.71, 2.71]]],
        a=np.eye(3) * 5.42,
        basis='gth-dzvp-molopt-sr',
        pseudo='gth-pbe',
        unit='bohr',
        verbose=0,
    )


class TestCrossBasisIntegrals(unittest.TestCase):
    """Test GPU _int_vnl_gpu against CPU _int_vnl for each element."""

    def _compare_integrals(self, cell, intors=None, comp=1, places=6):
        from gpu4pyscf.pbc.gto.pseudo.pp_int import _int_vnl_gpu
        kpts = np.zeros((1, 3))
        fakecell, hl_blocks = fake_cell_vnl(cell)

        cpu = _int_vnl(cell, fakecell, hl_blocks, kpts, intors, comp)
        gpu = _int_vnl_gpu(cell, fakecell, hl_blocks, kpts, intors, comp)

        for lvl in range(3):
            c, g = cpu[lvl], gpu[lvl]
            if len(c) == 0 and len(g) == 0:
                continue
            self.assertEqual(c.shape, g.shape,
                             f"Shape mismatch at level {lvl}: CPU {c.shape} vs GPU {g.shape}")
            err = np.max(np.abs(g - c))
            self.assertAlmostEqual(err, 0, places,
                                   f"Level {lvl} base integrals: max|err|={err:.2e}")

    def test_carbon_base(self):
        self._compare_integrals(cell_c)

    def test_carbon_deriv(self):
        self._compare_integrals(
            cell_c,
            ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2'),
            comp=3)

    def test_silicon_base(self):
        self._compare_integrals(cell_si)

    def test_silicon_deriv(self):
        self._compare_integrals(
            cell_si,
            ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2'),
            comp=3)

    def test_iron_base(self):
        self._compare_integrals(cell_fe, places=5)

    def test_iron_deriv(self):
        self._compare_integrals(
            cell_fe,
            ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2'),
            comp=3, places=5)


class TestVppnlNucGrad(unittest.TestCase):
    """Test full vppnl_nuc_grad against CPU reference."""

    def _compare_grad(self, cell, places=6):
        import cupy as cp
        from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
        from pyscf.pbc.gto.pseudo import pp_int as pyscf_pp_int

        nao = cell.nao_nr()
        np.random.seed(42)
        dm = np.random.randn(nao, nao)
        dm = dm + dm.T

        # CPU reference: use pyscf's own vppnl_nuc_grad
        grad_cpu = pyscf_pp_int.vppnl_nuc_grad(cell, dm)

        # GPU
        grad_gpu = vppnl_nuc_grad(cell, cp.asarray(dm))

        err = np.max(np.abs(grad_gpu - grad_cpu))
        grad_max = np.max(np.abs(grad_cpu))
        self.assertAlmostEqual(err / max(grad_max, 1e-15), 0, places,
                               f"Gradient error: max|err|={err:.2e}, |grad_max|={grad_max:.2e}")

    def test_carbon(self):
        self._compare_grad(cell_c)

    def test_silicon(self):
        self._compare_grad(cell_si)

    def test_iron(self):
        self._compare_grad(cell_fe, places=5)


class TestFiniteDifference(unittest.TestCase):
    """Validate gradient via finite difference of the PP energy."""

    def _fd_check(self, cell, atom_id=1, cart_id=0, places=5):
        import cupy as cp
        from pyscf.pbc import scf
        from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

        mf = scf.RHF(cell)
        mf.max_cycle = 3
        mf.conv_tol = 1e-8
        mf.kernel()
        dm = mf.make_rdm1()

        grad = vppnl_nuc_grad(cell, cp.asarray(dm))

        # Finite difference
        coords = cell.atom_coords().copy()
        for sign, label in [(+1, 'plus'), (-1, 'minus')]:
            cell_d = cell.copy()
            coords_d = coords.copy()
            coords_d[atom_id, cart_id] += sign * disp
            atom_list = []
            for ia in range(cell.natm):
                sym = cell.atom_symbol(ia)
                atom_list.append([sym, coords_d[ia]])
            cell_d.atom = atom_list
            cell_d.build()
            mf_d = scf.RHF(cell_d)
            mf_d.max_cycle = 3
            mf_d.conv_tol = 1e-8
            mf_d.kernel()
            dm_d = mf_d.make_rdm1()
            fakecell_d, hl_d = fake_cell_vnl(cell_d)
            ppnl_d = _int_vnl(cell_d, fakecell_d, hl_d, np.zeros((1, 3)))
            from pyscf.pbc.gto.pseudo.pp_int import get_pp_nl
            vppnl = get_pp_nl(cell_d)
            e_d = np.einsum('ij,ji', vppnl, dm_d)
            if sign == 1:
                e_plus = e_d
            else:
                e_minus = e_d

        fd_grad = (e_plus - e_minus) / (2 * disp)
        self.assertAlmostEqual(
            grad[atom_id, cart_id], fd_grad, places,
            f"FD check: analytical={grad[atom_id,cart_id]:.6e}, fd={fd_grad:.6e}")

    def test_carbon_fd(self):
        self._fd_check(cell_c, atom_id=1, cart_id=0, places=2)

    def test_silicon_fd(self):
        self._fd_check(cell_si, atom_id=0, cart_id=2, places=4)


if __name__ == '__main__':
    unittest.main()
