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
1. Cross-basis integrals (_int_vnl_gpu) against CPU _int_vnl  (gamma only —
   _int_vnl_gpu intentionally implements only the gamma path; vppnl_nuc_grad
   falls back to CPU _int_vnl at non-gamma k-points)
2. Full vppnl_nuc_grad against CPU reference, gamma point
3. Full vppnl_nuc_grad against CPU reference, non-zero k-points (single kpt
   and a k-mesh) — validates the GPU contraction logic with multi-k-point
   arrays of complex integrals from CPU _int_vnl
4. Finite difference consistency, gamma point
5. Multiple elements (C, Si, Fe) covering s/p/d/f projectors
"""

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl, _int_vnl
from pyscf.pbc.lib.kpts_helper import gamma_point
import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
import gpu4pyscf.pbc.dft.multigrid_v2 as multigrid_v2
import pytest

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


def _cpu_vppnl_nuc_grad(cell, dm, kpts=None):
    """CPU reference gradient using only stable pyscf APIs (_int_vnl + numpy)."""
    if kpts is None:
        kpts_lst = np.zeros((1, 3))
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))

    fakecell, hl_blocks = fake_cell_vnl(cell)
    intors_d = ('int1e_ipovlp', 'int1e_r2_origi_ip2', 'int1e_r4_origi_ip2')
    ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    ppnl_half_ip2 = _int_vnl(cell, fakecell, hl_blocks, kpts_lst, intors_d, comp=3)
    if len(ppnl_half_ip2[0]) > 0:
        for k in range(len(kpts_lst)):
            ppnl_half_ip2[0][k] *= -1

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()

    dm = np.asarray(dm).reshape(-1, nao, nao)
    if gamma_point(kpts_lst):
        dm = dm.real
    dm_dmH = dm + dm.transpose(0, 2, 1).conj()

    grad = np.zeros([cell.natm, 3], dtype=np.complex128)
    dppnl = np.zeros((nkpts, 3, nao, nao), dtype=np.complex128)

    for k in range(nkpts):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp = np.zeros((hl_dim, nd, nao), dtype=np.complex128)
            dilp = np.zeros((hl_dim, 3, nd, nao), dtype=np.complex128)
            for i in range(hl_dim):
                p0 = offset[i]
                if len(ppnl_half[i]) > 0:
                    ilp[i] = ppnl_half[i][k, p0:p0+nd]
                if len(ppnl_half_ip2[i]) > 0:
                    dilp[i] = ppnl_half_ip2[i][k, :, p0:p0+nd]
                offset[i] = p0 + nd
            dppnl_k = np.einsum('idlp,ij,jlq->dpq', dilp.conj(), hl, ilp)
            dppnl[k] += dppnl_k
            i_pp_atom = fakecell._bas[ib, 0]
            grad[i_pp_atom] += np.einsum('dpq,qp->d', dppnl_k, dm_dmH[k])

    aoslices = cell.aoslice_by_atom()
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia][2:]
        grad[ia] -= np.einsum('kdpq,kqp->d', dppnl[:, :, p0:p1, :],
                              dm_dmH[:, :, p0:p1])

    return grad.real


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

        nao = cell.nao_nr()
        np.random.seed(42)
        dm = np.random.randn(nao, nao)
        dm = dm + dm.T

        # CPU reference: compute using CPU _int_vnl + Python contraction
        # (avoids importing pyscf.vppnl_nuc_grad which may not exist in pyscf<2.11)
        grad_cpu = _cpu_vppnl_nuc_grad(cell, dm)

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


class TestVppnlNucGradKpts(unittest.TestCase):
    """Test full vppnl_nuc_grad with non-zero k-points."""

    @staticmethod
    def _build_random_dm_kpts(nao, nkpts, seed=42):
        """Build a hermitian-per-k random complex DM, shape (nkpts, nao, nao)."""
        rng = np.random.default_rng(seed)
        dm = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        for k in range(nkpts):
            a = rng.standard_normal((nao, nao)) + 1j * rng.standard_normal((nao, nao))
            dm[k] = a + a.conj().T  # hermitian
        return dm

    def _compare_grad_kpts(self, cell, kpts, places=6):
        import cupy as cp
        from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

        nao = cell.nao_nr()
        nkpts = len(kpts)
        dm = self._build_random_dm_kpts(nao, nkpts)

        grad_cpu = _cpu_vppnl_nuc_grad(cell, dm, kpts=kpts)
        grad_gpu = vppnl_nuc_grad(cell, cp.asarray(dm), kpts=kpts)

        err = np.max(np.abs(grad_gpu - grad_cpu))
        grad_max = np.max(np.abs(grad_cpu))
        self.assertAlmostEqual(err / max(grad_max, 1e-15), 0, places,
                               f"Gradient error at {nkpts} kpts: "
                               f"max|err|={err:.2e}, |grad_max|={grad_max:.2e}")

    def test_carbon_single_kpt(self):
        # Single non-zero kpt in 1/Bohr (~midpoint of the BZ)
        kpts = np.array([[0.1, 0.2, 0.3]])
        self._compare_grad_kpts(cell_c, kpts)

    def test_carbon_kpts_mesh(self):
        # 2x2x2 Monkhorst-Pack mesh — 8 k-points, including gamma. Even though
        # the list contains gamma, gamma_point(kpts_lst) is False for any list
        # not entirely at gamma, so vppnl_nuc_grad takes the CPU-_int_vnl path
        # and contracts on GPU — same code path as the single non-zero kpt test
        # but with 8 k-points instead of 1.
        kpts = cell_c.make_kpts([2, 2, 2])
        self._compare_grad_kpts(cell_c, kpts)

    def test_silicon_single_kpt(self):
        kpts = np.array([[0.1, 0.2, 0.3]])
        self._compare_grad_kpts(cell_si, kpts)

    def test_silicon_kpts_mesh(self):
        kpts = cell_si.make_kpts([2, 2, 2])
        self._compare_grad_kpts(cell_si, kpts)

    def test_iron_single_kpt(self):
        # Fe exercises the d-projector hl_dim=2 path with a complex DM.
        # Loose tolerance (places=5) consistent with TestVppnlNucGrad gamma case.
        kpts = np.array([[0.1, 0.2, 0.3]])
        self._compare_grad_kpts(cell_fe, kpts, places=5)


class TestFiniteDifference(unittest.TestCase):
    """Validate gradient via finite difference of the nonlocal PP energy."""

    def _fd_check(self, cell, atom_id=1, cart_id=0, places=5):
        import cupy as cp
        from pyscf.pbc import scf
        from pyscf.pbc.gto.pseudo.pp_int import get_pp_nl
        from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

        mf = scf.RHF(cell)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()

        grad = vppnl_nuc_grad(cell, cp.asarray(dm))

        coords = cell.atom_coords().copy()
        energies = {}
        for sign in (+1, -1):
            cell_d = cell.copy()
            coords_d = coords.copy()
            coords_d[atom_id, cart_id] += sign * disp
            cell_d.atom = [[cell.atom_symbol(ia), coords_d[ia]]
                           for ia in range(cell.natm)]
            cell_d.build()
            vppnl = get_pp_nl(cell_d)
            energies[sign] = np.einsum('ij,ji', vppnl, dm).real

        fd_grad = (energies[+1] - energies[-1]) / (2 * disp)
        self.assertAlmostEqual(
            grad[atom_id, cart_id], fd_grad, places,
            f"FD check: analytical={grad[atom_id,cart_id]:.6e}, "
            f"fd={fd_grad:.6e}, diff={grad[atom_id,cart_id]-fd_grad:.2e}")

    def test_carbon_fd(self):
        self._fd_check(cell_c, atom_id=1, cart_id=0, places=5)

    def test_silicon_fd(self):
        self._fd_check(cell_si, atom_id=0, cart_id=2, places=5)

    @pytest.mark.slow
    def test_iron_fd(self):
        self._fd_check(cell_fe, atom_id=1, cart_id=0, places=4)

    def test_pseudo_gradient_term_with_zero_nexp(self):
        cell = pyscf.M(
            a = np.array([
                [3.18693029, 0.0, 0.0],
                [1.593466157846262, 2.759963819342879, 0.0],
                [1.5934664345206309, 0.9199872811273334, 2.6021185638285855],
            ]),
            atom = """
                Ga 0 -0 0
                N 0.27 0.25 0.25
            """,
            unit = "Angstrom",
            fractional = True,
            basis = {"Ga": """
                Ga DZVP-MOLOPT-PBE-GTH-q13 DZVP-MOLOPT-GGA-GTH-q13
                1
                2 0 3 5 2 2 2 1
                        3.01656447397946    -1.48615548285917E-02    -4.82983759170633E-02     2.02987429358091E-02    -2.13010036960508E-02    -6.10380904630582E-01    -2.58289780414473E-01     5.55521291693774E-04
                        1.10154483600156     4.01248950871096E-01     3.52468529188874E-01     1.33450451355874E-01    -1.72653030419605E-01    -4.80974524787380E-01    -2.30702268754674E-01     1.31822285722899E-01
                        0.40429913834530    -2.68052946527183E-01    -4.34387846771425E-01    -3.62845354157947E-01     5.18953771543878E-01    -3.63556802820521E-01     6.94523681159656E-01     8.93294520205134E-01
                        0.15745703748705    -8.26716943904166E-01     2.37442900862471E-02    -7.36191953337155E-01     3.01123009465079E-01    -1.38575446856837E-01     5.57839790724372E-01     4.07171709684337E-01
                        0.05692267130844    -2.88855442630589E-01     8.27121240422123E-01    -5.55101142265710E-01    -7.80864171135513E-01    -4.97947071392160E-02     2.53060753645856E-01     1.37326888851730E-01
            """, # Largest exponent removed
            "N": """
                N DZVP-MOLOPT-PBE-GTH-q5 DZVP-MOLOPT-GGA-GTH-q5
                1
                2 0 2 4 2 2 1
                        2.81966237794259    -1.74044102679302E-01     4.24004069220353E-02    -2.98226721897575E-01     1.34580220978959E-01     1.14821628218594E-01
                        1.09114390677598     4.41288051809814E-01    -9.13540901978963E-02    -5.60270180894578E-01     2.83662239065009E-01     4.43570475758278E-01
                        0.43100900982055     8.58813021027908E-01    -6.51904223705357E-02    -6.89335539768301E-01     1.57118280926414E-01     8.82189870978279E-01
                        0.13893847703177     1.11941135993420E-01     9.91688841490055E-01    -3.33973496542287E-01    -9.35099219564131E-01     8.84793677023504E-02
            """}, # Largest exponent removed
            pseudo = {"Ga": """
                Ga GTH-PBE-q13 GTH-GGA-q13
                    2    1   10    0
                    0.49000018487159       0
                    3
                    0.41677483095310       3   10.48679119269639   -4.92176814704009    0.87070493953275
                                                                    7.77018207637078   -2.24815160599927
                                                                                        1.78441528219626
                    0.56962661099353       2    1.77860037827899    0.19586036552562
                                                                -0.23168154587648
                    0.23814730101676       1  -16.24818353736915
                """, "N": """
                    N  GTH-PBE-q5 GTH-GGA-q5
                        2    3    0    0
                        0.28382600053810       2  -12.41517350030142    1.86813618209744
                        1
                        0.25541754972811       1   13.63124869974610
                """}, # Unmodified from pyscf/pbc/gto/pseudo/POTENTIAL_UZH, made a copy because it requires a relatively new version of pyscf
            precision = 1e-6,
            verbose = 0,
        )

        kmesh = [1,3,3]
        kpts = cell.make_kpts(kmesh)
        mf = cell.KRKS(xc = "pbe", kpts = kpts)
        mf.conv_tol = 1e-1
        mf = mf.to_gpu()

        mf = mf.multigrid_numint()
        mf.kernel()

        ni = mf._numint
        dm0 = mf.make_rdm1()
        rho_g = multigrid_v2.evaluate_density_on_g_mesh(ni, dm0, kpts)
        rho_g = rho_g[0,0]

        dx = 1e-5
        numerical_gradient = np.zeros([cell.natm, 3])

        def get_pp_local_energy(cell):
            vpplocG = multigrid_v1.eval_vpplocG(cell, cell.mesh)
            e_vpplocG = cp.einsum("g,g->", rho_g.conj(), vpplocG) / cell.vol
            assert abs(e_vpplocG.imag) < 1e-8
            return float(e_vpplocG.real)

        cell_copy = cell.copy()
        cell_copy.fractional = False
        for i_atom in range(cell.natm):
            for i_xyz in range(3):
                xyz_p = cell.atom_coords(unit='Bohr')
                xyz_p[i_atom, i_xyz] += dx
                cell_copy.set_geom_(xyz_p, unit='Bohr')
                cell_copy.build()
                e_p = get_pp_local_energy(cell_copy)

                xyz_m = cell.atom_coords(unit='Bohr')
                xyz_m[i_atom, i_xyz] -= dx
                cell_copy.set_geom_(xyz_m, unit='Bohr')
                cell_copy.build()
                e_m = get_pp_local_energy(cell_copy)

                numerical_gradient[i_atom, i_xyz] = (e_p - e_m) / (2 * dx)

        analytical_gradient = multigrid_v1.eval_vpplocG_SI_gradient(cell, cell.mesh, rho_g)
        analytical_gradient = analytical_gradient.get()

        assert np.max(np.abs(numerical_gradient - analytical_gradient)) < 1e-8


if __name__ == '__main__':
    unittest.main()
