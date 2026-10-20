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
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import slab_dipole_correction
from gpu4pyscf.pbc.grad.slabdipole import dipole_grad


DISP = 1e-4 # finite difference step


def make_cell(dir_idx=2, boundary=False):
    a = np.diag([7., 8., 22.])
    coords = np.array([[2.1, 2.8, 10.], [2.8, 3.3, 11.7]])
    if boundary:
        coords[:, 2] = (coords[:, 2] - 10.8) % a[2, 2]
    order = [0, 1, 2]
    order[dir_idx], order[2] = order[2], order[dir_idx]
    return gto.M(
        a=a[np.ix_(order, order)],
        atom=list(zip(['F', 'H'], coords[:, order])), unit='Bohr',
        basis='gth-szv', pseudo='gth-pade', mesh=np.array([31, 33, 91])[order],
        precision=1e-10, verbose=0)


def make_mf(cell, name='RKS', dir_idx=2, cut=None, z0=None, becke=False,
            multigrid=False):
    if name.startswith('K'):
        kmesh = [2, 2, 2]
        kmesh[dir_idx] = 1
        mf = getattr(dft, name)(cell, cell.make_kpts(kmesh), xc='lda,vwn')
    else:
        mf = getattr(dft, name)(cell, xc='lda,vwn')
    if becke:
        mf.grids = dft.BeckeGrids(cell).set(level=2)
    if multigrid:
        mf = mf.multigrid_numint(mesh=np.asarray(cell.mesh) + 2)
    if cut is None:
        cut = .95 * cell.lattice_vectors()[dir_idx, dir_idx]
    mf = slab_dipole_correction(mf, dir_idx=dir_idx, cut=cut, z0=z0)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.chkfile = None
    return mf


def displaced_cell(cell, atom, axis, step):
    coords = cell.atom_coords()
    coords[atom, axis] += step
    displaced = cell.set_geom_(coords, unit='Bohr', inplace=False)
    displaced.mesh = cell.mesh.copy()
    displaced.rcut = cell.rcut
    return displaced


def finite_difference(cell, energy):
    de = np.zeros((cell.natm, 3))
    for atom in range(cell.natm):
        for axis in range(3):
            ep = energy(displaced_cell(cell, atom, axis, DISP))
            em = energy(displaced_cell(cell, atom, axis, -DISP))
            de[atom, axis] = (ep - em) / (2 * DISP)
    return de


class KnownValues(unittest.TestCase):
    def check_explicit(self, name='RKS', dir_idx=2, boundary=False,
                       becke=False, multigrid=False):
        cell = make_cell(dir_idx, boundary)
        length = cell.lattice_vectors()[dir_idx, dir_idx]
        cut = .5 * length if boundary else .95 * length

        z0 = (cut + length/2) % length
        def build(c):
            return make_mf(c, name, dir_idx, cut, z0, becke, multigrid)
        mf = build(cell)
        dm = mf.get_init_guess()

        rng = np.random.default_rng(12)
        ddm = rng.random(dm.shape)

        if name.startswith('K') and not multigrid:
            ddm = ddm + 1j * rng.random(dm.shape)
        ddm = ddm + ddm.swapaxes(-1, -2).conj()
        dm = dm + cp.asarray(ddm) * .01
        if name in ('UKS', 'KUKS'):
            dm[0] *= 1.1
            dm[1] *= .9
        grad = mf.nuc_grad_method()
        grad.grid_response = becke
        analytic = dipole_grad(grad, dm)
        numeric = finite_difference(cell, lambda c: build(c).get_veff(dm=dm).edip)
        self.assertGreater(abs(numeric).max(), 1e-5)
        np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=0)

    def test_explicit_finite_difference(self):
        for name in ('RKS', 'UKS', 'KRKS', 'KUKS'):
            with self.subTest(name=name):
                self.check_explicit(name)

    def test_other_normals_and_periodic_branch(self):
        for dir_idx in (0, 1):
            with self.subTest(dir_idx=dir_idx):
                self.check_explicit('KRKS', dir_idx=dir_idx, boundary=True)

    def test_multigrid_explicit_finite_difference(self):
        self.check_explicit('KUKS', multigrid=True)

    def check_scf(self, name='RKS', multigrid=False, becke=False):
        cell = make_cell()
        def build(c, z0=None):
            return make_mf(c, name, z0=z0, multigrid=multigrid, becke=becke)
        mf = build(cell)
        mf.kernel()
        self.assertTrue(mf.converged)

        grad = mf.nuc_grad_method()
        grad.grid_response = becke
        analytic = grad.kernel()
        self.assertTrue(np.isfinite(analytic).all())
        self.assertTrue(np.isnan(grad.stress).all())
 
        dm0 = mf.make_rdm1()
        def energy(c):
            displaced = build(c)
            e = displaced.kernel(dm0=dm0)
            self.assertTrue(displaced.converged)
            return e
        numeric = finite_difference(cell, energy)
        np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=0)

        # A constant potential shifts orbital energies and the Pulay term.
        # Check its cancellation against the explicit overlap derivative.
        shifted = build(cell, z0=.37 * cell.lattice_vectors()[2, 2])
        shifted.kernel(dm0=dm0)
        self.assertTrue(shifted.converged)
        shifted_grad = shifted.nuc_grad_method()
        shifted_grad.grid_response = becke
        np.testing.assert_allclose(shifted_grad.kernel(), analytic, atol=2e-7, rtol=0)
        self.assertAlmostEqual(shifted.e_tot, mf.e_tot, delta=1e-9)

    def test_scf_finite_difference(self):
        for name in ('RKS', 'UKS', 'KRKS', 'KUKS'):
            with self.subTest(name=name):
                self.check_scf(name)

    def test_multigrid_scf_finite_difference(self):
        self.check_scf('RKS', multigrid=True)

    def test_automatic_cut_and_scanner(self):
        cell = make_cell()
        mf = make_mf(cell)
        mf.cut = None
        mf.kernel()
        self.assertTrue(mf.converged)
        cut = mf.dipole_cut
        analytic = mf.nuc_grad_method().kernel()
        self.assertEqual(mf.dipole_cut, cut)
        mf.cut = cut
        scanner = mf.nuc_grad_method().as_scanner()
        ep, _ = scanner(displaced_cell(cell, 1, 2, DISP))
        self.assertTrue(scanner.base.converged)
        em, _ = scanner(displaced_cell(cell, 1, 2, -DISP))
        self.assertTrue(scanner.base.converged)
        self.assertEqual(scanner.base.dipole_cut, cut)
        self.assertAlmostEqual(analytic[1, 2], (ep-em)/(2*DISP), delta=1e-5)


if __name__ == '__main__':
    unittest.main()
