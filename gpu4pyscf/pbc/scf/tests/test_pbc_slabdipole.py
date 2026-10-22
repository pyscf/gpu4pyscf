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
from pyscf.lib.parameters import BOHR
from pyscf.pbc import gto, dft as cpu_dft
from pyscf.pbc.scf import addons as cpu_addons
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import slab_dipole_correction

cpu_correction = getattr(cpu_addons, 'slab_dipole_correction')

def make_cell(polar=True):
    return gto.M(
        a=np.diag([4., 4., 12.]),
        atom='F 2 2 5.00; H 2 2 5.92' if polar else 'H 2 2 5.6; H 2 2 6.4',
        basis='gth-szv', pseudo='gth-pade', mesh=[21, 21, 65],
        precision=1e-9, verbose=0)


class KnownValues(unittest.TestCase):
    def test_pyscf_potential_and_energy(self):
        cell = make_cell()
        kpts = cell.make_kpts([3, 1, 1])
        center = 5.46 / BOHR
        cut = center + cell.lattice_vectors()[2, 2]/2
        for name in ('KRKS', 'KUKS'):
            for becke in (False, True):
                with self.subTest(name=name, becke=becke):
                    ref = getattr(cpu_dft, name)(cell, kpts, xc='pbe')
                    mf = getattr(dft, name)(cell, kpts, xc='pbe')
                    ref.grids = cpu_dft.UniformGrids(cell)
                    if becke:
                        ref.grids = cpu_dft.BeckeGrids(cell).set(level=1).build()
                        mf.grids = dft.BeckeGrids(cell)
                        mf.grids.coords = cp.asarray(ref.grids.coords)
                        mf.grids.weights = cp.asarray(ref.grids.weights)
                    dm = ref.get_init_guess()
                    if name == 'KUKS':
                        dm[0] *= 1.1
                        dm[1] *= .9
                    dm_gpu = cp.asarray(dm)
                    v0_ref, v0 = ref.get_veff(dm=dm), mf.get_veff(dm=dm_gpu)
                    cpu_correction(ref, slab_center=center)
                    self.assertIs(slab_dipole_correction(mf, z0=center, cut=cut), mf)
                    self.assertEqual(mf.__class__.__name__,
                                     f'SlabDipoleCorrection{name}')
                    vref, v = ref.get_veff(dm=dm), mf.get_veff(dm=dm_gpu)
                    np.testing.assert_allclose((v-v0).get(), vref-v0_ref, atol=2e-9)
                    self.assertAlmostEqual(v.edip, vref.exc-v0_ref.exc, delta=2e-10)
                    self.assertAlmostEqual(v.ecoul, v0.ecoul, delta=1e-12)
                    h1 = mf.get_hcore()
                    de = mf.energy_elec(dm_gpu, h1, v)[0] - mf.energy_elec(dm_gpu, h1, v0)[0]
                    self.assertAlmostEqual(de, v.edip, delta=1e-12)

    def test_self_consistent_pyscf(self):
        for name in ('KRKS', 'KUKS'):
            with self.subTest(name=name):
                cell = make_cell()
                ref = getattr(cpu_dft, name)(cell, xc='pbe')
                ref.grids = cpu_dft.UniformGrids(cell)
                mf = slab_dipole_correction(getattr(dft, name)(cell, xc='pbe'))
                dm = ref.get_init_guess()
                mf.get_veff(dm=cp.asarray(dm))
                cut = mf.dipole_cut
                length = cell.lattice_vectors()[2, 2]
                center = (cut + length/2) % length
                mf.z0 = center
                mf.kernel(dm0=cp.asarray(dm))
                self.assertTrue(mf.converged)
                self.assertEqual(mf.dipole_cut, cut)

                dm_gpu = mf.make_rdm1()
                plain = getattr(dft, name)(cell, xc='pbe')
                v0 = plain.get_veff(dm=dm_gpu)
                v = mf.get_veff(dm=dm_gpu)

                dm_cpu = dm_gpu.get()
                v0_ref = ref.get_veff(dm=dm_cpu)
                cpu_correction(ref, slab_center=center)
                vref = ref.get_veff(dm=dm_cpu)

                np.testing.assert_allclose((v-v0).get(), vref-v0_ref, atol=2e-8)
                self.assertAlmostEqual(v.edip, vref.exc-v0_ref.exc, delta=2e-10)
                self.assertGreater(v.edip, 1e-5)

    def test_multigrid(self):
        cell = make_cell()
        kpts = cell.make_kpts([2, 1, 1])
        mf = dft.KUKS(cell, kpts, xc='pbe').multigrid_numint(mesh=[23, 23, 69])
        ref = cpu_dft.KUKS(cell, kpts, xc='pbe')
        ref.grids = cpu_dft.UniformGrids(cell).set(mesh=mf._numint.mesh)
        dm = ref.get_init_guess()
        dm_gpu = cp.asarray(dm)
        v0, v0_ref = mf.get_veff(dm=dm_gpu), ref.get_veff(dm=dm)
        slab_dipole_correction(mf, z0=10., cut=10. + cell.lattice_vectors()[2, 2]/2)
        cpu_correction(ref, slab_center=10.)
        v, vref = mf.get_veff(dm=dm_gpu), ref.get_veff(dm=dm)
        np.testing.assert_allclose((v-v0).get(), vref-v0_ref, atol=2e-9)
        self.assertAlmostEqual(v.edip, vref.exc-v0_ref.exc, delta=2e-10)

    def test_periodic_vacuum_and_reset(self):
        cell = make_cell()
        coords = cell.atom_coords()
        coords[:, 2] -= 5.5 / BOHR
        coords[:, 2] %= cell.lattice_vectors()[2, 2]
        cell.set_geom_(coords, unit='Bohr')
        mf = slab_dipole_correction(dft.KRKS(cell))
        dm = mf.get_init_guess()
        mf.get_veff(dm=dm)
        length = cell.lattice_vectors()[2, 2]
        cut = mf.dipole_cut
        self.assertLess(abs(cut-length/2), length/8)
        ref = cpu_dft.KRKS(cell)
        ref.grids = cpu_dft.UniformGrids(cell)
        mf.z0 = (cut + length/2) % length
        cpu_correction(ref, slab_center=mf.z0)
        np.testing.assert_allclose(mf.get_veff(dm=dm).get(),
                                   ref.get_veff(dm=dm.get()), atol=2e-8)
        self.assertIsNotNone(mf.dipole_offset)
        mf.reset()
        self.assertIsNone(mf.dipole_cut)
        self.assertIsNone(mf.dipole_z0)
        self.assertIsNone(mf.dipole_offset)

    def test_potential_gauges(self):
        cell = make_cell()
        length = cell.lattice_vectors()[2, 2]
        cut = .9 * length
        mf = dft.KRKS(cell, cell.make_kpts([3, 1, 1]), xc='lda,vwn')
        dm = mf.get_init_guess()

        slab_dipole_correction(mf, cut=cut)
        vavg = mf.get_veff(dm=dm)
        self.assertAlmostEqual(mf.dipole_z0, (cut + length/2) % length)

        mf.z0 = .37 * length
        vzero = mf.get_veff(dm=dm)
        self.assertAlmostEqual(mf.dipole_z0, mf.z0)
        self.assertAlmostEqual(vzero.dipole, vavg.dipole)
        self.assertAlmostEqual(vzero.edip, vavg.edip)

        field = 4 * np.pi * vavg.dipole / cell.vol
        offset = (mf.z0 - cut) % length - length/2
        shift = field * offset
        overlap = mf.get_ovlp()
        np.testing.assert_allclose(
            (vzero-vavg).get(), (shift*overlap).get(), atol=2e-10)

        hcore = mf.get_hcore()
        eavg, cavg = mf.eig(hcore+vavg, overlap)
        ezero, czero = mf.eig(hcore+vzero, overlap)
        np.testing.assert_allclose((ezero-eavg).get(), shift, atol=2e-10)

        dmavg = mf.make_rdm1(cavg, mf.get_occ(eavg, cavg))
        dmzero = mf.make_rdm1(czero, mf.get_occ(ezero, czero))
        np.testing.assert_allclose(dmzero.get(), dmavg.get(), atol=2e-9)

    def test_energy_derivative(self):
        cell = make_cell()
        mf = dft.KUKS(cell, cell.make_kpts([3, 1, 1]), xc='lda,vwn')
        dm = mf.get_init_guess()
        rng = np.random.default_rng(12)
        ddm = cp.asarray(rng.random(dm.shape) + 1j*rng.random(dm.shape))
        ddm = (ddm + ddm.swapaxes(-1, -2).conj()) * .01
        overlap = mf.get_ovlp()
        # remove electron number change
        ddm -= dm * (cp.einsum('skij,kji->', ddm, overlap).real /
                     cp.einsum('skij,kji->', dm, overlap).real)
        v0 = mf.get_veff(dm=dm)
        slab_dipole_correction(mf, cut=.9*cell.lattice_vectors()[2, 2])
        for z0 in (None, 10.):
            with self.subTest(z0=z0):
                mf.z0 = z0
                v = mf.get_veff(dm=dm)
                h = 1e-4
                ep = mf.get_veff(dm=dm+h*ddm).edip
                em = mf.get_veff(dm=dm-h*ddm).edip
                derivative = cp.einsum('skij,skji->', v-v0, ddm).real / len(mf.kpts)
                self.assertAlmostEqual((ep-em)/(2*h), float(derivative), delta=1e-9)

    def test_nonpolar(self):
        cell = make_cell(polar=False)
        mf = dft.KRKS(cell, xc='pbe')
        dm = mf.get_init_guess()
        slab_dipole_correction(mf)
        self.assertLess(mf.get_veff(dm=dm).edip, 1e-10)


if __name__ == '__main__':
    print("Full test for slab dipole correction.")
    unittest.main()
