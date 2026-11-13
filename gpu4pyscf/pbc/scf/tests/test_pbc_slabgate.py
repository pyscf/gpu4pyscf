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
from unittest import mock
import numpy as np
import cupy as cp
from pyscf.pbc import gto, dft as cpu_dft
from pyscf.pbc.dft import numint as cpu_numint
from gpu4pyscf.pbc import dft, df
from gpu4pyscf.pbc.scf import slabgate
from gpu4pyscf.pbc.scf.addons import gate_charge, slab_dipole_correction
from gpu4pyscf.pbc.scf.slabdipole import (
    SlabDipoleCorrection, _sawtooth, _slab_charge)
from gpu4pyscf.pbc.scf.slabgate import _sheet_potential, _barrier_potential


def make_cell(charge=2, dimension=3):
    length, zshift, nz = (60., -10., 249) if dimension == 2 else (22., 0., 91)
    return gto.M(
        a=np.diag([7., 8., length]), unit='Bohr',
        atom=f'F 2.1 2.8 {10.+zshift}; H 2.8 3.3 {11.7+zshift}',
        charge=charge, basis='gth-szv', pseudo='gth-pade',
        dimension=dimension, mesh=[31, 33, nz], precision=1e-10, verbose=0)


def make_mf(cell, name='RKS', multigrid=False, becke=False):
    if name.startswith('K'):
        mf = getattr(dft, name)(cell, cell.make_kpts([2, 1, 1]), xc='lda,vwn')
    else:
        mf = getattr(dft, name)(cell, xc='lda,vwn')
    if multigrid:
        mf = mf.multigrid_numint(mesh=np.asarray(cell.mesh) + 2)
    if becke:
        mf.grids = dft.BeckeGrids(cell).set(level=1)
    mf.chkfile = None
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    return mf


class KnownValues(unittest.TestCase):
    def test_profiles(self):
        length, area, qc, zgate = 22., 56., 2., 17.
        z = zgate + (np.arange(10000) + .5) * length / 10000
        phi = _sheet_potential(z, zgate, length, area, qc)
        # Both potentials use a zero cell average, and Eq. (5)
        self.assertAlmostEqual(float(phi.mean()), 0., delta=1e-7)
        self.assertAlmostEqual(float(_sheet_potential(zgate, zgate, length, area, qc)),
                               np.pi*qc*length/(3*area))
        for width in (0., 2.):
            saw = _sawtooth(z, zgate, length, width)
            self.assertAlmostEqual(float(saw.mean()), 0., delta=1e-12)

        step = 1e-5
        points = np.array([1.2, 9.4, 19.8])
        fd = (_sheet_potential(points+step, zgate, length, area, qc) -
              _sheet_potential(points-step, zgate, length, area, qc)) / (2*step)
        cp.testing.assert_allclose(fd, _sheet_potential(
            points, zgate, length, area, qc, deriv=1), atol=1e-10)

        # The barrier crosses the cell boundary and has two linear ramps.
        points = np.array([18., 18.5, 19.5, 21.9, .1, .5, 1.5, 2.1])
        vb = _barrier_potential(points, 18., 2., length, .05, 1.)
        cp.testing.assert_allclose(vb, [0., .025, .05, .05, .05, .05, .025, 0.])
        slope = _barrier_potential(
            np.array([18.5, 20., 1.5, 2.1]), 18., 2., length, .05, 1., deriv=1)
        cp.testing.assert_allclose(slope, [.05, 0., -.05, 0.])

    def test_cpu_reference(self):
        length, area, zgate = 22., 56., 17.
        for name in ('RKS', 'KUKS'):
            with self.subTest(name=name):
                cell = make_cell()
                mf = make_mf(cell, name)
                if name == 'KUKS':
                    mf.kpts = cell.make_kpts([2, 1, 1], scaled_center=[.17, 0, 0])
                dm = mf.get_init_guess()
                if name == 'KUKS':
                    dm[0] *= 1.1
                    dm[1] *= .9
                h0 = mf.get_hcore()
                v0 = mf.get_veff(dm=dm)
                e0 = mf.energy_tot(dm, h0, v0)
                gate_charge(mf, zgate=zgate, barrier=True, barrier_start=15.,
                            barrier_end=19., barrier_width=.7)
                h = mf.get_hcore()
                # Reuse v0 to isolate the gate and barrier energy.
                e = mf.energy_tot(dm, h, v0)

                coords = mf.grids.coords.get()
                weights = mf.grids.weights.get()
                aos = cpu_numint.eval_ao_kpts(cell, coords, kpts=mf.kpts)
                nkpts = len(mf.kpts)
                qc = -cell.charge / nkpts
                s = (coords[:, 2]-zgate+length/2) % length - length/2
                phi = 2*np.pi*qc/area * (-abs(s) + s*s/length + length/6)
                t = (coords[:, 2]-15.) % length
                barrier = .05 * np.clip(np.minimum(t, 4.-t)/.7, 0., 1.)
                ref = []
                for ao in aos:
                    ref.append(ao.conj().T @ ((weights*(-phi+barrier))[:, None]*ao))
                ref = np.asarray(ref)
                np.testing.assert_allclose((h-h0).get().reshape(ref.shape), ref,
                                           atol=2e-9)

                charge_dm = dm.get()
                if name == 'KUKS':
                    charge_dm = charge_dm[0] + charge_dm[1]
                charge_dm = charge_dm.reshape(ref.shape)
                electronic = np.einsum('kij,kji->', charge_dm, ref).real / nkpts
                zi = cell.atom_coords()[:, 2]
                si = (zi-zgate+length/2) % length - length/2
                ionic = np.dot(cell.atom_charges(),
                               2*np.pi*qc/area*(-abs(si)+si*si/length+length/6))
                self_energy = np.pi*qc**2*length/(6*area)
                # Ions lie outside the barrier; the sheet is on its plateau.
                gate_barrier = -qc*.05
                self.assertAlmostEqual(e-e0, electronic+ionic+self_energy+gate_barrier,
                                       delta=2e-9)
                self.assertAlmostEqual(e-e0, mf.scf_summary['egate'] +
                                       mf.scf_summary['ebarrier'], delta=1e-10)

    def check_smearing_energy(self, dimension):
        sigma, entropy = .02, 1.25
        for name in ('RKS', 'UKS', 'KRKS', 'KUKS'):
            cell = make_cell(dimension=dimension)
            zgate = cell.atom_coords()[0, 2] + 7.
            gate_kwargs = dict(zgate=zgate, barrier=True, barrier_start=zgate-2.,
                               barrier_end=zgate+2., barrier_width=.7)
            reference = gate_charge(make_mf(cell, name), **gate_kwargs)
            dm = reference.get_init_guess()
            h = reference.get_hcore()
            v = reference.get_veff(dm=dm)
            energy = reference.energy_tot(dm, h, v)
            for gate_first in (False, True):
                with self.subTest(name=name, gate_first=gate_first):
                    mf = make_mf(cell, name)
                    if gate_first:
                        mf = gate_charge(mf, **gate_kwargs).smearing(sigma)
                    else:
                        mf = gate_charge(mf.smearing(sigma), **gate_kwargs)
                    # Hold the density and entropy fixed to isolate energy
                    # accounting from the occupation and SCF algorithms.
                    mf.entropy = entropy
                    for _ in range(2):
                        self.assertAlmostEqual(mf.energy_tot(dm, h, v), energy,
                                               delta=1e-10)
                        self.assertAlmostEqual(mf.e_free, energy-sigma*entropy,
                                               delta=1e-10)
                        self.assertAlmostEqual(mf.e_zero, energy-sigma*entropy/2,
                                               delta=1e-10)
                        for key in ('egate', 'ebarrier'):
                            self.assertAlmostEqual(mf.scf_summary[key],
                                                   reference.scf_summary[key],
                                                   delta=1e-10)

    def test_smearing_energy(self):
        self.check_smearing_energy(dimension=3)

    def test_smearing_energy_2d(self):
        self.check_smearing_energy(dimension=2)

    def test_charge_per_cell(self):
        cell = make_cell()
        gamma = gate_charge(make_mf(cell))
        with mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', 2./22.):
            mf = gate_charge(make_mf(cell, 'KUKS'), zgate=17., dipole_gap=.1)
        self.assertEqual(_slab_charge(gamma, cell), 2.)
        self.assertEqual(_slab_charge(mf, cell), 1.)
        dm = mf.get_init_guess()
        ne = float(cp.einsum('skij,kji->', dm, mf.get_ovlp()).real) / len(mf.kpts)
        self.assertAlmostEqual(ne, cell.atom_charges().sum()-1., delta=1e-9)
        vc, _ = mf._gate_matrices(cell, mf.kpts)

        # At fixed density, removing the sheet removes its dipole contribution.
        v = mf.get_veff(dm=dm)
        self.assertAlmostEqual(mf.dipole_offset, 0., delta=1e-14)
        self.assertEqual(mf.dipole_z0, mf.z0)
        cell.charge = 0
        no_sheet = mf.get_veff(dm=dm)
        cell.charge = 2
        self.assertAlmostEqual(v.dipole-no_sheet.dipole, 9.9, delta=1e-11)

        # Scale the total charge to retain two holes per primitive cell.
        cell.charge = 4
        self.assertEqual(_slab_charge(mf, cell), 2.)
        vc_fixed, _ = mf._gate_matrices(cell, mf.kpts)
        cp.testing.assert_allclose(vc_fixed, 2*vc, atol=1e-12)
        dm = mf.get_init_guess()
        ne = float(cp.einsum('skij,kji->', dm, mf.get_ovlp()).real) / len(mf.kpts)
        self.assertAlmostEqual(ne, cell.atom_charges().sum()-2., delta=1e-9)

    @mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', .4/22.)
    def test_reset_and_parameters(self):
        cell = make_cell()
        mf = gate_charge(make_mf(cell), zgate=17., gate_side=-1,
                         dipole_gap=.1, barrier=True)
        self.assertEqual(mf.gate_side, -1)
        self.assertAlmostEqual(mf.cut, 16.7)
        self.assertAlmostEqual(mf.z0, 5.7)
        self.assertAlmostEqual(mf.barrier_start, 16.5)
        self.assertAlmostEqual(mf.barrier_end, 18.7)
        self.assertEqual(mf.barrier_width, mf.dipole_width)

        self.assertIs(gate_charge(mf, zgate=17., gate_side=1,
                                  dipole_gap=.1, barrier=True), mf)
        self.assertEqual(mf.gate_side, 1)
        self.assertAlmostEqual(mf.cut, 17.3)
        self.assertAlmostEqual(mf.z0, 6.3)
        self.assertAlmostEqual(mf.barrier_start, 15.3)
        self.assertAlmostEqual(mf.barrier_end, 17.5)
        mf.reset()
        self.assertEqual(mf.zgate, 17.)
        self.assertEqual(mf.gate_side, 1)
        self.assertAlmostEqual(mf.cut, 17.3)
        self.assertAlmostEqual(mf.z0, 6.3)
        self.assertAlmostEqual(mf.barrier_start, 15.3)
        self.assertAlmostEqual(mf.barrier_end, 17.5)

    def test_cell_dimension(self):
        cell = make_cell()
        mf = make_mf(cell)
        cell.dimension = 1
        with self.assertRaises(NotImplementedError):
            gate_charge(mf)

    @mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', .4/22.)
    def test_barrier_positions(self):
        cell = make_cell()
        # Defaults and explicit intervals, including both periodic directions.
        valid = (
            dict(zgate=5., gate_side=-1),
            dict(zgate=17., gate_side=1),
            dict(zgate=5., gate_side=-1, barrier_start=3., barrier_end=7.),
            dict(zgate=17., gate_side=1, barrier_start=15., barrier_end=19.),
            dict(zgate=.2, gate_side=-1, barrier_start=21., barrier_end=2.),
            dict(zgate=21.8, gate_side=1, barrier_start=20., barrier_end=1.),
            dict(zgate=22.2, gate_side=-1, barrier_start=-1., barrier_end=24.),
        )
        for kwargs in valid:
            with self.subTest(**kwargs):
                gate_charge(make_mf(cell), barrier=True,
                            dipole_gap=.1, barrier_width=.2, **kwargs)

        invalid = (
            # Gate outside the interval or exactly on an edge.
            (dict(zgate=17., barrier_start=13., barrier_end=16.), 'Gate plane'),
            (dict(zgate=17., barrier_start=17., barrier_end=19.), 'Gate plane'),
            (dict(zgate=17., barrier_start=15., barrier_end=17.), 'Gate plane'),
            # Gate enclosed, but part of the return region is left exposed.
            (dict(zgate=5., gate_side=-1, barrier_start=4.6, barrier_end=7.),
             'dipole return region'),
            (dict(zgate=17., gate_side=1, barrier_start=15., barrier_end=17.4),
             'dipole return region'),
            (dict(zgate=.2, gate_side=-1, barrier_start=21.8, barrier_end=2.),
             'dipole return region'),
            (dict(zgate=21.8, gate_side=1, barrier_start=20., barrier_end=.2),
             'dipole return region'),
            # Both dipole planes lie inside, but the interval cuts through
            # the return region across the periodic boundary.
            (dict(zgate=17., gate_side=1, barrier_start=17.3, barrier_end=17.2),
             'dipole return region'),
        )
        for kwargs, message in invalid:
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    gate_charge(make_mf(cell), barrier=True,
                                dipole_gap=.1, barrier_width=.2, **kwargs)
                # Unused barrier coordinates do not restrict barrier=False.
                gate_charge(make_mf(cell), barrier=False,
                            dipole_gap=.1, barrier_width=.2, **kwargs)

    def test_defaults_and_reset_2d(self):
        cell = make_cell(dimension=2)
        length = cell.lattice_vectors()[2, 2]
        zgate = cell.atom_coords()[0, 2] + 7.
        # The 2D geometry must not depend on 3D dipole parameters.
        with mock.patch.object(slabgate, 'DIPOLE_WIDTH_RATIO', .4):
            mf = gate_charge(make_mf(cell), zgate=zgate, barrier=True,
                             dipole_gap=10., gate_side=1)
        self.assertNotIsInstance(mf, SlabDipoleCorrection)
        self.assertAlmostEqual(mf.barrier_start, zgate-.05*length)
        self.assertAlmostEqual(mf.barrier_end, zgate+.05*length)
        self.assertAlmostEqual(mf.barrier_width, .01*length)
        mf.reset(cell.copy())
        self.assertEqual(mf.zgate, zgate)
        self.assertAlmostEqual(mf.barrier_start, zgate-.05*length)
        self.assertAlmostEqual(mf.barrier_end, zgate+.05*length)
        self.assertNotIsInstance(mf, SlabDipoleCorrection)
        with self.assertRaisesRegex(ValueError, 'dimensionality'):
            mf.reset(make_cell())

    def test_outside_cell_gate_2d(self):
        cell = make_cell(dimension=2)
        length = cell.lattice_vectors()[2, 2]
        area = cell.vol / length
        # Both gates are outside the numerical cell and more than L/2
        # from the slab. The addon must preserve these physical coordinates.
        for zgate in (-length-5., length+5.):
            with self.subTest(zgate=zgate):
                start, end = zgate-2., zgate+2.
                mf = gate_charge(make_mf(cell), zgate=zgate, barrier=True,
                                 barrier_start=start, barrier_end=end,
                                 barrier_width=.5)
                self.assertEqual(mf.zgate, zgate)
                self.assertEqual(mf.barrier_start, start)
                self.assertEqual(mf.barrier_end, end)
                params = mf._gate_parameters(cell)
                self.assertEqual(params[1:4], (zgate, start, end))
                qc = -cell.charge
                ionic = -2*np.pi*qc/area * np.dot(
                    cell.atom_charges(), abs(cell.atom_coords()[:, 2]-zgate))
                egate, ebarrier = mf._gate_ionic_energy(cell)
                self.assertAlmostEqual(egate, ionic, delta=1e-12)
                self.assertAlmostEqual(ebarrier, -qc*mf.barrier_height, delta=1e-12)

    @mock.patch('gpu4pyscf.pbc.scf.hf.ALLOWED_FFT_MESH_SIZE', 0)
    def test_scf_2d_vs_3d(self):
        # Use FFTDF on both sides, including the 3D nuclear potential, to
        # isolate boundary-condition effects from the automatic MultiGrid path.
        # Large vacuum suppresses the remaining nonplanar 3D image interactions.
        for length, nz in ((60., 249), (80., 333)):
            with self.subTest(length=length):
                cell_2d = gto.M(
                    a=np.diag([7., 8., length]), unit='Bohr', dimension=2,
                    atom='C 2.1 2.8 0.; H 2.8 3.3 1.7',
                    charge=1, spin=0, basis='gth-szv', pseudo='gth-pade',
                    mesh=[31, 33, nz], precision=1e-10, verbose=0)
                cell_3d = cell_2d.copy()
                cell_3d.dimension = 3
                cell_3d.build()
                mf_2d = gate_charge(make_mf(cell_2d), zgate=-5., barrier=False)
                mf_3d = gate_charge(make_mf(cell_3d), zgate=-5., barrier=False)
                mf_2d.xc = 'pbe'
                mf_3d.xc = 'pbe'

                # Keep the return region well outside the electron-density
                # tails, on the vacuum side of the gate. Do not use its default
                # location immediately next to the gate for this comparison.
                cut = (-length/3) % length
                slab_dipole_correction(mf_3d, cut=cut, width=1.,
                                       z0=(cut+length/2) % length)
                self.assertNotIsInstance(mf_2d, SlabDipoleCorrection)
                self.assertIsInstance(mf_3d, SlabDipoleCorrection)

                # Independent SCFs from the same initial density; do not seed
                # the second calculation with the first converged solution.
                dm0 = mf_2d.get_init_guess()
                energy_2d = mf_2d.kernel(dm0=dm0.copy())
                energy_3d = mf_3d.kernel(dm0=dm0.copy())
                self.assertTrue(mf_2d.converged, '2D SCF did not converge')
                self.assertTrue(mf_3d.converged, '3D SCF did not converge')
                # Compare full total energies without fitting a constant shift.
                # Do not demand bitwise equality for different electrostatics.
                self.assertAlmostEqual(energy_2d, energy_3d, delta=1e-6)
                cp.testing.assert_allclose(mf_2d.make_rdm1(), mf_3d.make_rdm1(),
                                            atol=1e-5, rtol=0)

    def test_backend_selection_is_unchanged(self):
        # Attaching a gate does not build or validate the base SCF backend.
        mf = make_mf(make_cell(dimension=2))
        mf.with_df = df.GDF(mf.cell)
        mf.xc = 'pbe0'
        backend, integrator = mf.with_df, mf._numint
        zgate = mf.cell.atom_coords()[0, 2] + 7.
        self.assertIs(gate_charge(mf, zgate=zgate), mf)
        self.assertIs(mf.with_df, backend)
        self.assertIs(mf._numint, integrator)
        self.assertEqual(mf.xc, 'pbe0')


if __name__ == '__main__':
    print("Full tests for slabgate.")
    unittest.main()
