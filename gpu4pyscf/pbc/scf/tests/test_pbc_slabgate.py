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
from pyscf.pbc.dft import numint as cpu_numint
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.addons import gate_charge
from gpu4pyscf.pbc.scf.slabdipole import _sawtooth, _slab_charge
from gpu4pyscf.pbc.scf.slabgate import _sheet_potential, _barrier_potential


def make_cell(charge=2):
    return gto.M(
        a=np.diag([7., 8., 22.]), unit='Bohr',
        atom='F 2.1 2.8 10.; H 2.8 3.3 11.7',
        charge=charge, basis='gth-szv', pseudo='gth-pade',
        mesh=[31, 33, 91], precision=1e-10, verbose=0)


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

    def test_charge_per_cell(self):
        cell = make_cell()
        gamma = gate_charge(make_mf(cell))
        mf = gate_charge(make_mf(cell, 'KUKS'), zgate=17.,
                         dipole_width=2., dipole_gap=.1)
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

    def test_reset_and_parameters(self):
        cell = make_cell()
        mf = gate_charge(make_mf(cell), zgate=17., gate_side=-1,
                         dipole_width=.4, dipole_gap=.1, barrier=True)
        self.assertEqual(mf.gate_side, -1)
        self.assertAlmostEqual(mf.cut, 16.7)
        self.assertAlmostEqual(mf.z0, 5.7)
        self.assertAlmostEqual(mf.barrier_start, 16.5)
        self.assertAlmostEqual(mf.barrier_end, 18.7)
        self.assertEqual(mf.barrier_width, mf.dipole_width)

        self.assertIs(gate_charge(mf, zgate=17., gate_side=1,
                                  dipole_width=.4, dipole_gap=.1, barrier=True), mf)
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


if __name__ == '__main__':
    unittest.main()
