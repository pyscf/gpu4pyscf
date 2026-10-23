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
from gpu4pyscf.pbc.scf.addons import gate_charge, slab_dipole_correction
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
        z = np.linspace(0., length, 10001, endpoint=False)
        phi = _sheet_potential(z, zgate, length, area, qc)
        self.assertAlmostEqual(phi.mean(), 0., delta=1e-7)
        # Independent reciprocal-space solution, excluding the sheet cusp.
        modes = np.arange(1, 100001)
        for point in (1.3, 8.2, 20.1):
            fourier = (2*qc*length/(np.pi*area) *
                       np.sum(np.cos(2*np.pi*modes*(point-zgate)/length)/modes**2))
            self.assertAlmostEqual(_sheet_potential(point, zgate, length, area, qc),
                                   fourier, delta=2e-9)
        step = 1e-5
        points = np.array([1.2, 9.4, 19.8])
        fd = (_sheet_potential(points+step, zgate, length, area, qc) -
              _sheet_potential(points-step, zgate, length, area, qc)) / (2*step)
        np.testing.assert_allclose(fd, _sheet_potential(
            points, zgate, length, area, qc, deriv=1), atol=1e-10)

        points = np.array([18., 18.5, 19.5, 21.9, .1, .5, 1.5, 2.1])
        vb = _barrier_potential(points, 18., 2., length, .05, 1.)
        np.testing.assert_allclose(vb, [0., .025, .05, .05, .05, .05, .025, 0.])
        interior = points[[1, 2, 3, 4, 5, 6, 7]]
        fd = (_barrier_potential(interior+step, 18., 2., length, .05, 1.) -
              _barrier_potential(interior-step, 18., 2., length, .05, 1.)) / (2*step)
        np.testing.assert_allclose(fd, _barrier_potential(
            interior, 18., 2., length, .05, 1., deriv=1), atol=1e-10)

    def check_reference(self, name, multigrid=False, becke=False, shifted=False,
                        charge=2):
        cell = make_cell(charge)
        mf = make_mf(cell, name, multigrid, becke)
        if shifted:
            mf.kpts = cell.make_kpts([2, 1, 1], scaled_center=[.17, 0, 0])
        dm = mf.get_init_guess()
        if name in ('UKS', 'KUKS'):
            dm[0] *= 1.1
            dm[1] *= .9
        h0 = mf.get_hcore()
        v0 = mf.get_veff(dm=dm)
        e0 = mf.energy_tot(dm, h0, v0)
        gate_charge(mf, zgate=17., barrier=True, barrier_start=15.,
                    barrier_end=19., barrier_width=.7)
        h = mf.get_hcore()
        e = mf.energy_tot(dm, h, v0)
        vc, vb = mf._gate_matrices(cell, mf.kpts)
        np.testing.assert_allclose((h-h0).get(),
                                   (vc+vb if name.startswith('K') else vc[0]+vb[0]).get(),
                                   atol=1e-12)
        np.testing.assert_allclose(vc.get(), vc.swapaxes(-1, -2).conj().get(), atol=1e-12)
        self.assertAlmostEqual(e-e0, mf.scf_summary['egate'] +
                               mf.scf_summary['ebarrier'], delta=1e-10)

        grids = mf.grids
        if multigrid:
            grids = dft.UniformGrids(cell).set(mesh=mf._numint.mesh)
        coords, weights = grids.coords.get(), grids.weights.get()
        aos = cpu_numint.eval_ao_kpts(cell, coords, kpts=mf.kpts)
        q = cell.atom_charges().sum() - cell.tot_electrons(len(mf.kpts))/len(mf.kpts)
        z = coords[:, 2]
        s = (z-17.+11.) % 22. - 11.
        phi = 2*np.pi*(-q)/56. * (-abs(s) + s*s/22. + 22./6)
        t = (z-15.) % 22.
        barrier = .05 * np.clip(np.minimum(t, 4.-t)/.7, 0., 1.)
        ref = np.asarray([ao.conj().T @ ((weights*(-phi+barrier))[:, None]*ao)
                          for ao in aos])
        np.testing.assert_allclose((vc+vb).get(), ref, atol=2e-9)
        charge_dm = dm.get().sum(axis=0) if name in ('UKS', 'KUKS') else dm.get()
        charge_dm = charge_dm.reshape(ref.shape)
        electronic = np.einsum('kij,kji->', charge_dm, ref).real / len(ref)
        zi = cell.atom_coords()[:, 2]
        si = (zi-17.+11.) % 22. - 11.
        ionic = np.dot(cell.atom_charges(),
                       2*np.pi*(-q)/56.*(-abs(si)+si*si/22.+22./6))
        self_energy = np.pi*q**2*22./(6*56.)
        gate_barrier = q*.05
        self.assertAlmostEqual(e-e0, electronic+ionic+self_energy+gate_barrier, delta=2e-9)

    def test_cpu_reference(self):
        for name in ('RKS', 'UKS', 'KRKS', 'KUKS'):
            with self.subTest(name=name):
                self.check_reference(name)

    def test_multigrid_and_becke_reference(self):
        self.check_reference('KUKS', multigrid=True)
        self.check_reference('KRKS', becke=True)

    def test_shifted_and_electron_doping(self):
        self.check_reference('KRKS', shifted=True)
        self.check_reference('UKS', charge=-2)

    def test_charge_per_cell(self):
        cell = make_cell()
        gamma = gate_charge(make_mf(cell))
        mf = gate_charge(make_mf(cell, 'KUKS'))
        self.assertEqual(_slab_charge(gamma, cell), 2.)
        self.assertEqual(_slab_charge(mf, cell), 1.)
        dm = mf.get_init_guess()
        ne = float(cp.einsum('skij,kji->', dm, mf.get_ovlp()).real) / len(mf.kpts)
        self.assertAlmostEqual(ne, cell.atom_charges().sum()-1., delta=1e-9)
        vc, _ = mf._gate_matrices(cell, mf.kpts)
        # Scale the total charge to retain two holes per primitive cell.
        cell.charge = 4
        self.assertEqual(_slab_charge(mf, cell), 2.)
        vc_fixed, _ = mf._gate_matrices(cell, mf.kpts)
        np.testing.assert_allclose(vc_fixed.get(), 2*vc.get(), atol=1e-12)
        dm = mf.get_init_guess()
        ne = float(cp.einsum('skij,kji->', dm, mf.get_ovlp()).real) / len(mf.kpts)
        self.assertAlmostEqual(ne, cell.atom_charges().sum()-2., delta=1e-9)

    def test_density_derivative_and_sheet_dipole(self):
        cell = make_cell()
        mf = make_mf(cell, 'KUKS')
        dm = mf.get_init_guess()
        v0 = mf.get_veff(dm=dm)
        h0 = mf.get_hcore()
        gate_charge(mf, zgate=17., barrier=True, barrier_start=15.,
                    barrier_end=19., barrier_width=.7)
        slab_dipole_correction(mf, cut=20., width=2.)
        rng = np.random.default_rng(20)
        ddm = cp.asarray(rng.random(dm.shape) + 1j*rng.random(dm.shape))
        ddm = (ddm + ddm.swapaxes(-1, -2).conj())*.01
        overlap = mf.get_ovlp()
        ddm -= dm * (cp.einsum('skij,kji->', ddm, overlap).real /
                     cp.einsum('skij,kji->', dm, overlap).real)
        v = mf.get_veff(dm=dm)
        h = mf.get_hcore()
        step = 1e-4
        def added_energy(d):
            mf.energy_tot(d)
            return mf.scf_summary['egate'] + mf.scf_summary['ebarrier'] + mf.get_veff(dm=d).edip
        derivative = cp.einsum('skij,skji->', ddm, h-h0+v-v0).real / len(mf.kpts)
        self.assertAlmostEqual((added_energy(dm+step*ddm)-added_energy(dm-step*ddm))/(2*step),
                               float(derivative), delta=2e-8)
        # Neutralizing sheet must be included even if it lies in the return region.
        mf.zgate = 20.3
        v = mf.get_veff(dm=dm)
        gate_dipole = -cell.charge/len(mf.kpts) * _sawtooth(mf.zgate, 20., 22., 2.)
        cell.charge = 0
        no_sheet = mf.get_veff(dm=dm)
        cell.charge = 2
        self.assertAlmostEqual(v.dipole-no_sheet.dipole, gate_dipole, delta=1e-11)

    def test_reset_and_parameters(self):
        cell = make_cell()
        mf = gate_charge(make_mf(cell))
        vc, _ = mf._gate_matrices(cell, mf.kpts)
        self.assertIs(mf._gate_matrices(cell, mf.kpts)[0], vc)
        mf.zgate = 17.
        self.assertIsNot(mf._gate_matrices(cell, mf.kpts)[0], vc)
        mf.reset()
        self.assertIsNone(mf._gate_cache)
        self.assertEqual(mf.zgate, 17.)
        with self.assertRaises(NotImplementedError):
            slab_dipole_correction(make_mf(cell), cut=20.)
        with self.assertRaises(ValueError):
            gate_charge(make_mf(cell), barrier=True, barrier_width=3.)
        with self.assertRaises(ValueError):
            slab_dipole_correction(gate_charge(make_mf(cell)), width=2.)
        # Both composition orders work on a neutral cell.
        cell = make_cell(charge=0)
        mf = gate_charge(slab_dipole_correction(make_mf(cell), cut=20., width=2.))
        self.assertLess(np.max(abs(mf._gate_matrices(cell, mf.kpts)[0].get())), 1e-15)
        self.assertIsNotNone(mf.Gradients())


if __name__ == '__main__':
    unittest.main()
