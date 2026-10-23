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

'''Charged-sheet field-effect doping, Phys. Rev. B 89, 245406 (2014).'''

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.scf.slabdipole import _slab_length, _dipole_grids, _slab_charge


def gate_charge(mf, zgate=None, barrier=False, barrier_start=None,
                barrier_end=None, barrier_height=.05, barrier_width=None):
    r'''Apply a compensating planar gate charge in place.

    The sheet carries qc = -q, where q = cell.charge/nkpts is the charge
    PER PRIMITIVE CELL. At a single k-point, q = cell.charge. Negative q
    adds electrons. The gate charge is determined solely by cell.charge
    and the number of sampled k-points. To fix a target q per primitive
    cell when changing the k mesh, set cell.charge = nkpts*q.
    Build the cell with a consistent charge/spin; KUKS spin counts are totals
    over the k-points. No electron-count setting is changed by this addon.
    The standard 3D Coulomb calculation retains its G=0 convention.
    With A = Omega/L and s = (z-zgate+L/2) % L - L/2, the sheet potential is

        phi_c(z) = 2*pi*qc/A * (-abs(s) + s**2/L + L/6).

    The electronic Hamiltonian gains -phi_c and, optionally, a positive
    barrier v_b. The added total energy (atomic units) is

        E_gate = -int n phi_c + sum_I Z_I phi_c(R_I)
                 + pi*qc**2*L/(6*A),
        E_barrier = int n v_b - sum_I Z_I v_b(R_I) - qc*v_b(zgate).

    Z_I are the effective ionic charges, including pseudopotentials. The
    sheet self-energy includes the factor 1/2. The barrier follows the
    Brumme convention of coupling to the full charge density; its ionic
    and sheet terms should vanish or remain constant when placed in vacuum.
    scf_summary['egate'] and ['ebarrier'] report these complete additions.
    Electronic contributions are already in scf_summary['e1'].

    Args:
        mf: GPU RKS/UKS/KRKS/KUKS for a three-dimensionally periodic slab.
            The slab normal is z, with the third lattice vector along positive
            z and the other two in the xy plane.
        zgate: Sheet position in Bohr; None means L/2.
        barrier: Enable the repulsive electronic barrier (default: False).
        barrier_start, barrier_end: Periodic barrier edges in Bohr; defaults
            are .45*L and .55*L. An interval can cross the cell boundary.
        barrier_height: Plateau height in Hartree (default: .05 = .1 Ry).
        barrier_width: Linear ramp width INSIDE each edge, in Bohr; None
            means .01*L. Zero gives a rectangular barrier.

    All lengths use Bohr, regardless of cell.unit; ABACUS fractional lengths
    must be multiplied by L. The sheet, barrier and cell remain fixed when
    differentiating atomic positions. Gradients are dE/dR; forces are -dE/dR.
    No translation-force subtraction is made for this external field.

    Compose with slab_dipole_correction(mf, cut=..., width=...) to cancel
    the periodic residual field. Its total dipole then includes the sheet;
    the barrier does not contribute to that dipole. Use the same normal and
    put the whole dipole return region in vacuum.

    Uniform and MultiGrid integration and frozen Becke grids are supported.
    Becke grid response, stress, Hessians, response kernels and band k-points
    are not implemented. reset() invalidates AO matrices; use reset(new_cell)
    after changing geometry or basis, as for the underlying SCF object.
    '''
    from gpu4pyscf.pbc.dft import rks, uks, krks, kuks
    if not isinstance(mf, (rks.RKS, uks.UKS, krks.KRKS, kuks.KUKS)):
        raise NotImplementedError('Gate charge requires RKS/UKS/KRKS/KUKS')
    _slab_length(mf.cell)
    if not isinstance(mf, SlabGateCharge):
        lib.set_class(mf, (SlabGateCharge, mf.__class__))
    mf.zgate = zgate
    mf.barrier = barrier
    mf.barrier_start = barrier_start
    mf.barrier_end = barrier_end
    mf.barrier_height = barrier_height
    mf.barrier_width = barrier_width
    mf._gate_cache = None
    mf._gate_parameters(mf.cell)
    return mf


def _sheet_potential(z, zgate, length, area, charge, deriv=0):
    '''Electrostatic potential of a sheet and its neutralizing background.'''
    s = (z - zgate + length/2) % length - length/2
    factor = 2*np.pi*charge/area
    if deriv:
        return factor * (-cp.get_array_module(z).sign(s) + 2*s/length)
    return factor * (-abs(s) + s*s/length + length/6)


def _barrier_potential(z, start, end, length, height, width, deriv=0):
    '''Periodic barrier with linear ramps inside its edges.'''
    xp = cp.get_array_module(z)
    t = (z - start) % length
    span = (end - start) % length
    if width == 0:
        return xp.zeros_like(z) if deriv else height * ((t < span) & (t >= 0))
    if deriv:
        return height/width * (xp.asarray(t < width, dtype=float)
                               - xp.asarray((t > span-width) & (t < span), dtype=float))
    return height * xp.maximum(0., xp.minimum(1., xp.minimum(t, span-t)/width))


class SlabGateCharge:
    _keys = {'zgate', 'barrier', 'barrier_start', 'barrier_end',
             'barrier_height', 'barrier_width'}

    def reset(self, cell=None):
        self._gate_cache = None
        return super().reset(cell)

    def _gate_parameters(self, cell):
        length = _slab_length(cell)
        zgate = (.5*length if self.zgate is None else self.zgate) % length
        start = (.45*length if self.barrier_start is None else self.barrier_start) % length
        end = (.55*length if self.barrier_end is None else self.barrier_end) % length
        width = .01*length if self.barrier_width is None else self.barrier_width
        if self.barrier:
            span = (end - start) % length
            if span == 0 or not 0 <= width <= span/2:
                raise ValueError('Barrier ramp width must lie between 0 and half the interval')
            if self.barrier_height < 0:
                raise ValueError('Barrier height must be nonnegative')
        return length, zgate, start, end, self.barrier_height, width

    def _gate_matrices(self, cell, kpts):
        from gpu4pyscf.pbc.dft import numint, gen_grid
        params = self._gate_parameters(cell)
        length, zgate, start, end, height, width = params
        kpts = np.asarray(kpts).reshape(-1, 3)
        grids = _dipole_grids(self, cell)
        qc = -_slab_charge(self, cell)
        if isinstance(grids, gen_grid.UniformGrids):
            grid_signature = (tuple(grids.mesh), id(grids._coords), id(grids._weights))
        else:
            grid_signature = (id(grids.coords), id(grids.weights))
        # Geometry and basis changes go through reset(), as in the SCF base.
        signature = (id(cell), qc, self.barrier, params,
                     kpts.tobytes(), grid_signature)
        if self._gate_cache is not None and self._gate_cache[0] == signature:
            return self._gate_cache[1:]
        charge_matrix = barrier_matrix = None
        ni = numint.KNumInt()
        area = cell.vol/length
        for ao_kpts, weight, coords in ni.block_loop(cell, grids, kpts=kpts):
            if charge_matrix is None:
                shape = (len(kpts), cell.nao, cell.nao)
                charge_matrix = cp.zeros(shape, dtype=ao_kpts.dtype)
                barrier_matrix = cp.zeros_like(charge_matrix)
            z = coords[:, 2]
            vc = -_sheet_potential(z, zgate, length, area, qc)
            vb = (_barrier_potential(z, start, end, length, height, width)
                  if self.barrier else cp.zeros_like(z))
            for k, ao in enumerate(ao_kpts):
                charge_matrix[k] += ao.conj().T @ ((weight*vc)[:, None] * ao)
                if self.barrier:
                    barrier_matrix[k] += ao.conj().T @ ((weight*vb)[:, None] * ao)
        self._gate_cache = signature, charge_matrix, barrier_matrix
        return charge_matrix, barrier_matrix

    def _gate_ionic_energy(self, cell):
        length, zgate, start, end, height, width = self._gate_parameters(cell)
        area = cell.vol/length
        qc = -_slab_charge(self, cell)
        z = cell.atom_coords()[:, 2]
        egate = np.dot(cell.atom_charges(),
                       _sheet_potential(z, zgate, length, area, qc))
        egate += np.pi*qc**2*length/(6*area)
        ebarrier = 0.
        if self.barrier:
            ebarrier = -np.dot(cell.atom_charges(),
                               _barrier_potential(z, start, end, length, height, width))
            ebarrier -= qc * _barrier_potential(zgate, start, end, length, height, width)
        return float(egate), float(ebarrier)

    def get_hcore(self, cell=None, kpts=None, *, kpt=None):
        if cell is None:
            cell = self.cell
        is_kpts = self.istype('KSCF')
        if kpts is None:
            kpts = self.kpts if is_kpts else (self.kpt if kpt is None else kpt)
        hcore = super().get_hcore(cell, kpts)
        vc, vb = self._gate_matrices(cell, kpts)
        return hcore + (vc + vb if is_kpts else vc[0] + vb[0])

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        energy = super().energy_tot(dm, h1e, vhf)
        egate, ebarrier = self._gate_ionic_energy(self.cell)
        vc, vb = self._gate_matrices(self.cell, self.kpts if self.istype('KSCF') else self.kpt)
        dm_charge = cp.asarray(dm)
        if self.istype('UHF') or self.istype('KUHF'):
            dm_charge = dm_charge[0] + dm_charge[1]
        dm_charge = dm_charge.reshape(vc.shape)
        self.scf_summary['egate'] = egate + float(
            cp.einsum('kij,kji->', dm_charge, vc).real) / len(vc)
        self.scf_summary['ebarrier'] = ebarrier + float(
            cp.einsum('kij,kji->', dm_charge, vb).real) / len(vb)
        logger.debug(self, 'Gate energy = %.10g, barrier energy = %.10g',
                     self.scf_summary['egate'], self.scf_summary['ebarrier'])
        # The electronic terms are included once through the corrected hcore.
        return energy + egate + ebarrier

    def Gradients(self):
        from gpu4pyscf.pbc.grad.slabgate import _SlabGateGradients
        grad = super().Gradients()
        return lib.set_class(grad, (_SlabGateGradients, grad.__class__))

    def get_bands(self, *args, **kwargs):
        raise NotImplementedError('Gate charge for band k-points')

    def Hessian(self):
        raise NotImplementedError('Gate charge Hessian')

    def gen_response(self, *args, **kwargs):
        raise NotImplementedError('Gate charge response kernel')

    def to_cpu(self):
        raise NotImplementedError('Gate charge is a GPU SCF addon')
