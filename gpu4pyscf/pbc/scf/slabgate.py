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

'''
Charged-sheet field-effect doping,
references: Phys. Rev. B 89, 245406 (2014).
'''

import numpy as np
import cupy as cp
from pyscf import lib, __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.scf.slabdipole import _slab_length, _dipole_grids, _slab_charge


# Dipole-plane separation as a fraction of the cell length along z.
DIPOLE_WIDTH_RATIO = getattr(__config__, 'pbc_scf_slabgate_dipole_width_ratio', .01)


def gate_charge(mf, zgate=None, barrier=False, barrier_start=None,
                barrier_end=None, barrier_height=.05, barrier_width=None,
                dipole_gap=1e-4, gate_side=-1, *,
                fractional_electrons=False):
    '''
    Apply a compensating planar gate and its dipole correction in place.

    The sheet carries qc = -q, where q is the slab charge per primitive cell.
    Negative q adds electrons. With fractional_electrons=False, the legacy
    convention q = cell.charge/nkpts is retained. To keep q fixed in this mode,
    set cell.charge = nkpts*q when changing the k mesh.

    For RKS/KRKS with spin=0, fractional_electrons=True enables real electron
    counts and requires an explicit cell.nelectron. In this mode cell.charge is PER
    PRIMITIVE CELL, and cell.nelectron + cell.charge must equal the sum of
    effective ionic charges. For example, after building an 8-valence-electron
    cell, set cell.charge = .3 and cell.nelectron = 7.7. No input is changed
    automatically. Setting cell.nelectron alone does not enable this mode.
    Occupations are filled globally at zero temperature, sharing any frontier
    degeneracy, or obtained with Fermi/Gaussian smearing at fixed electron
    number. Both orders of applying gate_charge and smearing are supported.
    Fractional-electron mode currently excludes UKS/KUKS, level shifting,
    fixed chemical potentials and second-order SCF. Nuclear gradients keep
    the electron number and external gate parameters fixed. With Fermi smearing
    at fixed sigma, they differentiate e_free, not e_tot or e_zero.
    In the legacy mode, build the cell with consistent charge/spin; KUKS spin
    counts are totals over the k-points. An explicit cell.nelectron must be
    an integer matching the legacy charge/nkpts convention. Restricted SCF
    requires an even total electron count in this mode.
    The standard 3D Coulomb calculation retains its G=0 convention.
    With A = Omega/L and s = (z-zgate+L/2) % L - L/2, the sheet potential is

        phi_c(z) = 2*pi*qc/A * (-abs(s) + s**2/L + L/6).

    The constant L/6 fixes the zero-average potential reference in
    Brumme Eq. (5). The dipole in Eq. (7) uses the same zero-cell-integral
    convention; zgate specifies the sheet position.

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

    The dipole-plane separation dipole_width is DIPOLE_WIDTH_RATIO*L
    (default: .01*L, as in Brumme). The return region is centered at
    cut = zgate + gate_side*(dipole_gap + dipole_width/2).
    For debugging, change this module's DIPOLE_WIDTH_RATIO before calling
    gate_charge, or set pbc_scf_slabgate_dipole_width_ratio in pyscf.__config__
    before importing this module.

    Args:
        mf: RKS/UKS/KRKS/KUKS for a three-dimensionally periodic slab.
            The slab normal is z, with the third lattice vector along positive
            z and the other two in the xy plane.
        zgate: Sheet position in Bohr; None means L/2.
        gate_side: Gate side relative to the slab along z: +1 means
            the higher-coordinate side, -1 the lower-coordinate side
            (default: -1, Brumme's orientation). zgate sets the actual
            sheet position; this sign places the dipole and default barrier.
        dipole_gap: Small positive distance epsilon in Bohr between the
            gate and the nearest dipole plane (default: 1e-4).
        barrier: Enable the repulsive electronic barrier (default: False).
        barrier_start, barrier_end: Periodic barrier edges in Bohr. For
            gate_side=-1, start defaults to cut-dipole_width/2 and end to
            start+.1*L. For gate_side=+1, end defaults to cut+dipole_width/2
            and start to end-.1*L. Explicit edges are coordinates along
            z, independent of gate_side. Intervals can cross the boundary.
            When barrier=True, the interval must contain zgate strictly
            inside and cover the entire dipole return region.
        fractional_electrons: Enable real per-cell electron counts (default:
            False). Requires explicit, consistent cell.nelectron and cell.charge.
            Integer targets are also supported. When reapplying gate_charge to
            an enabled object, pass True again; use a new SCF object to disable.
        barrier_height: Plateau height in Hartree (default: .05 = .1 Ry).
        barrier_width: Linear ramp width INSIDE each edge, in Bohr; None
            means dipole_width, as in Brumme. Zero gives a rectangular barrier.

    All lengths use Bohr, regardless of cell.unit; ABACUS fractional lengths
    must be multiplied by L. The sheet, barrier and cell remain fixed when
    differentiating atomic positions. Gradients are dE/dR; forces are -dE/dR.
    No translation-force subtraction is made for this external field.

    The dipole correction is added automatically to cancel the periodic
    residual field. An existing dipole addon is reused and its cut,
    width and z0 are reset to the Brumme convention. In particular,
    z0 = (cut+L/2) % L makes Eq. (7) vanish without a constant shift.
    Its total dipole includes the sheet; the barrier does not contribute.
    The sheet's dipole contribution has no atomic position derivative when
    its charge, position and the dipole geometry are fixed.
    The return region is on the gate's side away from the slab; the default
    barrier extends toward the slab. Put the gate and the entire return
    region in vacuum. Reapply gate_charge to update this placement when
    changing gate parameters.

    Uniform and MultiGrid integration and frozen Becke grids are supported.
    Becke grid response, stress, Hessians, response kernels and band k-points
    are not implemented. Use reset(new_cell)
    after changing geometry or basis, as for the underlying SCF object.
    '''
    from gpu4pyscf.pbc.dft import rks, uks, krks, kuks
    from gpu4pyscf.pbc.scf.slabdipole import slab_dipole_correction
    if not isinstance(mf, (rks.RKS, uks.UKS, krks.KRKS, kuks.KUKS)):
        raise NotImplementedError('Gate charge requires RKS/UKS/KRKS/KUKS')
    if isinstance(mf, GateFractionalOccupations) and not fractional_electrons:
        raise ValueError('Cannot disable fractional_electrons on an existing object; '
                         'create a new SCF object')
    if fractional_electrons:
        from gpu4pyscf.scf.soscf import _CIAH_SOSCF
        if not isinstance(mf, (rks.RKS, krks.KRKS)):
            raise NotImplementedError('Fractional gate electron counts require RKS/KRKS')
        if isinstance(mf, _CIAH_SOSCF):
            raise NotImplementedError('Second-order SCF with fractional gate occupations')
        GateFractionalOccupations._gate_nelectron(mf)
    else:
        charge = mf.cell.charge
        if not np.isfinite(charge) or charge != round(charge):
            raise ValueError('Fractional gate charge requires fractional_electrons=True '
                             'and an explicit cell.nelectron')
        if mf.cell._nelectron is not None:
            nelectron = float(mf.cell.nelectron)
            if nelectron != round(nelectron):
                raise ValueError('Fractional cell.nelectron requires fractional_electrons=True')
            nkpts = len(mf.kpts) if mf.istype('KSCF') else 1
            # Explicit N is per cell even in the legacy charge/nkpts mode.
            if abs(nelectron + charge / nkpts - mf.cell.atom_charges().sum()) > 1e-9:
                raise ValueError('cell.nelectron and cell.charge must match charge/nkpts '
                                 'in legacy mode; use fractional_electrons=True '
                                 'for per-cell charge')
            if isinstance(mf, (rks.RKS, krks.KRKS)) and round(nelectron * nkpts) % 2:
                raise ValueError('An odd restricted electron count requires '
                                 'fractional_electrons=True')
    length = _slab_length(mf.cell)
    if gate_side not in (-1, 1):
        raise ValueError('gate_side must be -1 or +1')
    zgate = (.5*length if zgate is None else zgate) % length
    dipole_width = DIPOLE_WIDTH_RATIO * length
    if not 0 < dipole_width < length:
        raise ValueError('Gate dipole width must satisfy 0 < width < cell length')
    if not 0 < dipole_gap < length-dipole_width:
        raise ValueError('Dipole gap must satisfy 0 < gap < cell length - dipole width')
    # z_dip - gate_side*d_dip/2 = z_mono + gate_side*epsilon.
    cut = (zgate + gate_side*(dipole_gap + dipole_width/2)) % length
    if gate_side == -1:
        if barrier_start is None:
            barrier_start = (cut - dipole_width/2) % length
        if barrier_end is None:
            barrier_end = (barrier_start + .1*length) % length
    else:
        if barrier_end is None:
            barrier_end = (cut + dipole_width/2) % length
        if barrier_start is None:
            barrier_start = (barrier_end - .1*length) % length
    if barrier_width is None:
        barrier_width = dipole_width
    if barrier:
        span = (barrier_end - barrier_start) % length
        # Unwrap relative to the barrier start so crossing z=0 is allowed.
        gate_pos = (zgate - barrier_start) % length
        if not 0 < gate_pos < span:
            raise ValueError('Gate plane must lie strictly inside the barrier interval')
        return_start = (gate_pos + gate_side*(dipole_gap + dipole_width/2)
                        - dipole_width/2)
        if return_start < -1e-10 or return_start + dipole_width > span + 1e-10:
            raise ValueError('Barrier interval must cover the entire dipole return region')
    if not isinstance(mf, SlabGateCharge):
        lib.set_class(mf, (SlabGateCharge, mf.__class__))
    if fractional_electrons and not isinstance(mf, GateFractionalOccupations):
        lib.set_class(mf, (GateFractionalOccupations, mf.__class__))
    mf.gate_side = gate_side
    mf.zgate = zgate
    mf.barrier = barrier
    mf.barrier_start = barrier_start
    mf.barrier_end = barrier_end
    mf.barrier_height = barrier_height
    mf.barrier_width = barrier_width
    mf._gate_parameters(mf.cell)
    # Mark the compensating gate before checking the slab's net charge.
    # The dipole factory adds its mixin only if absent, and resets parameters.
    # z0 = (cut+L/2) % L vanish the dipole potential constant shift.
    return slab_dipole_correction(mf, cut=cut,
                                  width=dipole_width, z0=(cut+length/2) % length)


def _sheet_potential(z, zgate, length, area, charge, deriv=0):
    '''
    Electrostatic potential of a sheet and its neutralizing background.
    '''
    z = cp.asarray(z)
    s = (z - zgate + length/2) % length - length/2
    factor = 2*np.pi*charge/area
    if deriv:
        return factor * (-cp.sign(s) + 2*s/length)
    return factor * (-cp.abs(s) + s*s/length + length/6)


def _barrier_potential(z, start, end, length, height, width, deriv=0):
    '''
    Periodic barrier with linear ramps inside its edges.
    '''
    z = cp.asarray(z)
    t = (z - start) % length
    span = (end - start) % length
    if width == 0:
        return cp.zeros_like(z) if deriv else height * ((t < span) & (t >= 0))
    if deriv:
        return height/width * (cp.asarray(t < width, dtype=float)
                               - cp.asarray((t > span-width) & (t < span), dtype=float))
    return height * cp.maximum(0., cp.minimum(1., cp.minimum(t, span-t)/width))


class SlabGateCharge:
    __name_mixin__ = 'Gate'

    _keys = {'gate_side', 'zgate', 'barrier', 'barrier_start',
             'barrier_end', 'barrier_height', 'barrier_width'}

    def _gate_parameters(self, cell):
        length = _slab_length(cell)
        zgate = self.zgate % length
        start = self.barrier_start % length
        end = self.barrier_end % length
        width = self.barrier_width
        if self.barrier:
            span = (end - start) % length
            if span == 0 or not 0 <= width <= span/2:
                raise ValueError('Barrier ramp width must lie between 0 and half the interval')
            if self.barrier_height < 0:
                raise ValueError('Barrier height must be nonnegative')
        return length, zgate, start, end, self.barrier_height, width

    def _gate_matrices(self, cell, kpts):
        from gpu4pyscf.pbc.dft import numint
        length, zgate, start, end, height, width = self._gate_parameters(cell)
        kpts = np.asarray(kpts).reshape(-1, 3)
        grids = _dipole_grids(self, cell)
        qc = -_slab_charge(self, cell)
        charge_matrix = barrier_matrix = None
        # TODO: may use multigrid to speed up
        ni = numint.KNumInt()
        area = cell.vol/length
        for ao_kpts, weight, coords in ni.block_loop(cell, grids, kpts=kpts):
            if charge_matrix is None:
                shape = (len(kpts), cell.nao, cell.nao)
                charge_matrix = cp.zeros(shape, dtype=ao_kpts.dtype)
                barrier_matrix = cp.zeros_like(charge_matrix)
            z = coords[:, 2]
            vc = -_sheet_potential(z, zgate, length, area, qc) # - for electron
            vb = (_barrier_potential(z, start, end, length, height, width)
                  if self.barrier else cp.zeros_like(z))
            for k, ao in enumerate(ao_kpts):
                charge_matrix[k] += ao.conj().T @ ((weight*vc)[:, None] * ao)
                if self.barrier:
                    barrier_matrix[k] += ao.conj().T @ ((weight*vb)[:, None] * ao)
        return charge_matrix, barrier_matrix

    def _gate_ionic_energy(self, cell):
        # electron energy will be added in energy_tot
        length, zgate, start, end, height, width = self._gate_parameters(cell)
        area = cell.vol/length
        qc = -_slab_charge(self, cell)
        z = cell.atom_coords()[:, 2]
        egate = np.dot(cell.atom_charges(),
                       _sheet_potential(z, zgate, length, area, qc).get())
        egate += np.pi*qc**2*length/(6*area)
        ebarrier = 0.
        if self.barrier:
            # TODO: the following two terms may be modified.
            # if barrier interacts with atoms this will appear
            ebarrier = -np.dot(cell.atom_charges(),
                               _barrier_potential(z, start, end, length, height, width).get())
            # this term can be ignored for this is a constant.
            ebarrier -= qc * float(
                _barrier_potential(zgate, start, end, length, height, width))
        return float(egate), float(ebarrier)

    def get_hcore(self, cell=None, kpts=None, *, kpt=None):
        if cell is None:
            cell = self.cell
        is_kpts = self.istype('KSCF')
        if kpts is None:
            kpts = self.kpts if is_kpts else (self.kpt if kpt is None else kpt)
        hcore = super().get_hcore(cell, kpts)
        # charge and barrier matrices
        vc, vb = self._gate_matrices(cell, kpts)
        return hcore + (vc + vb if is_kpts else vc[0] + vb[0])

    def energy_nuc(self):
        # Include the static terms before smearing constructs E(T), F and E0.
        egate, ebarrier = self._gate_ionic_energy(self.cell)
        return super().energy_nuc() + egate + ebarrier

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        energy = super().energy_elec(dm, h1e, vhf)
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
        return energy

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


# TODO: this may be applied to normal SCF calculations as well.
def _check_nelectron(cell):
    if cell._nelectron is None:
        raise ValueError('Set cell.nelectron explicitly for fractional gate occupations')
    nelectron = float(cell.nelectron)
    charge = float(cell.charge)
    if not np.isfinite(nelectron) or nelectron < 0 or not np.isfinite(charge):
        raise ValueError('Gate electron number must be finite and nonnegative')
    if abs(nelectron + charge - cell.atom_charges().sum()) > 1e-9:
        raise ValueError('cell.nelectron and cell.charge must match per primitive cell')
    if cell.spin != 0:
        raise NotImplementedError('Fractional gate occupations require spin=0 RKS/KRKS')
    return nelectron


def _zero_temperature_occ(mo_energy, nelectron):
    '''
    Global filling adapted from pbc.scf.khf.get_occ.
    '''
    energies = np.asarray(mo_energy)
    occ = np.zeros_like(energies, dtype=float)
    if nelectron == 0:
        return occ
    # Unlike khf.get_occ's N//2, include the partially occupied frontier state.
    fermi = np.sort(energies.ravel())[int(np.ceil(nelectron / 2)) - 1]
    # Replace the normal energy <= fermi -> 2 rule by sharing the remainder.
    below = energies < fermi - 1e-9
    frontier = abs(energies - fermi) <= 1e-9
    occ[below] = 2
    occ[frontier] = (nelectron - 2 * below.sum()) / frontier.sum()
    return occ


class GateFractionalOccupations:
    '''
    Opt-in, per-cell electron counting; ordinary SCF objects are unchanged.
    '''
    __name_mixin__ = 'Fractional'

    def _gate_nelectron(self, cell=None):
        nelectron = _check_nelectron(self.cell if cell is None else cell)
        if getattr(self, 'mu0', None) is not None:
            raise ValueError('Gate occupations fix the electron number; mu0 must be None')
        if self.level_shift:
            # TODO: how to do level shifting with fractional gate occupations?
            raise NotImplementedError('Level shifting with fractional gate occupations')
        return nelectron

    def build(self, cell=None):
        self._gate_nelectron(cell)
        return super().build(cell)

    def reset(self, cell=None):
        self._gate_nelectron(cell)
        return super().reset(cell)

    def _get_gate_occ(self, mo_energy=None, mo_coeff=None):
        '''
        Restricted-spin adaptation of _SmearingSCF/_SmearingKSCF.get_occ.
        '''
        from gpu4pyscf.scf import smearing

        nelectron = self._gate_nelectron()
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        if mo_energy is None:
            mo_energy = self.mo_energy
        energies = cp.asnumpy(mo_energy)
        # Do not use Cell.tot_electrons(nkpts): it rounds the explicit count.
        target = nelectron * nkpts
        if not 0 <= target <= 2 * energies.size:
            raise ValueError('Electron number exceeds the available orbital capacity')
        sigma = getattr(self, 'sigma', None)
        method = getattr(self, 'smearing_method', None)
        if sigma and method:
            if method.lower() == 'fermi':
                f_occ = smearing._fermi_smearing_occ
            elif method.lower() == 'gaussian':
                f_occ = smearing._gaussian_smearing_occ
            else:
                raise ValueError('Gate smearing supports fermi or gaussian')
            if target == 0 or target == 2 * energies.size:
                occ = np.full_like(energies, 0. if target == 0 else 2., dtype=float)
                self.entropy = 0.
            else:
                # Reuse the solver, replacing KRKS's (N+1)//2 by real N/2.
                mu, occ = smearing._smearing_optimize(
                    f_occ, energies.ravel(), target / 2, sigma)
                self.entropy = 2 * self._get_entropy(energies.ravel(), occ, mu) / nkpts
                occ = (2 * occ).reshape(energies.shape)
                logger.info(self, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                            sigma, mu, self.entropy)
        else:
            occ = _zero_temperature_occ(energies, target)
            if hasattr(self, 'entropy'):
                self.entropy = 0.
                self.e_free = self.e_zero = None
        if abs(occ.sum() / nkpts - nelectron) > 1e-9:
            raise RuntimeError('Gate occupations do not conserve the electron number')
        return cp.asarray(occ)

    get_occ = _get_gate_occ

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        from gpu4pyscf.scf import hf

        if cell is None:
            cell = self.cell
        target = self._gate_nelectron(cell)
        # Adapt pbc.scf.hf.RHF / khf.KRHF.get_init_guess: reuse the common
        # dispatcher, but normalize to the real target rather than tot_electrons.
        dm = hf.RHF.get_init_guess(self, cell, key)
        if s1e is None:
            s1e = self.get_ovlp(cell)
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        count = float(cp.einsum('...ij,...ji->...', dm, s1e).real.sum()) / nkpts
        if count == 0:
            if target != 0:
                raise ValueError('Cannot normalize an empty gate initial density')
            return dm

        scale = target / count
        dm *= scale
        if hasattr(dm, 'mo_occ'):
            dm.mo_occ *= scale
        return dm

    def _get_gate_grad(self, mo_coeff, mo_occ, fock=None):
        self._gate_nelectron()
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore() + self.get_veff(self.cell, dm)
        f_mo = mo_coeff.swapaxes(-1, -2).conj() @ fock @ mo_coeff
        # Normal hf/khf.get_grad only pairs occupied and empty orbitals.
        # Occupation differences also include full-to-partial rotations.
        gradient = f_mo * (mo_occ[..., None, :] - mo_occ[..., :, None])
        i, j = cp.tril_indices(mo_occ.shape[-1], -1)
        return gradient[..., i, j].ravel()

    get_grad = _get_gate_grad

    def get_fermi(self, mo_energy=None, mo_occ=None):
        target = self._gate_nelectron()
        if target == 0:
            return -np.inf
        if mo_energy is None:
            mo_energy = self.mo_energy
        nkpts = len(self.kpts) if self.istype('KSCF') else 1
        # Unlike normal get_fermi's integer truncation, include the partial state.
        # This remains a zero-temperature reference, not the smearing chemical potential.
        index = int(np.ceil(target * nkpts / 2)) - 1
        return float(cp.sort(cp.asarray(mo_energy).ravel())[index])

    def newton(self):
        raise NotImplementedError('Second-order SCF with fractional gate occupations')
