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
Macroscopic polarization from Wannier centers.
Reference:
- tutorial: https://arxiv.org/abs/1202.1831v1
- working equations: PhysRevB.47.1651, PhysRevB.89.155114
- reference codes: https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pywannier90.py
'''

from dataclasses import dataclass
import itertools
import numpy as np
import cupy as cp

from pyscf.data import nist
from gpu4pyscf.pbc.properties.berry.berry_phase import (
    berry_phase,
    diagonal_wannier_centers,
    hybrid_wannier_centers,
)
from gpu4pyscf.pbc.properties.berry.overlap import (
    KPointMesh,
    build_mmn,
)

__all__ = [
    'WannierCenterResult',
    'PolarizationResult',
    'eval_wannier_centers',
    'eval_berry_phase',
    'electronic_polarization',
    'ionic_polarization',
    'eval_polarization',
    'polarization_quantum',
    'unwrap_polarization',
    'polarization_difference',
]

AU_TO_C_PER_M2 = nist.E_CHARGE / nist.BOHR_SI**2


@dataclass
class WannierCenterResult:
    '''Directional centers and their sums for one polarization branch.

    Attributes:
        centers : tuple of three numpy.ndarray
            centers[d] has shape (nkpts/kmesh[d], noccupied) and contains
            dimensionless fractional centers in [0, 1) along lattice axis d.
            Wilson centers are sorted within each string; matching column
            indices on different strings do not identify the same band.
        berry_phases : tuple of three numpy.ndarray
            berry_phases[d] has shape (nkpts/kmesh[d],) and contains principal
            determinant phases in radians, in [-pi, pi).
        center_sums_fractional : numpy.ndarray, shape (3,)
            Transversely averaged occupied-center sums in lattice coordinates.
            Wilson sums use unwrapped determinant phases; diagonal sums use
            unwrapped localized band centers. No spin factor is included.
        center_sums_cartesian : numpy.ndarray, shape (3,)
            center_sums_fractional @ lattice_vectors, in Bohr.
        kmesh : numpy.ndarray of int, shape (3,)
            Sampling mesh dimensions.
        method : str
            'wilson' or 'diagonal'.
    '''

    centers: tuple
    berry_phases: tuple
    center_sums_fractional: np.ndarray
    center_sums_cartesian: np.ndarray
    kmesh: np.ndarray
    method: str


@dataclass
class PolarizationResult:
    '''Electronic, ionic, and total polarization on one branch.

    Attributes:
        electronic, ionic, total : numpy.ndarray, shape (3,)
            Cartesian polarization vectors in unit; total = electronic + ionic.
        quantum : numpy.ndarray, shape (3, 3)
            Primitive polarization-quantum vectors as rows, in unit.
            Equivalent branches differ by integer combinations of these rows.
        unit : str
            Normalized unit, either 'e/bohr^2' or 'C/m^2'.
        wannier : WannierCenterResult
            Centers and phases used for the electronic contribution.
    '''

    electronic: np.ndarray
    ionic: np.ndarray
    total: np.ndarray
    quantum: np.ndarray
    unit: str
    wannier: WannierCenterResult


def _unit_factor(unit):
    normalized = unit.lower().replace(' ', '')
    if normalized in ('au', 'a.u.', 'e/bohr^2', 'e/bohr2'):
        return 1., 'e/bohr^2'
    if normalized in ('c/m^2', 'c/m2'):
        return AU_TO_C_PER_M2, 'C/m^2'
    raise ValueError(
        f"Unsupported polarization unit {unit!r}; use 'au' or 'C/m^2'")


def _occupied_coefficients(mf, occupation_tol=1e-7):
    # TODO: in current version, kpoints must be in the same order as the kpts in mf
    assert mf.converged
    if mf.istype('KUHF'):
        raise NotImplementedError(
            'Unrestricted mean-field objects are not supported')

    coeff = cp.asarray(mf.mo_coeff)
    occupation = cp.asarray(mf.mo_occ)
    empty = cp.abs(occupation) <= occupation_tol
    occupied = cp.abs(occupation - 2.) <= occupation_tol
    if not bool(cp.all(empty | occupied).get()):
        raise ValueError(
            'Only closed-shell occupations equal to 0 or 2 are supported')

    counts = cp.count_nonzero(occupied, axis=1).get()
    if np.any(counts != counts[0]):
        raise ValueError(
            'The number of occupied bands changes across k-points; '
            'metals and smearing are unsupported')

    noccupied = int(counts[0])
    if noccupied == 0:
        return cp.empty(
            (coeff.shape[0], coeff.shape[1], 0), dtype=coeff.dtype)
    return cp.stack(
        [coeff[k][:, occupied[k]] for k in range(coeff.shape[0])])


def _apply_wannier_gauge(coeff, wannier_gauge, tol=1e-7):
    if wannier_gauge is None:
        return coeff

    gauge = cp.asarray(wannier_gauge)
    nocc = coeff.shape[2]
    expected = (coeff.shape[0], nocc, nocc)
    if gauge.shape != expected:
        raise ValueError(
            f'Wannier gauge must have shape {expected}, got {gauge.shape}')

    identity = cp.eye(nocc)
    error = cp.max(cp.abs(
        cp.matmul(gauge.conj().transpose(0, 2, 1), gauge) - identity))
    error = float(error.get())
    if not np.isfinite(error) or error > tol:
        raise ValueError(
            f'Wannier gauge is not unitary (error {error:.3e})')
    return cp.matmul(coeff, gauge)


def _unwrap_transverse_phases(phases, kmesh, direction):
    shape = phases.shape
    transverse_shape = tuple(
        kmesh[dim] for dim in range(3) if dim != direction)
    # (Nd1, Nd2) for phases or (Nd1, Nd2, nocc) for hybrid wannier centers
    phases = phases.reshape(transverse_shape + shape[1:])
    for axis in range(2):
        phases = cp.unwrap(phases, axis=axis)
    # Check all edges after both unwrap passes, including periodic closing
    # edges. A nonzero winding or an ambiguous pi jump has no unique lift.
    for axis in range(2):
        jumps = cp.roll(phases, -1, axis=axis) - phases
        if bool(cp.any(~cp.isfinite(jumps) | (cp.abs(jumps) >= np.pi - 1e-8)).get()):
            raise ValueError(
                'Cannot choose a continuous periodic transverse phase branch. '
                'Refine the k mesh and check for topological winding; for '
                'diagonal centers also check the localized gauge.')
    return phases.reshape(shape)


def _sum_diagonal_centers(centers, kmesh, direction):
    if centers.shape[1] == 0:
        return cp.asarray(0.)
    centers = _unwrap_transverse_phases(
        centers * (2. * np.pi), kmesh, direction) / (2. * np.pi)
    # \sum_occ 1/N_string \sum_string unwrappedCenter_n_string
    return cp.sum(cp.mean(centers, axis=0))


def _prepare_input(mf, kmesh, wannier_gauge=None):
    topology = KPointMesh(mf.cell, mf.kpts, kmesh)
    coeff = _occupied_coefficients(mf)
    coeff = _apply_wannier_gauge(coeff, wannier_gauge)
    return topology, coeff


def eval_wannier_centers(mf, kmesh=None, method='wilson',
                         wannier_gauge=None, singular_tol=1e-10):
    '''Evaluate directional Wannier centers from a converged k-point SCF.

    Args:
        mf : KRHF or KRKS
            Converged mean field on a full uniform k-point mesh in a 3D cell.
            Occupations must be 0 or 2 with the same occupied-band count at
            every k-point. mo_coeff and mo_occ must follow mf.kpts order.
        kmesh : array_like of int, shape (3,), optional
            Sampling mesh dimensions, inferred from mf.kpts if omitted.
        method : str
            'wilson' (default) computes gauge-invariant hybrid centers from
            unitary Wilson loops. 'diagonal' computes individual band centers
            in an externally localized Wannier gauge.
        wannier_gauge : array_like, shape (nkpts, noccupied, noccupied), optional
            Unitary rotations of occupied orbitals, C_new(k) = C_occ(k) U(k).
            Required for 'diagonal'; this function does not localize orbitals.
        singular_tol : float
            Minimum allowed singular value of a neighboring occupied overlap.

    Returns:
        WannierCenterResult
            centers[d]: NumPy array of shape (nkpts/kmesh[d], noccupied),
            fractional coordinates in [0, 1) along lattice vector d.
            berry_phases[d]: NumPy array of shape (nkpts/kmesh[d],), principal
            determinant phases in radians in [-pi, pi). Center sums are
            reported in fractional lattice coordinates and Cartesian Bohr;
            see WannierCenterResult for the branch and averaging conventions.

    Raises:
        numpy.linalg.LinAlgError
            An occupied overlap is singular or below singular_tol.
        ValueError
            Inputs are inconsistent, or transverse phases cannot be unwrapped
            to a continuous periodic branch. The latter requires a finer mesh
            or checking topological winding and, for 'diagonal', the gauge.
    '''
    if method not in ('wilson', 'diagonal'):
        raise ValueError(f"method must be 'wilson' or 'diagonal', got {method!r}")
    if method == 'diagonal' and wannier_gauge is None:
        raise ValueError(
            "method='diagonal' requires an externally localized wannier_gauge")

    topology, coeff = _prepare_input(mf, kmesh, wannier_gauge)

    centers = []
    phases = []
    center_sums_fractional = cp.zeros(3)

    all_overlaps = build_mmn(
        mf.cell, coeff, topology.kpts, topology.kmesh, topology=topology)
    for direction, overlaps in enumerate(all_overlaps):
        strings = topology.strings(direction)
        if method == 'wilson':
            center, phase = hybrid_wannier_centers(
                overlaps, strings, singular_tol=singular_tol)
            unwrapped_phase = _unwrap_transverse_phases(
                phase, topology.kmesh, direction)
            center_sum = -cp.mean(unwrapped_phase) / (2. * np.pi)
        else:
            phase = berry_phase(
                overlaps, strings, singular_tol=singular_tol)
            center = diagonal_wannier_centers(overlaps, strings)
            center_sum = _sum_diagonal_centers(
                center, topology.kmesh, direction)
        centers.append(cp.asnumpy(center))
        phases.append(cp.asnumpy(phase))
        center_sums_fractional[direction] = center_sum

    center_sums_fractional = cp.asnumpy(center_sums_fractional)
    lattice = np.asarray(mf.cell.lattice_vectors())
    center_sums_cartesian = center_sums_fractional @ lattice
    return WannierCenterResult(
        centers=tuple(centers),
        berry_phases=tuple(phases),
        center_sums_fractional=center_sums_fractional,
        center_sums_cartesian=center_sums_cartesian,
        kmesh=topology.kmesh.copy(),
        method=method,
    )


def eval_berry_phase(mf, kmesh=None, singular_tol=1e-10):
    '''Evaluate principal determinant Berry phases in three directions.

    Args:
        mf : KRHF or KRKS
            Converged closed-shell mean field on a full uniform mesh in a 3D
            cell, with constant occupied-band count and occupations 0 or 2.
        kmesh : array_like of int, shape (3,), optional
            Sampling mesh dimensions, inferred from mf.kpts if omitted.
        singular_tol : float or None
            Minimum allowed overlap singular value. None skips the SVD check;
            exactly singular determinants still raise LinAlgError.

    Returns:
        tuple of three numpy.ndarray
            phases[d] has shape (nkpts/kmesh[d],) and contains one phase per
            closed string along reciprocal axis d, in radians in [-pi, pi).
            Strings follow KPointMesh.strings(d). No Wilson-loop eigenproblem
            or transverse branch unwrapping is performed.
    '''
    topology, coeff = _prepare_input(mf, kmesh)
    phases = []
    all_overlaps = build_mmn(
        mf.cell, coeff, topology.kpts, topology.kmesh, topology=topology)
    for direction, overlaps in enumerate(all_overlaps):
        strings = topology.strings(direction)
        phases.append(cp.asnumpy(
            berry_phase(overlaps, strings, singular_tol=singular_tol)))
    return tuple(phases) # contains nstrings


def electronic_polarization(cell, wannier, unit='au'):
    '''Compute closed-shell electronic polarization -2 sum_n r_n / Omega.

    Args:
        cell : pyscf.pbc.gto.Cell
            Cell supplying the volume in Bohr^3.
        wannier : WannierCenterResult
            Occupied-center sums in Cartesian Bohr, on the selected branch.
        unit : str
            'au' (e/Bohr^2, default) or 'C/m^2'.

    Returns:
        numpy.ndarray, shape (3,)
            Cartesian electronic polarization in the requested unit. The
            factor -2 accounts for electron charge and double occupation.
    '''
    factor, _ = _unit_factor(unit)
    return -2. * wannier.center_sums_cartesian / cell.vol * factor


def ionic_polarization(cell, unit='au'):
    '''Compute ionic polarization sum_A Z_A R_A / Omega.

    Args:
        cell : pyscf.pbc.gto.Cell
            Cell supplying atom coordinates in Bohr and volume in Bohr^3.
            atom_charges provides valence charges for pseudopotential atoms
            and nuclear charges for all-electron atoms.
        unit : str
            'au' (e/Bohr^2, default) or 'C/m^2'.

    Returns:
        numpy.ndarray, shape (3,)
            Cartesian ionic polarization in the requested unit, using the
            atomic images stored in cell without wrapping their coordinates.
    '''
    factor, _ = _unit_factor(unit)
    return cell.atom_charges() @ cell.atom_coords() / cell.vol * factor


def polarization_quantum(cell, unit='au'):
    '''Return the primitive polarization-quantum vectors e a_d / Omega.

    Args:
        cell : pyscf.pbc.gto.Cell
            Cell supplying lattice vectors in Bohr and volume in Bohr^3.
        unit : str
            'au' (e/Bohr^2, default) or 'C/m^2'.

    Returns:
        numpy.ndarray, shape (3, 3)
            Cartesian quantum vectors as rows in the requested unit.
            An integer branch vector n shifts polarization by n @ quantum.
    '''
    factor, _ = _unit_factor(unit)
    return np.asarray(cell.lattice_vectors()) / cell.vol * factor


def eval_polarization(mf, kmesh=None, unit='au',
                      method='wilson', wannier_gauge=None,
                      singular_tol=1e-10, return_details=False):
    '''Evaluate electronic plus ionic macroscopic polarization.

    Args:
        mf : KRHF or KRKS
            Converged closed-shell mean field on a full uniform mesh in a 3D
            cell, with constant occupied-band count and occupations 0 or 2.
        kmesh : array_like of int, shape (3,), optional
            Sampling mesh dimensions, inferred from mf.kpts if omitted.
        unit : str
            'au' (e/Bohr^2, default) or 'C/m^2'.
        method : str
            'wilson' (default) or 'diagonal', as in eval_wannier_centers.
        wannier_gauge : array_like, shape (nkpts, noccupied, noccupied), optional
            Unitary occupied-orbital rotations; an externally localized gauge
            is required for 'diagonal'.
        singular_tol : float
            Minimum allowed neighboring-overlap singular value.
        return_details : bool
            Return PolarizationResult instead of only the total vector.

    Returns:
        numpy.ndarray, shape (3,), or PolarizationResult
            Cartesian total polarization in the requested unit. With
            return_details=True, includes electronic and ionic vectors,
            quantum vectors, normalized unit, and the WannierCenterResult.

    Notes:
        The Wilson branch follows the continuously unwrapped determinant phase
        over transverse strings. The diagonal method uses localized band-center
        sums, a gauge-dependent finite-mesh estimator that need not agree with
        the determinant estimator. Use unwrap_polarization when comparing
        nearby structures at the same lattice.
    '''
    wannier = eval_wannier_centers(
        mf, kmesh=kmesh, method=method,
        wannier_gauge=wannier_gauge, singular_tol=singular_tol)
    electronic = electronic_polarization(mf.cell, wannier, unit)
    ionic = ionic_polarization(mf.cell, unit)
    _, normalized_unit = _unit_factor(unit)
    result = PolarizationResult(
        electronic=electronic,
        ionic=ionic,
        total=electronic + ionic,
        quantum=polarization_quantum(mf.cell, unit),
        unit=normalized_unit,
        wannier=wannier,
    )
    if return_details:
        return result
    return result.total


def unwrap_polarization(polarization, reference, cell, unit='au',
                        return_branch=False):
    '''Move a polarization onto the branch nearest to reference.

    Args:
        polarization, reference : array_like, shape (3,)
            Finite Cartesian polarization vectors in unit, at the same lattice.
        cell : pyscf.pbc.gto.Cell
            Cell defining the common polarization quantum.
        unit : str
            'au' (e/Bohr^2, default) or 'C/m^2'; used for both input and output.
        return_branch : bool
            Also return the integer coefficients of the removed quantum.

    Returns:
        unwrapped : numpy.ndarray, shape (3,)
            polarization - branch @ polarization_quantum(cell, unit), chosen
            to minimize Cartesian Euclidean distance to reference. The search
            accounts for nonorthogonal lattices.
        branch : numpy.ndarray of int, shape (3,), optional
            Removed quantum coefficients; returned only if return_branch=True.

    Notes:
        Nearest-branch matching assumes nearby states. It cannot recover a
        multi-quantum change without intermediate points along the path.
    '''
    polarization = np.asarray(polarization, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if polarization.shape != (3,) or reference.shape != (3,):
        raise ValueError('polarization and reference must both have shape (3,)')
    if not np.all(np.isfinite(polarization)) or not np.all(np.isfinite(reference)):
        raise ValueError('polarization and reference must be finite')

    quantum = polarization_quantum(cell, unit)
    delta = polarization - reference
    fractional = np.linalg.solve(quantum.T, delta)
    initial = np.rint(fractional).astype(int)
    best_branch = initial
    best_distance = np.linalg.norm(delta - initial @ quantum)

    smallest_singular_value = np.linalg.svd(quantum, compute_uv=False)[-1]
    radius = best_distance / smallest_singular_value + 1e-12
    lower = np.floor(fractional - radius).astype(int)
    upper = np.ceil(fractional + radius).astype(int)
    for branch in itertools.product(
            *(range(lower[dim], upper[dim] + 1) for dim in range(3))):
        branch = np.asarray(branch)
        distance = np.linalg.norm(delta - branch @ quantum)
        if distance < best_distance:
            best_distance = distance
            best_branch = branch

    unwrapped = polarization - best_branch @ quantum
    if return_branch:
        return unwrapped, best_branch
    return unwrapped


def polarization_difference(final, initial, cell, unit='au',
                            return_branch=False):
    '''Return the nearest-branch polarization change from initial to final.

    Args:
        final, initial : array_like, shape (3,)
            Cartesian polarization vectors at the same lattice, in unit.
        cell : pyscf.pbc.gto.Cell
            Cell defining the common polarization quantum.
        unit : str
            'au' (e/Bohr^2, default) or 'C/m^2'; used for both input and output.
        return_branch : bool
            Also return the integer quantum coefficients removed from final.

    Returns:
        difference : numpy.ndarray, shape (3,)
            unwrap_polarization(final, initial, cell, unit) - initial,
            in the requested unit. Intended for nearby states at fixed lattice.
        branch : numpy.ndarray of int, shape (3,), optional
            Removed quantum coefficients; returned only if return_branch=True.
    '''
    matched, branch = unwrap_polarization(
        final, initial, cell, unit=unit, return_branch=True)
    difference = matched - np.asarray(initial)
    if return_branch:
        return difference, branch
    return difference
