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

'''Macroscopic polarization from Wannier centers.'''

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
    build_mmn_channels,
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
    '''Directional Wannier-center data for all spin channels.

    ``centers[spin][direction]`` has shape ``(nstring, noccupied)`` and
    contains fractional hybrid centers in ``[0, 1)``.
    '''

    centers: tuple
    berry_phases: tuple
    center_sums_fractional: np.ndarray
    center_sums_cartesian: np.ndarray
    spin_weights: np.ndarray
    kmesh: np.ndarray
    method: str


@dataclass
class PolarizationResult:
    '''Electronic, ionic, and total polarization on one branch.'''

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
    if getattr(mf, 'converged', True) is False:
        raise RuntimeError('The mean-field calculation is not converged')
    if not hasattr(mf, 'mo_coeff') or not hasattr(mf, 'mo_occ'):
        raise ValueError('mf must provide mo_coeff and mo_occ')

    coeff = cp.asarray(mf.mo_coeff)
    occupation = cp.asarray(mf.mo_occ)
    if coeff.ndim == 3 and occupation.ndim == 2:
        coeff_channels = (coeff,)
        occupation_channels = (occupation,)
    elif (coeff.ndim == 4 and occupation.ndim == 3 and
          coeff.shape[0] == occupation.shape[0]):
        coeff_channels = tuple(coeff[spin] for spin in range(coeff.shape[0]))
        occupation_channels = tuple(
            occupation[spin] for spin in range(occupation.shape[0]))
    else:
        raise NotImplementedError(
            'Only full-k-mesh restricted and unrestricted mean-field '
            'objects are supported')

    occupied_coefficients = []
    spin_weights = []
    for spin, (coeff_kpts, occupation_kpts) in enumerate(
            zip(coeff_channels, occupation_channels)):
        occupied = occupation_kpts > occupation_tol
        counts = cp.count_nonzero(occupied, axis=1).get()
        if np.any(counts != counts[0]):
            raise ValueError(
                f'The number of occupied bands changes across k-points '
                f'in spin channel {spin}; metals and smearing are unsupported')

        noccupied = int(counts[0])
        if noccupied == 0:
            occupied_coefficients.append(
                cp.empty((coeff_kpts.shape[0], coeff_kpts.shape[1], 0),
                         dtype=coeff_kpts.dtype))
            spin_weights.append(1.)
            continue

        occupied_values = occupation_kpts[occupied]
        weight = float(cp.mean(occupied_values).get())
        if (not np.isclose(weight, 1., atol=occupation_tol) and
                not np.isclose(weight, 2., atol=occupation_tol)):
            raise ValueError(
                'Fractional occupations are unsupported in Berry-phase '
                f'polarization; spin channel {spin} has occupation {weight:g}')
        if not bool(cp.all(cp.abs(occupied_values - weight) < occupation_tol).get()):
            raise ValueError(
                'Occupied bands must have one common integer occupation')

        selected = cp.stack(
            [coeff_kpts[k][:, occupied[k]]
             for k in range(coeff_kpts.shape[0])])
        occupied_coefficients.append(selected)
        spin_weights.append(weight)

    return tuple(occupied_coefficients), np.asarray(spin_weights)


def _apply_wannier_gauge(coeff_channels, wannier_gauge, tol=1e-7):
    if wannier_gauge is None:
        return coeff_channels

    if len(coeff_channels) == 1:
        gauges = (cp.asarray(wannier_gauge),)
    else:
        gauges = tuple(cp.asarray(gauge) for gauge in wannier_gauge)
    if len(gauges) != len(coeff_channels):
        raise ValueError('One Wannier gauge is required for each spin channel')

    transformed = []
    for spin, (coeff, gauge) in enumerate(zip(coeff_channels, gauges)):
        noccupied = coeff.shape[2]
        expected = (coeff.shape[0], noccupied, noccupied)
        if gauge.shape != expected:
            raise ValueError(
                f'Wannier gauge {spin} must have shape {expected}, got {gauge.shape}')
        if noccupied == 0:
            transformed.append(coeff)
            continue
        identity = cp.eye(noccupied)
        error = cp.max(cp.abs(
            cp.matmul(gauge.conj().transpose(0, 2, 1), gauge) - identity))
        if float(error.get()) > tol:
            raise ValueError(
                f'Wannier gauge {spin} is not unitary (error {float(error.get()):.3e})')
        transformed.append(cp.matmul(coeff, gauge))
    return tuple(transformed)


def _unwrap_transverse_phases(phases, kmesh, direction):
    transverse_shape = tuple(
        kmesh[dim] for dim in range(3) if dim != direction)
    phases = phases.reshape(transverse_shape)
    for axis in range(phases.ndim):
        phases = cp.unwrap(phases, axis=axis)
    return phases.ravel()


def _sum_diagonal_centers(centers, kmesh, direction):
    if centers.shape[1] == 0:
        return cp.asarray(0.)

    transverse_shape = tuple(
        kmesh[dim] for dim in range(3) if dim != direction)
    center_phases = centers.reshape(
        transverse_shape + (centers.shape[1],)) * (2. * np.pi)
    for axis in range(len(transverse_shape)):
        center_phases = cp.unwrap(center_phases, axis=axis)
    centers = center_phases.reshape(-1, centers.shape[1]) / (2. * np.pi)
    return cp.sum(cp.mean(centers, axis=0))


def eval_wannier_centers(mf, kmesh=None, batch_size=None, method='wilson',
                         wannier_gauge=None, singular_tol=1e-10):
    '''Evaluate directional Wannier centers from a converged k-point SCF.

    ``method='wilson'`` returns gauge-invariant hybrid Wannier centers and is
    the default. ``method='diagonal'`` is only meaningful with an externally
    supplied localized ``wannier_gauge``.
    '''
    if not hasattr(mf, 'cell') or not hasattr(mf, 'kpts'):
        raise ValueError('mf must be a periodic k-point mean-field object')
    if method not in ('wilson', 'diagonal'):
        raise ValueError(f"method must be 'wilson' or 'diagonal', got {method!r}")
    if method == 'diagonal' and wannier_gauge is None:
        raise ValueError(
            "method='diagonal' requires an externally localized wannier_gauge")

    topology = KPointMesh(mf.cell, mf.kpts, kmesh)
    coeff_channels, spin_weights = _occupied_coefficients(mf)
    if any(coeff.shape[0] != len(topology.kpts) for coeff in coeff_channels):
        raise ValueError('mo_coeff and kpts contain different numbers of k-points')
    coeff_channels = _apply_wannier_gauge(
        coeff_channels, wannier_gauge)

    centers = [[] for _ in coeff_channels]
    phases = [[] for _ in coeff_channels]
    center_sums_fractional = cp.zeros((len(coeff_channels), 3))

    for direction in range(3):
        overlaps = build_mmn_channels(
            mf.cell, coeff_channels, topology.kpts, topology.kmesh,
            direction, batch_size=batch_size, topology=topology)
        strings = topology.strings(direction)

        for spin, mmn in enumerate(overlaps):
            phase = berry_phase(mmn, strings)
            if method == 'wilson':
                center, _ = hybrid_wannier_centers(
                    mmn, strings, singular_tol=singular_tol)
                phase = _unwrap_transverse_phases(
                    phase, topology.kmesh, direction)
                center_sum = -cp.mean(phase) / (2. * np.pi)
            else:
                center = diagonal_wannier_centers(mmn, strings)
                center_sum = _sum_diagonal_centers(
                    center, topology.kmesh, direction)
            centers[spin].append(cp.asnumpy(center))
            phases[spin].append(cp.asnumpy(phase))
            center_sums_fractional[spin, direction] = center_sum
        del overlaps

    center_sums_fractional = cp.asnumpy(center_sums_fractional)
    lattice = np.asarray(mf.cell.lattice_vectors())
    center_sums_cartesian = center_sums_fractional @ lattice
    return WannierCenterResult(
        centers=tuple(tuple(channel) for channel in centers),
        berry_phases=tuple(tuple(channel) for channel in phases),
        center_sums_fractional=center_sums_fractional,
        center_sums_cartesian=center_sums_cartesian,
        spin_weights=spin_weights,
        kmesh=topology.kmesh.copy(),
        method=method,
    )


def eval_berry_phase(mf, kmesh=None, batch_size=None, singular_tol=1e-10):
    '''Return determinant Berry phases as ``phases[spin][direction]``.'''
    result = eval_wannier_centers(
        mf, kmesh=kmesh, batch_size=batch_size, method='wilson',
        singular_tol=singular_tol)
    return result.berry_phases


def electronic_polarization(cell, wannier, unit='au'):
    r'''Electronic polarization ``-sum_n f_n r_n / Omega``.'''
    factor, _ = _unit_factor(unit)
    weighted_centers = np.einsum(
        's,sx->x', wannier.spin_weights, wannier.center_sums_cartesian)
    return -weighted_centers / cell.vol * factor


def ionic_polarization(cell, unit='au'):
    r'''Ionic polarization ``sum_A Z_A R_A / Omega``.

    ``Cell.atom_charges`` supplies valence charges for pseudopotential atoms
    and nuclear charges for all-electron atoms.
    '''
    factor, _ = _unit_factor(unit)
    charges = cp.asarray(cell.atom_charges(), dtype=cp.float64)
    coordinates = cp.asarray(cell.atom_coords(), dtype=cp.float64)
    polarization = cp.einsum('a,ax->x', charges, coordinates) / cell.vol
    return cp.asnumpy(polarization) * factor


def polarization_quantum(cell, unit='au'):
    '''Return the three primitive polarization-quantum vectors as rows.'''
    factor, _ = _unit_factor(unit)
    return np.asarray(cell.lattice_vectors()) / cell.vol * factor


def eval_polarization(mf, kmesh=None, unit='au', batch_size=None,
                      method='wilson', wannier_gauge=None,
                      singular_tol=1e-10, return_details=False):
    '''Evaluate total macroscopic polarization through Wannier centers.

    The returned branch follows the continuously unwrapped determinant phase
    over transverse k-point strings. Use :func:`unwrap_polarization` when
    comparing two structures.
    '''
    wannier = eval_wannier_centers(
        mf, kmesh=kmesh, batch_size=batch_size, method=method,
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
    '''Move a polarization onto the branch nearest to ``reference``.'''
    polarization = np.asarray(polarization, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if polarization.shape != (3,) or reference.shape != (3,):
        raise ValueError('polarization and reference must both have shape (3,)')

    quantum = polarization_quantum(cell, unit)
    delta = polarization - reference
    fractional = delta @ np.linalg.inv(quantum)
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
    '''Return the minimum-branch polarization change from initial to final.'''
    matched, branch = unwrap_polarization(
        final, initial, cell, unit=unit, return_branch=True)
    difference = matched - np.asarray(initial)
    if return_branch:
        return difference, branch
    return difference
