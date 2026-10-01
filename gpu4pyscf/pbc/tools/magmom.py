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

'''Tools for constructing atom-resolved magnetic moments in periodic systems.'''

import numbers
import cupy as cp
import numpy as np
import scipy.linalg
from pyscf import scf
from pyscf import gto
from pyscf.data import elements
from gpu4pyscf.lib import logger


__all__ = [
    'get_init_guess_with_magmom',
    'get_spin_flip_magmom',
]


def get_spin_flip_magmom(cell, dm, atom_indices):
    '''Swap alpha and beta density blocks for the specified atoms.

    Args:
        cell:
            Periodic cell that defines the atom-to-AO mapping.
        dm:
            Spin-resolved density matrix with shape ``(2, nkpts, nao, nao)``.
            The number of k-points is inferred from this array.
        atom_indices:
            Zero-based indices of the atoms whose local spins are flipped.
    '''
    if not isinstance(dm, cp.ndarray):
        raise TypeError('dm must be a CuPy ndarray')

    nao = cell.nao_nr()
    if (dm.ndim != 4 or dm.shape[0] != 2 or dm.shape[1] == 0 or
            dm.shape[-2:] != (nao, nao)):
        raise ValueError(
            f'dm must have shape (2, nkpts, {nao}, {nao}) with nkpts > 0; '
            f'got {dm.shape}')

    atom_indices = np.asarray(atom_indices)
    if (atom_indices.ndim != 1 or
            (atom_indices.size > 0 and
             not np.issubdtype(atom_indices.dtype, np.integer))):
        raise TypeError('atom_indices must be a sequence of atom indices')

    assert np.all((0 <= atom_indices) & (atom_indices < cell.natm))

    dm_flipped = dm.copy()
    aoslices = cell.aoslice_by_atom()
    for ia in atom_indices:
        p0, p1 = aoslices[ia, 2:]
        dm_flipped[0, :, p0:p1, p0:p1] = dm[1, :, p0:p1, p0:p1]
        dm_flipped[1, :, p0:p1, p0:p1] = dm[0, :, p0:p1, p0:p1]
    return dm_flipped


def _get_valence_ao_indices(cell, ia):

    labels = cell.ao_labels(fmt=False)
    p0, p1 = cell.aoslice_by_atom()[ia, 2:]
    angular_symbols = 'spdfghi'
    angular_momenta = {symbol: l for l, symbol in enumerate(angular_symbols)}

    ao_angular = []
    for ao in range(p0, p1):
        nl = str(labels[ao][2]).lower()
        angular = [angular_momenta[c] for c in nl if c in angular_momenta]
        if not angular:
            raise ValueError(
                f'Cannot determine angular momentum from AO label {labels[ao]!r}')
        ao_angular.append(max(angular))

    if not ao_angular:
        raise ValueError(f'Atom {ia} has no atomic orbitals')

    nuc = gto.charge(cell.atom_pure_symbol(ia))
    atomic_configuration = elements.NRSRHF_CONFIGURATION[nuc]
    core_shells = gto.ecp.core_configuration(
        cell.atom_nelec_core(ia), atom_symbol=cell.atom_pure_symbol(ia))
    occupied_l = []
    for l, nelectron_l in enumerate(atomic_configuration):
        core_electrons_l = core_shells[l] * (4 * l + 2)
        if nelectron_l > core_electrons_l and l in ao_angular:
            occupied_l.append(l)  # both ecp and occupied orbitals
    if not occupied_l:
        raise ValueError(f'Cannot identify valence orbitals for atom {ia}')
    highest_l = max(occupied_l)
    return np.asarray([ao for ao, l in zip(range(p0, p1), ao_angular)
                       if l == highest_l], dtype=np.int32)


def _make_atomic_mol(cell, ia, spin):
    atm = cell.copy(deep=False)
    atm._atom = [cell._atom[ia]]
    atm.atom = atm._atom
    atm._atm = cell._atm[ia:ia+1].copy()

    bas_mask = cell._bas[:, gto.ATOM_OF] == ia
    atm._bas = cell._bas[bas_mask].copy()
    atm._bas[:, gto.ATOM_OF] = 0

    ecpbas_mask = cell._ecpbas[:, gto.ATOM_OF] == ia
    atm._ecpbas = cell._ecpbas[ecpbas_mask].copy()
    if len(atm._ecpbas) > 0:
        atm._ecpbas[:, gto.ATOM_OF] = 0

    atm.charge = 0
    atm.spin = spin
    atm._nelectron = None
    atm.symmetry = False
    atm.enuc = 0
    atm._built = True
    return atm


def _get_spin_sad(cell, kpts, magmoms):
    aoslices = cell.aoslice_by_atom()
    dma_blocks = []
    dmb_blocks = []
    # Isolated-atom densities are position-independent.
    density_cache = {}

    def get_atomic_density(spin):
        cache_key = (cell.atom_symbol(ia), int(spin))
        if cache_key in density_cache:
            return density_cache[cache_key]

        # An odd-electron unpolarized density.
        if spin == 0 and nelectron % 2 == 1:
            dma_ref, dmb_ref = get_atomic_density(1)
            dm = (dma_ref + dmb_ref) * .5
            density_cache[cache_key] = (dm, dm)
            return density_cache[cache_key]

        atm.spin = spin
        atm_mf = scf.UHF(atm)

        if max(atm.nelec) < atm.nao_nr():
            atm_mf = scf.addons.frac_occ(atm_mf)
        atm_mf.verbose = cell.verbose
        atm_mf.kernel()
        if not atm_mf.converged:
            logger.warn(
                cell, 'Atomic UHF for atom %d (%s) did not converge',
                ia, cell.atom_symbol(ia))
        density_cache[cache_key] = atm_mf.make_rdm1()
        return density_cache[cache_key]

    for ia, (p0, p1) in enumerate(aoslices[:, 2:]):
        nao_atm = p1 - p0
        magmom = magmoms.get(ia, 0.)
        if nao_atm == 0:
            if magmom:
                raise ValueError(f'Atom {ia} has no basis functions')
            dma_blocks.append(np.zeros((0, 0)))
            dmb_blocks.append(np.zeros((0, 0)))
            continue

        atm = _make_atomic_mol(cell, ia, 0)
        nelectron = atm.nelectron
        if nelectron == 0:
            if magmom:
                raise ValueError(
                    f'Cannot assign a magnetic moment to ghost atom {ia}')
            dma_blocks.append(np.zeros((nao_atm, nao_atm)))
            dmb_blocks.append(np.zeros((nao_atm, nao_atm)))
            continue

        target_spin = abs(magmom)
        if target_spin > nelectron + 1e-9:
            raise ValueError(
                f'Atomic magnetic moment {target_spin} for atom {ia} exceeds '
                f'{nelectron} electrons')
        target_spin = min(target_spin, float(nelectron))

        max_spin = min(nelectron, 2 * nao_atm - nelectron)
        if target_spin > max_spin:
            raise ValueError(
                f'Atomic magnetic moment {target_spin} for atom {ia} cannot '
                f'be represented by {nao_atm} orbitals and {nelectron} electrons')

        if nelectron % 2 == 0:
            spin_states = np.arange(0, max_spin + 1, 2)
        else:
            spin_states = np.append(0, np.arange(1, max_spin + 1, 2))

        # For non-integer spin, find the closest integer spin states.
        upper_index = np.searchsorted(spin_states, target_spin)
        if upper_index < len(spin_states) and np.isclose(
                target_spin, spin_states[upper_index], atol=1e-12, rtol=0):
            lower_spin = upper_spin = spin_states[upper_index]
        else:
            lower_spin = spin_states[upper_index - 1]
            upper_spin = spin_states[upper_index]

        dma, dmb = get_atomic_density(lower_spin)
        if lower_spin != upper_spin:
            dma_upper, dmb_upper = get_atomic_density(upper_spin)
            upper_weight = (
                (target_spin - lower_spin) / (upper_spin - lower_spin)
            )
            lower_weight = 1. - upper_weight
            dma = lower_weight * dma + upper_weight * dma_upper
            dmb = lower_weight * dmb + upper_weight * dmb_upper

        if magmom < 0:
            dma, dmb = dmb, dma
        dma_blocks.append(dma)
        dmb_blocks.append(dmb)

    dm_r0 = np.asarray((scipy.linalg.block_diag(*dma_blocks),
                        scipy.linalg.block_diag(*dmb_blocks)))

    dm_kpts = cp.repeat(cp.asarray(dm_r0[:,None]), len(kpts), axis=1)
    return dm_kpts


def get_init_guess_with_magmom(cell, kpts, magmoms_dict, method='spin_sad',
                               key='minao', dm_init=None):
    r'''Generate a KUHF/KUKS initial density with atom-resolved moments.

    ``magmoms_dict`` maps zero-based atom indices to
    :math:`M_A=N_A^\alpha-N_A^\beta`. The returned density matrix has shape
    ``(2, nkpts, nao, nao)``.

    Args:
        cell:
            Periodic cell.
        kpts:
            Equally weighted k-point coordinates with shape ``(nkpts, 3)``.
            A single k-point with shape ``(3,)`` is also accepted.
        magmoms_dict:
            Mapping from atom indices to finite, real local magnetic moments
            per primitive cell. Unspecified atoms have zero initial moment.
            An empty mapping or all-zero moments with no ``dm_init`` preserve
            the native KUHF initial guess, including its total spin.
        method:
            ``'uniform'``, ``'valence'``, or ``'spin_sad'``.
        key:
            Initial charge-density method used by ``uniform`` and ``valence``.
        dm_init:
            Optional CuPy density with shape ``(2, nkpts, nao, nao)`` for
            ``uniform`` and ``valence``. Its spin-summed density is preserved.

    Nonzero total moments are allowed. In KUHF/KUKS, ``cell.spin`` is the
    alpha-minus-beta electron count over the entire k-point mesh, not per
    primitive cell. A matching initial guess therefore requires
    ``len(kpts) * sum(magmoms_dict.values()) == cell.spin``. A mismatch is
    logged as a warning: the requested initial moments are retained, and
    neither ``cell.spin`` nor ``kpts`` is changed. Subsequent SCF occupations
    follow the SCF object's electron counts, not these initial local moments.
    Invalid electron-count/spin parity or orbital capacity raises ValueError.

    For ``spin_sad``, fractional moments and integer moments incompatible with
    the neutral atom's spin parity are formed by linearly interpolating the
    neighboring allowed isolated-atom spin states. This preserves the atomic
    charge while producing the requested average moment. All-electron,
    molecular ECP, and GTH pseudopotential cells are supported. The polarized
    ``spin_sad`` construction requires a neutral cell; use ``uniform`` or
    ``valence`` for a charged cell.
    '''

    if not isinstance(method, str):
        raise TypeError('method must be a string')
    method = method.lower()
    if method not in ('uniform', 'valence', 'spin_sad'):
        raise ValueError(f'Unknown magnetic-moment initial guess method {method!r}')
    if not isinstance(magmoms_dict, dict):
        raise TypeError('magmoms_dict must be a dict of atom indices and moments')
    magmoms = magmoms_dict

    kpts = np.asarray(kpts)
    if kpts.shape == (3,):
        kpts = kpts.reshape(1, 3)
    if kpts.ndim != 2 or kpts.shape[1] != 3 or len(kpts) == 0:
        raise ValueError('kpts must have shape (nkpts, 3) with nkpts > 0')
    if (not np.issubdtype(kpts.dtype, np.number) or np.iscomplexobj(kpts) or
            not np.isfinite(kpts).all()):
        raise ValueError('kpts must contain only finite real coordinates')

    for ia, magmom in magmoms.items():
        if isinstance(ia, (bool, np.bool_)) or not isinstance(ia, numbers.Integral):
            raise TypeError(f'Atom index {ia!r} must be an integer')
        if not 0 <= ia < cell.natm:
            raise IndexError(f'Atom index {ia} is outside [0, {cell.natm})')
        if (isinstance(magmom, (bool, np.bool_)) or
                not isinstance(magmom, numbers.Real)):
            raise TypeError(f'Magnetic moment for atom {ia} must be a real number')
        if not np.isfinite(magmom):
            raise ValueError(f'Magnetic moment for atom {ia} must be finite')

    nkpts = len(kpts)
    nelectron = cell.tot_electrons(nkpts)
    spin = cell.spin
    if (isinstance(spin, (bool, np.bool_)) or
            not isinstance(spin, numbers.Integral)):
        raise ValueError('cell.spin must be an integer')
    if abs(spin) > nelectron or (nelectron + spin) % 2:
        raise ValueError(
            f'cell.spin={spin} is incompatible with {nelectron} electrons '
            f'over {nkpts} k-points')
    nao = cell.nao_nr()
    if (nelectron + abs(spin)) / 2 > nkpts * nao:
        raise ValueError('cell.spin exceeds the available orbital capacity')

    if dm_init is not None:
        if method == 'spin_sad':
            raise ValueError('dm_init is only supported by uniform and valence')
        if not isinstance(dm_init, cp.ndarray):
            raise TypeError('dm_init must be a CuPy ndarray')
        if dm_init.shape != (2, nkpts, nao, nao):
            raise ValueError(
                f'dm_init must have shape (2, {nkpts}, {nao}, {nao})')
        if not cp.isfinite(dm_init).all():
            raise ValueError('dm_init must contain only finite values')

    # Preserve the native initial guess exactly when no spin polarization was
    # requested. In particular, keep any tagged orbital information.
    if not any(magmoms.values()) and dm_init is None:
        from gpu4pyscf.pbc.scf.kuhf import KUHF
        return KUHF(cell, kpts=kpts).get_init_guess(key=key)

    if (method == 'spin_sad' and
            nelectron != cell.atom_charges().sum() * nkpts):
        raise ValueError(
            'spin_sad requires a neutral cell; use uniform or valence '
            'for a charged cell')

    total_magmom = sum(magmoms.values())
    if abs(total_magmom) > nelectron / nkpts + 1e-9:
        raise ValueError('Total magnetic moment exceeds the number of electrons')
    if not np.isclose(total_magmom * nkpts, spin, atol=1e-9, rtol=0):
        logger.warn(
            cell, 'Initial magnetic moment per cell %g times %d k-points '
            'does not match cell.spin=%g. The requested initial moments are '
            'retained; SCF occupations will follow the SCF electron counts.',
            total_magmom, nkpts, spin)

    if method == 'spin_sad':
        return _get_spin_sad(cell, kpts, magmoms)

    if dm_init is None:
        from gpu4pyscf.pbc.scf.khf import KRHF
        dm_charge = KRHF(cell, kpts=kpts).get_init_guess(key=key)
    else:
        dm_charge = dm_init[0] + dm_init[1]
    dm = cp.repeat(dm_charge[None] * .5, 2, axis=0)
    aoslices = cell.aoslice_by_atom()
    for ia, magmom in magmoms.items():
        if magmom == 0:
            continue
        if method == 'uniform':
            p0, p1 = aoslices[ia, 2:]
            ao_indices = np.arange(p0, p1, dtype=np.int32)
        else:
            ao_indices = _get_valence_ao_indices(cell, ia)
        if len(ao_indices) == 0:
            raise ValueError(f'Atom {ia} has no orbitals for method={method!r}')

        delta_q = magmom / (2 * len(ao_indices))
        for k in range(len(kpts)):
            dm[0, k, ao_indices, ao_indices] += delta_q
            dm[1, k, ao_indices, ao_indices] -= delta_q
    return dm
