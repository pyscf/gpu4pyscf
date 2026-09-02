# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
Unrestricted Hartree-Fock for periodic systems with k-point sampling
'''

__all__ = [
    'KUHF',
    'get_spin_flip_magmom',
]

import functools
from typing import Sequence
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.data.nist import HARTREE2EV
from pyscf.pbc.scf import kuhf as kuhf_cpu
from gpu4pyscf.scf import hf as mol_hf
from gpu4pyscf.pbc.scf import khf
from gpu4pyscf.pbc.scf import uhf as pbcuhf
from gpu4pyscf.pbc.scf.rsjk import PBCJKMatrixOpt
from gpu4pyscf.pbc.scf.j_engine import PBCJMatrixOpt
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import (
    return_cupy_array, contract, tag_array, sandwich_dot, asarray)


def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    '''Alpha and beta spin one particle density matrices for all k-points.

    Returns:
        dm_kpts : (2, nkpts, nao, nao) ndarray
    '''
    mo_occ_kpts = cp.asarray(mo_occ_kpts)
    mo_coeff_kpts = cp.asarray(mo_coeff_kpts)
    assert mo_occ_kpts.dtype == np.float64
    c = mo_coeff_kpts * mo_occ_kpts[:,:,None,:]
    dm = contract('nkpi,nkqi->nkpq', mo_coeff_kpts, c.conj())
    return tag_array(dm, mo_coeff=mo_coeff_kpts, mo_occ=mo_occ_kpts)

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
    if isinstance(atom_indices, (str, bytes)) or not isinstance(
            atom_indices, Sequence):
        raise TypeError('atom_indices must be a sequence of atom indices')

    checked_indices = []
    for ia in atom_indices:
        if isinstance(ia, (bool, np.bool_)) or not isinstance(
                ia, (int, np.integer)):
            raise TypeError(f'Atom index {ia!r} is not an integer')
        if not 0 <= ia < cell.natm:
            raise IndexError(
                f'Atom index {ia} is outside [0, {cell.natm})')
        checked_indices.append(int(ia))

    dm_flipped = dm.copy()
    aoslices = cell.aoslice_by_atom()
    for ia in checked_indices:
        p0, p1 = aoslices[ia, 2:]
        dm_flipped[0, :, p0:p1, p0:p1] = dm[1, :, p0:p1, p0:p1]
        dm_flipped[1, :, p0:p1, p0:p1] = dm[0, :, p0:p1, p0:p1]
    return dm_flipped

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    f_kpts = h1e_kpts + vhf_kpts
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f_kpts

    if s_kpts is None: s_kpts = mf.get_ovlp()
    if dm_kpts is None: dm_kpts = mf.make_rdm1()

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if damp_factor is None:
        damp_factor = mf.damp
    if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
        if isinstance(damp_factor, (tuple, list, np.ndarray)):
            dampa, dampb = damp_factor
        else:
            dampa = dampb = damp_factor
        f_a = []
        f_b = []
        for k in range(len(s_kpts)):
            f_a.append(asarray(mol_hf.damping(f_kpts[0][k], fock_last[0][k], dampa)))
            f_b.append(asarray(mol_hf.damping(f_kpts[1][k], fock_last[1][k], dampb)))
        f_kpts = cp.asarray([f_a, f_b])
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts, f_prev=fock_last)

    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if level_shift_factor is not None:
        if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
            shifta, shiftb = level_shift_factor
        else:
            shifta = shiftb = level_shift_factor
        f_kpts =([asarray(mol_hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta))
                  for k, s in enumerate(s_kpts)],
                 [asarray(mol_hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb))
                  for k, s in enumerate(s_kpts)])
    return cp.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''A pair of Fermi level for spin-up and spin-down orbitals
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ
    assert isinstance(mo_energy_kpts, cp.ndarray) and mo_energy_kpts.ndim == 3
    assert isinstance(mo_occ_kpts, cp.ndarray) and mo_occ_kpts.ndim == 3

    nocca, noccb = mf.nelec
    fermi_a = cp.partition(mo_energy_kpts[0].ravel(), nocca-1)[nocca-1]
    fermi_b = cp.partition(mo_energy_kpts[1].ravel(), noccb-1)[noccb-1]

    if mf.verbose >= logger.DEBUG:
        for k, mo_e in enumerate(mo_energy_kpts[0]):
            mo_occ = mo_occ_kpts[0][k]
            if mo_occ[mo_e > fermi_a].sum() > 0.5:
                logger.warn(mf, 'Alpha occupied band above Fermi level: \n'
                            'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
        for k, mo_e in enumerate(mo_energy_kpts[1]):
            mo_occ = mo_occ_kpts[1][k]
            if mo_occ[mo_e > fermi_b].sum() > 0.5:
                logger.warn(mf, 'Beta occupied band above Fermi level: \n'
                            'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    fermi_a = float(fermi_a.get())
    fermi_b = float(fermi_b.get())
    return (fermi_a, fermi_b)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''

    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    assert isinstance(mo_energy_kpts, cp.ndarray)

    nocc_a, nocc_b = mf.nelec
    mo_energy_a = cp.sort(mo_energy_kpts[0].ravel())
    nmo = mo_energy_a.size
    if nocc_a > nmo or nocc_b > nmo:
        raise RuntimeError('Failed to assign mo_occ. '
                           f'Nocc ({nocc_a}, {nocc_b}) > Nmo ({nmo})')
    fermi_a = mo_energy_a[nocc_a-1]
    mo_occ_kpts = cp.zeros_like(mo_energy_kpts)
    mo_occ_kpts[0] = (mo_energy_kpts[0] <= fermi_a).astype(np.float64)
    if nocc_b > 0:
        mo_energy_b = cp.sort(mo_energy_kpts[1].ravel())
        fermi_b = mo_energy_b[nocc_b-1]
        mo_occ_kpts[1] = (mo_energy_kpts[1] <= fermi_b).astype(np.float64)

    if nocc_a < nmo and nocc_b < nmo:
        homo = homo_a = fermi_a
        homo_b = None
        if nocc_b > 0:
            homo = max(homo, fermi_b)
        lumo = lumo_b = mo_energy_b[nocc_b]
        lumo_a = None
        if nocc_a < nmo:
            lumo_a = mo_energy_a[nocc_a]
            lumo = min(lumo, lumo_a)
        gap = (lumo - homo) * HARTREE2EV
        mf.scf_summary['gap'] = gap
        if mf.verbose >= logger.INFO:
            if lumo_a is not None:
                logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', homo_a, lumo_a)
            else:
                logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', homo_a)
            if homo_b is not None:
                logger.info(mf, 'beta HOMO = %.12g  LUMO = %.12g', homo_b, lumo_b)
            else:
                logger.info(mf, 'beta               LUMO = %.12g', lumo_b)
            if homo+1e-3 > lumo:
                logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', homo, lumo)
            else:
                logger.info(mf, '  HOMO = %.12g  LUMO = %.12g  gap/eV = %.5f',
                            homo, lumo, gap)

        if (mf.time_reversal_symmetry and
            (lumo_a is not None and homo_a+1e-5 > lumo_a) and
            (homo_b is not None and homo_b+1e-5 > lumo_b)):
            idx = np.array([(k, k_conj) for k, k_conj in mf.iter_kpt_pairs()
                            if k_conj is not None])
            if not cp.array_equal(mo_occ_kpts[0,idx[:,0]], mo_occ_kpts[0,idx[:,1]]):
                logger.warn(mf, 'k/-k pairs have unequal alpha occupations.')
            if not cp.array_equal(mo_occ_kpts[1,idx[:,0]], mo_occ_kpts[1,idx[:,1]]):
                logger.warn(mf, 'k/-k pairs have unequal beta occupations.')
    return mo_occ_kpts


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None or getattr(vhf_kpts, 'ecoul', None) is None:
        vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(h1e_kpts)
    e1 = 1./nkpts * cp.einsum('skij,kji->', dm_kpts, h1e_kpts).get()
    e2 = 1./nkpts * cp.einsum('skij,skji->', dm_kpts, vhf_kpts).get() * 0.5
    ecoul = vhf_kpts.ecoul
    exx = e2 - ecoul
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e2.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exx.real
    logger.debug(mf, 'E1 = %s  E2 = %s  Ecoul = %s  Exc = %s', e1, e2, ecoul, exx)
    if abs(e2.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e2.imag)
    return (e1+e2).real, e2.real

def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)
    ea, ca = khf.canonicalize(mf, mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
    eb, cb = khf.canonicalize(mf, mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
    mo_energy = cp.stack([ea, eb])
    mo_coeff = cp.stack([ca, cb])
    return mo_energy, mo_coeff

def _cast_mol_init_guess(fn):
    @functools.wraps(fn)
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = fn(mf, cell)
        assert dm.ndim == 3
        nkpts = len(kpts)
        if hasattr(dm, 'mo_coeff'):
            idx = np.where(cp.asnumpy(dm.mo_occ.sum(axis=0)) > 0)[0]
            mo_coeff = cp.repeat(asarray(dm.mo_coeff[:,None,:,idx]), nkpts, axis=1)
            mo_occ = cp.repeat(asarray(dm.mo_occ[:,None,idx]), nkpts, axis=1)
            dm = cp.repeat(asarray(dm[:,None]), nkpts, axis=1)
            dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        else:
            dm = cp.repeat(asarray(dm[:,None]), nkpts, axis=1)
        return dm
    return fn_init_guess

def _validate_magmom_input(cell, kpts, magmoms_dict, method):
    if not hasattr(magmoms_dict, 'items'):
        raise TypeError('magmoms_dict must be a mapping of atom indices to magnetic moments')

    kpts = np.asarray(kpts)
    if kpts.ndim == 1:
        kpts = kpts.reshape(1, -1)
    if kpts.ndim != 2 or kpts.shape[1] != 3 or len(kpts) == 0:
        raise ValueError('kpts must have shape (nkpts, 3)')
    if not np.isfinite(kpts).all():
        raise ValueError('kpts must contain only finite coordinates')

    if not isinstance(method, str):
        raise TypeError('method must be a string')
    method = method.lower()
    if method not in ('uniform', 'valence', 'spin_sad'):
        raise ValueError(f'Unknown magnetic-moment initial guess method {method!r}')

    magmoms = {}
    for ia, magmom in magmoms_dict.items():
        if isinstance(ia, (bool, np.bool_)) or not isinstance(ia, (int, np.integer)):
            raise TypeError(f'Atom index {ia!r} is not an integer')
        if not 0 <= ia < cell.natm:
            raise IndexError(f'Atom index {ia} is outside [0, {cell.natm})')
        try:
            magmom = float(magmom)
        except (TypeError, ValueError) as err:
            raise TypeError(f'Magnetic moment for atom {ia} is not a number') from err
        if not np.isfinite(magmom):
            raise ValueError(f'Magnetic moment for atom {ia} must be finite')
        magmoms[int(ia)] = magmom
    return kpts, magmoms, method

def _get_valence_ao_indices(cell, ia):
    from pyscf import gto
    from pyscf.data import elements

    labels = cell.ao_labels(fmt=False)
    p0, p1 = cell.aoslice_by_atom()[ia, 2:]
    angular_symbols = 'spdfghi'
    angular_momenta = {symbol: l for l, symbol in enumerate(angular_symbols)}

    ao_angular = []
    for ao in range(p0, p1):
        nl = str(labels[ao][2]).lower()
        angular = [angular_momenta[c] for c in nl if c in angular_momenta]
        if not angular:
            raise ValueError(f'Cannot determine angular momentum from AO label {labels[ao]!r}')
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
            occupied_l.append(l) # both ecp and occupied orbitals
    if not occupied_l:
        raise ValueError(f'Cannot identify valence orbitals for atom {ia}')
    highest_l = max(occupied_l)
    return np.asarray([ao for ao, l in zip(range(p0, p1), ao_angular)
                       if l == highest_l], dtype=np.int32)

def _make_atomic_mol(cell, ia, spin):
    from pyscf import gto

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
    atm.symmetry = False
    atm.enuc = 0
    atm._built = True
    return atm

def _get_spin_sad(cell, kpts, magmoms):
    import scipy.linalg
    from pyscf import scf

    aoslices = cell.aoslice_by_atom()
    dma_blocks = []
    dmb_blocks = []
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
                raise ValueError(f'Cannot assign a magnetic moment to ghost atom {ia}')
            dma_blocks.append(np.zeros((nao_atm, nao_atm)))
            dmb_blocks.append(np.zeros((nao_atm, nao_atm)))
            continue

        target_spin = abs(magmom)
        if target_spin > nelectron + 1e-9:
            raise ValueError(
                f'Atomic magnetic moment {target_spin} for atom {ia} exceeds '
                f'{nelectron} electrons')
        target_spin = min(target_spin, float(nelectron))

        density_cache = {}

        def get_atomic_density(spin):
            if spin in density_cache:
                return density_cache[spin]

            # An odd-electron unpolarized density.
            if spin == 0 and nelectron % 2:
                dma_ref, dmb_ref = get_atomic_density(1)
                dm = (dma_ref + dmb_ref) * .5
                density_cache[spin] = (dm, dm)
                return density_cache[spin]

            atm.spin = spin
            atm_mf = scf.UHF(atm)

            if max(atm.nelec) < atm.nao_nr():
                atm_mf = scf.addons.frac_occ(atm_mf)
            atm_mf.verbose = cell.verbose
            atm_mf.kernel()
            if not atm_mf.converged:
                logger.warn(cell, 'Atomic UHF for atom %d (%s) did not converge',
                            ia, cell.atom_symbol(ia))
            density_cache[spin] = atm_mf.make_rdm1()
            return density_cache[spin]

        spin_states = list(range(nelectron % 2, nelectron + 1, 2))
        if spin_states[0] != 0:
            spin_states.insert(0, 0)

        # for non-integer spin, find the closest integer spin states
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

    translations = np.zeros((1, 3))
    phase = np.exp(-1j * np.dot(kpts, translations.T))
    dm_kpts = np.einsum('kR,sRij->skij', phase, dm_r0[:, None])
    return cp.asarray(np.real_if_close(dm_kpts))

def get_init_guess_with_magmom(cell, kpts, magmoms_dict, method='valence',
                               key='minao', dm_init=None):
    r'''Generate a KUHF/KUKS initial density with atom-resolved moments.

    ``magmoms_dict`` maps zero-based atom indices to
    :math:`M_A=N_A^\alpha-N_A^\beta`. The returned density matrix has shape
    ``(2, nkpts, nao, nao)``.

    Args:
        cell:
            Periodic cell.
        kpts:
            K-point coordinates with shape ``(nkpts, 3)``.
        magmoms_dict:
            Mapping from atom indices to local magnetic moments.
        method:
            ``'uniform'``, ``'valence'``, or ``'spin_sad'``.
        key:
            Initial charge-density method used by ``uniform`` and ``valence``.

    For ``spin_sad``, fractional moments and integer moments incompatible with
    the neutral atom's spin parity are formed by linearly interpolating the
    neighboring allowed isolated-atom spin states. This preserves the atomic
    charge while producing the requested average moment. All-electron,
    molecular ECP, and GTH pseudopotential cells are supported.
    '''
    kpts, magmoms, method = _validate_magmom_input(
        cell, kpts, magmoms_dict, method)

    # Preserve the native initial guess exactly when no spin polarization was
    # requested. In particular, keep any tagged orbital information.
    if not any(magmoms.values()):
        return KUHF(cell, kpts=kpts).get_init_guess(key=key)

    if method == 'spin_sad':
        return _get_spin_sad(cell, kpts, magmoms)

    if dm_init is None:
        dm0 = KUHF(cell, kpts=kpts).get_init_guess(key=key)
    else:
        dm0 = dm_init
    dm_charge = dm0[0] + dm0[1]
    dm = cp.stack((dm_charge * .5, dm_charge * .5))
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

class KUHF(khf.KSCF):
    '''UHF class with k-point sampling.
    '''
    conv_tol_grad = kuhf_cpu.KUHF.conv_tol_grad
    init_guess_breaksym = kuhf_cpu.KUHF.init_guess_breaksym

    _keys = kuhf_cpu.KUHF._keys

    def __init__(self, cell, kpts=None, exxdiv='ewald'):
        khf.KSCF.__init__(self, cell, kpts, exxdiv)
        self.nelec = None

    def dump_flags(self, verbose=None):
        khf.KSCF.dump_flags(self, verbose)
        logger.info(self, 'number of electrons per cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    nelec = kuhf_cpu.KUHF.nelec

    init_guess_by_minao = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_atom)
    init_guess_by_huckel = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_huckel)
    init_guess_by_mod_huckel = _cast_mol_init_guess(pbcuhf.UHF.init_guess_by_mod_huckel)
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    energy_elec = energy_elec
    get_rho = khf.get_rho
    canonicalize = canonicalize

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        logger.info(self, 'Initial guess from hcore.')
        h = self.get_hcore(cell)
        s = self.get_ovlp(cell)
        e, c = self.eig((h, h), s)
        mo_occ = self.get_occ(e, c)
        nocc = int((mo_occ > 0).sum(axis=2).max())
        dm = self.make_rdm1(c[:,:,:,:nocc], mo_occ[:,:,:nocc])
        return dm

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm = cp.asarray(mol_hf.SCF.get_init_guess(self, cell, key))
        nkpts = len(self.kpts)
        assert dm.ndim == 4 and dm.shape[:2] == (2, nkpts)

        ne = cp.einsum('xkij,kji->x', dm, s1e).real.get()
        nelec = self.nelec
        if any(abs(ne - nelec) > 0.01*nkpts):
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons (%g, %g)',
                         ne.mean()/nkpts, nelec[0]/nkpts, nelec[1]/nkpts)
            ne[1] += 1e-300 # Number of beta electrons may be 0
            dm[0] *= nelec[0] / ne[0]
            dm[1] *= nelec[1] / ne[1]
            if hasattr(dm, 'mo_coeff'):
                dm.mo_occ[0] *= nelec[0] / ne[0]
                dm.mo_occ[1] *= nelec[1] / ne[1]
        return dm

    def get_veff(self, cell=None, dm_kpts=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None):
        return _get_veff(self, cell, dm_kpts, dm_last, vhf_last, hermi, kpts, kpts_band)

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        nkpts, nao = mo_occ_kpts.shape[1:3]

        def grad(mo, mo_occ, fock):
            omask = mo_occ > 0
            vmask = ~omask
            nocc = cp.count_nonzero(omask, axis=1).get()
            if all(nocc[0] == nocc):
                o = mo.transpose(0,2,1)[omask].reshape(nkpts,-1,nao)
                v = mo.transpose(0,2,1)[vmask].reshape(nkpts,-1,nao)
                g = contract('kpq,kjq->kpj', fock, o)
                g = contract('kpj,kip->kij', g, v.conj())
                return g.ravel()

            g = [ck[:,vmask[k]].conj().T.dot(fk.dot(ck[:,omask[k]])).ravel()
                 for k, (fk, ck) in enumerate(zip(fock, mo))]
            return cp.hstack(g).ravel()

        return cp.hstack([
            grad(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0]),
            grad(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])])

    def eig(self, h_kpts, s_kpts, overwrite=False, x=None, time_reversal_symmetry=None):
        e_a, c_a = khf.KSCF.eig(self, h_kpts[0], s_kpts, False, x, time_reversal_symmetry)
        e_b, c_b = khf.KSCF.eig(self, h_kpts[1], s_kpts, overwrite, x, time_reversal_symmetry)
        return cp.asarray((e_a,e_b)), cp.asarray((c_a,c_b))

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None: mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None: mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = kpts_band.ndim == 1
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
        fock += self.get_hcore(cell, kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)

        x = self.check_linear_dependency(s1e, time_reversal_symmetry=False)
        e, c = self.eig(fock, s1e, overwrite=True, x=x, time_reversal_symmetry=False)

        if single_kpt_band:
            e = e[:,0]
            c = c[:,0]
        return e, c

    init_guess_by_chkfile = return_cupy_array(kuhf_cpu.KUHF.init_guess_by_chkfile)

    mulliken_meta = NotImplemented
    mulliken_meta_spin = NotImplemented
    mulliken_pop = NotImplemented
    dip_moment = NotImplemented
    spin_square = NotImplemented
    stability = NotImplemented
    to_ks = NotImplemented
    convert_from_ = NotImplemented

    density_fit = khf.KRHF.density_fit
    x2c = x2c1e = sfx2c1e = khf.KRHF.sfx2c1e

    def Gradients(self):
        from gpu4pyscf.pbc.grad.kuhf import Gradients
        return Gradients(self)

    def to_cpu(self):
        mf = kuhf_cpu.KUHF(self.cell)
        with lib.temporary_env(self, _numint=None):
            utils.to_cpu(self, out=mf)
        return mf

    def analyze(self, verbose=None, **kwargs):
        '''Analyze the given SCF object:  print orbital energies, occupancies;
        print orbital coefficients; Mulliken population analysis; Dipole moment
        '''
        from pyscf.pbc.scf.kuhf import mulliken_meta
        if verbose is None:
            verbose = self.verbose
        log = logger.new_logger(self, verbose)
        mo_energy = self.mo_energy.get()
        mo_occ = self.mo_occ.get()
        cell = self.cell
        kpts = self.kpts
        if log.verbose >= logger.NOTE:
            self.dump_scf_summary(log)
            log.note('**** MO energy ****')
            log.note('                           alpha                               | beta')
            log.note('k-point                    nocc    HOMO/AU         LUMO/AU     | nocc    HOMO/AU         LUMO/AU')
            for k, kpt in enumerate(cell.get_scaled_kpts(kpts)):
                nocca = np.count_nonzero(mo_occ[0,k])
                noccb = np.count_nonzero(mo_occ[1,k])
                homoa = mo_energy[0,k,nocca-1]
                homob = mo_energy[1,k,noccb-1]
                lumoa = mo_energy[0,k,nocca  ]
                lumob = mo_energy[1,k,noccb  ]
                log.note('%2d (%6.3f %6.3f %6.3f) %2d   %15.9f %15.9f |%2d   %15.9f %15.9f',
                         k, kpt[0], kpt[1], kpt[2], nocca, homoa, lumoa, noccb, homob, lumob)

        log.note('**** Population analysis for atoms in the reference cell ****')
        s = self.get_ovlp(kpts=kpts).get()
        dm = self.make_rdm1().get()
        pop, chg = mulliken_meta(cell, dm, kpts=kpts, s=s, verbose=verbose)
        dip = None
        return (pop, chg), dip

    def gen_response(self, mo_coeff=None, mo_occ=None,
                     with_j=True, hermi=0, max_memory=None, with_nlc=False):
        cell = self.cell
        kpts = self.kpts
        with_j = with_j and hermi != 2
        def vind(dm1, kshift=0):
            assert kshift == 0
            vhf = _get_veff(self, cell, dm1, hermi=hermi, kpts=kpts,
                            with_j=with_j, with_ecoul=False)
            return vhf.view(cp.ndarray)
        return vind

    def newton(self):
        from gpu4pyscf.pbc.scf import soscf
        return soscf.newton(self)

def _get_veff(mf, cell=None, dm_kpts=None, dm_last=None, vhf_last=None,
              hermi=1, kpts=None, kpts_band=None, with_j=True, with_ecoul=True):
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts

    def get_vhf_(vj, vk):
        vk *= -1.
        if with_j and vj is not None:
            vk += vj
        return vk

    def trace(dm, vj):
        if kpts_band is not None:
            return None
        if not with_ecoul:
            return None
        if vj.ndim == 2:
            return cp.einsum('nij,ji->', dm_kpts, vj).real.get() * .5
        return cp.einsum('nKij,Kji->', dm_kpts, vj).real.get() * .5

    j_engine = None
    if with_j:
        j_engine = mf.j_engine

    if mf.rsjk or isinstance(j_engine, (PBCJKMatrixOpt, PBCJMatrixOpt)):
        incremental_veff = dm_last is not None and hasattr(vhf_last, 'sr')
        ddm = dm_kpts
        if incremental_veff:
            ddm = dm_kpts - dm_last

        vk_sr = 0
        vj = vj_sr = None
        ecoul = ecoul_sr = None
        if with_j:
            if isinstance(j_engine, (PBCJKMatrixOpt, PBCJMatrixOpt)):
                if j_engine.supmol is None:
                    j_engine.build(kpts)
                vj_sr = j_engine._get_j_sr(ddm.sum(axis=0), hermi, kpts, kpts_band)
                vj = j_engine._get_j_lr(dm_kpts.sum(axis=0), hermi, kpts, kpts_band)
                if with_ecoul:
                    if incremental_veff:
                        if hasattr(vhf_last, 'ecoul_sr'):
                            ecoul_sr = trace(dm_last, vj_sr) * 2
                            ecoul_sr += trace(ddm, vj_sr)
                            ecoul_sr += vhf_last.ecoul_sr
                            ecoul = trace(dm_kpts, vj) + ecoul_sr
                    else:
                        ecoul_sr = trace(dm_kpts, vj_sr)
                        ecoul = trace(dm_kpts, vj) + ecoul_sr
            else:
                vj = mf.get_j(cell, dm_kpts.sum(axis=0), hermi, kpts, kpts_band)
                if with_ecoul:
                    ecoul = trace(dm_kpts, vj)

        if mf.rsjk:
            if mf.rsjk.supmol is None:
                mf.rsjk.build(kpts)
            vk_sr = mf.rsjk._get_k_sr(ddm, hermi, kpts, kpts_band, mf.exxdiv)
            vk = mf.rsjk._get_k_lr(dm_kpts, hermi, kpts, kpts_band, mf.exxdiv)
        else:
            vk = mf.get_k(cell, dm_kpts, hermi, kpts, kpts_band)

        vhf_sr = get_vhf_(vj_sr, vk_sr)
        if incremental_veff:
            vhf_sr += vhf_last.sr
        vhf = get_vhf_(vj, vk) + vhf_sr
        vhf = tag_array(vhf, sr=vhf_sr)

        if with_ecoul and ecoul is not None:
            vhf.ecoul = ecoul
            if ecoul_sr is not None:
                vhf.ecoul_sr = ecoul_sr
    else:
        vj, vk = mf.with_df.get_jk(
            dm_kpts, hermi, kpts, kpts_band, with_j=with_j, with_k=True,
            exxdiv=mf.exxdiv)
        vj = vj.sum(axis=0)
        if with_j and with_ecoul:
            ecoul = trace(dm_kpts, vj)
            vhf = tag_array(get_vhf_(vj, vk), ecoul=ecoul)
        else:
            vhf = get_vhf_(vj, vk)
    return vhf
