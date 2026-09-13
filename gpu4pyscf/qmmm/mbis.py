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

import cupy as cp
import numpy as np

from gpu4pyscf.lib import logger
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf.solvent.pcm import natm_without_ghost

MBIS_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-10

def _period_of_element(Z):
    for period, upper in enumerate((2, 10, 18, 36, 54, 86, 118), start = 1):
        if Z <= upper:
            return period
    raise ValueError(f"Z = {Z} not supported by _period_of_element()")

def _neutral_atom_shell_populations(Z):
    orbital_occupations = (
        (1, 2),
        (2, 2),
        (2, 6),
        (3, 2),
        (3, 6),
        (4, 2),
        (3, 10),
        (4, 6),
        (5, 2),
        (4, 10),
        (5, 6),
        (6, 2),
        (4, 14),
        (5, 10),
        (6, 6),
        (7, 2),
        (5, 14),
        (6, 10),
        (7, 6),
    )
    n_shell = _period_of_element(Z)
    population_of_shell = np.zeros(n_shell)
    remaining = Z
    for principal_n, capacity in orbital_occupations:
        occupation = min(remaining, capacity)
        population_of_shell[principal_n - 1] += occupation
        remaining -= occupation
        if remaining == 0:
            break
    assert remaining == 0, f"Z = {Z} not supported by _neutral_atom_shell_populations()"
    return population_of_shell

def _initial_width(Z):
    n_shell = _period_of_element(Z)
    if n_shell == 1:
        return np.array([0.5 / Z])
    i_shell = np.arange(n_shell, dtype=float)
    Z_exponent = 1.0 - i_shell / (n_shell - 1)
    return 0.5 / np.power(float(Z), Z_exponent) # Eq between 19 and 20, \sigma_{Ai} = a_0 / 2 Z_A^{1 - (i-1)/(m_A-1)}

def _mbis_density_on_grid(atom_grid_rij, shell_atom_indices, populations, widths):
    shell_grid_rij = atom_grid_rij[shell_atom_indices]
    prefactors = populations / (8.0 * np.pi * widths**3)
    return prefactors[:, None] * cp.exp(-shell_grid_rij / widths[:, None]) # Eq 7

def mbis(mol, grids, dm, conv_tol = 1e-8, max_cycle = 500, damping = 0.1, compute_multipoles = True):
    """
        Implementation follows:
        Verstraelen, T., Vandenbrande, S., Heidar-Zadeh, F., Vanduyfhuys, L., Van Speybroeck, V., Waroquier, M., & Ayers, P. W. (2016).
        Minimal basis iterative stockholder: atoms in molecules for force-field development.
        Journal of Chemical Theory and Computation, 12(8), 3894-3912.
    """
    assert 0.0 <= damping and damping <= 1.0

    log = logger.new_logger(mol)

    dm = cp.asarray(dm)
    if dm.ndim == 3:
        assert dm.shape[0] == 2
        dm = dm[0] + dm[1]
    assert dm.shape == (mol.nao, mol.nao)

    natm = natm_without_ghost(mol)
    if natm != mol.natm:
        raise NotImplementedError("Ghost atoms are not supported in MBIS yet")
    if len(mol._ecpbas) > 0:
        raise NotImplementedError("ECP is not supported in MBIS yet")
    if mol.pseudo:
        raise NotImplementedError("GTH pseudopotential is not supported in MBIS yet")

    if mol.charge < 0:
        log.warn("MBIS does not handle negative charge well, particularly when the negative charge is diffuse. "
                 "If you see a shell with huge width on one atom, it is likely due to the diffused charge got fitted onto that random atom.")

    atom_coords = cp.asarray(mol.atom_coords())
    atom_charges = cp.asarray(mol.atom_charges(), dtype = cp.int32)
    assert cp.all(atom_charges > 0)

    if grids.coords is None:
        grids.build()

    grid_coords = cp.asarray(grids.coords)
    grid_weights = cp.asarray(grids.weights)
    ni = NumInt()
    grid_rho = ni.get_rho(mol, dm, grids)

    rho_nonzero_mask = cp.logical_and(
        grid_rho >= MBIS_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cp.abs(grid_weights) > 1e-14,
    )

    grid_rho = grid_rho[rho_nonzero_mask]
    grid_coords = cp.ascontiguousarray(grid_coords[rho_nonzero_mask, :])
    grid_weights = grid_weights[rho_nonzero_mask]
    grid_w_rho = grid_weights * grid_rho

    nelec = float(cp.sum(grid_w_rho))
    log.info(f"Total number of electrons integrated by MBIS grid = {nelec}")

    atom_grid_vecrij = grid_coords[None, :, :] - atom_coords[:, None, :]
    atom_grid_rij = cp.linalg.norm(atom_grid_vecrij, axis=2)
    del atom_grid_vecrij

    shell_populations = []
    shell_widths = []
    shell_atom_indices = []
    atom_shell_offsets = []
    n_shell_offset = 0
    for i_atom in range(mol.natm):
        Z = int(atom_charges[i_atom])
        shell_population = _neutral_atom_shell_populations(Z)
        shell_populations.append(shell_population)
        shell_width = _initial_width(Z)
        shell_widths.append(shell_width)
        n_shell_of_atom = shell_population.shape[0]
        shell_atom_indices.extend([i_atom] * n_shell_of_atom)
        atom_shell_offsets.append(n_shell_offset)
        n_shell_offset += n_shell_of_atom
    atom_shell_offsets.append(n_shell_offset)
    shell_populations = cp.asarray(np.concatenate(shell_populations))
    shell_widths = cp.asarray(np.concatenate(shell_widths))
    shell_atom_indices = np.asarray(shell_atom_indices, dtype = np.int32)

    converged = False
    for i_cycle in range(max_cycle):
        old_shell_rho = _mbis_density_on_grid(atom_grid_rij, shell_atom_indices, shell_populations, shell_widths)
        old_rho0 = cp.sum(old_shell_rho, axis = 0)

        rho0_nonzero_mask = old_rho0 >= MBIS_REMOVE_ZERO_RHO_GRID_THRESHOLD
        integration_prefactor = cp.zeros_like(grid_rho)
        integration_prefactor[rho0_nonzero_mask] = grid_w_rho[rho0_nonzero_mask] / old_rho0[rho0_nonzero_mask] # w (from integration) * rho / rho0 in Eq 18,19
        del old_rho0, rho0_nonzero_mask
        target_shell_populations = old_shell_rho @ integration_prefactor # Eq 18
        target_shell_widths = (old_shell_rho * atom_grid_rij[shell_atom_indices]) @ integration_prefactor
        target_shell_widths /= 3.0 * target_shell_populations # Eq 19
        del integration_prefactor

        assert cp.all(target_shell_populations >= 0.0)

        new_shell_populations = damping * shell_populations + (1.0 - damping) * target_shell_populations
        new_shell_widths = damping * shell_widths + (1.0 - damping) * target_shell_widths

        new_shell_rho = _mbis_density_on_grid(atom_grid_rij, shell_atom_indices, new_shell_populations, new_shell_widths)
        max_delta_rhoA0 = 0
        for i_atom in range(mol.natm):
            s0, s1 = atom_shell_offsets[i_atom], atom_shell_offsets[i_atom + 1]
            old_rhoA0 = cp.sum(old_shell_rho[s0:s1, :], axis = 0)
            new_rhoA0 = cp.sum(new_shell_rho[s0:s1, :], axis = 0)
            delta_rhoA0 = cp.sum(grid_weights * (new_rhoA0 - old_rhoA0)**2) # Eq 20
            max_delta_rhoA0 = max(max_delta_rhoA0, float(delta_rhoA0))
            del old_rhoA0, new_rhoA0
        max_delta_rhoA0 = float(np.sqrt(max_delta_rhoA0))
        del old_shell_rho, new_shell_rho
        log.info(f"MBIS i_cycle = {i_cycle:4d} max_delta_rhoA0 = {max_delta_rhoA0:.6e}")

        shell_populations = new_shell_populations
        shell_widths = new_shell_widths
        if max_delta_rhoA0 < conv_tol:
            converged = True
            break

    assert converged, f"MBIS not converged in {max_cycle} iterations; last max_delta_rhoA0 = {max_delta_rhoA0:.3e} > conv_tol = {conv_tol:.3e}"
    log.info("MBIS converged!")

    n_shell = shell_atom_indices.shape[0]
    for i_shell in range(n_shell):
        log.info(f"MBIS shell {i_shell} from atom {shell_atom_indices[i_shell]} has "
                 "population = {shell_populations[i_shell]} and width = {shell_widths[i_shell]}")

    if not compute_multipoles:
        return shell_populations.get(), shell_widths.get(), shell_atom_indices

    shell_rho = _mbis_density_on_grid(atom_grid_rij, shell_atom_indices, shell_populations, shell_widths)
    atom_rho = []
    for i_atom in range(mol.natm):
        s0, s1 = atom_shell_offsets[i_atom], atom_shell_offsets[i_atom + 1]
        atom_rho.append(cp.sum(shell_rho[s0:s1, :], axis = 0))
    atom_rho = cp.vstack(atom_rho)
    rho0 = cp.sum(atom_rho, axis = 0)
    rho0_nonzero_mask = rho0 >= MBIS_REMOVE_ZERO_RHO_GRID_THRESHOLD
    atom_partition = cp.zeros_like(atom_rho)
    atom_partition[:, rho0_nonzero_mask] = atom_rho[:, rho0_nonzero_mask] / rho0[rho0_nonzero_mask]
    del rho0, rho0_nonzero_mask, atom_rho

    # MBIS Multipoles

    partitioned_w_rho = atom_partition * grid_w_rho[None, :]
    partitioned_nelec = cp.sum(partitioned_w_rho, axis = 1)
    charges = atom_charges.astype(cp.float64) - partitioned_nelec

    log.info("MBIS Charge (a.u.)")
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {charges[i_atom]:13.8f}")

    atom_grid_vecrij = grid_coords[None, :, :] - atom_coords[:, None, :]

    dipoles = -cp.einsum("Ag,Agx->Ax", partitioned_w_rho, atom_grid_vecrij)

    log.info("MBIS Dipole x y z (a.u.)")
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {dipoles[i_atom, 0]:13.8f}  {dipoles[i_atom, 1]:13.8f}  {dipoles[i_atom, 2]:13.8f}")

    # ORCA did not remove trace, and we follow them
    quadrupoles = -cp.einsum("Ag,Agx,Agy->Axy", partitioned_w_rho, atom_grid_vecrij, atom_grid_vecrij)

    log.info("MBIS Quadrupole xx yy zz xy xz yz (a.u.)")
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {quadrupoles[i_atom, 0, 0]:13.8f}  {quadrupoles[i_atom, 1, 1]:13.8f}  {quadrupoles[i_atom, 2, 2]:13.8f}  "
                 f"{0.5 * (quadrupoles[i_atom, 0, 1] + quadrupoles[i_atom, 1, 0]):13.8f}  "
                 f"{0.5 * (quadrupoles[i_atom, 0, 2] + quadrupoles[i_atom, 2, 0]):13.8f}  "
                 f"{0.5 * (quadrupoles[i_atom, 1, 2] + quadrupoles[i_atom, 2, 1]):13.8f}")

    # ORCA did not remove trace, and we follow them
    octupoles = -cp.einsum("Ag,Agx,Agy,Agz->Axyz", partitioned_w_rho, atom_grid_vecrij, atom_grid_vecrij, atom_grid_vecrij)

    log.info("MBIS Octupole xxx yyy zzz xxy xxz xyy xyz xzz yyz yzz (a.u.)")
    octupole_xyz_term = octupoles[i_atom, 0, 1, 2] + octupoles[i_atom, 0, 2, 1] + octupoles[i_atom, 1, 0, 2] \
                        + octupoles[i_atom, 1, 2, 0] + octupoles[i_atom, 2, 0, 1] + octupoles[i_atom, 2, 1, 0] # Just to make linter happy
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {octupoles[i_atom, 0, 0, 0]:13.8f}  {octupoles[i_atom, 1, 1, 1]:13.8f}  {octupoles[i_atom, 2, 2, 2]:13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 0, 1] + octupoles[i_atom, 0, 1, 0] + octupoles[i_atom, 1, 0, 0]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 0, 2] + octupoles[i_atom, 0, 2, 0] + octupoles[i_atom, 2, 0, 0]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 1, 1] + octupoles[i_atom, 1, 0, 1] + octupoles[i_atom, 1, 1, 0]):13.8f}  "
                 f"{1.0/6.0 * octupole_xyz_term:13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 2, 2] + octupoles[i_atom, 2, 0, 2] + octupoles[i_atom, 2, 2, 0]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 1, 1, 2] + octupoles[i_atom, 1, 2, 1] + octupoles[i_atom, 2, 1, 1]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 1, 2, 2] + octupoles[i_atom, 2, 1, 2] + octupoles[i_atom, 2, 2, 1]):13.8f}  "
                 f"")

    return charges.get(), dipoles.get(), quadrupoles.get(), octupoles.get()
