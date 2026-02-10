# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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


"""
Fewest Switches Surface Hopping (FSSH) Implementation

This module provides an enhanced implementation of the FSSH algorithm
for nonadiabatic molecular dynamics simulations.

References:
    1. Molecular dynamics with electronic transitions.
       John C. Tully
       J. Chem. Phys. 93, 1061 (1990).
       DOI: 10.1063/1.459170

    2. Nonadiabatic Field on Quantum Phase Space: A Century after Ehrenfest
       Baihua Wu, Xin He, and Jian Liu
       J. Phys. Chem. Lett. 15, 644 (2024).
       DOI: 10.1021/acs.jpclett.3c03385

    3. Critical appraisal of the fewest switches algorithm for surface hopping
       Giovanni Granucci, Maurizio Persico
       J. Chem. Phys. 126 (13): 134114 (2007).
       DOI: 10.1063/1.2715585
"""

from typing import Tuple, Optional, List
from collections import deque
import json
import numpy as np
import cupy as cp
import h5py
from pyscf.data.nist import AMU2AU, BOHR, HARTREE2J, PLANCK
from gpu4pyscf.lib import logger

# Physical constants for unit conversions
FS2AUTIME = 2*np.pi * HARTREE2J / PLANCK * 1e-15  # 41.34137: femtoseconds to atomic time units
A2BOHR = 1/BOHR  # Conversion factor: Angstrom to Bohr radius

class FSSH:
    """
    This class implements the FSSH algorithm for nonadiabatic molecular dynamics simulations.

    The FSSH method treats nuclear motion classically while quantum mechanically
    describing electronic transitions between different potential energy surfaces.

    Attributes:
        tddft: Time-dependent density functional theory object
        tdgrad: Nuclear gradient scanner for force calculations
        states (List[int]): List of electronic states to include in simulation
        cur_state (int): Current active electronic state
        mass (np.ndarray): Nuclear masses in atomic units
        dt (float): Time step in atomic units
        nsteps (int): Number of simulation steps
        decoherence: Whether to perform decoherence
        alpha: parameter for the strength of decoherence
        seed: Random seed for hopping
    """

    seed = None
    tdc_method = 'nac'
    save_force = False
    decoherence = True
    alpha = 0.1

    def __init__(self, mol, states: list[int]):
        """
        Initialize the FSSH simulation with comprehensive parameter validation.

        Args:
            mol: Mole object
            states (List[int]): Electronic states to include in simulation
        """
        # Validate input parameters
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("At least two electronic states must be specified")

        if any(not isinstance(s, int) or s < 0 for s in states):
            raise ValueError("All state indices must be non-negative integers")

        self.mol = mol
        self.verbose = 5

        # Set up electronic state configuration
        self.states = list(states)
        self.cur_state = states[0]  # Start from the first specified state

        # Calculate nuclear masses and convert to atomic units
        self.mass = mol.atom_mass_list(True) * AMU2AU # (Na,1)  Unit: a.u.

        # Set default simulation parameters
        self.dt = 0.5 * FS2AUTIME  # Default: 0.5 fs in atomic units
        self.nsteps = 1
        self.filename = 'trajectory.h5'
        self.callback = None

        self.position = None
        self.velocity = None
        self.coefficient = None

        # Don't modify the following attributes. They are used to restart a calculation
        self._step_skip = 0

    @property
    def time_step(self):
        '''time step length in fs'''
        return self.dt / FS2AUTIME
    @time_step.setter
    def time_step(self, x):
        self.dt = x * FS2AUTIME

    def kTDC(self,
             energy_t: np.ndarray,
             energy_p: np.ndarray,
             energy_pp: np.ndarray) -> np.ndarray:
        """
        Calculate κTDC (kappa Time-Derivative Coupling) matrix elements.

        The κTDC method provides an efficient approximation to nonadiabatic coupling
        vectors by using finite differences of potential energy surfaces. This approach
        avoids the computational cost of explicit derivative coupling calculations.

        The coupling is calculated as:
        κ_ij = sqrt(max(0, d²(E_j - E_i)/dt² / (E_j - E_i)))

        where the second derivative is approximated using three-point finite differences:
        d²(ΔE)/dt² ≈ (ΔE_t - 2*ΔE_p + ΔE_pp) / dt²

        Args:
            energy_t (np.ndarray): Time-dependent energy array (Nstates,)
            energy_p (np.ndarray): Time-dependent energy array (Nstates,)
            energy_pp (np.ndarray): Time-dependent energy array (Nstates,)

        Returns:
            np.ndarray: Nonadiabatic coupling matrix (Nstates, Nstates)

        Note:
            The resulting matrix is antisymmetric: κ_ij = -κ_ji
            Diagonal elements are zero by construction
        """
        # Initialize coupling matrix
        states = self.states
        nstates = len(states)
        nact = np.zeros((nstates, nstates))

        # Indices for nonadiabatic coupling calculations. By default,
        # all pairs (i,j) where i < j within self.states are evaluated.
        nac_idx = [(i,j) for i in range(nstates-1) for j in range(i+1, nstates)]

        # Calculate coupling for each unique state pair
        for idx in nac_idx:
            i, j = idx

            # Energy differences at three time points
            dVt = energy_t[j] - energy_t[i]    # Current time
            dVp = energy_p[j] - energy_p[i]    # Previous time
            dVpp = energy_pp[j] - energy_pp[i] # Two steps ago

            # Second derivative using three-point finite difference
            d2Vdt2 = (dVt - 2 * dVp + dVpp) / (self.dt**2)
            # Calculate κTDC matrix element
            sqrt_part = d2Vdt2 / dVt
            if sqrt_part > 0:
                kappa_value = np.sqrt(sqrt_part)
            else:
                kappa_value = 0.0

            nact[j, i] = kappa_value
            nact[i, j] = -kappa_value

        return nact

    def compute_electronic(self, position: np.ndarray, with_nacv=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate electronic energies, nuclear forces and nonadiabatic coupling for all states.

        This method computes the potential energies and nonadiabatic coupling for all electronic
        states at the given nuclear configuration. The forces are obtained as the
        negative gradient of the potential energy surface at the current state.

        Args:
            position (np.ndarray): Nuclear coordinates in Bohr (Natoms * 3)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - energy: Electronic energies for all states (Nstates,) in Hartree
                - force: Nuclear forces for current state (Natoms * 3) in Ha/Bohr
                - Nacv: Nonadiabatic coupling vectors for all states (Nstates, Nstates, Natoms, 3) in 1/bohr
        """
        raise NotImplementedError

    def exp_propagator(self, c: np.ndarray, Veff: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate quantum coefficients using matrix exponential.
        dc/dt = -i V_eff(R,P) c

        The matrix exponential is computed efficiently using eigenvalue decomposition:
        exp(-i * V_eff * dt) = U * diag(exp(-i * λ_k * dt)) * U†

        Args:
            c (np.ndarray): Current quantum coefficients (Nstates,)
            Veff (np.ndarray): Effective Hamiltonian matrix (Nstates * Nstates)
            dt (float): Time step in atomic units

        Returns:
            np.ndarray: Updated quantum coefficients (Nstates,)

        Note:
            The effective Hamiltonian includes both diagonal energies and
            off-diagonal nonadiabatic coupling terms.
        """
        # Diagonalize the effective Hamiltonian
        diags, coeff = np.linalg.eigh(Veff)

        # Compute the matrix exponential
        U = coeff @ np.diag(np.exp(-1j * diags * dt)) @ coeff.T.conj()

        # Apply propagator to coefficients
        c_new = np.dot(U, c)

        # Normalize coefficients
        c_new = c_new / np.linalg.norm(c_new)

        return c_new

    def update_coefficient(self, coeffs: np.ndarray,
                           energy: np.ndarray,
                           nact: np.ndarray) -> np.ndarray:
        """
        Update quantum coefficients using the effective Hamiltonian.

        The effective Hamiltonian in the FSSH method combines:
        1. Diagonal electronic energies: E_ii(R)
        2. Off-diagonal nonadiabatic coupling: -i * κ_ij

        V_eff = E(R) - i * d(R) * P/m = diag(E) - i * κTDC

        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            energy (np.ndarray): Electronic energies (Nstates,)
            nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)

        Returns:
            np.ndarray: Updated quantum coefficients (Nstates,)
        """
        # Construct effective Hamiltonian
        Veff = np.diag(energy) - 1j * nact

        # Propagate coefficients
        c_new = self.exp_propagator(coeffs, Veff, self.dt)

        return c_new

    def compute_hopping_probability(self,
                                    coeffs: np.ndarray,
                                    nact: np.ndarray) -> np.ndarray:
        """
        Calculate surface hopping probabilities using Tully's formula.

        The hopping probability from the current state i to state j is:
        g_ij = (2 * Re(κ_ij * c_i* * c_j) - 2 / ħ * Im(V_ij * c_i* * c_j)) * dt / |c_i|²
        p_ij = max(0, g_ij)
        p_ij = min(1, p_ij)

        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)

        Returns:
            np.ndarray: Hopping probabilities from current state (Nstates,)
        """

        # Get index of current state in the states list
        state_idx = self.states.index(self.cur_state)

        # Current state coefficient
        c_i = coeffs[state_idx]

        # Calculate hopping probabilities
        g_ij = 2 * (nact[state_idx] * c_i.conj() * coeffs).real * self.dt / (np.abs(c_i)**2)

        # Adjust hopping probabilities
        p_ij = np.where(g_ij < 0, 0, g_ij)
        p_ij = np.where(p_ij > 1, 1, p_ij)

        return p_ij

    def check_hop(self, r: float, p_ij: np.ndarray) -> int:
        """
        Determine if a surface hop occurs.

        The hopping decision is made by comparing a random number r ∈ [0,1)
        with cumulative probabilities. A hop to state k occurs if:
        Σ_{j=0}^{k-1} p_j < r ≤ Σ_{j=0}^{k} p_j

        Args:
            r (float): Random number between 0 and 1
            p_ij (np.ndarray): Hopping probabilities (Nstates,)

        Returns:
            int: Index of target state (-1 if no hop occurs)

        Note:
            Returns -1 if no hop occurs (r falls in the "stay" probability region)
        """
        # Calculate cumulative probabilities
        cumu_p_ij = np.cumsum(p_ij)

        # Check each state for hopping condition
        for k, u_bound in enumerate(cumu_p_ij):
            l_bound = 0.0 if k == 0 else cumu_p_ij[k-1]

            if l_bound < r <= u_bound:
                return k

        return -1

    def rescale_velocity(self,
                         hop_index: int,
                         energy: np.ndarray,
                         velocity: np.ndarray,
                         d_vec: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Rescale nuclear velocities to conserve total energy after surface hopping.

        When a surface hop occurs, the nuclear kinetic energy must be adjusted to
        compensate for the change in electronic energy. This is achieved by solving
        the energy conservation equation:

        1/2m(v')² + E_new = 1/2mv² + E_old

        The new velocity is:
        v' = v - gamma * d_vec / mass

        if delta > 0:
            gamma = (b +- sqrt(b^2 - 4ac)) / 2a
            a = sum_i (d_i^2 / 2m_i)
            b = sum_i (v_i * d_i)
            c = E_new - E_old
        else:
            gamma = b / a

        Args:
            hop_index (int): Index of target state in states list
            energy (np.ndarray): Electronic energies for all states
            velocity (np.ndarray): Current nuclear velocities (Natoms × 3)
            d_vec (np.ndarray): Difference vector for velocity adjustment

        Returns:
            Tuple[bool, np.ndarray]:
                - hop_allowed: Whether the hop is energetically allowed
                - velocity: Updated nuclear velocities
        """

        # To conserve energy, the new velocity v' = v - gamma * d_vec / mass must satisfy
        # the energy conservation equation, which leads to a quadratic equation for the
        # scaling factor gamma:
        #     a*gamma^2 - b*gamma + c = 0
        # where:
        #     a = sum_i (d_i^2 / 2m_i)
        #     b = sum_i (v_i * d_i)
        #     c = E_new - E_old

        # Get index of current state in the states list
        state_idx = self.states.index(self.cur_state)

        # Coefficients for the quadratic equation
        a = np.sum(d_vec**2 / (2 * self.mass[:,None]))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[state_idx]

        # Discriminant of the quadratic equation
        delta = b**2 - 4 * a * c

        if delta >= 0:
            gamma = (b + np.sqrt(delta)) / (2 * a) if b < 0 else (b - np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass[:,None]
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass[:,None]
            return False, velocity

    # NOT TESTED YET!!!!
    def compute_decoherence(self,
                    coeffs: np.ndarray,
                    velocity: np.ndarray,
                    energy: np.ndarray) -> np.ndarray:
        """
        Decoherence.

        c_j = c_j * exp(-dt / tau_ji)
        c_i = c_i * sqrt((1 - sum_j(j!=i) |c_j|**2) / |c_i|**2)
        tau_ji = ħ / |E_jj - E_ii| * (1 + a / E_kin)
        """

        E_kin = (0.5 * self.mass[:,None] * np.sum(velocity ** 2)).sum()
        cumu_sum = 0
        cur_idx = self.states.index(self.cur_state)

        for i in range(len(coeffs)):
            if i != cur_idx:
                tau_ji = 1 / np.abs(energy[i] - energy[cur_idx]) * (1 + self.alpha / E_kin)
                coeffs[i] = coeffs[i] * np.exp(-self.dt / tau_ji)
                cumu_sum += np.abs(coeffs[i]) ** 2

        coeffs[cur_idx] = np.sqrt((1 - cumu_sum) / np.abs(coeffs[cur_idx]) ** 2) * coeffs[cur_idx]
        return coeffs

    def write_trajectory(self,
                         step: int,
                         position: np.ndarray,
                         velocity: np.ndarray,
                         energy: np.ndarray,
                         coeffs: np.ndarray,
                         **kwargs) -> None:
        """
        Write current trajectory frame

        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            energy (np.ndarray): Electronic energies in Hartree
            coeffs (np.ndarray): Quantum coefficients
        """
        mode = 'w' if step == 0 else 'a'

        with h5py.File(self.filename, mode) as f:
            if step == 0:
                f['configuration'] = json.dumps({
                    'elements': self.mol.elements,
                    'dt': self.dt,
                    'decoherence': self.decoherence,
                    'alpha': self.alpha,
                    'seed': self.seed,
                    'states': self.states,
                })
                f['velocity'] = velocity
            else:
                f['velocity'][:] = velocity

            h5subset = f.create_group(str(step))
            h5subset['position'] = position
            h5subset['energy'] = energy
            h5subset['coeffs'] = coeffs
            h5subset['cur_state'] = self.cur_state
            if kwargs:
                for k, v in kwargs:
                    h5subset[k] = v

    def random_uniform(self):
        return np.random.rand()

    def restore(self, trajectory_file):
        '''
        Restore a MD simulation from a trajectory file.

        This operation overwrites the attributes of the current instance with the
        data stored in the trajectory file. After restoration, calling the kernel
        method will resume the calculation from the saved state.

        Parameters:
            trajectory_file : str
                The trajectory file in HDF5 format
        '''
        self.filename = trajectory_file
        with h5py.File(trajectory_file, 'r') as f:
            configuration = json.loads(f['configuration'][()])
            self.elements = configuration['elements']
            self.dt = configuration['dt']
            self.seed = configuration['seed']
            self.states = configuration['states']
            self.decoherence = configuration['decoherence']
            self.alpha = configuration['alpha']

            # Two additional keys (configuration, velocity) along with the steps
            # are stored in the trajectory file.
            step_n = len(f.keys()) - 2
            step_n -= 1 # exclude step_0
            self._step_skip = step_n

            self.velocity = np.asarray(f['velocity']) * (FS2AUTIME * 1e3) / A2BOHR  # Angstrom/fs
            self.position = np.asarray(f[f'{step_n}/position']) / A2BOHR  # Angstrom
            self.coefficient = np.asarray(f[f'{step_n}/coeffs'])
            self.cur_state = int(f[f'{step_n}/cur_state'][()])

        if self.seed is not None:
            np.random.seed(self.seed)
            # Skip the first step_n random numbers, to ensure the "restart"
            # calculation reproduce the run from beginning.
            np.random.rand(step_n)

        return self

    def check_sanity(self):
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.nsteps <= 0:
            raise ValueError("Number of steps must be positive")

    def kernel(self,
               position: Optional[np.ndarray] = None,
               velocity: Optional[np.ndarray] = None,
               coefficient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the main FSSH trajectory simulation.

        This method implements the complete FSSH algorithm using the velocity Verlet
        integration scheme.

        Integration Frame Ref:
            Nonadiabatic Field on Quantum Phase Space: A Century after Ehrenfest
            Baihua Wu, Xin He, and Jian Liu
            The Journal of Physical Chemistry Letters 2024 15 (2), 644-658
            DOI: 10.1021/acs.jpclett.3c03385

        Args:
            position (Optional[np.ndarray]): Initial nuclear coordinates in Angstrom
                If None, uses equilibrium geometry from TDDFT object
            velocity (Optional[np.ndarray]): Initial nuclear velocities in Angstrom/fs
                Must be provided for dynamics simulation
            coefficient (Optional[np.ndarray]): Initial quantum coefficients
                If None, starts in the first specified state

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Final nuclear positions in Angstrom
                - Final nuclear velocities in Angstrom/fs
                - Final quantum coefficients
        """
        self.check_sanity()

        log = logger.new_logger(self.mol, self.verbose)
        start_timing = log.init_timer()

        log.info("Starting FSSH trajectory simulation")
        Nstates = len(self.states)
        log.info(f"FSSH simulation initialized with {Nstates} states, "
                 f"dt={self.dt/FS2AUTIME:.3f} fs, {self.nsteps} steps")

        if self.seed is not None:
            np.random.seed(self.seed)

        if position is None:
            position = self.position
        if position is None:
            position = self.mol.atom_coords(unit='Bohr')
        else:
            position = position * A2BOHR   # (Na,D) a.u.

        if velocity is None:
            velocity = self.velocity
            assert velocity is not None
        velocity = velocity * A2BOHR / (FS2AUTIME * 1e3) # (Na,D) Bohr/a.u.Time

        if coefficient is None:
            coefficient = self.coefficient
            assert coefficient is not None
        assert len(coefficient) == len(self.states)
        norm = np.linalg.norm(coefficient)
        coefficient /= norm

        # Calculate initial electronic structure
        energy, force = self.compute_electronic(position, with_nacv=False)

        if self._step_skip == 0:
            if self.tdc_method == 'curvature':
                # Initialize energy history for κTDC calculation
                energy_list = deque(maxlen=3)
                energy_list.append(energy)

                log.info("Performing initial steps to establish energy history")
                # pure velocity verlet to get energy_p and energy_pp
                velocity = velocity + 0.5 * self.dt * force / self.mass[:,None]
                position = position + self.dt * velocity
                energy, force = self.compute_electronic(position, with_nacv=False)
                energy_list.append(energy)
                velocity = velocity + 0.5 * self.dt * force / self.mass[:,None]

            # Write initial trajectory frame
            self.write_trajectory(0, position, velocity, energy, coefficient)
            log.info(f"Starting main simulation loop for {self.nsteps} steps")
        else:
            if self.tdc_method == 'curvature':
                energy_list = deque(maxlen=3)
                prev_step = self._step_skip - 1
                with h5py.File(self.filename, 'r') as f:
                    energy_list.append(np.asarray(f[f'{prev_step}/energy']))
                energy_list.append(energy)

            # For a "restart" calculation, skip the trajectory initialization
            assert h5py.is_hdf5(self.filename)
            log.info(f'Skipping {self._step_skip} steps and resuming the simulation.')

        step_start = self._step_skip + 1

        total_time = step_start * self.dt / FS2AUTIME

        iter_timing = start_timing

        for step in range(step_start, self.nsteps+1):
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass[:,None]

            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity

            # 3. calculte new energy, force, and nacv
            if self.tdc_method == 'nac':
                energy, force, nacv = self.compute_electronic(position, with_nacv=True)
                nact = np.einsum('ijnd,nd->ij', nacv, velocity)
            elif self.tdc_method == 'curvature':
                energy, force = self.compute_electronic(position, with_nacv=False)
                energy_list.append(energy)
                nact = self.kTDC(energy_list[2], energy_list[1], energy_list[0])
            elif self.tdc_method == 'overlap':
                raise NotImplementedError
            else:
                raise RuntimeError(f'TDC method {self.tdc_method} not supported')

            # 4. update the electronic amplitude within a full-time step
            coefficient = self.update_coefficient(coefficient, energy, nact)

            # 5. evaluate the switching probability
            p_ij = self.compute_hopping_probability(coefficient, nact)
            r = self.random_uniform()
            hop_index = self.check_hop(r, p_ij)
            log.debug(f"Switching probability: {p_ij}, Random number: {r}")

            # 6. adjust nuclear velocity
            cur_idx = self.states.index(self.cur_state)
            if hop_index != -1 and hop_index != cur_idx:

                # Attempt velocity rescaling
                d_vec = nacv[cur_idx, hop_index]
                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)

                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]

                    log.info(f"Hop: {old_state} → {self.cur_state} at step {step}")

                else:
                    ke = (0.5 * self.mass[:,None] * velocity ** 2).sum()
                    log.debug(f"Hop to state {self.states[hop_index]} rejected "
                              f"due to insufficient kinetic energy, "
                              f"current kinetic energy: {ke:.8f} Ha"
                              f"energy difference: {energy[cur_idx] - energy[hop_index]:.8f} Ha")

            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass[:,None]

            # 8. update total time
            total_time += self.dt / FS2AUTIME

            # 9. decoherence
            if self.decoherence:
                # TODO: disable decoherence near the avoid-crossing region, and
                # perform decoherence when the trajectory moves to the
                # well-separated surface region
                coefficient = self.compute_decoherence(coefficient, velocity, energy)

            kwargs = {}
            if self.save_force:
                kwargs['force'] = force
                if self.tdc_method == 'nac':
                    kwargs['nacv'] = nacv
            self.write_trajectory(step, position, velocity, energy, coefficient, **kwargs)

            current_idx = self.states.index(self.cur_state)
            current_energy = energy[current_idx]
            populations = np.abs(coefficient)**2

            if callable(self.callback):
                self.callback(locals())

            # Format output
            log.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, "
                     f"Energy {current_energy:12.8f} Ha, Populations: {populations}")

            iter_timing = log.timer(f'FSSH step {step}', *iter_timing)

        # Simulation completed successfully
        log.timer("FSSH simulation", *start_timing)
        log.info("FSSH simulation completed successfully")

        # Convert results back to user units
        final_position = position / A2BOHR  # Angstrom
        final_velocity = velocity * (FS2AUTIME * 1e3) / A2BOHR  # Angstrom/fs

        self.position = final_position
        self.velocity = velocity
        self.coefficient = coefficient
        return final_position, final_velocity, coefficient

def h5_to_xyz(h5file, trajectory_file):
    with h5py.File(h5file, 'r') as h5f, open(trajectory_file, 'w') as f:
        configuration = json.loads(h5f['configuration'][()])
        elements = configuration['elements']
        natm = len(elements)
        dt = configuration['dt']
        states = configuration['states']

        nsteps = len(h5f.keys()) - 2
        print(f'Converting {nsteps} steps of trajectory data to xyz')
        for step in range(nsteps):
            # Write number of atoms
            f.write(f'{natm}\n')

            # Write comment line with simulation data
            time_fs = step * dt / FS2AUTIME
            energy = np.asarray(h5f[f'{step}/energy'])
            coeffs = np.asarray(h5f[f'{step}/coeffs'])
            cur_state = int(h5f[f'{step}/cur_state'][()])
            current_energy = energy[states.index(cur_state)]

            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')

            position = np.asarray(h5f[f'{step}/position']) / A2BOHR  # Convert to Angstrom
            # Write atomic coordinates
            for i, (x, y, z) in enumerate(position):
                symbol = elements[i]
                f.write(f'{symbol:4s} {x:12.6f} {y:12.6f} {z:12.6f}\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert an HDF5 trajectory file to XYZ format.'
    )
    parser.add_argument(
        'h5file',
        type=str,
        help='Input HDF5 file containing the trajectory data.'
    )
    parser.add_argument(
        'trajectory_file',
        type=str,
        help='Output XYZ trajectory file.'
    )

    args = parser.parse_args()

    h5_to_xyz(args.h5file, args.trajectory_file)
