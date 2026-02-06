# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import numpy as np
import time
import sys

import logging
from typing import Tuple, Optional, List
from pathlib import Path

# Physical constants for unit conversions
FS2AUTIME = 41.34137        # Conversion factor: femtoseconds to atomic time units
A2BOHR = 1.889726           # Conversion factor: Angstrom to Bohr radius
AMU2AU = 1822.8884858012984 # Conversion factor: atomic mass units to atomic units

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    """
    
    def __init__(self, 
                 tddft, 
                 states:list[int],
                 decoherence:bool=True,
                 alpha:float=0.1,
                 **kwargs):
        """
        Initialize the FSSH simulation with comprehensive parameter validation.
        
        Args:
            tddft: Time-dependent DFT object providing electronic structure
            states (List[int]): Electronic states to include in simulation
            **kwargs: Additional simulation parameters including:
                - dt (float): Time step in femtoseconds (default: 0.5)
                - nsteps (int): Number of simulation steps (default: 1)
                - output_dir (str): Directory for output files (default: current)
                - verbose (bool): Enable verbose output (default: True)
        """
        # Validate input parameters
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("At least two electronic states must be specified")
        
        if any(not isinstance(s, int) or s < 0 for s in states):
            raise ValueError("All state indices must be non-negative integers")
        
        # Initialize core simulation objects
        self.tddft = tddft
        self.tdgrad = self.tddft.nuc_grad_method().as_scanner()
        self.tdnac = self.tddft.nac_method().as_scanner()
        
        # Set up electronic state configuration
        self.states = list(states)
        self.Nstates = len(states)
        self.cur_state = states[0]  # Start from the first specified state
        
        # Calculate nuclear masses and convert to atomic units
        self.mass = self.tddft.mol.atom_mass_list(True).reshape(-1, 1) * AMU2AU # (Na,1)  Unit: a.u.
        
        # Generate indices for nonadiabatic coupling calculations
        # Only consider unique pairs (i,j) where i < j to avoid redundancy
        self.nac_idx = [(i, j) for i in range(self.Nstates-1) 
                        for j in range(i+1, self.Nstates)]
        
        self.decoh = decoherence
        self.alpha = alpha
                     
        # Set default simulation parameters
        self.dt = 0.5 * FS2AUTIME  # Default: 0.5 fs in atomic units
        self.nsteps = 1
        self.output_dir = Path('.')
        self.filename = 'trajectory.xyz'
        
        # Override defaults with user-provided parameters
        for key, value in kwargs.items():
            if key == 'dt' and isinstance(value, (int, float)):
                if value <= 0:
                    raise ValueError("Time step must be positive")
                self.dt = value * FS2AUTIME
            elif key == 'nsteps' and isinstance(value, int):
                if value <= 0:
                    raise ValueError("Number of steps must be positive")
                self.nsteps = value
            elif key == 'output_dir':
                self.output_dir = Path(value)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            elif key == 'filename':
                self.filename = value
                if not self.filename.endswith('.xyz'):
                    self.filename = self.filename + '.xyz'
            else:
                setattr(self, key, value)
        
        logger.info(f"FSSH simulation initialized with {self.Nstates} states, "
                   f"dt={self.dt/FS2AUTIME:.3f} fs, {self.nsteps} steps")
    
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
        nact = np.zeros((self.Nstates, self.Nstates))

        # Calculate coupling for each unique state pair
        for idx in self.nac_idx:
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
    
    def calc_electronic(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        # Set the current state for gradient calculation
        self.tdgrad.state = self.cur_state

        # Calculate energy and gradient for the current state
        _, grad = self.tdgrad(position,self.cur_state)
            
        ground_energy = self.tdgrad.base._scf.e_tot
        excited_energies = self.tdgrad.e_tot
        if hasattr(excited_energies, 'get'):
            excited_energies = excited_energies.get()
        energy = np.concatenate([[ground_energy], excited_energies])[self.states]  # (Nstates,)  Unit: Ha
        force = -grad  # (Na,D)  Unit: Ha/bohr
            
        # calculate nacv
        Nacv = np.zeros((self.Nstates, self.Nstates, position.shape[0], position.shape[1]))  # (Ns, Ns, Na, D)  Unit: 1/bohr
        for state_index in self.nac_idx:
            state = tuple(self.states[i] for i in state_index)
            self.tdnac.states = state
            nk = self.tdnac(position, state)[4]
            Nacv[state_index[0],state_index[1]] = nk
            Nacv[state_index[1],state_index[0]] = -nk
            
        return energy, force, Nacv
            
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
        a = np.sum(d_vec**2 / (2 * self.mass))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[state_idx]

        # Discriminant of the quadratic equation
        delta = b**2 - 4 * a * c

        if delta >= 0:
            gamma = (b + np.sqrt(delta)) / (2 * a) if b < 0 else (b - np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass
            return False, velocity
    
    # NOT TESTED YET!!!!
    def decoherence(self,
                    coeffs: np.ndarray,
                    velocity: np.ndarray,
                    energy: np.ndarray) -> np.ndarray:
        """
        Decoherence.

        c_j = c_j * exp(-dt / tau_ji)
        c_i = c_i * sqrt((1 - sum_j(j!=i) |c_j|**2) / |c_i|**2)
        tau_ji = ħ / |E_jj - E_ii| * (1 + a / E_kin)
        """

        E_kin = (0.5 * self.mass * np.sum(velocity ** 2)).sum()
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
                         ) -> None:
        """
        Write current trajectory frame to XYZ file with comprehensive metadata.
        
        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            energy (np.ndarray): Electronic energies in Hartree
            coeffs (np.ndarray): Quantum coefficients
            filename (str): Output filename
        """
        filepath = self.output_dir / self.filename
        mode = 'w' if step == 0 else 'a'
        
        with open(filepath, mode) as f:
            # Write number of atoms
            f.write(f'{self.tddft.mol.natm}\n')
            
            # Write comment line with simulation data
            time_fs = step * self.dt / FS2AUTIME
            current_energy = energy[self.states.index(self.cur_state)]
            
            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {self.cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')
            
            # Write atomic coordinates
            for i, coord in enumerate(position):
                symbol = self.tddft.mol.atom_pure_symbol(i)
                x, y, z = coord / A2BOHR  # Convert to Angstrom
                f.write(f'{symbol:4s} {x:12.6f} {y:12.6f} {z:12.6f}\n')
    
    def write_restart(self,
                         step: int,
                         position: np.ndarray,
                         velocity: np.ndarray,
                         energy: np.ndarray,
                         coeffs: np.ndarray,
                         ) -> None:
        """
        Write current simulation state to a restart file.

        The restart file stores coordinates, velocities, and quantum coefficients.
        The file is overwritten at each step to save space.

        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            energy (np.ndarray): Electronic energies in Hartree
            coeffs (np.ndarray): Quantum coefficients
        """
        filepath = self.output_dir / self.filename.replace('.xyz', '.rst')
        # Always overwrite the restart file
        with open(filepath, 'w') as f:
            # Write number of atoms
            f.write(f'{self.tddft.mol.natm}\n')

            # Write comment line with simulation data
            time_fs = step * self.dt / FS2AUTIME
            current_energy = energy[self.states.index(self.cur_state)]

            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {self.cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')

            # Write atomic coordinates and velocities
            for i, (coord, vel) in enumerate(zip(position, velocity)):
                symbol = self.tddft.mol.atom_pure_symbol(i)
                x, y, z = coord / A2BOHR  # Convert to Angstrom
                vx, vy, vz = vel # atomic units
                f.write(f'{symbol:4s} {x:12.6f} {y:12.6f} {z:12.6f} {vx:12.6f} {vy:12.6f} {vz:12.6f}\n')

    def print_step_info(self, 
                        step: int, 
                        total_time: float, 
                        energy: np.ndarray,
                        coeffs: np.ndarray, 
                        ) -> None:
        """
        Print detailed information about the current simulation step.
        
        Args:
            step (int): Current step number
            total_time (float): Total simulation time in fs
            energy (np.ndarray): Electronic energies
            coeffs (np.ndarray): Quantum coefficients
            nact (np.ndarray): κTDC coupling matrix
            hop_occurred (bool): Whether a hop occurred in this step
        """
        
        current_idx = self.states.index(self.cur_state)
        current_energy = energy[current_idx]
        populations = np.abs(coeffs)**2
            
        # Format output
        logger.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, "
              f"Energy {current_energy:12.8f} Ha, Populations: {populations}")

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

        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"Starting FSSH trajectory simulation at {now_str}")
        start_time = time.time()
        
        # Initialize or validate input parameters
        if position is None:
            position = self.tddft.mol.atom_coords()
        else:
            position = position * A2BOHR   # (Na,D) a.u.

        velocity = velocity * A2BOHR / (FS2AUTIME * 1e3) # (Na,D) Bohr/a.u.Time

        norm = np.linalg.norm(coefficient)
        coefficient /= norm
        
        # Calculate initial electronic structure
        energy, force, nacv = self.calc_electronic(position)
        
        # Write initial trajectory frame
        self.write_trajectory(0, position, velocity, energy, coefficient)
        self.write_restart(0, position, velocity, energy, coefficient)
        
        total_time = 0.0
        
        # Main simulation loop
        logger.info(f"Starting main simulation loop for {self.nsteps} steps")
        
        for i in range(self.nsteps):
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity
            
#QS: API/hook to support calculating NACV using other methods
#QS: consider the kTDC in fssh_new.py
#QS: Does it make sense to pass TDDFT initial guess?
            # 3. calculte new energy, force, and nacv
            energy, force, nacv = self.calc_electronic(position)
            
            # 4. update the electronic amplitude within a full-time step
            nact = np.einsum('ijnd,nd->ij', nacv, velocity)
            coefficient = self.update_coefficient(coefficient, energy, nact)
            
#QS: API/hook here for hopping probability (see Chaoyuan Zhu)
            # 5. evaluate the switching probability
            p_ij = self.compute_hopping_probability(coefficient, nact,
                                                    additional_kwargs)
            r = np.random.rand()
            hop_index = self.check_hop(r, p_ij)

            logger.debug(f"Switching probability: {p_ij}, Random number: {r}")
            
            # 6. adjust nuclear velocity
            cur_idx = self.states.index(self.cur_state)
            if hop_index != -1 and hop_index != cur_idx:
          
                # Attempt velocity rescaling
                d_vec = nacv[cur_idx, hop_index]
                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)
                
                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]
                    
                    logger.info(f"Hop: {old_state} → {self.cur_state} at step {i + 1}")

                else:
                    ke = (0.5 * self.mass * velocity ** 2).sum()
                    logger.debug(f"Hop to state {self.states[hop_index]} rejected "
                                 f"due to insufficient kinetic energy, "
                                 f"current kinetic energy: {ke:.8f} Ha"
                                 f"energy difference: {energy[cur_idx] - energy[hop_index]:.8f} Ha")
                    
            
            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 8. update total time
            total_time += self.dt / FS2AUTIME

#QS: API/hook? Is it necessary/reasonable to disable decoherence near the avoid-crossing region,
# and perform decoherence only when the trajectory moves to the well-separated surface region
            # 9. decoherence
            if self.decoh:
                coefficient = self.decoherence(coefficient, velocity, energy,
                                               additional_kwarg)
            
#QS: Option to store meta data: geometry, velocity, nacv, ...
#QS: store in HDF5 format?
            self.write_trajectory(i + 1, position, velocity, energy, coefficient)   
            self.write_restart(i + 1, position, velocity, energy, coefficient)

            self.print_step_info(i + 1, total_time, energy, coefficient)
#QS: print wall time
        
        # Simulation completed successfully
        elapsed_time = time.time() - start_time
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"FSSH simulation completed successfully at {now_str}")
        logger.info(f"Total simulation time: {elapsed_time:.2f} s")
            
        # Convert results back to user units
        final_position = position / A2BOHR  # Angstrom
        final_velocity = velocity * (FS2AUTIME * 1e3) / A2BOHR  # Angstrom/fs
        
        return final_position, final_velocity, coefficient
