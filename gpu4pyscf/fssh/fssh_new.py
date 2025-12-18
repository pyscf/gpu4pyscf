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
with support for both direct nonadiabatic coupling vectors (NACV) and
κTDC (kappa Time-Derivative Coupling) methods for nonadiabatic molecular dynamics simulations.

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
       
    4. Nonadiabatic Dynamics Algorithms with Only Potential Energies and Gradients: 
       Curvature-Driven Coherent Switching with Decay of Mixing and 
       Curvature-Driven Trajectory Surface Hopping
       Yinan Shu, Linyao Zhang, Xiye Chen, Shaozeng Sun, Yudong Huang, Donald G. Truhlar
       J. Chem. Theory Comput. 2022, 18, 3, 1320-1328
       DOI: 10.1021/acs.jctc.1c01080
"""

import numpy as np
import time
import sys

import logging
from typing import Tuple, Optional, List
from pathlib import Path
from gpu4pyscf.lib import logger

# Physical constants for unit conversions
FS2AUTIME = 41.34137        # Conversion factor: femtoseconds to atomic time units
A2BOHR = 1.889726           # Conversion factor: Angstrom to Bohr radius
AMU2AU = 1822.8884858012984 # Conversion factor: atomic mass units to atomic units

# Configure logging for debugging and monitoring
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


class FSSH:
    """
    This class implements the FSSH algorithm for nonadiabatic molecular dynamics simulations.
    It supports both direct nonadiabatic coupling vectors (NACV) and κTDC methods.

    The FSSH method treats nuclear motion classically while quantum mechanically
    describing electronic transitions between different potential energy surfaces.
    
    Attributes:
        tddft: Time-dependent density functional theory object
        tdgrad: Nuclear gradient scanner for force calculations
        tdnac: Nonadiabatic coupling vector scanner (if using direct method)
        states (List[int]): List of electronic states to include in simulation
        cur_state (int): Current active electronic state
        mass (np.ndarray): Nuclear masses in atomic units
        dt (float): Time step in atomic units
        nsteps (int): Number of simulation steps
        coupling_method (str): Method for calculating nonadiabatic coupling ('direct' or 'ktdc')
    """
    
    def __init__(self, 
                 tddft, 
                 states:list[int],
                 dt:float = 0.5,
                 nsteps:int = 1,
                 coupling_method: str = 'direct',
                 decoherence:bool=True,
                 alpha:float=0.1,
                 verbose:int=4,
                 **kwargs):
        """
        Initialize the FSSH simulation with comprehensive parameter validation.
        
        Args:
            tddft: Time-dependent DFT object providing electronic structure
            states (List[int]): Electronic states to include in simulation
            dt (float): Time step in atomic units
            nsteps (int): Number of simulation steps
            coupling_method (str): Method for calculating nonadiabatic coupling ('direct' or 'ktdc')
            decoherence (bool): Enable decoherence correction
            alpha (float): Decoherence parameter
            **kwargs: Additional simulation parameters including:
                - output_dir (str): Directory for output files (default: current)
                - filename (str): Output trajectory filename (default: 'trajectory.xyz')
        """
        # Validate input parameters
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("At least two electronic states must be specified")
        
        if any(not isinstance(s, int) or s < 0 for s in states):
            raise ValueError("All state indices must be non-negative integers")
        
        if coupling_method not in ['direct', 'ktdc']:
            raise ValueError("coupling_method must be either 'direct' or 'ktdc'")
        
        # Initialize core simulation objects
        self.tddft = tddft
        self.verbose = verbose
        self.log = logger.new_logger(self, verbose)
        if self.tddft.mol.unit.lower() != 'Bohr':
            self.tddft.mol.unit = 'Bohr'
            self.tddft.mol.set_geom_(self.tddft.mol.atom_coords())
        self.tdgrad = self.tddft.nuc_grad_method().as_scanner()
        self.coupling_method = coupling_method
        
        # Only initialize tdnac if using direct coupling method
        if coupling_method == 'direct':
            self.tdnac = self.tddft.nac_method().as_scanner()
        
        # Set up electronic state configuration
        self.states = list(states)
        self.Nstates = len(states)
        self.cur_state = states[-1]  # Start from the last specified state
        
        # Calculate nuclear masses and convert to atomic units
        self.mass = self.tddft.mol.atom_mass_list(True).reshape(-1, 1) * AMU2AU # (Na,1)  Unit: a.u.
        
        # Generate indices for nonadiabatic coupling calculations
        # Only consider unique pairs (i,j) where i < j to avoid redundancy
        self.nac_idx = [(i, j) for i in range(self.Nstates-1) 
                        for j in range(i+1, self.Nstates)]
        
        # Decoherence parameters
        self.decoh = decoherence
        self.alpha = alpha
                      
        # Set default simulation parameters
        self.dt = dt * FS2AUTIME  # Default: 0.5 fs in atomic units
        self.nsteps = nsteps
        self.output_dir = Path('.')
        self.filename = 'trajectory.xyz'
        
        # Override defaults with user-provided parameters
        for key, value in kwargs.items():
            if key == 'output_dir':
                self.output_dir = Path(value)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            elif key == 'filename':
                self.filename = value
                if not self.filename.endswith('.xyz'):
                    self.filename = self.filename + '.xyz'
            else:
                setattr(self, key, value)
        
        self.log.info(f"FSSH simulation initialized with states {self.states}, current state: {self.cur_state}\n"
                      f"dt={self.dt/FS2AUTIME:.3f} fs, total steps: {self.nsteps}\n"
                      f"coupling_method={coupling_method}\n"
                      f"decoherence={decoherence}, alpha={alpha}\n"
                      f"Trajectory will be saved to {self.output_dir / self.filename}\n")
    
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
    
    def calc_electronic(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Calculate electronic energies, nuclear forces and nonadiabatic coupling for all states.
        
        This method computes the potential energies and either:
        - For direct method: nonadiabatic coupling vectors for all electronic states
        - For κTDC method: only energies and forces (coupling will be calculated later)
        
        Args:
            position (np.ndarray): Nuclear coordinates in Bohr (Natoms * 3)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
                - energy: Electronic energies for all states (Nstates,) in Hartree
                - force: Nuclear forces for current state (Natoms * 3) in Ha/Bohr
                - Nacv: Nonadiabatic coupling vectors (only for direct method) or None
        """

        # Set the current state for gradient calculation
        self.tdgrad.state = self.cur_state

        # Calculate energy and gradient for the current state
        _, grad = self.tdgrad(position, self.cur_state)
            
        ground_energy = self.tdgrad.base._scf.e_tot
        excited_energies = self.tdgrad.e_tot
        if hasattr(excited_energies, 'get'):
            excited_energies = excited_energies.get()
        energy = np.concatenate([[ground_energy], excited_energies])[self.states]  # (Nstates,)  Unit: Ha
        force = -grad  # (Na,D)  Unit: Ha/bohr
            
        # Calculate nonadiabatic coupling vectors if using direct method
        Nacv = None
        if self.coupling_method == 'direct':
            Nacv = np.zeros((self.Nstates, self.Nstates, position.shape[0], position.shape[1]))  # (Ns, Ns, Na, D)  Unit: 1/bohr
            for state_index in self.nac_idx:
                state = tuple(self.states[i] for i in state_index)
                self.tdnac.states = state
                nk = self.tdnac(position, state)[4]
                Nacv[state_index[0], state_index[1]] = nk
                Nacv[state_index[1], state_index[0]] = -nk
            
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

    def update_coefficient(self,
                           coeffs: np.ndarray, 
                           energy: np.ndarray, 
                           nact: np.ndarray) -> np.ndarray:
        """
        Update quantum coefficients using the effective Hamiltonian.
        
        The effective Hamiltonian in the FSSH method combines:
        1. Diagonal electronic energies: E_ii(R)
        2. Off-diagonal nonadiabatic coupling: -i * κ_ij or -i * NACV·P/m
        
        V_eff = E(R) - i * d(R) * P/m
        
        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            energy (np.ndarray): Electronic energies (Nstates,)
            nact (np.ndarray): Nonadiabatic coupling matrix (Nstates * Nstates)
        
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
        g_ij = (2 * Re(κ_ij * c_i* * c_j)) * dt / |c_i|²
        p_ij = max(0, g_ij)
        p_ij = min(1, p_ij)

        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            nact (np.ndarray): Nonadiabatic coupling matrix (Nstates * Nstates)
        
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
    
    def decoherence(self,
                    coeffs: np.ndarray,
                    velocity: np.ndarray,
                    energy: np.ndarray) -> np.ndarray:
        """
        Apply decoherence correction to quantum coefficients.

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
                         coeffs: np.ndarray) -> None:
        """
        Write current trajectory frame to XYZ file with comprehensive metadata.
        
        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            energy (np.ndarray): Electronic energies in Hartree
            coeffs (np.ndarray): Quantum coefficients
        """
            
        filepath = self.output_dir / self.filename
        mode = 'w' if step == 0 else 'a'
        
        with open(filepath, mode) as f:
            # Write number of atoms
            f.write(f'{self.tddft.mol.natm}\n')
            
            # Write comment line with simulation data
            time_fs = step * self.dt / FS2AUTIME
            
            comment = f'Step {step}, Time {time_fs:.3f} fs, State {self.cur_state}, Coefficient {coeffs}'
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
                      coeffs: np.ndarray) -> None:
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

            comment = f'Step {step}, Time {time_fs:.3f} fs, '
            comment += f'State {self.cur_state}, Energy {energy} Ha, '
            comment += f'Coefficient {coeffs}'
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
                        coeffs: np.ndarray) -> None:
        """
        Print detailed information about the current simulation step.
        
        Args:
            step (int): Current step number
            total_time (float): Total simulation time in fs
            energy (np.ndarray): Electronic energies
            coeffs (np.ndarray): Quantum coefficients
        """
        
        populations = np.abs(coeffs)**2
            
        # Format output
        self.log.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, "
                      f"Energy {energy} Ha, Populations: {populations}")

    def kernel(self, 
               position: Optional[np.ndarray] = None, 
               velocity: Optional[np.ndarray] = None, 
               coefficient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the main FSSH trajectory simulation.
        
        This method implements the complete FSSH algorithm using the velocity Verlet
        integration scheme with support for both direct NACV and κTDC coupling methods.

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
        self.log.info(f"Starting FSSH trajectory simulation at {now_str}")
        start_time = time.time()
        
        # Initialize or validate input parameters
        if position is None:
            position = self.tddft.mol.atom_coords()
        else:
            position = position * A2BOHR   # (Na,D) a.u.

        if velocity is None:
            raise ValueError("Velocity must be provided for dynamics simulation")
        velocity = velocity * A2BOHR / (FS2AUTIME * 1e3) # (Na,D) Bohr/a.u.Time

        if coefficient is None:
            # Start in the first specified state
            coefficient = np.zeros(self.Nstates, dtype=complex)
            coefficient[self.cur_state] = 1.0
        else:
            # Normalize coefficients
            norm = np.linalg.norm(coefficient)
            coefficient /= norm
        
        # For κTDC method, we need to establish energy history
        energy_list = None
        if self.coupling_method == 'ktdc':
            self.log.info("Performing initial steps to establish energy history for κTDC")
            energy_list = [None, None, None]
            
            # Get initial energy
            energy, force, _ = self.calc_electronic(position)
            energy_list[0] = energy.copy()
            
            # Perform initial velocity Verlet steps to populate energy history
            velocity = velocity + 0.5 * self.dt * force / self.mass
            position = position + self.dt * velocity
            energy, force, _ = self.calc_electronic(position)
            energy_list[1] = energy.copy()
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # Write initial trajectory frame
            self.write_trajectory(0, position, velocity, energy, coefficient)
            self.write_restart(0, position, velocity, energy, coefficient)
        else:
            # For direct method, just calculate initial electronic structure
            energy, force, nacv = self.calc_electronic(position)
            
            # Write initial trajectory frame
            self.write_trajectory(0, position, velocity, energy, coefficient)
            self.write_restart(0, position, velocity, energy, coefficient)
        
        total_time = 0.0
        
        # Main simulation loop
        self.log.info(f"Starting main simulation loop for {self.nsteps} steps")
        
        for i in range(self.nsteps):
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity
            
            # 3. calculate new energy, force, and nacv (if using direct method)
            if self.coupling_method == 'direct':
                energy, force, nacv = self.calc_electronic(position)
                # 4. update the electronic amplitude using direct NACV
                nact = np.einsum('ijnd,nd->ij', nacv, velocity)
            else:  # ktdc method
                energy, force, _ = self.calc_electronic(position)
                # 4. update the electronic amplitude using κTDC
                energy_list[(i+2)%3] = energy.copy()
                nact = self.kTDC(energy_list[(i+2)%3], energy_list[(i+1)%3], energy_list[i%3])
            
            coefficient = self.update_coefficient(coefficient, energy, nact)
            
            # 5. evaluate the switching probability
            p_ij = self.compute_hopping_probability(coefficient, nact)
            r = np.random.rand()
            hop_index = self.check_hop(r, p_ij)

            self.log.debug1(f"Switching probability: {p_ij}, Random number: {r}")
            
            hop_occurred = False
            # 6. adjust nuclear velocity
            cur_idx = self.states.index(self.cur_state)
            if hop_index != -1 and hop_index != cur_idx:
                # Calculate d_vec for velocity rescaling
                if self.coupling_method == 'direct':
                    d_vec = nacv[cur_idx, hop_index]
                else:  # ktdc method
                    # For κTDC, we need to calculate the gradient difference
                    self.tdgrad.state = self.states[hop_index]
                    _, dVh = self.tdgrad(position, self.states[hop_index])
                    dVc = -force
                    d_vec = dVh - dVc
                
                # Attempt velocity rescaling
                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)
                
                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]
                    self.log.info(f"Hop: {old_state} → {self.cur_state} at step {i + 1}")

                else:
                    self.log.debug1(f"Hop to state {self.states[hop_index]} rejected ")
            
            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 8. update total time
            total_time += self.dt / FS2AUTIME

            # 9. apply decoherence if enabled
            if self.decoh:
                coefficient = self.decoherence(coefficient, velocity, energy)
            
            # Write trajectory and restart files
            step = i + 1
            self.write_trajectory(step, position, velocity, energy, coefficient)
            self.write_restart(step, position, velocity, energy, coefficient)

            # Print step information
            self.print_step_info(step, total_time, energy, coefficient, hop_occurred)
        
        # Simulation completed successfully
        elapsed_time = time.time() - start_time
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log.info(f"FSSH simulation completed successfully at {now_str}")
        elapsed_time_str = self._format_time(elapsed_time)
        self.log.info(f"Total simulation time: {elapsed_time_str}")
            
        # Convert results back to user units
        final_position = position / A2BOHR  # Angstrom
        final_velocity = velocity * (FS2AUTIME * 1e3) / A2BOHR  # Angstrom/ps
        
        return final_position, final_velocity, coefficient
    
    def _format_time(self, seconds):
        """将秒数转换为天、小时、分钟、秒的格式"""
        days = int(seconds // (24 * 3600))
        seconds %= 24 * 3600
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        seconds_float = seconds
        
        if days > 0:
            return f"{days} days, {hours} hours, {minutes} minutes, {seconds_float:.2f} seconds"
        elif hours > 0:
            return f"{hours} hours, {minutes} minutes, {seconds_float:.2f} seconds"
        elif minutes > 0:
            return f"{minutes} minutes, {seconds_float:.2f} seconds"
        else:
            return f"{seconds_float:.2f} seconds"