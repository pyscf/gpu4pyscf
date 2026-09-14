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
import dataclasses
import json
import numpy as np
import cupy as cp
import h5py
from pyscf.data.nist import AMU2AU, BOHR, HARTREE2J, PLANCK
from pyscf.data.elements import COMMON_ISOTOPE_MASSES, NUC
from gpu4pyscf.lib import logger
import datetime

# Physical constants for unit conversions
FS2AUTIME = 2*np.pi * HARTREE2J / PLANCK * 1e-15  # 41.34137: femtoseconds to atomic time units
H5_FORMAT = 'gpu4pyscf.fssh'
H5_VERSION = 2
SUPPORTED_EXTRA_DUMP = ('force', 'nacv', 'velocity')
H5_REQUIRED_CONFIGURATION = {
    'elements', 'mass', 'dt', 'nsteps', 'decoherence', 'alpha', 'seed',
    'states', 'coupling_method', 'extra_dump',
}


def _validate_fssh_configuration(configuration):
    """Validate fields required to reconstruct an FSSH trajectory."""
    missing = H5_REQUIRED_CONFIGURATION.difference(configuration)
    if missing:
        raise ValueError(f"HDF5 trajectory configuration is missing {sorted(missing)}")
    return configuration


class H5Trajectory:
    """Generic storage operations for one versioned HDF5 trajectory."""

    def __init__(self, filename, mode):
        self.filename = str(filename)
        self._file = h5py.File(self.filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._file.id.valid:
            self._file.close()

    def write_configuration(self, configuration):
        self._file.attrs['format'] = H5_FORMAT
        self._file.attrs['h5_version'] = H5_VERSION
        self._file['configuration'] = json.dumps(configuration)
        self._file.flush()

    def read_configuration(self):
        file_format = self._file.attrs.get('format')
        h5_version = self._file.attrs.get('h5_version')
        if file_format != H5_FORMAT or h5_version != H5_VERSION:
            raise ValueError(
                f"HDF5 trajectory format must be {H5_FORMAT!r} "
                f"schema version {H5_VERSION}")
        if 'configuration' not in self._file:
            raise ValueError("HDF5 trajectory configuration is missing")
        configuration = json.loads(self._file['configuration'][()])
        if not isinstance(configuration, dict):
            raise ValueError("HDF5 trajectory configuration must be a mapping")
        return configuration

    def write_checkpoint(self, datasets):
        for name, value in datasets.items():
            if name in self._file:
                dataset = self._file[name]
                if dataset.shape == np.shape(value):
                    dataset[...] = value
                    continue
                del self._file[name]
            self._file[name] = value
        self._file.flush()

    def read_checkpoint(self):
        return {
            name: item[()]
            for name, item in self._file.items()
            if name != 'configuration' and isinstance(item, h5py.Dataset)
        }

    def write_frame(self, step, datasets):
        frame = self._file.create_group(str(step))
        for name, value in datasets.items():
            frame[name] = value
        self._file.flush()

    def read_frame(self, step):
        frame = self._file[str(step)]
        return {
            name: dataset[()]
            for name, dataset in frame.items()
            if isinstance(dataset, h5py.Dataset)
        }

    def frame_steps(self):
        return sorted(
            int(name) for name, item in self._file.items()
            if name.isdigit() and isinstance(item, h5py.Group)
        )

    def read_last_frame(self):
        steps = self.frame_steps()
        if not steps:
            raise ValueError("HDF5 trajectory contains no frame groups")
        step = steps[-1]
        return step, self.read_frame(step)


def _ktdc_from_curvature(energy, curvature, gap_tol):
    """Build an antisymmetric κTDC matrix from energy curvatures."""
    energy = np.asarray(energy)
    curvature = np.asarray(curvature)
    nstates = len(energy)
    nact = np.zeros((nstates, nstates))

    for i in range(nstates - 1):
        for j in range(i + 1, nstates):
            energy_gap = energy[j] - energy[i]
            if np.abs(energy_gap) < gap_tol:
                energy_gap = np.copysign(
                    gap_tol,
                    energy_gap if energy_gap != 0 else 1.0
                )

            curvature_gap = curvature[j] - curvature[i]
            sqrt_part = curvature_gap / energy_gap
            if sqrt_part > 0:
                kappa_value = 0.5 * np.sqrt(sqrt_part)
                nact[j, i] = kappa_value
                nact[i, j] = -kappa_value
    return nact


def ktdc_energy(energy_t, energy_p, energy_pp, dt, gap_tol=1e-6):
    """Calculate curvature-driven TDCs from three consecutive energies.

    Args:
        energy_t: State energies at the current time, shape ``(nstates,)``.
        energy_p: State energies one time step earlier.
        energy_pp: State energies two time steps earlier.
        dt: Time step in atomic units.
        gap_tol: Energy-gap threshold below which κTDC is set to zero.

    Returns:
        An antisymmetric κTDC matrix with shape ``(nstates, nstates)``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    energy_t = np.asarray(energy_t)
    energy_p = np.asarray(energy_p)
    energy_pp = np.asarray(energy_pp)
    curvature = (energy_t - 2 * energy_p + energy_pp) / dt**2
    return _ktdc_from_curvature(energy_t, curvature, gap_tol)


def ktdc_gradient(energy_t, gradient_t, velocity_t,
                  gradient_p, velocity_p, dt, gap_tol=1e-6):
    """Calculate curvature-driven TDCs from gradients and velocities.

    The time derivative of each state energy is evaluated as
    ``dE/dt = gradient · velocity``. Its backward finite difference gives
    the energy curvature used in the κTDC approximation.

    Args:
        energy_t: State energies at the current time, shape ``(nstates,)``.
        gradient_t: Current energy gradients, shape ``(nstates, natoms, 3)``.
        velocity_t: Current nuclear velocities, shape ``(natoms, 3)``.
        gradient_p: Energy gradients one time step earlier.
        velocity_p: Nuclear velocities one time step earlier.
        dt: Time step in atomic units.
        gap_tol: Energy-gap threshold below which κTDC is set to zero.

    Returns:
        An antisymmetric κTDC matrix with shape ``(nstates, nstates)``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    gradient_t = np.asarray(gradient_t)
    velocity_t = np.asarray(velocity_t)
    gradient_p = np.asarray(gradient_p)
    velocity_p = np.asarray(velocity_p)
    energy_rate_t = np.einsum('ind,nd->i', gradient_t, velocity_t)
    energy_rate_p = np.einsum('ind,nd->i', gradient_p, velocity_p)
    curvature = (energy_rate_t - energy_rate_p) / dt
    return _ktdc_from_curvature(energy_t, curvature, gap_tol)


def exp_propagator(c, H_eff, dt):
    """Propagate electronic coefficients with a Hermitian Hamiltonian.
    dc/dt = -i H_eff(R,P) c
    The matrix exponential is computed efficiently using eigenvalue decomposition:
    exp(-i * H_eff * dt) = U * diag(exp(-i * λ_k * dt)) * U†

    Args:
        c (np.ndarray): Current quantum coefficients (Nstates,)
        Veff (np.ndarray): Effective Hamiltonian matrix (Nstates * Nstates)
        dt (float): Time step in atomic units

    Returns:
        np.ndarray: Updated quantum coefficients (Nstates,)
    """
    c = np.asarray(c)
    H_eff = np.asarray(H_eff)
    eigenval, eigenvec = np.linalg.eigh(H_eff)
    propagator = (eigenvec @ np.diag(np.exp(-1j * eigenval * dt))@ eigenvec.T.conj())
    propagated = propagator @ c
    return propagated / np.linalg.norm(propagated)


def update_coefficient(coeffs, energy, nact, dt):
    """Update quantum coefficients using the effective Hamiltonian.
    H_eff = E(R) - i * d(R) * P/m = diag(E) - i * κTDC
    
    Args:
        coeffs (np.ndarray): Current quantum coefficients (Nstates,)
        energy (np.ndarray): Electronic energies (Nstates,)
        nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)

    Returns:
        np.ndarray: Updated quantum coefficients (Nstates,)
    """
    H_eff = np.diag(np.asarray(energy)) - 1j * np.asarray(nact)
    return exp_propagator(coeffs, H_eff, dt)


def compute_hopping_probability(coeffs, nact, active_index, dt):
    """Return Tully hopping probabilities from one active-state index.
    The hopping probability from the current state i to state j is:
    g_ij = (2 * Re(κ_ij * c_i* * c_j) - 2 / ħ * Im(V_ij * c_i* * c_j)) * dt / |c_i|²
    p_ij = max(0, g_ij)
    p_ij = min(1, p_ij)
    """
    coeffs = np.asarray(coeffs)
    nact = np.asarray(nact)
    active_coeff = coeffs[active_index]
    probabilities = (
        2 * (nact[active_index] * active_coeff.conj() * coeffs).real * dt
        / np.abs(active_coeff)**2
    )
    return np.clip(probabilities, 0, 1)


def check_hop(rand_num, probs):
    """Select a target index from cumulative hopping probabilities.
    The hopping decision is made by comparing a random number r ∈ [0,1)
    with cumulative probabilities. A hop to state k occurs if:
    Σ_{j=0}^{k-1} p_j < r ≤ Σ_{j=0}^{k} p_j

    Args:
        r (float): Random number between 0 and 1
        p_ij (np.ndarray): Hopping probabilities (Nstates,)

    Returns:
        int: Index of target state (-1 if no hop occurs)
    """
    cumu_prob = np.cumsum(np.asarray(probs))
    for index, upper_bound in enumerate(cumu_prob):
        lower_bound = 0.0 if index == 0 else cumu_prob[index - 1]
        if lower_bound < rand_num <= upper_bound:
            return index
    return -1


def rescale_velocity(velocity, mass, energy, active_index, target_index, direction):
    """Return a rescaled velocity and whether the requested hop is allowed."""
    velocity = np.array(velocity, copy=True)
    mass = np.asarray(mass)
    energy = np.asarray(energy)
    direction = np.asarray(direction)
    a = np.sum(direction**2 / (2 * mass))
    b = np.sum(velocity * direction)
    c = energy[target_index] - energy[active_index]
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        gamma = ((b + np.sqrt(discriminant)) / (2 * a) if b < 0
                 else (b - np.sqrt(discriminant)) / (2 * a))
        velocity -= gamma * direction / mass
        return True, velocity

    gamma = b / a
    velocity -= gamma * direction / mass
    return False, velocity


def kinetic_energy(velocity, mass):
    """Return the classical nuclear kinetic energy in Hartree."""
    velocity = np.asarray(velocity)
    mass = np.asarray(mass)
    return np.sum(0.5 * mass * velocity**2)


def edc_decoherence(coeffs, energy, active_index, kinetic_energy,
                    dt, alpha):
    """Apply the energy-based decoherence correction to a coefficient copy.
    Ref:
    [1] Critical appraisal of the fewest switches algorithm for surface
        hopping. DOI: 10.1063/1.2715585
    [2] Nonadiabatic excited-state molecular dynamics: Treatment of
        electronic decoherence. DOI: 10.1063/1.4809568

    c_j = c_j * exp(-dt / tau_ji)
    c_i = c_i * sqrt((1 - sum_j(j!=i) |c_j|**2) / |c_i|**2)
    tau_ji = ħ / |E_jj - E_ii| * (1 + a / E_kin)
    """
    coeffs = np.array(coeffs, copy=True)
    energy = np.asarray(energy)
    inactive_population = 0.0

    for index in range(len(coeffs)):
        if index != active_index:
            tau_ji = (
                1 / np.abs(energy[index] - energy[active_index])
                * (1 + alpha / kinetic_energy)
            )
            coeffs[index] *= np.exp(-dt / tau_ji)
            inactive_population += np.abs(coeffs[index])**2

    coeffs[active_index] *= np.sqrt(
        (1 - inactive_population) / np.abs(coeffs[active_index])**2
    )
    return coeffs


@dataclasses.dataclass
class PES:
    '''
    energy: Electronic energies for all states (Nstates,) in Hartree
    force: Nuclear forces for current state (Natoms * 3) in Ha/Bohr
    nacv: Nonadiabatic coupling vectors for all states (Nstates, Nstates, Natoms, 3) in 1/bohr
    '''
    energy: np.ndarray = None
    force: np.ndarray = None
    nacv: np.ndarray = None


class FSSH:
    """
    This class implements the FSSH algorithm for nonadiabatic molecular dynamics simulations.

    The FSSH method treats nuclear motion classically while quantum mechanically
    describing electronic transitions between different potential energy surfaces.

    Attributes:
        tddft: Time-dependent density functional theory object
        tdgrad: Nuclear gradient scanner for force calculations
        states (List[int]): List of electronic states to include in simulation
        mass (np.ndarray): Nuclear masses in atomic units
        dt (float): Time step in atomic units
        nsteps (int): Number of simulation steps
        decoherence (str): Decoherence scheme, either ``'none'`` or ``'edc'``
        alpha (float): EDC strength parameter in Hartree
        seed: seed for random number generator

    Saved Results:
        cur_state (int): Current active electronic state
        energy: The electronic structure energies for the simulated states
        position: Nuclear coordinates in atomic units (Bohr)
        velocity: Nuclear velocities in atomic units (1 au = 21.877 A/fs)
        coefficient: Quantum coefficients for adiabatic states
    """

    seed = None
    coupling_method = 'nac'
    decoherence = 'edc'
    alpha = 0.1  # Hartree
    ktdc_gap_tol = 1e-6

    def __init__(self, mol, states: list[int], extra_dump=None):
        """
        Initialize the FSSH simulation with comprehensive parameter validation.

        Args:
            mol: Mole object
            states (List[int]): Electronic states to include in simulation
            extra_dump: Per-frame optional data selected from ``force``,
                ``nacv``, and ``velocity``.
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
        self.extra_dump = tuple(extra_dump or ())

        # Calculate nuclear masses and convert to atomic units
        masses = [COMMON_ISOTOPE_MASSES[NUC[x]] for x in mol.elements]
        self.mass = np.array(masses) * AMU2AU # (Na,1)  Unit: a.u.

        # Set default simulation parameters
        self.dt = 0.5 * FS2AUTIME  # Default: 0.5 fs in atomic units
        self.nsteps = 1
        self.filename = 'trajectory.h5'
        self._trajectory = None
        self.callback = None

        # State of the current step
        self.energy = None
        self.position = None
        self.velocity = None
        self.coefficient = None
        self.cur_step = 0
        self._history = {}

    @property
    def timestep_fs(self):
        '''time step length in fs'''
        return self.dt / FS2AUTIME
    @timestep_fs.setter
    def timestep_fs(self, x):
        self.dt = x * FS2AUTIME

    def evaluate_pes(self, position: np.ndarray, cur_state: int, with_nacv=True) -> PES:
        """
        Calculate electronic energies, nuclear forces and nonadiabatic coupling for all states.

        This method computes the potential energies and nonadiabatic coupling for all electronic
        states at the given nuclear configuration. The forces are obtained as the
        negative gradient of the potential energy surface at the current state.

        Args:
            position (np.ndarray): Nuclear coordinates in Bohr (Natoms * 3)
            cur_state (int): Current active electronic state

        Returns:
            PES instance:
                - energy: Electronic energies for all states (Nstates,) in Hartree
                - force: Nuclear forces for current state (Natoms * 3) in Ha/Bohr
                - nacv: Nonadiabatic coupling vectors for all states (Nstates, Nstates, Natoms, 3) in 1/bohr
        """
        raise NotImplementedError

    def _evaluate_hop(self, coeffs, nact, cur_state):
        """Draw a random number and select a hopping target index."""
        active_index = self.states.index(cur_state)
        p_ij = compute_hopping_probability(coeffs, nact, active_index, self.dt)
        r = self.random_uniform()
        hop_index = check_hop(r, p_ij)
        logger.debug(self.mol, f"Switching probability: {p_ij}, Random number: {r}")
        return hop_index

    def write_trajectory(self,
                         step: int,
                         position: np.ndarray,
                         velocity: np.ndarray,
                         pes: PES,
                         coeffs: np.ndarray,
                         cur_state: int,
                         **kwargs) -> None:
        """
        Write current trajectory frame

        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            pes (PES): Electronic structure energy, force, and nacv
            coeffs (np.ndarray): Quantum coefficients
            cur_state (int): Current active electronic state
        """
        if self._trajectory is None:
            mode = 'w' if step == 0 else 'a'
            self._trajectory = H5Trajectory(self.filename, mode)

        if step == 0:
            self._trajectory.write_configuration({
                'elements': self.mol.elements,
                'mass': self.mass.tolist(),
                'dt': self.dt,
                'nsteps': self.nsteps,
                'decoherence': self.decoherence,
                'alpha': self.alpha,
                'seed': self.seed,
                'states': self.states,
                'coupling_method': self.coupling_method,
                'extra_dump': list(self.extra_dump),
            })

        self._trajectory.write_checkpoint({'velocity': velocity})
        frame = {
            'position': position,
            'energy': pes.energy,
            'coeffs': coeffs,
            'cur_state': cur_state,
        }
        if 'velocity' in self.extra_dump:
            frame['velocity'] = velocity
        if 'force' in self.extra_dump:
            frame['force'] = pes.force
        if 'nacv' in self.extra_dump:
            if pes.nacv is None:
                raise ValueError("nacv was requested in extra_dump but is unavailable")
            frame['nacv'] = pes.nacv
        frame.update(kwargs)
        self._trajectory.write_frame(step, frame)

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
        with H5Trajectory(trajectory_file, 'r') as trajectory:
            configuration = _validate_fssh_configuration(
                trajectory.read_configuration())
            checkpoint = trajectory.read_checkpoint()
            cur_step, frame = trajectory.read_last_frame()

        if 'velocity' not in checkpoint:
            raise ValueError("HDF5 trajectory checkpoint velocity is missing")
        required_frame = {'position', 'energy', 'coeffs', 'cur_state'}
        missing_frame = required_frame.difference(frame)
        if missing_frame:
            raise ValueError(
                f"HDF5 trajectory frame is missing {sorted(missing_frame)}")

        elements = configuration['elements']
        if list(elements) != list(self.mol.elements):
            raise ValueError(
                "Trajectory elements do not match the current molecule")
        self.elements = elements
        self.mass = np.asarray(configuration['mass'], dtype=float)
        self.dt = configuration['dt']
        self.nsteps = configuration['nsteps']
        self.seed = configuration['seed']
        self.states = configuration['states']
        self.decoherence = configuration['decoherence']
        self.alpha = configuration['alpha']
        self.coupling_method = configuration['coupling_method']
        self.extra_dump = tuple(configuration['extra_dump'])

        self.cur_step = cur_step
        self.cur_state = int(frame['cur_state'])
        self.velocity = np.asarray(checkpoint['velocity'])
        self.position = np.asarray(frame['position'])
        self.energy = np.asarray(frame['energy'])
        self.coefficient = np.asarray(frame['coeffs'])

        return self

    def check_sanity(self):
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.nsteps <= 0:
            raise ValueError("Number of steps must be positive")
        supported_coupling = ['nac', 'direct', 'curvature', 'ktdc']
        if self.coupling_method not in supported_coupling:
            raise ValueError(f"coupling_method must be one of {supported_coupling}")
        supported_decoherence = ['none', 'edc']
        if self.decoherence not in supported_decoherence:
            raise ValueError(
                f"decoherence must be one of {supported_decoherence}")
        if isinstance(self.extra_dump, str):
            raise ValueError(
                f"extra_dump must contain values from {SUPPORTED_EXTRA_DUMP}")
        extra_dump = tuple(self.extra_dump)
        unsupported = set(extra_dump).difference(SUPPORTED_EXTRA_DUMP)
        if unsupported:
            raise ValueError(
                f"extra_dump contains unsupported values {sorted(unsupported)}; "
                f"choose from {SUPPORTED_EXTRA_DUMP}")
        if len(extra_dump) != len(set(extra_dump)):
            raise ValueError("extra_dump must not contain duplicate values")
        if ('nacv' in extra_dump and
                self.coupling_method not in ('nac', 'direct')):
            raise ValueError(
                "nacv in extra_dump requires coupling_method 'nac' or 'direct'")
        self.extra_dump = extra_dump

    def _finalize(self):
        if self._trajectory is not None:
            self._trajectory.close()
            self._trajectory = None

    def _history_requirements(self):
        """Return rolling algorithm-cache lengths for the coupling method."""
        if self.coupling_method in ('ktdc', 'curvature'):
            return {'energy': 3}
        return {}

    def _record_history(self, **values):
        """Append copies of available step data to configured cache fields."""
        for name, value in values.items():
            if name in self._history:
                self._history[name].append(np.array(value, copy=True))

    def _initialize_history(self, pes, velocity):
        """Create and populate the algorithm cache for a new or resumed run."""
        self._history = {
            name: deque(maxlen=maxlen)
            for name, maxlen in self._history_requirements().items()
        }
        if self.cur_step > 0 and 'energy' in self._history:
            previous_step = self.cur_step - 1
            with H5Trajectory(self.filename, 'r') as trajectory:
                previous_frame = trajectory.read_frame(previous_step)
            if 'energy' not in previous_frame:
                raise ValueError("HDF5 trajectory history frame energy is missing")
            previous_energy = np.asarray(previous_frame['energy'])
            self._record_history(energy=previous_energy)
        self._record_history(energy=pes.energy, force=pes.force, velocity=velocity)

    def _init_conditions(self, position, velocity, coefficient):
        """Resolve restart/default inputs and normalize coefficients."""
        if position is None:
            position = self.position
        if position is None:
            position = self.mol.atom_coords(unit='Bohr')

        if velocity is None:
            velocity = self.velocity
        if velocity is None:
            velocity = np.zeros_like(position)
            logger.warn("Velocity is set to zero if not provided")

        if coefficient is None:
            coefficient = self.coefficient
        if coefficient is None:
            coefficient = np.zeros(len(self.states), dtype=complex)
            coefficient[self.states.index(self.cur_state)] = 1.
        assert len(coefficient) == len(self.states)
        coefficient /= np.linalg.norm(coefficient)
        return position, velocity, coefficient

    def _prepare_run(self, position, velocity, coefficient, log):
        """Initialize random state, PES data, trajectory, and kTDC history."""
        if self.seed is not None:
            np.random.seed(self.seed)
            np.random.rand(self.cur_step)

        position, velocity, coefficient = self._init_conditions(position, velocity, coefficient)
        pes = self.evaluate_pes(position, self.cur_state, with_nacv=('nacv' in self.extra_dump))
        self._initialize_history(pes, velocity)

        if self.cur_step == 0:
            if self.coupling_method in ('ktdc', 'curvature'):
                log.info("Building κTDC energy history during the first two steps")

            self.write_trajectory(0, position, velocity, pes, coefficient, self.cur_state)
            log.info(f"Starting main simulation loop for {self.nsteps} steps")
        else:
            assert h5py.is_hdf5(self.filename)
            log.info(f'Skipping {self.cur_step} steps and resuming the simulation.')

        return position, velocity, coefficient, pes

    def _compute_nact(self, position, velocity, cur_state):
        """Evaluate the new PES and its time-derivative coupling matrix."""
        if self.coupling_method in ('nac', 'direct'):
            pes = self.evaluate_pes(position, cur_state, with_nacv=True)
            nact = np.einsum('ijnd,nd->ij', pes.nacv, velocity)
        elif self.coupling_method in ('ktdc', 'curvature'):
            pes = self.evaluate_pes(position, cur_state, with_nacv=False)
            self._record_history(energy=pes.energy, force=pes.force, velocity=velocity)
            energy_history = self._history['energy']
            if len(energy_history) == 3:
                nact = ktdc_energy(
                    energy_history[2], energy_history[1], energy_history[0],
                    self.dt, self.ktdc_gap_tol)
            else:
                nstates = len(self.states)
                nact = np.zeros((nstates, nstates))
        elif self.coupling_method == 'overlap':
            raise NotImplementedError
        else:
            raise RuntimeError(f'TDC method {self.coupling_method} not supported')
        return pes, nact

    def _attempt_hop(self, hop_index, cur_state, position, velocity, pes,
                     step, log):
        """Attempt a selected hop and return the resulting trajectory state."""
        current_index = self.states.index(cur_state)
        if hop_index == -1 or hop_index == current_index:
            return cur_state, velocity, pes

        target_pes = None
        if self.coupling_method in ('nac', 'direct'):
            direction = pes.nacv[current_index, hop_index]
        elif self.coupling_method in ('ktdc', 'curvature'):
            target_pes = self.evaluate_pes(position, self.states[hop_index], with_nacv=False)
            target_gradient = -target_pes.force
            current_gradient = -pes.force
            direction = target_gradient - current_gradient

        hop_allowed, velocity = rescale_velocity(
            velocity, self.mass[:,None], pes.energy, current_index, hop_index, direction)
        if hop_allowed:
            old_state = cur_state
            cur_state = self.states[hop_index]
            if target_pes is None:
                pes = self.evaluate_pes(position, cur_state, with_nacv=True)
            else:
                pes = target_pes
            log.info(f"Hop: {old_state} → {cur_state} at step {step}")
        else:
            log.debug(
                f"Hop to state {self.states[hop_index]} rejected.")
        return cur_state, velocity, pes

    def _apply_decoherence(self, coefficient, pes, cur_state, E_kin):
        """Apply the configured decoherence scheme."""
        if self.decoherence == 'none':
            return coefficient

        # TODO: disable decoherence near the avoid-crossing region, and
        # perform decoherence when the trajectory moves to the
        # well-separated surface region.
        return edc_decoherence(
            coefficient, pes.energy, self.states.index(cur_state),
            E_kin, self.dt, self.alpha)

    def _update_runtime_state(self, step, position, velocity, coefficient,
                              pes, cur_state):
        """Commit one completed integration step to public runtime state."""
        self.cur_state = cur_state
        self.energy = pes.energy
        self.position = position
        self.velocity = velocity
        self.coefficient = coefficient
        self.cur_step = step

    def _log_step(self, log, step, pes, cur_state, kinetic_energy_value,
                  iter_timing):
        """Log one completed step and return the next timing checkpoint."""
        current_index = self.states.index(cur_state)
        potential_energy = pes.energy[current_index]
        total_energy = potential_energy + kinetic_energy_value
        populations = np.abs(self.coefficient)**2
        total_time = step * self.timestep_fs
        log.info(
            f"Step {step:4d}: Time {total_time:8.3f} fs, "
            f"State {cur_state:2d}, "
            f"Potential {potential_energy:14.8f} Ha, "
            f"Kinetic {kinetic_energy_value:14.8f} Ha, "
            f"Total {total_energy:14.8f} Ha, "
            f"Populations: {populations}")
        log.debug('Energies for all electronic states: %s', pes.energy)
        return log.timer(f'FSSH step {step}', *iter_timing)

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
            position (Optional[np.ndarray]): Initial nuclear coordinates in a.u.
                If None, uses the geometry from geometry provided by mol.
            velocity (Optional[np.ndarray]): Initial nuclear velocities in a.u.
                Must be provided for dynamics simulation
            coefficient (Optional[np.ndarray]): Initial quantum coefficients
                If None, starts in the first specified state

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Final nuclear positions in a.u.
                - Final nuclear velocities in a.u.
                - Final quantum coefficients
        """
        self.check_sanity()

        log = logger.new_logger(self.mol, self.verbose)
        start_timing = log.init_timer()

        log.info(f"FSSH simulation initialized with states {self.states}, current state: {self.cur_state}\n"
                 f"dt={self.timestep_fs:.3f} fs, total steps: {self.nsteps}\n"
                 f"coupling_method={self.coupling_method}\n"
                 f"decoherence={self.decoherence}, alpha={self.alpha}\n"
                 f"Trajectory will be saved to {self.filename}\n")
        log.info(f"Starting FSSH trajectory simulation at {datetime.datetime.now()}")

        position, velocity, coefficient, pes = self._prepare_run(
            position, velocity, coefficient, log)

        iter_timing = start_timing

        for step in range(self.cur_step+1, self.nsteps+1):
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * pes.force / self.mass[:,None]

            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity
            cur_state = self.cur_state

            # 3. calculate new energy, force, and time-derivative coupling
            pes, nact = self._compute_nact(position, velocity, cur_state)

            # 4. update the electronic amplitude within a full-time step
            coefficient = update_coefficient(coefficient, pes.energy, nact, self.dt)

            # 5. evaluate the switching probability
            hop_index = self._evaluate_hop(coefficient, nact, cur_state)

            # 6. adjust nuclear velocity
            cur_state, velocity, pes = self._attempt_hop(
                hop_index, cur_state, position, velocity, pes, step, log)

            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * pes.force / self.mass[:,None]
            E_kin = kinetic_energy(velocity, self.mass[:,None])

            # 8. decoherence
            coefficient = self._apply_decoherence(coefficient, pes, cur_state, E_kin)

            self.write_trajectory(step, position, velocity, pes, coefficient, cur_state)

            self._update_runtime_state(step, position, velocity, coefficient, pes, cur_state)

            if callable(self.callback):
                self.callback(locals())

            iter_timing = self._log_step(log, step, pes, cur_state, E_kin, iter_timing)

        # Simulation completed successfully
        log.timer("FSSH simulation", *start_timing)
        log.info(f"FSSH simulation completed successfully at {datetime.datetime.now()}")

        self._finalize()

        return position, velocity, coefficient

def h5_to_xyz(h5file, trajectory_file):
    with H5Trajectory(h5file, 'r') as trajectory, open(trajectory_file, 'w') as f:
        configuration = _validate_fssh_configuration(
            trajectory.read_configuration())
        elements = configuration['elements']
        natm = len(elements)
        dt = configuration['dt']
        states = configuration['states']

        frame_steps = trajectory.frame_steps()
        print(f'Converting {len(frame_steps)} steps of trajectory data to xyz')
        for step in frame_steps:
            frame = trajectory.read_frame(step)
            required_frame = {'position', 'energy', 'coeffs', 'cur_state'}
            missing_frame = required_frame.difference(frame)
            if missing_frame:
                raise ValueError(
                    f"HDF5 trajectory frame {step} is missing "
                    f"{sorted(missing_frame)}")
            # Write number of atoms
            f.write(f'{natm}\n')

            # Write comment line with simulation data
            time_fs = step * dt / FS2AUTIME
            energy = np.asarray(frame['energy'])
            coeffs = np.asarray(frame['coeffs'])
            cur_state = int(frame['cur_state'])
            current_energy = energy[states.index(cur_state)]

            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')

            position = np.asarray(frame['position']) * BOHR  # Convert to Angstrom
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
