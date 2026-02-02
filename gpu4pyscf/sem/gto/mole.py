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

from pyscf import gto
from gpu4pyscf.sem.gto import params as params_gpu4pyscf
import sys
import numpy as np
from pyscf.gto.mole import format_atom
from pyscf.data import elements
from pyscf.lib import logger

class Mole:
    """
    A standalone Molecule class designed for PM6 Semi-Empirical methods.
    
    This class mimics the API of pyscf.gto.Mole (Duck Typing) to ensure 
    compatibility with PySCF-style workflows, but removes all libcint 
    dependencies and legacy Fortran indexing complexities.

    Attributes:
        atom (list/str): Raw atom input.
        params (SEMParams): Parameter object containing semi-empirical constants.
        charge (int): Total system charge.
        spin (int): Spin multiplicity (N_alpha - N_beta).
        
        natm (int): Number of atoms.
        nao (int): Number of atomic orbitals.
        nelec (int): Number of valence electrons.
        
        hcore (np.ndarray): Core Hamiltonian matrix (nao, nao).
        eri (np.ndarray): Two-electron integrals in sparse/packed 1D format.
    """

    def __init__(self, atom, params=None, charge=0, spin=0, verbose=0, **kwargs):
        """
        Initialize the PM6Mole object.

        Args:
            atom: Geometry string or list (e.g., "H 0 0 0; F 0 0 1").
            params: Initialized SEMParams object.
            charge: System charge (default 0).
            spin: 2S (N_alpha - N_beta). 0
            verbose: Logging level.
        """
        self.verbose = verbose
        self.stdout = sys.stdout
        
        self.atom = atom
        self.method = 'PM6'
        self.params = params
        self.charge = charge
        self.spin = spin
        
        self.natm = 0
        self.nao = 0
        self.nelec = 0
        self.eta_1e = None
        self.eta_2e = None
        
        self._atom = []        # Internal format [[Z, [x,y,z]], ...]
        self._atom_ids = None  # Array of Atomic Numbers (Z)
        self._coords = None    # Array of Coordinates (Bohr)
        self._aoslice = None   # Array (natm, 2) -> [start_idx, end_idx]
        self.energy_nuc = None # function
        self._enuc = None
        
        self.uspd = None       # One-center energies
        self.atheat = None     # Heat of formation term
        self.unit = 'Angstrom'
        
        self._built = False

    def build(self):
        """
        Main initialization routine.
        Parses geometry, establishes topology, and allocates arrays.
        """
        raw_atom_data = format_atom(self.atom, unit=self.unit)
        
        self.natm = len(raw_atom_data)
        self._atom = []
        self._atom_ids = np.zeros(self.natm, dtype=np.int32)
        self._coords = np.zeros((self.natm, 3), dtype=np.float64)
        
        for i, (symb, coord) in enumerate(raw_atom_data):
            z = elements.charge(symb)
            self._atom.append([z, coord])
            self._atom_ids[i] = z
            self._coords[i] = coord

        if self.params is None:
            self.params = params_gpu4pyscf.load_sem_params(method=self.method)
        self._build_topology()
        self._count_electrons()
        self._init_model_arrays()

        try:
            self._compute_integrals()
        except NotImplementedError:
            if self.verbose > logger.WARN:
                logger.warn(self, "Integral computation skipped (Not Implemented). Arrays are empty.")

        try:
            self._compute_heat_formation()
        except NotImplementedError:
            pass

        self._built = True
        return self

    def _build_topology(self):
        """
        Determines the number of orbitals per atom and builds the slice index.
        Replaces legacy 'nfirst', 'nlast', 'natorb'.
        """
        self._aoslice = np.zeros((self.natm, 2), dtype=np.int32)
        
        cursor = 0
        for i in range(self.natm):
            z = self._atom_ids[i]
            n_orb = self.params.norbitals_per_atom[z-1]
            if n_orb == 0:
                raise ValueError(f"Element Z={z} is not supported by {self.params.method} or has no orbitals defined.")
                
            self._aoslice[i, 0] = cursor
            self._aoslice[i, 1] = cursor + n_orb
            cursor += n_orb
        self.nao = cursor

    def _count_electrons(self):
        """
        Counts valence electrons using core charges from SEMParams.
        """
        n_val = 0.0
        for z in self._atom_ids:
            n_val += self.params.core_charges[z-1] # 0-based index for params
            
        self.nelec = int(n_val - self.charge)
        if (self.nelec + self.spin) % 2 != 0:
            raise ValueError("Inconsistent electron count and spin.")

    def _init_model_arrays(self):
        """
        Initializes one-center parameters (USPD).
        """
        self.uspd = np.zeros(self.nao, dtype=np.float64)
        self.eta_1e = np.zeros(self.nao, dtype=np.float64)
        self.eta_2e = np.zeros(self.nao, dtype=np.float64)
        
        energy_core_s = self.params.get_parameter('energy_core_s', to_gpu=False)
        energy_core_p = self.params.get_parameter('energy_core_p', to_gpu=False)
        energy_core_d = self.params.get_parameter('energy_core_d', to_gpu=False)
        exponent_s = self.params.get_parameter('exponent_s', to_gpu=False)
        exponent_p = self.params.get_parameter('exponent_p', to_gpu=False)
        exponent_d = self.params.get_parameter('exponent_d', to_gpu=False)
        exponent_internal_s = self.params.get_parameter('exponent_internal_s', to_gpu=False)
        exponent_internal_p = self.params.get_parameter('exponent_internal_p', to_gpu=False)
        exponent_internal_d = self.params.get_parameter('exponent_internal_d', to_gpu=False)

        for i in range(self.natm):
            z = self._atom_ids[i]
            idx = z - 1 # 0-based index for parameter arrays
            start, end = self._aoslice[i]
            n_orb = end - start
            if n_orb >= 1: # s
                self.uspd[start] = energy_core_s[idx]
                self.eta_1e[start] = exponent_s[idx]
                self.eta_2e[start] = exponent_internal_s[idx]
            if n_orb >= 4: # p
                self.uspd[start+1 : start+4] = energy_core_p[idx]
                self.eta_1e[start+1 : start+4] = exponent_p[idx]
                self.eta_2e[start+1 : start+4] = exponent_internal_p[idx]
            if n_orb >= 9: # d
                self.uspd[start+4 : start+9] = energy_core_d[idx]
                self.eta_1e[start+4 : start+9] = exponent_d[idx]
                self.eta_2e[start+4 : start+9] = exponent_internal_d[idx]

    def _compute_integrals(self):
        """
        Calculates 1-electron Hcore matrix and 2-electron integral buffer.
        """
        # Placeholder for 'h1elec', 'rotate', 'wstore' replacements.
        # This will be implemented in the next phase using GPU kernels.
        raise NotImplementedError("Integral engine (h1elec/rotate/wstore) needs to be rewritten for GPU.")

    def _compute_heat_formation(self):
        """
        Calculates atomic heat of formation.
        """
        # Placeholder for 'compute_atheat_pm6_mol' replacement.
        raise NotImplementedError("Atomic heat calculation needs to be ported.")

    
    def atom_coords(self, unit='Bohr'):
        """Returns coordinates. Default is Bohr (internal storage)."""
        if unit.upper()[0] == 'A':
            # Conversion factor: Bohr to Angstrom
            return self._coords * 0.52917721092 
        return self._coords

    def atom_charge(self, atom_id):
        """Returns nuclear charge Z for a given atom index."""
        return self._atom_ids[atom_id]

    def aoslice_by_atom(self):
        """
        Returns [[shell_start, shell_end, ao_start, ao_end], ...].
        Since we don't track shells in the PySCF sense, we fake shell indices (0,0).
        Format: (natm, 4)
        """
        res = np.zeros((self.natm, 4), dtype=np.int32)
        res[:, 2] = self._aoslice[:, 0] # ao_start
        res[:, 3] = self._aoslice[:, 1] # ao_end
        return res

    @property
    def natorb_per_atom(self):
        """Returns array of orbitals per atom."""
        return self._aoslice[:, 1] - self._aoslice[:, 0]

    def intor(self, intor_name, *args, **kwargs):
        """
        Raises error for standard PySCF integral calls.
        """
        raise NotImplementedError(
            f"Integral '{intor_name}' is not supported in PM6Mole. "
            "Use parameterized integrals."
        )
    
    @property
    def enuc(self):
        """Nuclear repulsion energy (including core-core corrections)."""
        if self._enuc is None:
            self._enuc = self.energy_nuc()
        return self._enuc