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
import os
from pyscf import lib
import numpy as np
from pyscf.gto.mole import format_atom
from pyscf.data import elements
from pyscf.data.nist import BOHR
from pyscf.lib import logger

class Mole(lib.StreamObject):
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

    def __init__(self, atom, method='PM6', params=None, charge=0, spin=0, verbose=0, **kwargs):
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
        self.output = kwargs.get('output', None)
        self.atom = atom
        self.method = method
        self.params = params
        self.charge = charge
        self.spin = spin
        
        self.natm = 0
        self.nao = 0
        self.nelec_per_atom = None
        self.nelec = None
        self.nelectron = None
        self.eta_1e = None
        self.eta_2e = None
        
        self._atom = []        # Internal format [[Z, [x,y,z]], ...]
        self._atom_ids = None  # Array of Atomic Numbers (Z)
        self._coords = None    # Array of Coordinates (Bohr)
        self._aoslice = None   # Array (natm, 2) -> [start_idx, end_idx]
        self._enuc = None
        
        self.uspd = None       # One-center energies
        self.atheat = None     # Heat of formation term
        self.unit = kwargs.get('unit', 'Angstrom')
        self._check_input(kwargs)
        
        self._built = False

    def build(self):
        """
        Main initialization routine.
        Parses geometry, establishes topology, and allocates arrays.
        """
        if (self.output is not None
            # StringIO() does not have attribute 'name'
            and getattr(self.stdout, 'name', None) != self.output):

            if self.verbose > logger.QUIET:
                if os.path.isfile(self.output):
                    print('overwrite output file: %s' % self.output)
                else:
                    print('output file: %s' % self.output)

            if self.output == '/dev/null':
                self.stdout = open(os.devnull, 'w', encoding='utf-8')
            else:
                self.stdout = open(self.output, 'w', encoding='utf-8')

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
        self.dump_input()
        return self

    def _check_input(self, kwargs):
        """
        Validates input parameters and throws errors for invalid combinations.
        """
        if 'basis' in kwargs:
            raise ValueError("Basis set specification is not supported.")


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
        self.nelec_per_atom = np.zeros(self.natm, dtype=np.int32)
        for i, z in enumerate(self._atom_ids):
            self.nelec_per_atom[i] = self.params.core_charges[z-1] # 0-based index for params
            
        self.nelectron = int(np.sum(self.nelec_per_atom) - self.charge)
        if (self.nelectron + self.spin) % 2 != 0:
            raise ValueError("Inconsistent electron count and spin.")
        nalpha = (self.nelectron + self.spin) // 2
        nbeta = nalpha - self.spin
        self.nelec = (nalpha, nbeta) 

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
        if unit.upper().startswith('A'):
            return self._coords * BOHR
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
        res[:, 2] = self._aoslice[:, 0]
        res[:, 3] = self._aoslice[:, 1]
        ishell   = 0
        for i in range(self.natm):
            if self._aoslice[i, 1] - self._aoslice[i, 0] == 1:
                res[i, 0] = ishell
                res[i, 1] = ishell + 1
                ishell += 1
            elif self._aoslice[i, 1] - self._aoslice[i, 0] == 4:
                res[i, 0] = ishell
                res[i, 1] = ishell + 2
                ishell += 2
            elif self._aoslice[i, 1] - self._aoslice[i, 0] == 9:
                res[i, 0] = ishell
                res[i, 1] = ishell + 3
                ishell += 3
        return res

    @property
    def natorb_per_atom(self):
        """Returns array of orbitals per atom."""
        return self._aoslice[:, 1] - self._aoslice[:, 0]

    def ao_labels(self, fmt=None):
        """
        Returns a list of AO labels consistent with PySCF format.
        
        Args:
            fmt (str, optional): Format string. Not fully implemented as in PySCF,
                                 but kept for API compatibility.
                                 
        Returns:
            list: List of strings, e.g., ['0 O 2s', '0 O 2px', '0 O 2py', ...]
        """
        # PM6 Specific Basis Ordering (Spherical Harmonic Order for p/d)
        # s
        # p: y, z, x (m = -1, 0, +1)
        # d: xy, yz, z^2, xz, x^2-y^2 (m = -2, -1, 0, +1, +2)
        
        # Note: PySCF standard label for p is (x,y,z). 
        # However, our internal storage is Spherical (y,z,x).
        # To make it user-friendly, we label them as they correspond to the internal storage.
        
        labels = []
        atom_symbs = [elements._symbol(z) for z in self._atom_ids]
        
        for i in range(self.natm):
            symb = atom_symbs[i]
            z = self._atom_ids[i]
            
            n_s = self.params.principal_quantum_number_matrix[z-1, 0]
            n_p = self.params.principal_quantum_number_matrix[z-1, 1]
            n_d = self.params.principal_quantum_number_matrix[z-1, 2]
            
            start, end = self._aoslice[i]
            n_orb = end - start
            
            prefix = f"{i} {symb}"
            
            if n_orb >= 1:
                labels.append(f"{prefix} {n_s}s")
            
            if n_orb >= 4:
                labels.append(f"{prefix} {n_p}py")
                labels.append(f"{prefix} {n_p}pz")
                labels.append(f"{prefix} {n_p}px")
            
            if n_orb >= 9: 
                labels.append(f"{prefix} {n_d}dxy")
                labels.append(f"{prefix} {n_d}dyz")
                labels.append(f"{prefix} {n_d}dz^2")
                labels.append(f"{prefix} {n_d}dxz")
                labels.append(f"{prefix} {n_d}dx^2-y^2")
                
        if fmt is None:
            return labels
        else:
            return labels

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

    def energy_nuc(self, *args):
        raise NotImplementedError("Nuclear repulsion energy is not supported in PM6Mole.")

    def get_hcore(self, *args):
        raise NotImplementedError("Hcore matrix is not supported in PM6Mole.")

    def get_ovlp(self, *args):
        raise NotImplementedError("Overlap matrix is not supported in PM6Mole.")

    def dump_input(self):
        import __main__
        if hasattr(__main__, '__file__'):
            try:
                filename = os.path.abspath(__main__.__file__)
                finput = open(filename, 'r')
                self.stdout.write('#INFO: **** input file is %s ****\n' % filename)
                self.stdout.write(finput.read())
                self.stdout.write('#INFO: ******************** input file end ********************\n')
                self.stdout.write('\n')
                self.stdout.write('\n')
                finput.close()
            except IOError:
                logger.warn(self, 'input file does not exist')

        self.stdout.write('\n'.join(lib.misc.format_sys_info()))

        self.stdout.write('\n\n')
        for key in os.environ:
            if 'PYSCF' in key:
                self.stdout.write('[ENV] %s %s\n' % (key, os.environ[key]))

        self.stdout.write('[INPUT] verbose = %d\n' % self.verbose)
        if self.verbose >= logger.DEBUG:
            self.stdout.write('[INPUT] num. atoms = %d\n' % self.natm)
            self.stdout.write('[INPUT] num. electrons = %d\n' % self.nelectron)
            self.stdout.write('[INPUT] charge = %d\n' % self.charge)
            self.stdout.write('[INPUT] spin (= nelec alpha-beta = 2S) = %d\n' % self.spin)
            self.stdout.write('[INPUT] Mole.unit = %s\n' % self.unit)
            self.stdout.write('[INPUT] Basis in spherical coordinates\n')

            self.stdout.write('[INPUT] Symbol           X                Y                Z      unit'
                             '          X                Y                Z       unit\n')
        for ia,atom in enumerate(self._atom):
            coorda = tuple([x * BOHR for x in atom[1]])
            coordb = tuple(atom[1])
            self.stdout.write('[INPUT]%3d %-4s %16.12f %16.12f %16.12f AA  '
                              '%16.12f %16.12f %16.12f Bohr\n'
                              % ((ia+1, elements._symbol(atom[0])) + coorda + coordb))

        def dump_basis_info(self, eta_list):
            for ia in range(self.natm):
                start, end = self._aoslice[ia]
                n_orb = end - start
                shells = []
                if n_orb == 1:
                    shells = [(0, 0, 1)]
                elif n_orb == 3:
                    shells = [(1, 0, 3)]
                elif n_orb == 4:
                    shells = [(0, 0, 1), (1, 1, 3)]
                elif n_orb == 9:
                    shells = [(0, 0, 1), (1, 1, 3), (2, 4, 5)]
                elif n_orb == 5:
                    shells = [(2, 0, 5)]
                for i_sh, (l, offset, count) in enumerate(shells):
                    expnt = eta_list[start + offset]
                    self.stdout.write('[INPUT]   %3d   |   %2d  | %d | %16.12f\n' % 
                                    (ia, i_sh, l, expnt))


        if self.verbose >= logger.DEBUG:
            self.stdout.write('[INPUT] ---------------- BASIS SET for hcore ---------------- \n')
            self.stdout.write('[INPUT]   atom   l,   expnt\n')
            dump_basis_info(self, self.eta_1e)

            self.stdout.write('[INPUT] ---------------- BASIS SET for 2c2e ---------------- \n')
            self.stdout.write('[INPUT]   atom   l,   expnt\n')
            dump_basis_info(self, self.eta_2e)

        if self.verbose >= logger.INFO:
            self.stdout.write('\n')
            # logger.info(self, 'nuclear repulsion = %.15g', self.enuc) #TODO: remove if code has been done.
            logger.info(self, 'number of shells = %d', self.aoslice_by_atom()[-1, 1])
            logger.info(self, 'number of basis = %d', self.nao)
            logger.info(self, 'basis = Slater basis')
        if self.verbose >= logger.DEBUG2:
            for i in range(len(self._bas)):
                exps = self.bas_exp(i)
                logger.debug1(self, 'bas %d, expnt(s) = %s', i, str(exps))

        logger.info(self, 'CPU time: %12.2f', logger.process_clock())
        return self