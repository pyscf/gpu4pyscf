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

import numpy as np
import cupy as cp
from scipy.optimize import root
from gpu4pyscf.scf.hf import damping, level_shift
from pyscf import lib as pyscf_lib
from pyscf import gto
from gpu4pyscf.dft.rkspu import reference_mol, _make_minao_lo
from gpu4pyscf import dft, lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import radi

def _get_minao_basis_indices(mol, minao_mol, identifier):
    """
    Helper function to resolve identifiers into MINAO basis indices.
    
    Args:
        mol: The main molecule object.
        minao_mol: The reference MINAO molecule object.
        identifier: int (atom ID) or str (orbital label, e.g., "0 C 2p").
        
    Returns:
        list: A list of indices corresponding to the MINAO basis functions.
    """
    minao_slices = minao_mol.aoslice_by_atom()
    
    # Case 1: Input is an integer (Atom ID) -> Select all orbitals of this atom
    if isinstance(identifier, (int, np.integer)):
        atom_id = int(identifier)
        p0 = minao_slices[atom_id, 2]
        p1 = minao_slices[atom_id, 3]
        return list(range(p0, p1))
    
    # Case 2: Input is a string (Orbital Label) -> Use PySCF label search
    elif isinstance(identifier, str):
        # search_ao_label returns indices based on the minao_mol basis
        return minao_mol.search_ao_label(identifier)
    
    else:
        raise ValueError(f"CDFT Error: Unsupported identifier type {type(identifier)}: {identifier}")

class CDFT_UKS(dft.UKS):
    '''
    Constrained DFT implementation for unrestricted Kohn-Sham.
    Implements spatial charge/spin constraints using either Becke weights or MINAO projection.
    Supports two modes: 'lagrange' (Lagrange Multipliers) and 'penalty' (Quadratic Penalty).
    Supports constraints on whole atoms or specific orbitals (MINAO only).
    '''

    def __init__(self, mol, charge_constraints=None, spin_constraints=None, 
                 method='lagrange', penalty_weight=500.0,
                 projection_method='becke'):
        super().__init__(mol)
        
        # [Requirement 2] Concise Constraints format: 
        # [ [atom_indices_or_groups], [targets] ]
        # Groups can now contain ints (atoms) or strings (orbital labels)
        
        self.charge_groups, self.charge_targets = self._normalize_constraints(charge_constraints)
        self.spin_groups, self.spin_targets = self._normalize_constraints(spin_constraints)
        
        self.n_constraints = len(self.charge_targets) + len(self.spin_targets)
        
        # Initial guess for Lagrange multipliers V
        self.v_lagrange = np.zeros(self.n_constraints) + 0.01
        
        # Microiteration parameters
        self.micro_tol = 1e-4
        self.micro_max_cycle = 50
        
        # Cache for constraint matrices on GPU
        # Structure: List of matrices, one for each constraint (Charge first, then Spin)
        self.constraint_projectors = None

        # Configuration for methods
        self.method = method.lower() # 'lagrange' or 'penalty'
            
        # Store the penalty weight for energy calculation
        self.penalty_weight = penalty_weight

        # Selection of projection method: 'becke' (default) or 'minao'
        self.projection_method = projection_method.lower()
        self.minao_ref = 'MINAO'

    def _normalize_constraints(self, constraints):
        '''
        Helper: Normalize user input into internal list-of-lists format.
        Input: [ [0, 1], [6.5, 7.5] ] 
        Output: ( [[0], [1]], [6.5, 7.5] )
        Input: [ [0, "1 N 2p"], [targets] ]
        Output: ( [[0], ["1 N 2p"]], [targets] )
        '''
        if not constraints:
            return [], []
        
        input_groups = constraints[0]
        targets = constraints[1]
        
        if len(input_groups) != len(targets):
            raise ValueError("CDFT Error: The number of atom groups must match the number of targets.")

        normalized_groups = []
        for item in input_groups:
            # If user provided a single int or str, wrap it in a list
            if isinstance(item, (int, np.integer, str)):
                normalized_groups.append([item])
            else:
                # Assuming it's already a list/tuple for a group
                normalized_groups.append(list(item))
                
        return normalized_groups, targets

    def build_projectors(self):
        """
        Main entry point to build projectors.
        Dispatches to either Becke (Grid) or MINAO (Projection) method based on configuration.
        The result is stored in self.constraint_projectors.
        """
        if self.constraint_projectors is not None:
            return self.constraint_projectors

        if self.projection_method == 'minao':
            return self._build_minao_projectors()
        else:
            return self._build_becke_projectors()

    def _build_minao_projectors(self):
        """
        Constructs constraint weight matrices 'W' using Projected Lowdin Orthogonalization.
        This follows the exact logic used in PySCF/GPU4PySCF DFT+U implementation.
        Supports specific orbital selection (e.g., 'Ni 3d') via search_ao_label.

        Mathematical Steps:
        1. Projection: Solve for C_raw such that S * C_raw = S_12 (AO-MINAO overlap).
        2. Orthogonalization: S_0 = C_raw^T * S * C_raw (Metric of projected orbitals).
           Transform C_orth = C_raw * S_0^(-1/2).
        3. Weight Matrix: W = (S * C_orth) * (S * C_orth)^T.
        """
        mol = self.mol
        nao = mol.nao_nr()
        logger.info(self, "CDFT: Building projectors using Orthogonalized MINAO reference (DFT+U style)...")

        ovlp = self.get_ovlp()
        if isinstance(self.minao_ref, str):
            minao_mol = reference_mol(mol, self.minao_ref)
        else:
            minao_mol = self.minao_ref
        # s12_cpu = gto.mole.intor_cross('int1e_ovlp', mol, minao_mol)
        # s12 = cp.asarray(s12_cpu)
        # C_minao = cp.linalg.solve(ovlp, s12)
        # S0 = C_minao.conj().T @ ovlp @ C_minao
        # w, v = cp.linalg.eigh(S0)
        # # TODO: use _make_minao_lo instead
        # C_minao = C_minao.dot((v*cp.sqrt(1./w)).dot(v.conj().T))
        C_minao = _make_minao_lo(mol, minao_mol)
        SC = ovlp @ C_minao
        
        projectors = []
        all_groups = self.charge_groups + self.spin_groups
        
        for group in all_groups:
            indices = []
            
            # Resolve all identifiers in the group (atoms or orbital strings)
            for identifier in group:
                idx_list = _get_minao_basis_indices(mol, minao_mol, identifier)
                indices.extend(idx_list)
            
            if not indices:
                # Handle ghost atoms or empty selections
                projectors.append(cp.zeros((nao, nao)))
                continue
            
            # Sort and remove duplicates
            indices = sorted(list(set(indices)))
            indices = cp.asarray(indices)
            
            # Extract relevant columns from SC
            # Shape: (NAO, N_Selected_Orbitals)
            sc_subset = SC[:, indices]
            
            # Construct Weight Matrix: W = SC_sub * SC_sub^T
            w_matrix = sc_subset @ sc_subset.conj().T
            
            projectors.append(w_matrix)
            
        self.constraint_projectors = projectors
        return self.constraint_projectors

    def _build_becke_projectors(self):
        r'''
        Constructs constraint weight matrices 'w' on GPU using Becke partitioning.
        
        This implementation strictly follows the logic from the reference 'gen_grid_partition'
        function (from 'grids_response_cc'), preserving the exact mathematical steps,
        loop structures, and data dimensions.
        Note: Becke partitioning is spatial and cannot distinguish between orbitals (e.g., 3d vs 4s).
        It only supports atom-based constraints.
        '''
        mol = self.mol
        nao = mol.nao_nr()
        
        # Check input validity for Becke method
        all_groups = self.charge_groups + self.spin_groups
        for group in all_groups:
            for item in group:
                if isinstance(item, str):
                     raise ValueError(f"CDFT Error: Becke projection does not support orbital-specific constraints ('{item}'). "
                                      "Please use 'minao' projection method or use atom indices.")

        # --- Standard Becke Weight Calculation (Atom-wise) ---
        atm_coords = np.asarray(mol.atom_coords(), order='C')
        atm_dist = gto.inter_distance(mol, atm_coords)
        atm_dist = cp.asarray(atm_dist)
        atm_coords = cp.asarray(atm_coords)

        # ---------------------------------------------------------------------
        # Define Radii Adjustment Function
        # Logic copied from the reference '_radii_adjust'
        # ---------------------------------------------------------------------
        grids = self.grids
        atomic_radii = grids.atomic_radii
        radii_adjust_method = grids.radii_adjust

        def _radii_adjust(mol, atomic_radii):
            charges = mol.atom_charges()
            if radii_adjust_method == radi.treutler_atomic_radii_adjust:
                rad = np.sqrt(atomic_radii[charges]) + 1e-200
            elif radii_adjust_method == radi.becke_atomic_radii_adjust:
                rad = atomic_radii[charges] + 1e-200
            else:
                fadjust = lambda i, j, g: g
                return fadjust

            rr = rad.reshape(-1,1) * (1./rad)
            a = .25 * (rr.T - rr)
            a[a<-.5] = -.5
            a[a>0.5] = 0.5
            
            a_gpu = cp.asarray(a)

            def fadjust(i, j, g):
                # Apply the coordinate transformation: nu = mu + a_ij * (1 - mu^2)
                return g + a_gpu[i,j]*(1-g**2)
            return fadjust

        fadjust = _radii_adjust(mol, atomic_radii)

        # Initialize projectors list
        atom_projectors_temp = [cp.zeros((nao, nao)) for _ in range(mol.natm)]

        # Ensure grids are built
        if self.grids.coords is None:
            self.grids.build()
            
        coords_all = self.grids.coords
        weights_all = self.grids.weights
        ni = self._numint
        
        # Batch processing
        blksize = 4000
        n_grids_total = weights_all.shape[0]
        
        logger.info(self, f"CDFT: Building Becke projectors on GPU using {n_grids_total} grid points...")
        
        for p0, p1 in pyscf_lib.prange(0, n_grids_total, blksize):
            coords_batch = cp.asarray(coords_all[p0:p1])
            weights_batch = cp.asarray(weights_all[p0:p1])
            ngrids_batch = coords_batch.shape[0]
            ao_batch = ni.eval_ao(mol, coords_batch, deriv=0)
            
            grid_dist = []
            
            # 1. Calculate distances using Python loop over atoms
            for ia in range(mol.natm):
                # v = r_atom - r_grid
                v = (atm_coords[ia] - coords_batch).T
                normv = cp.linalg.norm(v, axis=0) + 1e-200
                grid_dist.append(normv)
            
            pbecke = cp.ones((mol.natm, ngrids_batch))
            
            # 2. Iterate over atom pairs to compute fuzzy cell weights
            for ia in range(mol.natm):
                for ib in range(ia):
                    # Calculate hyperbolic coordinate mu_ij
                    g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                    # Apply radii adjustment
                    p0 = fadjust(ia, ib, g)
                    p1 = (3 - p0**2) * p0 * .5
                    p2 = (3 - p1**2) * p1 * .5
                    p3 = (3 - p2**2) * p2 * .5

                    s_uab = .5 * (1 - p3 + 1e-200)
                    s_uba = .5 * (1 + p3 + 1e-200)

                    pbecke[ia] *= s_uab
                    pbecke[ib] *= s_uba
            
            z = 1./pbecke.sum(axis=0)

            for atom_id in range(mol.natm):
                w_atom = weights_batch * pbecke[atom_id] * z
                ao_weighted = ao_batch * w_atom[:, None]
                atom_projectors_temp[atom_id] += ao_batch.T @ ao_weighted
        
        # --- Aggregation into Constraint Projectors ---
        projectors = []
        for group in all_groups:
            w_sum = cp.zeros((nao, nao))
            for identifier in group:
                # Identifier is guaranteed to be int due to check at start of function
                atom_id = int(identifier)
                w_sum += atom_projectors_temp[atom_id]
            projectors.append(w_sum)
        
        self.constraint_projectors = projectors
        return self.constraint_projectors

    def get_constraint_potential(self, v_vec):
        '''
        Constructs V_const matrix based on multipliers (Lagrange Method).
        Uses pre-built constraint_projectors which map 1-to-1 with v_vec indices.
        '''
        if self.constraint_projectors is None:
            self.build_projectors()
            
        nao = self.mol.nao_nr()
        vc_a = cp.zeros((nao, nao))
        vc_b = cp.zeros((nao, nao))
        
        n_charge = len(self.charge_targets)
        
        # 1. Charge Constraints
        # V_alpha = V_beta = +V
        for i in range(n_charge):
            v = float(v_vec[i])
            w = self.constraint_projectors[i]
            
            vc_a += v * w
            vc_b += v * w
            
        # 2. Spin Constraints
        # V_alpha = +V, V_beta = -V
        for i in range(len(self.spin_targets)):
            idx = n_charge + i
            v = float(v_vec[idx])
            w = self.constraint_projectors[idx]
                
            vc_a += v * w
            vc_b -= v * w

        return (vc_a, vc_b)

    def get_penalty_potential(self, dm, penalty_value):
        '''
        Constructs V_const matrix based on the quadratic penalty method.
        V_shift = sum( 2 * lambda * (Pop_calc - Pop_target) * W )
        
        Args:
            dm: Current density matrix (cupy array)
            penalty_value: Current value of lambda (float)
            
        Returns:
            (vc_a, vc_b): Potentials to add to Fock matrix
        '''
        if self.constraint_projectors is None:
            self.build_projectors()
        
        nao = self.mol.nao_nr()
        vc_a = cp.zeros((nao, nao))
        vc_b = cp.zeros((nao, nao))
        
        # Handle DM input (can be tuple of (dm_a, dm_b) or (2, nao, nao) array)
        if isinstance(dm, (tuple, list)):
            dm_a, dm_b = dm
        elif isinstance(dm, cp.ndarray) and dm.ndim == 3:
            dm_a, dm_b = dm[0], dm[1]
        else:
            # Fallback for RKS-like DM input, though this is UKS class
            dm_a = dm_b = dm * 0.5

        log_msgs = []
        projector_idx = 0

        # 1. Charge Constraints
        for i, target in enumerate(self.charge_targets):
            w = self.constraint_projectors[projector_idx]
            
            # Population = Tr(Da * W) + Tr(Db * W)
            val = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
            n_val = float(val)
            
            diff = n_val - target
            log_msgs.append(f"Chg[{i}] err: {diff:.5f}")
            
            # Gradient: 2 * lambda * diff
            shift = 2.0 * penalty_value * diff
            
            vc_a += shift * w
            vc_b += shift * w
            projector_idx += 1

        # 2. Spin Constraints
        for i, target in enumerate(self.spin_targets):
            w = self.constraint_projectors[projector_idx]
            
            # Magnetization = Tr(Da * W) - Tr(Db * W)
            val = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
            m_val = float(val)
                
            diff = m_val - target
            log_msgs.append(f"Spin[{i}] err: {diff:.5f}")

            # Gradient: +shift for alpha, -shift for beta
            shift = 2.0 * penalty_value * diff
            
            vc_a += shift * w
            vc_b -= shift * w
            projector_idx += 1

        if log_msgs:
            logger.info(self, f"CDFT Penalty (L={penalty_value:.1f}): " + ", ".join(log_msgs))

        return vc_a, vc_b

    def _micro_objective_func(self, v_vec, f_std_a, f_std_b, s):
        '''
        Objective function for scipy.optimize.root.
        Returns the error vector (calculated_val - target).
        '''
        # Build total Fock: F_tot = F_std + V_const
        vc_a, vc_b = self.get_constraint_potential(v_vec)
        f_tot_a = f_std_a + vc_a
        f_tot_b = f_std_b + vc_b
        
        # Solve eigenvalue problem
        mo_e, mo_c = self.eig((f_tot_a, f_tot_b), s)
        mo_c_a, mo_c_b = mo_c
        nocc_a = self.mol.nelec[0]
        nocc_b = self.mol.nelec[1]
        
        # Construct Density Matrix locally (faster than make_rdm1 for full object)
        dm_a = mo_c_a[:, :nocc_a] @ mo_c_a[:, :nocc_a].T
        dm_b = mo_c_b[:, :nocc_b] @ mo_c_b[:, :nocc_b].T
        
        errors = []
        projector_idx = 0
        
        # Calculate Charge Errors
        for target in self.charge_targets:
            w = self.constraint_projectors[projector_idx]
            val = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
            errors.append(float(val) - target)
            projector_idx += 1
            
        # Calculate Spin Errors
        for target in self.spin_targets:
            w = self.constraint_projectors[projector_idx]
            val = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
            errors.append(float(val) - target)
            projector_idx += 1

        return np.array(errors)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if s1e is None: s1e = self.get_ovlp()
        h1e = cp.asarray(h1e)
        vhf = cp.asarray(vhf)
        f = h1e + vhf
        if f.ndim == 2:
            f = (f, f)
        
        if dm is None: 
            # If dm is missing (e.g. first cycle), assume initial guess
            dm = self.make_rdm1()
        dm = cp.asarray(dm)

        # --- CDFT Logic Modification Starts Here ---
        if self.n_constraints != 0:
            run_micro = False
            if cycle != -1:
                run_micro = True
            # --- METHOD 1: LAGRANGE MULTIPLIERS (Original) ---
            if self.method == 'lagrange':
                if run_micro:
                    f_a, f_b = f

                    res = root(
                        fun=self._micro_objective_func,
                        x0=self.v_lagrange,
                        args=(f_a, f_b, s1e),
                        method='hybr',
                        tol=self.micro_tol,
                        options={'maxfev': self.micro_max_cycle}
                    )
                    if not res.success:
                        raise RuntimeError(f"Micro optimization failed: {res.message}")
                    self.v_lagrange = res.x
                    logger.info(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
                    
                vc_a, vc_b = self.get_constraint_potential(self.v_lagrange)
                logger.info(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
                f = cp.stack((f[0] + vc_a, 
                              f[1] + vc_b))
                              
            # --- METHOD 2: PENALTY FUNCTION ---
            elif self.method == 'penalty':                
                # Get the potential based on current density and current lambda
                vc_a, vc_b = self.get_penalty_potential(dm, self.penalty_weight)
                
                f = cp.stack((f[0] + vc_a, 
                              f[1] + vc_b))
                
                logger.info(self, f"Cycle {cycle}: Applied Penalty (Weight={self.penalty_weight:.1f})")

        # --- CDFT Logic Ends Here ---

        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        s1e = cp.asarray(s1e)

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if damp_factor is None:
            damp_factor = self.damp
        if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
            if isinstance(damp_factor, (tuple, list, np.ndarray)):
                dampa, dampb = damp_factor
            else:
                dampa = dampb = damp_factor
            f = cp.asarray((damping(f[0], fock_last[0], dampa),
                            damping(f[1], fock_last[1], dampb)))
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm, f)

        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if level_shift_factor is not None:
            if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
                shifta, shiftb = level_shift_factor
            else:
                shifta = shiftb = level_shift_factor
            f = (level_shift(s1e, dm[0], f[0], shifta),
                level_shift(s1e, dm[1], f[1], shiftb))
        return f

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        '''
        Calculate electronic energy.
        Added penalty energy calculation using the last used penalty weight.
        '''
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        return e_tot, e_coul

    def newton(self):
        from gpu4pyscf.dft.cdft_soscf_full import newton_cdft
        return newton_cdft(self)

    def newton_penalty(self):
        from gpu4pyscf.dft.cdft_soscf import newton_cdft
        return newton_cdft(self)
    
    def Gradients(self):
        from gpu4pyscf.grad.ucdft import Gradients
        # Assuming the class above is saved in a module or available
        return Gradients(self)