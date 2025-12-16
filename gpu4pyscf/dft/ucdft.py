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
from gpu4pyscf import dft, lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import radi


def make_constant_schedule(weight):
    """
    Returns a function that always returns the same weight.
    Used for backward compatibility or fixed penalty.
    """
    return lambda cycle: float(weight)


def make_ramp_schedule(start_weight=1.0, end_weight=200.0, ramp_width=30, start_step=10):
    """
    Returns a function that linearly ramps the penalty weight.
    
    Phases:
    1. Hold: cycle < start_step -> constant start_weight
    2. Ramp: start_step <= cycle < start_step + ramp_width -> linear increase
    3. End:  cycle >= start_step + ramp_width -> constant end_weight
    
    Args:
        start_weight: Initial weight value.
        end_weight: Final target weight value.
        ramp_width: How many cycles the ramping phase takes.
        start_step: The cycle number when ramping begins (delay).
    """
    def scheduler(cycle):
        # Phase 1: Initial Hold / Delay
        if cycle < start_step:
            return start_weight
            
        # Phase 3: Reached target
        # Note: The total cycles to reach max is start_step + ramp_width
        if cycle >= (start_step + ramp_width):
            return end_weight
        
        # Phase 2: Linear Ramp
        # Shift the cycle so that 0 corresponds to start_step
        effective_cycle = cycle - start_step
        progress = effective_cycle / float(ramp_width)
        
        return start_weight + (end_weight - start_weight) * progress
    
    return scheduler


def make_power_schedule(start_weight=1.0, end_weight=100.0, ramp_width=30, start_step=1, power=3):
    """
    Returns a function that ramps the penalty weight using a power law with a delay.
    
    Phases:
    1. Hold: cycle < start_step -> constant start_weight
    2. Ramp: start_step <= cycle < start_step + ramp_width -> power law increase
    3. End:  cycle >= start_step + ramp_width -> constant end_weight
    
    Args:
        start_weight: Initial weight value.
        end_weight: Final target weight value.
        ramp_width: How many cycles the ramping phase takes.
        start_step: The cycle number when ramping begins (delay).
        power: Exponent for the curve (3 is recommended for smooth start).
    """
    def scheduler(cycle):
        # Phase 1: Initial Hold / Delay
        if cycle < start_step:
            return start_weight
            
        # Phase 3: Reached target
        if cycle >= (start_step + ramp_width):
            return end_weight
        
        # Phase 2: Power Law Ramp
        # Shift the cycle so that 0 corresponds to start_step
        effective_cycle = cycle - start_step
        
        # Calculate ratio (0.0 to 1.0) based on the ramp width
        ratio = effective_cycle / float(ramp_width)
        
        # Apply power law
        progress = ratio ** power
        
        return start_weight + (end_weight - start_weight) * progress
    
    return scheduler

class CDFT_UKS(dft.UKS):
    '''
    Constrained DFT implementation for unrestricted Kohn-Sham.
    Implements Voronoi-based spatial charge/spin constraints.
    Supports two modes: 'lagrange' (Lagrange Multipliers) and 'penalty' (Quadratic Penalty).
    '''

    def __init__(self, mol, charge_constraints=None, spin_constraints=None, 
                 method='lagrange', penalty_weight=500.0):
        super().__init__(mol)
        
        # [Requirement 2] Concise Constraints format: 
        # [ [atom_indices_or_groups], [targets] ]
        
        self.charge_groups, self.charge_targets = self._normalize_constraints(charge_constraints)
        self.spin_groups, self.spin_targets = self._normalize_constraints(spin_constraints)
        
        self.n_constraints = len(self.charge_targets) + len(self.spin_targets)
        
        # Initial guess for Lagrange multipliers V
        self.v_lagrange = np.zeros(self.n_constraints) + 0.01
        
        # Microiteration parameters
        self.micro_tol = 1e-4
        self.micro_max_cycle = 50
        
        # Cache for constraint matrices on GPU
        self.atom_projectors = None

        # Configuration for methods
        self.method = method.lower() # 'lagrange' or 'penalty'
        
        # Handle penalty weight scheduling
        # If the user passes a number, convert it to a constant scheduler.
        # If the user passes a callable (function), use it directly.
        if callable(penalty_weight):
            self.penalty_scheduler = penalty_weight
        else:
            self.penalty_scheduler = make_constant_schedule(penalty_weight)
            
        # Store the last used weight for energy calculation
        self.last_penalty_weight = 0.0

    def _normalize_constraints(self, constraints):
        '''
        Helper: Normalize user input into internal list-of-lists format.
        Input: [ [0, 1], [6.5, 7.5] ] 
        Output: ( [[0], [1]], [6.5, 7.5] )
        '''
        if not constraints:
            return [], []
        
        input_groups = constraints[0]
        targets = constraints[1]
        
        if len(input_groups) != len(targets):
            raise ValueError("CDFT Error: The number of atom groups must match the number of targets.")

        normalized_groups = []
        for item in input_groups:
            # If user provided a single int (e.g. 0), wrap it in a list [0]
            if isinstance(item, (int, np.integer)):
                normalized_groups.append([int(item)])
            else:
                # Assuming it's already a list/tuple for a group of atoms
                normalized_groups.append(list(item))
                
        return normalized_groups, targets

    # def build_atom_projectors(self):
    #     r'''
    #     Constructs constraint weight matrices 'w' on GPU using Voronoi division.
        
    #     Method:
    #     1. Generate DFT integration grid.
    #     2. For each grid point, assign it to the nearest atom (Voronoi cell).
    #     3. Integrate AO products over these regions: 
    #        W_A_munu = Sum_k weight_k * mask_A(r_k) * phi_mu(r_k) * phi_nu(r_k)
    #     '''
    #     if self.atom_projectors is not None:
    #         return self.atom_projectors

    #     mol = self.mol
    #     nao = mol.nao_nr()
    #     atom_coords = cp.asarray(mol.atom_coords())

    #     # Initialize projectors for all atoms (List of CuPy arrays)
    #     projectors = [cp.zeros((nao, nao)) for _ in range(mol.natm)]

    #     # Initialize Grids if not present
    #     if self.grids.coords is None:
    #         self.grids.build()

    #     # Access low-level grid data
    #     coords_all = self.grids.coords
    #     weights_all = self.grids.weights

    #     # Use NumInt for AO evaluation
    #     ni = self._numint

    #     blksize = 4000
    #     n_grids = weights_all.shape[0]

    #     logger.debug(self, f"CDFT: Building Voronoi projectors on GPU using {n_grids} grid points...")

    #     # TODO: using block loop in DFT
    #     # TODO: Becke grid should be used.
    #     for p0, p1 in pyscf_lib.prange(0, n_grids, blksize):
    #         # 1. Load batch
    #         coords_batch = cp.asarray(coords_all[p0:p1])
    #         weights_batch = cp.asarray(weights_all[p0:p1])

    #         # 2. Evaluate AO on grid: (N_grid, N_ao)
    #         ao_batch = ni.eval_ao(mol, coords_batch, deriv=0)

    #         # 3. Voronoi Assignment (Find nearest atom for each grid point)
    #         # Calculate distance squared: |r - R|^2 = r^2 + R^2 - 2rR
    #         r_sq = cp.sum(coords_batch**2, axis=1)[:, None] # (N_grid, 1)
    #         R_sq = cp.sum(atom_coords**2, axis=1)[None, :]  # (1, N_atom)
    #         interaction = coords_batch @ atom_coords.T      # (N_grid, N_atom)
    #         dist_sq = r_sq + R_sq - 2 * interaction

    #         # Index of the nearest atom for each grid point
    #         nearest_atom_idxs = cp.argmin(dist_sq, axis=1)

    #         # Loop only over atoms present in this batch might be an optim, 
    #         # but looping all atoms is safer code-wise.
    #         for atom_id in range(mol.natm):
    #             # Create boolean mask: 1 if grid point belongs to atom_id
    #             mask = (nearest_atom_idxs == atom_id)

    #             # If no grid points in this batch belong to this atom, skip
    #             if not cp.any(mask):
    #                 continue

    #             # Effective weights for this atom
    #             w_eff = weights_batch * mask

    #             ao_weighted = ao_batch * w_eff[:, None]
    #             projectors[atom_id] += ao_batch.T @ ao_weighted

    #     self.atom_projectors = projectors
    #     return self.atom_projectors

    def build_atom_projectors(self):
        r'''
        Constructs constraint weight matrices 'w' on GPU using Becke partitioning.
        
        This implementation strictly follows the logic from the reference 'gen_grid_partition'
        function (from 'grids_response_cc'), preserving the exact mathematical steps,
        loop structures, and data dimensions.
        '''
        mol = self.mol
        nao = mol.nao_nr()
        
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
        projectors = [cp.zeros((nao, nao)) for _ in range(mol.natm)]

        # Ensure grids are built
        if self.grids.coords is None:
            self.grids.build()
            
        coords_all = self.grids.coords
        weights_all = self.grids.weights
        ni = self._numint
        
        # Batch processing
        blksize = 4000
        n_grids_total = weights_all.shape[0]
        
        logger.info(self, f"CDFT: Building Becke projectors (Ref-logic) on GPU using {n_grids_total} grid points...")
        
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
                projectors[atom_id] += ao_batch.T @ ao_weighted
        
        self.atom_projectors = projectors
        return self.atom_projectors

    def get_constraint_potential(self, v_vec):
        '''
        Constructs V_const matrix based on multipliers (Lagrange Method).
        '''
        if self.atom_projectors is None:
            self.build_atom_projectors()
            
        nao = self.mol.nao_nr()
        vc_a = cp.zeros((nao, nao))
        vc_b = cp.zeros((nao, nao))
        
        idx = 0
        log_msgs = []
        # 1. Charge Constraints (V_alpha = V_beta = +V)
        for i, atom_group in enumerate(self.charge_groups):
            target = self.charge_targets[i]
            v = float(v_vec[idx])
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                vc_a += v * w
                vc_b += v * w
            # diff = n_val - target
            # log_msgs.append(f"Chg[{i}] err: {diff:.5f}")
            idx += 1
            
        # 2. Spin Constraints (V_alpha = +V, V_beta = -V)
        for i, atom_group in enumerate(self.spin_groups):
            target = self.spin_groups[i]
            v = float(v_vec[idx])
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                vc_a += v * w
                vc_b -= v * w
            idx += 1
        # if log_msgs:
        #     print(f"CDFT Penalty (L={penalty_value:.1f}): " + ", ".join(log_msgs))
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
        if self.atom_projectors is None:
            self.build_atom_projectors()
        
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

        # 1. Charge Constraints
        for i, atom_group in enumerate(self.charge_groups):
            target = self.charge_targets[i]
            
            # Calculate current population
            n_val = 0.0
            w_sum = cp.zeros((nao, nao))
            
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                val = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
                n_val += float(val)
                w_sum += w
            
            diff = n_val - target
            log_msgs.append(f"Chg[{i}] err: {diff:.5f}")
            
            # Gradient: 2 * lambda * diff
            shift = 2.0 * penalty_value * diff
            
            vc_a += shift * w_sum
            vc_b += shift * w_sum

        # 2. Spin Constraints
        for i, atom_group in enumerate(self.spin_groups):
            target = self.spin_targets[i]
            
            m_val = 0.0
            w_sum = cp.zeros((nao, nao))
            
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                val = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
                m_val += float(val)
                w_sum += w
                
            diff = m_val - target
            log_msgs.append(f"Spin[{i}] err: {diff:.5f}")

            # Gradient: 2 * lambda * diff (alpha), -2 * lambda * diff (beta)
            shift = 2.0 * penalty_value * diff
            
            vc_a += shift * w_sum
            vc_b -= shift * w_sum

        if log_msgs:
            logger.info(self, f"CDFT Penalty (L={penalty_value:.1f}): " + ", ".join(log_msgs))

        return vc_a, vc_b

    def _micro_objective_func(self, v_vec, f_std_a, f_std_b, s):
        '''
        Objective function for scipy.optimize.root.
        '''
        # Build total Fock: F_tot = F_std + V_const
        vc_a, vc_b = self.get_constraint_potential(v_vec)
        f_tot_a = f_std_a + vc_a
        f_tot_b = f_std_b + vc_b
        
        # Solve eigenvalue problem
        mo_e, mo_c = self.eig((f_tot_a, f_tot_b), s)
        mo_e_a, mo_e_b = mo_e
        mo_c_a, mo_c_b = mo_c
        nocc_a = self.mol.nelec[0]
        nocc_b = self.mol.nelec[1]
        mo_occ_a = cp.zeros((self.mol.nao))
        mo_occ_b = cp.zeros((self.mol.nao))
        mo_occ_a[:nocc_a] = 1.0
        mo_occ_b[:nocc_b] = 1.0
        dm_a, dm_b = self.make_rdm1( 
            (mo_c_a, mo_c_b), 
            (cp.asarray(mo_occ_a), cp.asarray(mo_occ_b)) )
        
        errors = []
        
        # Calculate Charge Errors
        for i, atom_group in enumerate(self.charge_groups):
            target = self.charge_targets[i]
            n_val = 0.0
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                val = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
                n_val += float(val)
            errors.append(n_val - target)
            
        # Calculate Spin Errors
        for i, atom_group in enumerate(self.spin_groups):
            target = self.spin_targets[i]
            m_val = 0.0
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                val = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
                m_val += float(val)
            errors.append(m_val - target)
        
        # logger.info(self, f"errors: {errors}   val: {n_val}  v_vec: {v_vec}")
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
                        pass
                    self.v_lagrange = res.x
                    logger.info(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
                    
                vc_a, vc_b = self.get_constraint_potential(self.v_lagrange)
                logger.info(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
                f = cp.stack((f[0] + vc_a, 
                              f[1] + vc_b))
                              
            # --- METHOD 2: PENALTY FUNCTION ---
            elif self.method == 'penalty':
                # Determine current penalty weight (lambda) based on the scheduler
                if run_micro:
                    current_lambda = self.penalty_scheduler(cycle)
                    self.last_penalty_weight = current_lambda
                elif self.last_penalty_weight == 0.0:
                    current_lambda = self.penalty_scheduler(0) # Assume initial lambda from scheduler
                    # TODO: This may also be set for soscf
                    self.last_penalty_weight = current_lambda
                    logger.debug(self, f"Cycle {cycle}: Auto-initialized Penalty Weight to {current_lambda}")
                
                # Get the potential based on current density and current lambda
                vc_a, vc_b = self.get_penalty_potential(dm, self.last_penalty_weight)
                
                f = cp.stack((f[0] + vc_a, 
                              f[1] + vc_b))
                
                logger.info(self, f"Cycle {cycle}: Applied Penalty (Weight={self.last_penalty_weight:.1f})")

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
        # 1. Get standard DFT energy (E_KS)
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        return e_tot, e_coul

    def newton_fock_only(self):
        from gpu4pyscf.scf.soscf import newton
        return newton(self)

    def newton(self):
        from gpu4pyscf.dft.cdft_soscf_full import newton_cdft
        return newton_cdft(self)

    def newton_penalty(self):
        from gpu4pyscf.dft.cdft_soscf import newton_cdft
        return newton_cdft(self)