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
from gpu4pyscf import dft, lib
from gpu4pyscf.lib import logger

class CDFT_UKS(dft.UKS):
    '''
    Constrained DFT implementation for unrestricted Kohn-Sham.
    Implements Voronoi-based spatial charge/spin constraints.
    '''

    def __init__(self, mol, charge_constraints=None, spin_constraints=None):
        super().__init__(mol)
        
        # [Requirement 2] Concise Constraints format: 
        # [ [atom_indices_or_groups], [targets] ]
        # Example: [ [0, 1], [6.5, 7.5] ] or [ [0, [1,3] ], [6.5, 6.5] ]
        
        self.charge_groups, self.charge_targets = self._normalize_constraints(charge_constraints)
        self.spin_groups, self.spin_targets = self._normalize_constraints(spin_constraints)
        
        self.n_constraints = len(self.charge_targets) + len(self.spin_targets)
        
        # Initial guess for Lagrange multipliers V
        self.v_lagrange = np.zeros(self.n_constraints)
        
        # Microiteration parameters
        self.micro_tol = 1e-4
        self.micro_max_cycle = 50
        
        # Cache for constraint matrices on GPU
        self.atom_projectors = None

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

    def build_atom_projectors(self):
        r'''
        Constructs constraint weight matrices 'w' on GPU using Voronoi division.
        
        Method:
        1. Generate DFT integration grid.
        2. For each grid point, assign it to the nearest atom (Voronoi cell).
        3. Integrate AO products over these regions: 
           W_A_munu = Sum_k weight_k * mask_A(r_k) * phi_mu(r_k) * phi_nu(r_k)
        '''
        if self.atom_projectors is not None:
            return self.atom_projectors

        mol = self.mol
        nao = mol.nao_nr()
        atom_coords = cp.asarray(mol.atom_coords())
        
        # Initialize projectors for all atoms (List of CuPy arrays)
        projectors = [cp.zeros((nao, nao)) for _ in range(mol.natm)]
        
        # Initialize Grids if not present
        if self.grids.coords is None:
            self.grids.build()
            
        # Access low-level grid data
        coords_all = self.grids.coords
        weights_all = self.grids.weights
        
        # Use NumInt for AO evaluation
        ni = self._numint
        
        blksize = 4000
        n_grids = weights_all.shape[0]
        
        logger.debug(self, f"CDFT: Building Voronoi projectors on GPU using {n_grids} grid points...")
        
        # TODO: using block loop in DFT
        # TODO: Becke grid should be used.
        for p0, p1 in pyscf_lib.prange(0, n_grids, blksize):
            # 1. Load batch
            coords_batch = cp.asarray(coords_all[p0:p1])
            weights_batch = cp.asarray(weights_all[p0:p1])
            
            # 2. Evaluate AO on grid: (N_grid, N_ao)
            ao_batch = ni.eval_ao(mol, coords_batch, deriv=0)
            
            # 3. Voronoi Assignment (Find nearest atom for each grid point)
            # Calculate distance squared: |r - R|^2 = r^2 + R^2 - 2rR
            r_sq = cp.sum(coords_batch**2, axis=1)[:, None] # (N_grid, 1)
            R_sq = cp.sum(atom_coords**2, axis=1)[None, :]  # (1, N_atom)
            interaction = coords_batch @ atom_coords.T      # (N_grid, N_atom)
            dist_sq = r_sq + R_sq - 2 * interaction
            
            # Index of the nearest atom for each grid point
            nearest_atom_idxs = cp.argmin(dist_sq, axis=1)
            
            # Loop only over atoms present in this batch might be an optim, 
            # but looping all atoms is safer code-wise.
            for atom_id in range(mol.natm):
                # Create boolean mask: 1 if grid point belongs to atom_id
                mask = (nearest_atom_idxs == atom_id)
                
                # If no grid points in this batch belong to this atom, skip
                if not cp.any(mask):
                    continue
                
                # Effective weights for this atom
                w_eff = weights_batch * mask
                
                ao_weighted = ao_batch * w_eff[:, None]
                projectors[atom_id] += ao_batch.T @ ao_weighted
        
        self.atom_projectors = projectors
        return self.atom_projectors

    def get_constraint_potential(self, v_vec):
        '''
        Constructs V_const matrix based on multipliers.
        '''
        if self.atom_projectors is None:
            self.build_atom_projectors()
            
        nao = self.mol.nao_nr()
        vc_a = cp.zeros((nao, nao))
        vc_b = cp.zeros((nao, nao))
        
        idx = 0
        
        # 1. Charge Constraints (V_alpha = V_beta = +V)
        for i, atom_group in enumerate(self.charge_groups):
            v = float(v_vec[idx])
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                vc_a += v * w
                vc_b += v * w
            idx += 1
            
        # 2. Spin Constraints (V_alpha = +V, V_beta = -V)
        for i, atom_group in enumerate(self.spin_groups):
            v = float(v_vec[idx])
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                vc_a += v * w
                vc_b -= v * w
            idx += 1
            
        return (vc_a, vc_b)

    def _micro_objective_func(self, v_vec, f_std_a, f_std_b, s):
        '''
        Objective function for scipy.optimize.root.
        Input: v_vec
        Output: errors
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
                # Tr(P * W) is the integrated population
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
        
        logger.debug(self, f"errors: {errors}   val: {n_val}  v_vec: {v_vec}")
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
        
        # Begin to change
        if self.n_constraints != 0:
            run_micro = False
            if cycle != -1:
                run_micro = True
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
                    # log warning
                    pass
                self.v_lagrange = res.x
                logger.debug(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
                
            else:
                pass

            vc_a, vc_b = self.get_constraint_potential(self.v_lagrange)

            f = cp.stack((f[0] + vc_a, 
                          f[1] + vc_b))
        # End to change

        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if dm is None: dm = self.make_rdm1()
        s1e = cp.asarray(s1e)
        dm = cp.asarray(dm)

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

    # def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
    #              diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
    #              fock_last=None):
    #     '''
    #     Override get_fock to inject the CDFT microiterations.
    #     Most matrices here (h1e, vhf, dm) are cupy arrays.
    #     '''
    #     # Standard Fock Construction
    #     # TODO: Check if DIIS should be changed place
    #     f_std = super().get_fock(h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle,
    #                              level_shift_factor, damp_factor, fock_last)
    #     print("cycle:", cycle)
    #     print("f_std:", f_std)
        
    #     if self.n_constraints == 0:
    #         return f_std

    #     run_micro = False
    #     if cycle != -1:
    #         run_micro = True
        
    #     if run_micro:
    #         s = self.get_ovlp()
    #         f_std_a, f_std_b = f_std

    #         res = root(
    #             fun=self._micro_objective_func,
    #             x0=self.v_lagrange,
    #             args=(f_std_a, f_std_b, s),
    #             method='hybr',
    #             tol=self.micro_tol,
    #             options={'maxfev': self.micro_max_cycle}
    #         )
    #         if not res.success:
    #             # log warning
    #             pass
    #         self.v_lagrange = res.x
    #         print(f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
            
    #     else:
    #         pass

    #     vc_a, vc_b = self.get_constraint_potential(self.v_lagrange)
    #     print("vc_a:", vc_a)
    #     print("vc_b:", vc_b)

    #     return (f_std[0] + vc_a, f_std[1] + vc_b)

def energy_elec(self, dm=None, h1e=None, vhf=None):
        '''
        Calculate electronic energy, properly defining the CDFT Lagrangian W.
        
        Ref: Eq 16 in Chem. Rev. 2012, 112, 321-370 .
        W = E_KS + Sum[ V_k * (N_calc_k - N_target_k) ]
          = E_KS + Tr(P * V_const) - Sum(V_k * N_target_k)
        
        Returns:
            Total Electronic Energy (Lagrangian W), Coulomb Energy
        '''
        # 1. Get standard DFT energy (E_KS)
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        
        # 2. Calculate Constraint Potential Energy: Tr(P * V_const)
        if self.atom_projectors is None:
            self.build_atom_projectors()
            
        vc_a, vc_b = self.get_constraint_potential(self.v_lagrange)
        
        if isinstance(dm, (tuple, list)) or (isinstance(dm, cp.ndarray) and dm.ndim == 3):
            dm_a, dm_b = dm[0], dm[1]
        else:
            dm_a = dm_b = dm * 0.5 

        # Term 1: V * N_calc
        e_interaction = cp.trace(dm_a @ vc_a) + cp.trace(dm_b @ vc_b)
        e_interaction = float(e_interaction)
        
        # 3. Calculate Constant Shift: Sum(V_k * N_target_k)
        # This term makes the energy correspond to the Lagrangian W
        e_shift = 0.0
        idx = 0
        
        # Charge Targets
        for target in self.charge_targets:
            v = self.v_lagrange[idx]
            e_shift += v * target
            idx += 1
            
        # Spin Targets
        for target in self.spin_targets:
            v = self.v_lagrange[idx]
            e_shift += v * target
            idx += 1
            
        # Final Lagrangian W = E_KS + (V * N_calc) - (V * N_target)
        # e_tot += (e_interaction - e_shift)
        # e_tot += e_interaction
        
        return e_tot, e_coul

