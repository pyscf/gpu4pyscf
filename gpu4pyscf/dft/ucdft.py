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
from pyscf import scf, lib as pyscf_lib
from gpu4pyscf import dft, lib

class CDFT_UKS(dft.UKS):
    '''
    GPU-accelerated Constrained DFT implementation using GPU4PySCF.
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
        
        # Initial guess for Lagrange multipliers V (CPU array for scipy)
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
        if not self.grids.coords:
            self.grids.build()
            
        # Access low-level grid data
        coords_all = self.grids.coords
        weights_all = self.grids.weights
        
        # Use NumInt for AO evaluation
        ni = self._numint
        
        blksize = 4000
        n_grids = weights_all.shape[0]
        
        print(f"CDFT: Building Voronoi projectors on GPU using {n_grids} grid points...")
        
        for p0, p1 in pyscf_lib.prange(0, n_grids, blksize):
            # 1. Load batch to GPU
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
            
            # 4. Accumulate W matrices for each atom
            # Optim: Pre-calculate (weights * ao) to save multiplications
            # But weights depend on the atom mask.
            
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
                
                # Integration: W_mn = Sum_k w_k * phi_m(k) * phi_n(k)
                # Matrix algebra: W = AO.T @ diag(w) @ AO
                # Efficiently: W = AO.T @ (AO * w[:, None])
                
                ao_weighted = ao_batch * w_eff[:, None]
                projectors[atom_id] += ao_batch.T @ ao_weighted
        
        self.atom_projectors = projectors
        return self.atom_projectors

    def get_constraint_potential_gpu(self, v_vec_cpu):
        '''
        Constructs V_const matrix on GPU based on CPU multipliers.
        '''
        if self.atom_projectors is None:
            self.build_atom_projectors()
            
        nao = self.mol.nao_nr()
        vc_a = cp.zeros((nao, nao))
        vc_b = cp.zeros((nao, nao))
        
        idx = 0
        
        # 1. Charge Constraints (V_alpha = V_beta = +V)
        for i, atom_group in enumerate(self.charge_groups):
            v = float(v_vec_cpu[idx])
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                vc_a += v * w
                vc_b += v * w
            idx += 1
            
        # 2. Spin Constraints (V_alpha = +V, V_beta = -V)
        for i, atom_group in enumerate(self.spin_groups):
            v = float(v_vec_cpu[idx])
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                vc_a += v * w
                vc_b -= v * w
            idx += 1
            
        return (vc_a, vc_b)

    def _micro_objective_func(self, v_vec_cpu, f_std_a_gpu, f_std_b_gpu, s_gpu):
        '''
        Objective function for scipy.optimize.root.
        Input: v_vec (CPU numpy array)
        Output: errors (CPU numpy array)
        '''
        # Build total Fock on GPU: F_tot = F_std + V_const
        vc_a, vc_b = self.get_constraint_potential_gpu(v_vec_cpu)
        f_tot_a = f_std_a_gpu + vc_a
        f_tot_b = f_std_b_gpu + vc_b
        
        # Solve eigenvalue problem on GPU
        # GPU4PySCF's eig solver handles S implicitly via generalized diagonalization
        mo_e_a, mo_c_a = self.eig(f_tot_a, s_gpu)
        mo_e_b, mo_c_b = self.eig(f_tot_b, s_gpu)
        
        # Get occupation and density on GPU
        mo_occ_a, mo_occ_b = self.get_occ((mo_e_a, mo_e_b), self.mo_coeff)
        dm_a = self.make_rdm1( (mo_c_a, mo_c_b), (mo_occ_a, mo_occ_b) )[0]
        dm_b = self.make_rdm1( (mo_c_a, mo_c_b), (mo_occ_a, mo_occ_b) )[1]
        
        errors = []
        
        # Calculate Charge Errors
        for i, atom_group in enumerate(self.charge_groups):
            target = self.charge_targets[i]
            n_val = 0.0
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                # Tr(P * W) is the integrated population
                val_gpu = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
                n_val += float(val_gpu)
            errors.append(n_val - target)
            
        # Calculate Spin Errors
        for i, atom_group in enumerate(self.spin_groups):
            target = self.spin_targets[i]
            m_val = 0.0
            for atom_id in atom_group:
                w = self.atom_projectors[atom_id]
                val_gpu = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
                m_val += float(val_gpu)
            errors.append(m_val - target)
            
        return np.array(errors)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        '''
        Override get_fock to inject the CDFT microiterations.
        Most matrices here (h1e, vhf, dm) are CuPy arrays (GPU).
        '''
        # 1. Standard Fock Construction (GPU)
        f_std = super().get_fock(h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle,
                                 level_shift_factor, damp_factor)
        
        if self.n_constraints == 0:
            return f_std

        # 2. Microiterations
        s_gpu = self.get_ovlp()
        
        # Handle f_std format (ensure tuple for UKS)
        if isinstance(f_std, tuple):
            f_std_a, f_std_b = f_std
        elif isinstance(f_std, cp.ndarray) and f_std.ndim == 3:
            f_std_a, f_std_b = f_std[0], f_std[1]
        else:
            f_std_a = f_std_b = f_std

        # Run optimization using scipy (CPU) calling objective (GPU)
        res = root(
            fun=self._micro_objective_func,
            x0=self.v_lagrange, # CPU array
            args=(f_std_a, f_std_b, s_gpu), # GPU arrays passed as constant args
            method='hybr',
            tol=self.micro_tol,
            options={'maxfev': self.micro_max_cycle}
        )
        
        # Update stored Lagrange multipliers (CPU)
        self.v_lagrange = res.x
        
        # 3. Add optimized V_const to Fock (GPU)
        vc_a, vc_b = self.get_constraint_potential_gpu(self.v_lagrange)
        
        return (f_std_a + vc_a, f_std_b + vc_b)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        '''
        Calculate electronic energy. 
        '''
        return super().energy_elec(dm, h1e, vhf)


if __name__ == '__main__':
    print("Initializing GPU4PySCF...")
    
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 3.0' 
    mol.basis = 'def2-svp'
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.build()

    # Define Constraints (List of Lists)
    # Constrain Left N (atom 0) to 6.5 electrons
    # Constrain Right N (atom 1) to 7.5 electrons
    charge_constraints = [ [0, 1], [6.5, 7.5] ]
    
    # Initialize CDFT on GPU
    mf = CDFT_UKS(mol, charge_constraints=charge_constraints)
    mf.xc = 'b3lyp'
    
    print(">>> Starting Voronoi-CDFT Calculation (GPU)...")
    mf.kernel()
    
    print("\n>>> Analysis of Results")
    print(f"Converged Lagrange Multipliers V (CPU): {mf.v_lagrange}")
    
    # Verification
    dm_gpu = mf.make_rdm1()
    projs_gpu = mf.build_atom_projectors()
    
    # Calculate populations on GPU
    n_left_gpu = cp.trace(dm_gpu[0] @ projs_gpu[0]) + cp.trace(dm_gpu[1] @ projs_gpu[0])
    n_right_gpu = cp.trace(dm_gpu[0] @ projs_gpu[1]) + cp.trace(dm_gpu[1] @ projs_gpu[1])
    
    print(f"Target N (Left):  {charge_constraints[1][0]}")
    print(f"Result N (Left):  {float(n_left_gpu):.6f}")
    print(f"Target N (Right): {charge_constraints[1][1]}")
    print(f"Result N (Right): {float(n_right_gpu):.6f}")