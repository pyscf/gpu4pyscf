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

import numpy as np
import cupy as cp
from gpu4pyscf import scf, dft
from pyscf.geomopt import geometric_solver

class ConicalIntersectionOptimizer:
    """
    Implements the direct method for locating the lowest energy point on a potential surface crossing,
    as described in Chemical Physics Letters 223 (1994) 269-274.
    
    This version uses pyscf.geomopt.geometric_solver as the optimizer.
    
    Args:
        td : A time-dependent HF or DFT object.
        states (tuple): A tuple of two integers specifying the electronic states to be coupled (e.g., (0, 1) for S0/S1).
        crossing_type (str): Type of intersection, 'n-2' for conical intersection (same spin)
                             or 'n-1' for different spin multiplicity crossing.
    """

    def __init__(self, td, states=(0, 1), crossing_type='n-2'):
        if len(states) != 2:
            raise ValueError("`states` must be a tuple of two state indices.")

        self.td = td
        self.mf = td._scf
        self.mol = self.mf.mol
        self.states = states
        self.crossing_type = crossing_type
        
        # --- Caching mechanism to avoid re-computing ---
        # One optimization step calls for energy first, then gradient.
        # We run the full calculation once and store the results.
        self._last_geom = None
        self._last_energy = None
        self._last_grad = None

    class _GradScanner:
        """Internal gradient wrapper to be returned by _SolverWrapper.nuc_grad_method()."""
        def __init__(self, solver_wrapper):
            self.solver_wrapper = solver_wrapper

        def kernel(self, *args, **kwargs):
            # This returns the gradient that was cached by the solver wrapper's __call__ method.
            return self.solver_wrapper._last_grad

    class _SolverWrapper:
        """Internal solver wrapper that mimics a PySCF method object for geometric_solver."""
        def __init__(self, ci_optimizer):
            self.ci_optimizer = ci_optimizer
            self.mol = self.ci_optimizer.mol
            self._last_grad = None # Cached gradient
            
        def __call__(self, *args, **kwargs):
            # The optimizer calls this to get the energy.
            # We run the full calculation here and cache both energy and gradient.
            energy, grad = self.ci_optimizer.get_eff_energy_and_gradient()
            self._last_grad = grad
            return energy

        def nuc_grad_method(self):
            # The optimizer calls this to get the gradient object.
            return self.ci_optimizer._GradScanner(self)


    def get_eff_energy_and_gradient(self):
        """
        Calculates the effective gradient for the CI optimization and the energy for the optimizer.
        This is the core implementation of the paper's algorithm.
        """
        # Check if the geometry has changed. If not, return cached results.
        current_geom = self.mol.atom_coords(unit='bohr')
        if self._last_geom is not None and np.array_equal(current_geom, self._last_geom):
            return self._last_energy, self._last_grad

        print("\n--- CI Optimizer Step (using pyscf.geomopt.geometric_solver) ---")

        self.mf.kernel()
        self.td.kernel()
        e_states = self.td.e
        
        E1 = e_states[self.states[0]]
        E2 = e_states[self.states[1]]
        
        print(f"  Energies: E1={E1:.6f}, E2={E2:.6f}")
        print(f"  Energy Gap (E2-E1): {E2-E1:.6f} Ha")

        # 1. Calculate analytical gradients for both states
        grad_method = self.td.nuc_grad_method()
        g1 = grad_method.kernel(state=self.states[0])
        g2 = grad_method.kernel(state=self.states[1])

        # 2. Define the branching space vectors x1 and x2
        # x1 is the gradient difference vector
        x1 = g1 - g2
        x1_norm_val = np.linalg.norm(x1)
        x1_norm_vec = x1 / x1_norm_val if x1_norm_val > 1e-9 else np.zeros_like(x1)

        # For minimum energy crossing point, x2 is the non-adiabatic coupling vector
        if self.crossing_type == 'n-2':
            nac_method = self.td.nac_method()
            nac_method.states=(self.states[0], self.states[1])
            x2 = nac_method.kernel()[1]
            x2_norm_val = np.linalg.norm(x2)
            x2_norm_vec = x2 / x2_norm_val if x2_norm_val > 1e-9 else np.zeros_like(x2)
            g_on_plane = project_on_plane_lstsq(g2, x1_norm_vec, x2_norm_vec)
            g_proj = g2 - g_on_plane
        
        elif self.crossing_type == 'n-1':
            # For crossings of different spin, x2 is zero
            raise NotImplementedError("n-1 crossing type not implemented yet.")
            g_proj = 0

        # 3. Calculate components of the effective gradient
        
        # f: component driving states to degeneracy
        f = (E2 - E1) * x1_norm_vec
        # 4. Total effective gradien
        g_bar = g_proj + f

        print(f"  ||Projected Grad (g)||: {np.linalg.norm(g_proj):.6f}")
        print(f"  ||Degeneracy Grad (f)||: {np.linalg.norm(f):.6f}")
        print(f"  ||Total Effective Grad||: {np.linalg.norm(g_bar):.6f}")
        print("----------------------------------------------------------------")
        
        # The optimizer minimizes a single energy value. We provide the average.
        energy_for_optimizer = (E1 + E2) / 2.0
        
        # Cache the results for the current geometry
        self._last_geom = np.copy(current_geom)
        self._last_energy = energy_for_optimizer
        self._last_grad = g_bar
        
        return energy_for_optimizer, g_bar

    def optimize(self, geom=None, **kwargs):
        """
        Runs the geometry optimization using pyscf.geomopt.geometric_solver.
        
        Args:
            geom (str or np.ndarray): Initial geometry. If None, uses the geometry
                                      from the molecule in the MCSCF object.
            **kwargs: Additional keyword arguments to pass to the optimizer.
                      e.g., max_cycle=100, trajectory='traj.xyz'
        """
        if geom is not None:
            self.mol.atom = geom
            self.td.mol = self.mol

        # Create the wrapper object that geometric_solver will use
        solver_wrapper = self._SolverWrapper(self)

        # Call the PySCF optimizer with our custom wrapper
        # The optimizer will call solver_wrapper() for energy and
        # solver_wrapper.nuc_grad_method().kernel() for the gradient.
        optimized_mol = geometric_solver.optimize(solver_wrapper, **kwargs)
        
        self.mol = optimized_mol
        return self.mol

def project_on_plane_lstsq(x3, x1, x2):
    A = np.column_stack([x1, x2])
    c, _, _, _ = np.linalg.lstsq(A, x3, rcond=None)
    projection = A @ c
    return projection
