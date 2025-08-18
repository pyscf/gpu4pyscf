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
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.geomopt import geometric_solver
from gpu4pyscf.lib import logger
from gpu4pyscf import tdscf


# TODO: Add support for S0/S1
class ConicalIntersectionOptimizer:
    """
    Implements the direct method for locating the lowest energy point on a potential surface crossing,
    as described in Chemical Physics Letters 223 (1994) 269-274.
    
    This version uses pyscf.geomopt.geometric_solver as the optimizer.
    
    Args:
        td : A time-dependent HF or DFT object.
        states (tuple): A tuple of two integers specifying the electronic states to be coupled (e.g., (0, 1) for S0/S1).
        crossing_type (str): Type of intersection, 'n-2' for conical intersection (same spin)
                             or 'n-1' for different spin multiplicity crossing. (This is not implemented)
    """

    def __init__(self, td, states=(1, 2), crossing_type='n-2'):
        if len(states) != 2:
            raise ValueError("`states` must be a tuple of two state indices.")

        self.td = td
        self.mf = td._scf
        self.mol = self.mf.mol
        self.states = states
        self.verbose = self.td.verbose
        self.crossing_type = crossing_type
        self.stdout = self.td.stdout
        self.log = logger.new_logger(self, self.verbose)
        
        self._last_geom = None
        self._last_energy = None
        self._last_grad = None

    class _GradScanner:
        def __init__(self, solver_wrapper):
            self.solver_wrapper = solver_wrapper
            self.mol = self.solver_wrapper.mol
            self.verbose = self.solver_wrapper.verbose
            self.stdout = self.solver_wrapper.stdout

        def __call__(self, mol_or_geom, **kwargs):
            """This is the scanner function."""
            if isinstance(mol_or_geom, gto.MoleBase):
                assert mol_or_geom.__class__ == gto.Mole
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            self.solver_wrapper.ci_optimizer.td.reset(mol)
            self.solver_wrapper.ci_optimizer.mol = self.solver_wrapper.ci_optimizer.td.mol
            self.solver_wrapper.mol = self.solver_wrapper.ci_optimizer.td.mol
            if isinstance(self.solver_wrapper.ci_optimizer.td, tdscf.ris.TDA) or \
                    isinstance(self.solver_wrapper.ci_optimizer.td, tdscf.ris.TDDFT):
                self.solver_wrapper.ci_optimizer.td.n_occ = None
                self.solver_wrapper.ci_optimizer.td.n_vir = None
                self.solver_wrapper.ci_optimizer.td.rest_occ = None
                self.solver_wrapper.ci_optimizer.td.rest_vir = None
                self.solver_wrapper.ci_optimizer.td.C_occ_notrunc = None
                self.solver_wrapper.ci_optimizer.td.C_vir_notrunc = None
                self.solver_wrapper.ci_optimizer.td.C_occ_Ktrunc = None
                self.solver_wrapper.ci_optimizer.td.C_vir_Ktrunc = None
                self.solver_wrapper.ci_optimizer.td.delta_hdiag = None
                self.solver_wrapper.ci_optimizer.td.hdiag = None
                self.solver_wrapper.ci_optimizer.td.eri_tag = None
                self.solver_wrapper.ci_optimizer.td.auxmol_J = None
                self.solver_wrapper.ci_optimizer.td.auxmol_K = None
                self.solver_wrapper.ci_optimizer.td.lower_inv_eri2c_J = None
                self.solver_wrapper.ci_optimizer.td.lower_inv_eri2c_K = None
                self.solver_wrapper.ci_optimizer.td.RKS = True
                self.solver_wrapper.ci_optimizer.td.UKS = False
                self.solver_wrapper.ci_optimizer.td.mo_coeff = cp.asarray(self.solver_wrapper.ci_optimizer.mf.mo_coeff, 
                    dtype=self.solver_wrapper.ci_optimizer.td.dtype)
                self.solver_wrapper.ci_optimizer.td.build()
            return self.kernel(**kwargs)

        def kernel(self, *args, **kwargs):
            self.solver_wrapper.ci_optimizer.get_eff_energy_and_gradient()
            return self.solver_wrapper.ci_optimizer._last_energy, self.solver_wrapper.ci_optimizer._last_grad

        def as_scanner(self):
            """Returns the scanner object."""
            return self

        def converged(self):
            td_scanner = self.base
            return all((td_scanner._scf.converged,
                        td_scanner.converged[self.states[0]-1],
                        td_scanner.converged[self.states[1]-1]))

    class _SolverWrapper:
        def __init__(self, ci_optimizer):
            self.ci_optimizer = ci_optimizer
            self.mol = self.ci_optimizer.mol
            self.verbose = self.ci_optimizer.verbose
            self.stdout = self.ci_optimizer.stdout
            
        def __call__(self, *args, **kwargs):
            energy, grad = self.ci_optimizer.get_eff_energy_and_gradient()
            return energy, grad

        def nuc_grad_method(self):
            return self.ci_optimizer._GradScanner(self)


    def get_eff_energy_and_gradient(self):
        current_geom = self.mol.atom_coords(unit='bohr')

        self.log.info("\n--- CI Optimizer Step (using pyscf.geomopt.geometric_solver) ---")

        self.mf.mol = self.mol
        self.td.mol = self.mol

        self.mf.kernel()
        self.td.kernel()
        if (isinstance(self.td, tdscf.rhf.TDA) or isinstance(self.td, tdscf.rhf.TDHF)
            or isinstance(self.td, tdscf.rks.TDA) or isinstance(self.td, tdscf.rks.TDDFT)):
            e_states = self.td.e
        else:
            e_states = self.td.energies/HARTREE2EV
        assert self.states[0] <= self.states[1]
        assert self.states[0] != 0
        
        E1 = float(e_states[self.states[0]-1] + self.mf.e_tot)
        E2 = float(e_states[self.states[1]-1] + self.mf.e_tot)
        self.log.info(f"  Total Energies: E1={E1:.6f}, E2={E2:.6f}")
        self.log.info(f"  Energy Gap (E2-E1): {E2-E1:.6f} Ha")

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
            nac_method.states=self.states
            nac_vect = nac_method.kernel()
            x2 = nac_vect[1]
            x2_norm_val = np.linalg.norm(x2)
            x2_norm_vec = x2 / x2_norm_val if x2_norm_val > 1e-9 else np.zeros_like(x2)
            
            # Project the gradient of second state onto the seam space and take the component outside
            natom = g2.shape[0]
            g_on_plane = project_on_plane_lstsq(g2, x1_norm_vec, x2_norm_vec)
            g_on_plane = g_on_plane.reshape(natom, 3)
            g_proj = g2 - g_on_plane
        
        elif self.crossing_type == 'n-1':
            raise NotImplementedError("n-1 crossing type not implemented yet.")

        # 3. Calculate component of the effective gradient driving states to degeneracy
        f = (E1 - E2) * x1_norm_vec  # Note: The original paper has (E1-E2)
        
        # 4. Total effective gradient
        g_bar = g_proj + f

        self.log.info(f"  ||Seam Grad (g_proj)||: {np.linalg.norm(g_proj):.6f}")
        self.log.info(f"  ||Degeneracy Grad (f)||: {np.linalg.norm(f):.6f}")
        self.log.info(f"  ||Total Effective Grad||: {np.linalg.norm(g_bar):.6f}")
        self.log.info("----------------------------------------------------------------")
        
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
                      e.g., max_cycle=100, prefix='traj.xyz'
            TODO: kwargs seem not work!
        """
        if geom is not None:
            self.mol.atom = geom

        # Create the wrapper object that geometric_solver will use
        solver_wrapper = self._SolverWrapper(self)

        # Call the PySCF optimizer with our custom wrapper
        # The optimizer will call solver_wrapper() for energy and
        # solver_wrapper.nuc_grad_method().as_scanner()(mol) for the gradient.
        optimized_mol = geometric_solver.optimize(solver_wrapper, **kwargs)
        
        self.mol = optimized_mol
        return self.mol


def project_on_plane_lstsq(x3, x1, x2):
    """
    Project x3 onto the plane defined by x1 and x2 using least squares.
    
    Args:
        x3 (np.ndarray): Vector to project.
        x1 (np.ndarray): First basis vector of the plane.
        x2 (np.ndarray): Second basis vector of the plane.
    
    Returns:
        np.ndarray: Projection of x3 onto the plane.
    """
    x3 = x3.reshape(-1)
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    A = np.column_stack([x1, x2])
    c, _, _, _ = np.linalg.lstsq(A, x3, rcond=None)
    projection = A @ c
    return projection
