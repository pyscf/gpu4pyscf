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
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.geomopt import geometric_solver
from gpu4pyscf.lib import logger
from gpu4pyscf import tdscf

def project_on_plane_lstsq(x3, x1, x2):
    """
    Project vector x3 onto the plane defined by vectors x1 and x2
    using a least-squares approach.

    Args:
        x3 (np.ndarray): Vector to project.
        x1 (np.ndarray): First basis vector of the plane.
        x2 (np.ndarray): Second basis vector of the plane.

    Returns:
        np.ndarray: Projection of x3 onto the plane.
    """
    x3 = x3.ravel()
    x1 = x1.ravel()
    x2 = x2.ravel()
    A = np.column_stack([x1, x2])
    # Solve Ax = x3 for x, which gives the coefficients for the projection
    coeffs, _, _, _ = np.linalg.lstsq(A, x3, rcond=None)
    projection = A @ coeffs
    return projection.reshape(-1, 3)

class MECPScanner:
    """
    A scanner class compatible with pyscf.geomopt.geometric_solver for
    finding minimum energy crossing points (MECP).

    This scanner computes an effective energy and gradient for the MECP
    optimization problem based on the direct method described in
    Chem. Phys. Lett. 223 (1994) 269-274.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.mol = optimizer.mol
        self.td = optimizer.td
        self.mf = optimizer.mf
        self.states = optimizer.states
        self.crossing_type = optimizer.crossing_type
        self.log = optimizer.log
        self.verbose = self.td.verbose
        self.base = self # For compatibility with some pyscf functions
        self.converged = False

        # Create scanners for the underlying SCF and TD-SCF objects
        self._mf_scanner = self.mf.as_scanner()
        self._td_scanner = self.td.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        """
        This is the main function called by the geometry optimizer at each step.

        It takes a new geometry, performs the necessary quantum chemistry
        calculations, and returns the effective energy and gradient for
        the MECP optimization.
        """
        if isinstance(mol_or_geom, gto.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.log.info("\n--- CI Optimizer Step (using pyscf.geomopt.geometric_solver) ---")

        # 1. Run SCF and TD-SCF calculations for the new geometry using scanners
        # This is the safe and recommended way to handle geometry changes.
        e_tot = self._mf_scanner(mol)
        self._td_scanner(mol)
        self.converged = all(self._td_scanner.converged)

        if (isinstance(self.td, (tdscf.rhf.TDA, tdscf.rhf.TDHF,
                                 tdscf.rks.TDA, tdscf.rks.TDDFT))):
            e_states = self.td.e
        else: # For UKS/UHF based TD
            e_states = self.td.energies / HARTREE2EV

        E1 = float(e_states[self.states[0]-1] + e_tot)
        E2 = float(e_states[self.states[1]-1] + e_tot)
        self.log.info(f"  Total Energies: E1={E1:.6f}, E2={E2:.6f}")
        self.log.info(f"  Energy Gap (E2-E1): {E2-E1:.6f} Ha")

        # 2. Calculate analytical gradients for both states
        # The td_scanner object also serves as the gradient method
        grad_method = self._td_scanner.Gradients()
        g1 = grad_method.kernel(state=self.states[0])
        g2 = grad_method.kernel(state=self.states[1])

        # 3. Define the branching space vectors x1 and x2
        # x1 is the gradient difference vector
        x1 = g1 - g2
        x1_norm_val = np.linalg.norm(x1)
        x1_norm_vec = x1 / x1_norm_val if x1_norm_val > 1e-9 else np.zeros_like(x1)

        # For MECP, x2 is the non-adiabatic coupling vector
        if self.crossing_type == 'n-2':
            nac_method = self._td_scanner.nac_method()
            nac_method.states = self.states
            nac_vect = nac_method.kernel()
            x2 = nac_vect[1]
            x2_norm_val = np.linalg.norm(x2)
            x2_norm_vec = x2 / x2_norm_val if x2_norm_val > 1e-9 else np.zeros_like(x2)

            # Project g2 onto the plane spanned by x1 and x2 (the branching plane)
            # The component of the gradient outside this plane drives the system
            # along the seam of intersection.
            g_on_plane = project_on_plane_lstsq(g2, x1_norm_vec, x2_norm_vec)
            g_proj = g2 - g_on_plane

        elif self.crossing_type == 'n-1':
            raise NotImplementedError("n-1 crossing type not implemented yet.")
        else:
            raise ValueError(f"Unknown crossing_type: {self.crossing_type}")


        # 4. Calculate the component of the gradient that drives the states to degeneracy
        f = (E1 - E2) * x1_norm_vec  # Note: The original paper has (E1-E2)

        # 5. The total effective gradient is the sum of the seam-following part and the
        #    degeneracy-driving part.
        g_bar = g_proj + f

        self.log.info(f"  ||Seam Grad (g_proj)||: {np.linalg.norm(g_proj):.6f}")
        self.log.info(f"  ||Degeneracy Grad (f)||: {np.linalg.norm(f):.6f}")
        self.log.info(f"  ||Total Effective Grad||: {np.linalg.norm(g_bar):.6f}")
        self.log.info("----------------------------------------------------------------")

        # The optimizer minimizes a single energy value. We provide the average energy.
        energy_for_optimizer = (E1 + E2) / 2.0

        return energy_for_optimizer, g_bar

    # The following methods make our scanner object behave like a gradient object
    # itself, which is expected by geometric_solver.
    def as_scanner(self):
        return self

    def Gradients(self):
        return self

    def nuc_grad_method(self):
        return self.Gradients()


class ConicalIntersectionOptimizer:
    """
    Implements the direct method for locating the lowest energy point on a
    potential energy surface crossing, as described in
    Chemical Physics Letters 223 (1994) 269-274.

    This class serves as a high-level driver that uses
    pyscf.geomopt.geometric_solver as the core optimizer.

    Args:
        td (TDSCF object): A converged time-dependent HF or DFT object from gpu4pyscf.
        states (tuple): A tuple of two integers (1-indexed) specifying the
                        electronic states, e.g., (1, 2) for S1/S2.
                        Note: Ground state (S0) is not supported in this formalism.
        crossing_type (str): Type of intersection. Currently supports:
                             'n-2' for a conical intersection (same spin multiplicity).
    """

    def __init__(self, td, states=(1, 2), crossing_type='n-2'):
        if len(states) != 2:
            raise ValueError("`states` must be a tuple of two state indices.")
        if 0 in states:
            raise ValueError("This method is for excited state crossings. "
                             "State indices must be > 0.")

        self.td = td
        self.mf = td._scf
        self.mol = self.mf.mol
        # Ensure states are sorted, e.g., (1, 2) not (2, 1)
        self.states = tuple(sorted(states))
        self.crossing_type = crossing_type
        self.verbose = self.td.verbose
        self.stdout = self.td.stdout
        self.log = logger.new_logger(self, self.verbose)

    def kernel(self, geom=None, **kwargs):
        """
        Alias for the optimize method.
        """
        return self.optimize(geom, **kwargs)

    def optimize(self, geom=None, **kwargs):
        """
        Runs the geometry optimization to find the MECP.

        Args:
            geom (str or np.ndarray): Initial geometry. If None, uses the geometry
                                      from the molecule in the TD-SCF object.
            **kwargs: Additional keyword arguments to pass to the
                      pyscf.geomopt.geometric_solver.optimize function.
                      e.g., max_cycle=50, dump_input=False

        Returns:
            Mole: An optimized pyscf Mole object.
        """
        if geom is not None:
            self.mol.atom = geom

        # Create the scanner object that geometric_solver will use.
        # This scanner encapsulates all the logic for one optimization step.
        mecp_scanner = MECPScanner(self)

        # Call the PySCF optimizer with our custom scanner.
        optimized_mol = geometric_solver.optimize(mecp_scanner, **kwargs)

        self.mol = optimized_mol
        return self.mol
