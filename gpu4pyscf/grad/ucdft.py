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
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, contract
from gpu4pyscf.grad import uks
from gpu4pyscf.dft.rkspu import reference_mol, _make_minao_lo
from gpu4pyscf.dft.ucdft import _get_minao_basis_indices
from gpu4pyscf.grad.rkspu import generate_first_order_local_orbitals


class Gradients(uks.Gradients):
    '''
    CDFT Gradients for Unrestricted Kohn-Sham.
    Adds the constraint force contribution to the standard UKS gradient.
    '''

    def __init__(self, method):
        super().__init__(method)
        self._dE_constraint = None

    def _get_constraint_force(self, mol, dm):
        """
        Calculates the gradient contribution from the CDFT constraints (Minao method).

        Core derivatives can refer to grad.rkspu
        
        Formula:
            F_c = V_c * sum_sigma Tr( P_sigma * d(w)/dx )
        Where w is the weight matrix: w = SC * SC^T
        """
        mf = self.base
        
        # Only support Minao currently as requested
        if mf.projection_method != 'minao':
            raise NotImplementedError("CDFT Gradient: Only 'minao' projection gradient is fully implemented.")

        # Ensure we have the converged Lagrange multipliers
        if mf.method == 'lagrange':
            v_vec = mf.v_lagrange
        elif mf.method == 'penalty':
            raise NotImplementedError("Penalty method not implemented for CDFT gradient.")
            # # For penalty, V_eff = 2 * lambda * (N - N_target)
            # # We construct an effective V vector to reuse the same logic
            # # This requires calculating the current population again, or storing it.
            # # Simplified: Assuming user wants gradient at convergence. 
            # # If not converged, this is an approximation.
            # # We calculate the effective V based on current P.
            # # Re-calculate populations:
            # errors = mf._micro_objective_func(np.zeros(mf.n_constraints), 
            #                                   cp.zeros_like(dm[0]), cp.zeros_like(dm[1]), 
            #                                   mf.get_ovlp())
            # # V_eff = 2 * penalty * error
            # v_vec = 2.0 * mf.penalty_weight * errors
        else:
            raise NotImplementedError(f"CDFT Gradient: Method {mf.method} not implemented.")

        pmol = reference_mol(mol, mf.minao_ref)
        C_minao = _make_minao_lo(mol, pmol) # (NAO, N_MINAO) orthogonalized coeffs or C^{orth}
        ovlp0 = mf.get_ovlp()
        ovlp1 = asarray(mol.intor('int1e_ipovlp')) # nabla_r integral (requires sign flip for nuclear grad)
        
        # Function to generate dC/dx or dC^{orth}/dx
        f_local_ao = generate_first_order_local_orbitals(mol, pmol)

        ao_slices = mol.aoslice_by_atom()
        dE_c = cp.zeros((mol.natm, 3))
        
        charge_indices_list = []
        for group in mf.charge_groups:
            indices = []
            for identifier in group:
                indices.extend(_get_minao_basis_indices(mol, pmol, identifier))
            charge_indices_list.append(sorted(list(set(indices))))
            
        spin_indices_list = []
        for group in mf.spin_groups:
            indices = []
            for identifier in group:
                indices.extend(_get_minao_basis_indices(mol, pmol, identifier))
            spin_indices_list.append(sorted(list(set(indices))))

        # SC = S * C (The undifferentiated matrix)
        SC = ovlp0.dot(C_minao)
        # === Loop over Atoms (Gradient Coordinate) ===
        for atm_id, (p0, p1) in enumerate(ao_slices[:,2:]):
            
            # 1. Calculate derivative of SC matrix: d(SC)/dx
            # SC = S * C_orth
            # d(SC) = (dS) * C + S * (dC)
            
            # Term A: S * dC
            C1 = f_local_ao(atm_id) # Shape (3, NAO, N_MINAO)
            SC1 = contract('pq,xqi->xpi', ovlp0, C1)
            
            # Term B: (dS) * C
            # dS is non-zero only for basis functions on atom `atm_id`
            # ovlp1 is <nabla_r p | q>. Nuclear grad is -<nabla_r p | q>.
            
            # Contribution from row derivative (bra on atom A)
            SC1 -= contract('xqp,qi->xpi', ovlp1[:,p0:p1], C_minao[p0:p1])
            
            # Contribution from col derivative (ket on atom A)
            SC1[:,p0:p1] -= contract('xpq,qi->xpi', ovlp1[:,p0:p1], C_minao)
            # Now we have SC1 = d(SC)/dx_A for all Minao columns

            # Loop over Constraints CORE PART!
            # 1. Charge Constraints (N_alpha + N_beta)
            # Term: Vc * Tr( (Pa + Pb) * dW/dx )
            # dW/dx = d(SC * SC^T)/dx = SC1 * SC^T + SC * SC1^T
            # Trace = 2 * Re( Tr( P * SC1 * SC^T ) )
            
            constraint_idx = 0
            P_tot = dm[0] + dm[1]
            
            for i, indices in enumerate(charge_indices_list):
                if not indices: continue
                idx_arr = asarray(indices)
                vc = float(v_vec[constraint_idx])
                
                # Extract columns corresponding to this constraint
                sc1_sub = SC1[:, :, idx_arr] # (3, NAO, N_subset)
                sc_sub = SC[:, idx_arr]      # (NAO, N_subset)
                
                # Compute Gradient Contribution
                # Force = Vc * Tr( P_tot * (SC1 * SC^T + h.c.) )
                #       = 2 * Vc * Re( Tr( P_tot * SC1 * SC^T ) )
                # Contract: P_tot[p,q] * SC1[x,q,i] * SC[p,i]^*
                term = cp.einsum('pq, xqi, pi -> x', P_tot, sc1_sub, sc_sub.conj())
                dE_c[atm_id] += 2.0 * vc * term.real
                
                constraint_idx += 1

            # 2. Spin Constraints (N_alpha - N_beta)
            # Term: Vc * Tr( (Pa - Pb) * dW/dx )
            P_spin = dm[0] - dm[1]
            
            for i, indices in enumerate(spin_indices_list):
                if not indices: continue
                idx_arr = asarray(indices)
                vc = float(v_vec[constraint_idx])
                
                sc1_sub = SC1[:, :, idx_arr]
                sc_sub = SC[:, idx_arr]
                
                term = cp.einsum('pq, xqi, pi -> x', P_spin, sc1_sub, sc_sub.conj())
                dE_c[atm_id] += 2.0 * vc * term.real
                
                constraint_idx += 1

        return dE_c.get()

    def get_veff(self, mol=None, dm=None):
        """
        Calculate the gradient response from the constraint potential.
        This is called during the gradient calculation loop.
        """
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        
        logger.info(self.base, "CDFT: Calculating constraint gradient contributions (Minao)...")
        self._dE_constraint = self._get_constraint_force(mol, dm)
        
        return super().get_veff(mol, dm)

    def extra_force(self, atom_id, envs):
        force = super().extra_force(atom_id, envs)
        
        if self._dE_constraint is not None:
            force += self._dE_constraint[atom_id]
            
        return force
