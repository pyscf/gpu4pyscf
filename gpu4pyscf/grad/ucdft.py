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
    '''

    def __init__(self, method):
        super().__init__(method)
        self._dE_constraint = None

    def _get_constraint_force(self, mol, dm):
        """
        Calculates the gradient contribution from the CDFT constraints (Minao method).
        See gpu4pyscf.grad.rkspu for AO derivatives.
        """
        mf = self.base
        
        if mf.projection_method != 'minao':
            raise NotImplementedError("Only 'minao' projection gradient is fully implemented.")

        if mf.method == 'lagrange':
            v_vec = mf.v_lagrange
        elif mf.method == 'penalty':
            raise NotImplementedError("Penalty method is not implemented for gradient.")
        else:
            raise NotImplementedError(f"Method {mf.method} not implemented.")

        pmol = reference_mol(mol, mf.minao_ref)
        C_orth = _make_minao_lo(mol, pmol) # (NAO, N_MINAO)
        ovlp0 = mf.get_ovlp()
        ovlp1 = asarray(mol.intor('int1e_ipovlp')) # sign flip needed
        
        # dC^{orth}/dx
        f_local_ao = generate_first_order_local_orbitals(mol, pmol)

        ao_slices = mol.aoslice_by_atom()
        dE_c = cp.zeros((mol.natm, 3))
        
        charge_indices_list = []
        for group in mf.charge_groups:
            indices = []
            for label in group:
                indices.extend(_get_minao_basis_indices(pmol, label))
            charge_indices_list.append(sorted(list(set(indices))))
            
        spin_indices_list = []
        for group in mf.spin_groups:
            indices = []
            for label in group:
                indices.extend(_get_minao_basis_indices(pmol, label))
            spin_indices_list.append(sorted(list(set(indices))))

        SC = ovlp0.dot(C_orth)
        for atm_id, (p0, p1) in enumerate(ao_slices[:,2:]):
            C1 = f_local_ao(atm_id) # (3, NAO, N_MINAO)
            SC1 = contract('pq,xqi->xpi', ovlp0, C1)
            
            SC1 -= contract('xqp,qi->xpi', ovlp1[:,p0:p1], C_orth[p0:p1]) # bra
            SC1[:,p0:p1] -= contract('xpq,qi->xpi', ovlp1[:,p0:p1], C_orth) # kst
            
            constraint_idx = 0
            P_tot = dm[0] + dm[1]         
            for i, indices in enumerate(charge_indices_list):
                if not indices: 
                    continue
                idx_arr = asarray(indices)
                vc = float(v_vec[constraint_idx])
                
                sc1_sub = SC1[:, :, idx_arr]
                sc_sub = SC[:, idx_arr] 
                
                # Force = Vc * Tr( P_tot * (SC1 * SC^T + h.c.) )
                #       = 2 * Vc * Tr( P_tot * SC1 * SC^T )
                term = cp.einsum('pq, xqi, pi -> x', P_tot, sc1_sub, sc_sub)
                dE_c[atm_id] += 2.0 * vc * term
                
                constraint_idx += 1

            P_spin = dm[0] - dm[1]
            
            for i, indices in enumerate(spin_indices_list):
                if not indices: 
                    continue
                idx_arr = asarray(indices)
                vc = float(v_vec[constraint_idx])
                
                sc1_sub = SC1[:, :, idx_arr]
                sc_sub = SC[:, idx_arr]
                
                term = cp.einsum('pq, xqi, pi -> x', P_spin, sc1_sub, sc_sub)
                dE_c[atm_id] += 2.0 * vc * term.real
                
                constraint_idx += 1

        return dE_c.get()

    def get_veff(self, mol=None, dm=None):
        """
        Calculate the gradient response from the constraint potential.
        """
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        
        logger.info(self.base, "Calculating constraint gradient contributions (Minao)...")
        # Note: Do not add force here. Veff is doubled in get_elec.
        self._dE_constraint = self._get_constraint_force(mol, dm)
        
        return super().get_veff(mol, dm)

    def extra_force(self, atom_id, envs):
        force = super().extra_force(atom_id, envs)
        
        if self._dE_constraint is not None:
            force += self._dE_constraint[atom_id]
            
        return force
