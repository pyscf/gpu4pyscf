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

import cupy as cp
import numpy as np
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, contract
from gpu4pyscf.pbc.grad import kuks as kuks_grad
from gpu4pyscf.pbc.dft.krkspu import reference_mol, _make_minao_lo
from gpu4pyscf.pbc.grad.krkspu import generate_first_order_local_orbitals
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.dft.ucdft import _get_minao_basis_indices


class Gradients(kuks_grad.Gradients):

    def __init__(self, method):
        super().__init__(method)
        self._dE_constraint = None

    def _get_constraint_force_kpts(self, dm=None, kpts=None):

        mf = self.base
        if mf.projection_method != 'minao':
            raise NotImplementedError("Only 'minao' projection gradient is fully implemented.")
        
        if mf.method == 'lagrange':
            v_vec = mf.v_lagrange
        elif mf.method == 'penalty':
            raise NotImplementedError("Penalty method is not implemented for gradient.")
        else:
            raise NotImplementedError(f"Method {mf.method} not implemented.")

        if dm is None:
            dm = mf.make_rdm1()
        if kpts is None:
            kpts = mf.kpts.reshape(-1, 3)
            
        cell = mf.cell
        nkpts = len(kpts)
        weight = getattr(kpts, "weights_ibz", np.repeat(1.0/nkpts, nkpts))
        weight = cp.array(weight)
        
        pcell = reference_mol(cell, mf.minao_ref)
        C_ao_lo = _make_minao_lo(cell, pcell, kpts=kpts) 
        ovlp0 = int1e.int1e_ovlp(cell, kpts)
        ovlp1 = int1e.int1e_ipovlp(cell, kpts) 

        f_local_ao = generate_first_order_local_orbitals(cell, pcell, kpts)
        
        charge_indices_list = []
        for group in mf.charge_groups:
            indices = []
            for label in group:
                indices.extend(_get_minao_basis_indices(pcell, label))
            charge_indices_list.append(sorted(list(set(indices))))
            
        spin_indices_list = []
        for group in mf.spin_groups:
            indices = []
            for label in group:
                indices.extend(_get_minao_basis_indices(pcell, label))
            spin_indices_list.append(sorted(list(set(indices))))

        ao_slices = cell.aoslice_by_atom()
        natm = cell.natm
        dE_c = cp.zeros((natm, 3))
        
        SC_list = [contract('pq, qi -> pi', s, c) for s, c in zip(ovlp0, C_ao_lo)]
        for atm_id, (p0, p1) in enumerate(ao_slices[:,2:]):
            # C1_all_k[k] shape: (3, NAO, N_MINAO)
            C1_all_k = f_local_ao(atm_id)
            
            for k in range(nkpts):
                w_k = weight[k]
                C0_k = C_ao_lo[k]
                SC_k = SC_list[k]
                
                C1_k = C1_all_k[k] 
                SC1 = contract('pq, xqi -> xpi', ovlp0[k], C1_k)
                
                SC1 -= contract('xqp, qi -> xpi', ovlp1[k][:, p0:p1].conj(), C0_k[p0:p1])
                SC1[:, p0:p1] -= contract('xpq, qi -> xpi', ovlp1[k][:, p0:p1], C0_k)

                P_tot = dm[0][k] + dm[1][k]
                P_spin = dm[0][k] - dm[1][k]
                constraint_idx = 0
                for indices in charge_indices_list:
                    if not indices: 
                        continue
                    idx_arr = asarray(indices)
                    vc = float(v_vec[constraint_idx])
                    
                    sc1_sub = SC1[:, :, idx_arr]
                    sc_sub = SC_k[:, idx_arr]
                    
                    term = cp.einsum('pq, xqi, pi -> x', P_tot, sc1_sub, sc_sub.conj())
                    dE_c[atm_id] += w_k * 2.0 * vc * term.real
                    
                    constraint_idx += 1

                for indices in spin_indices_list:
                    if not indices: 
                        continue
                    idx_arr = asarray(indices)
                    vc = float(v_vec[constraint_idx])
                    
                    sc1_sub = SC1[:, :, idx_arr]
                    sc_sub = SC_k[:, idx_arr]
                    
                    term = cp.einsum('pq, xqi, pi -> x', P_spin, sc1_sub, sc_sub.conj())
                    dE_c[atm_id] += weight * 2.0 * vc * term.real
                    
                    constraint_idx += 1

        return dE_c.get()

    def get_veff(self, dm=None, kpts=None):
        if dm is None: dm = self.base.make_rdm1()
        if kpts is None: kpts = self.base.kpts
        
        logger.info(self.base, "Calculating constraint gradient contributions (PBC Minao)...")
        self._dE_constraint = self._get_constraint_force_kpts(dm, kpts)
        
        return super().get_veff(dm, kpts)

    def extra_force(self, atom_id, envs):
        force = super().extra_force(atom_id, envs)
        if self._dE_constraint is not None:
            force += self._dE_constraint[atom_id]
        return force