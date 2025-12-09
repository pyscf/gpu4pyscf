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
import numpy as np
from gpu4pyscf.scf.soscf import _SecondOrderUHF
from gpu4pyscf.lib import logger
from pyscf import lib

class CDFTSecondOrderUHF(_SecondOrderUHF):
    '''
    Second Order SCF solver for Constrained DFT (UKS) with exact Hessian correction.
    Although, the second term (shift) is often unhelpful.
    
    Derivation based on Chain Rule & First-order Orbital Mixing:
    Objective: E_pen = lambda * (N - Nt)^2
    
    Gradient (dE/dx):
      dE/dx = (dE/dN) * (dN/dx)
            = 2*lambda*(N-Nt) * (2*W_vo)
            = 4*lambda*(N-Nt)*W_vo
            
    Hessian (d2E/dx2):
      Apply product rule to Gradient: d(u*v)/dx = u'*v + u*v'
      
      Term 1 (Rank-1): u' * v
         u' = d(2*lambda*(N-Nt))/dx = 2*lambda * (2*W_vo) = 4*lambda*W_vo
         Term 1 = (4*lambda*W_vo) * (2*W_vo)^T = 8*lambda * W_vo * W_vo^T
      
      NOTE: Term 2 is covered by the original h_op from the fock matrix, so we
      do not need to include it here.
      Term 2 (Shift): u * v'
         v' = d(2*W_vo)/dx = 2 * d(W_vo)/dx
         Using perturbation: dW_vo = W_vv * x - x * W_oo
         Term 2 = 2*lambda*(N-Nt) * 2 * (W_vv*x - x*W_oo)
                = 4*lambda*(N-Nt) * (W_vv*x - x*W_oo)
    '''

    def gen_g_hop(self, mo_coeff, mo_occ, fock_ao=None, h1e=None):
        # 1. Get standard electronic gradient and Hessian
        g, h_op_orig, h_diag = super().gen_g_hop(mo_coeff, mo_occ, fock_ao, h1e)

        if getattr(self, 'n_constraints', 0) == 0:
            return g, h_op_orig, h_diag
            
        penalty_weight = getattr(self, 'last_penalty_weight', 0.0)
        if penalty_weight == 0.0:
            return g, h_op_orig, h_diag
        
        if self.atom_projectors is None:
            self.build_atom_projectors()

        occidxa = mo_occ[0] > 0
        viridxa = mo_occ[0] == 0
        occidxb = mo_occ[1] > 0
        viridxb = mo_occ[1] == 0

        orb_oa = mo_coeff[0][:, occidxa]
        orb_va = mo_coeff[0][:, viridxa]
        orb_ob = mo_coeff[1][:, occidxb]
        orb_vb = mo_coeff[1][:, viridxb]

        nvira, nocca = orb_va.shape[1], orb_oa.shape[1]
        nvirb, noccb = orb_vb.shape[1], orb_ob.shape[1]
        ndim_a = nvira * nocca
        
        # Helper to compute all blocks W_oo, W_vv, W_vo
        def get_w_blocks(atom_indices):
            w_ao_sum = sum(self.atom_projectors[i] for i in atom_indices)

            wa_vo = orb_va.conj().T.dot(w_ao_sum).dot(orb_oa)
            wb_vo = orb_vb.conj().T.dot(w_ao_sum).dot(orb_ob)
            
            return (wa_vo, wb_vo)

        constraint_data = []
        
        for i, group in enumerate(self.charge_groups):
            w_vo = get_w_blocks(group)
            constraint_data.append(('charge', w_vo))
            
        for i, group in enumerate(self.spin_groups):
            w_vo = get_w_blocks(group)
            constraint_data.append(('spin', w_vo))

        processed_constraints = []
        for c_type, (wa_vo, wb_vo) in constraint_data:
            processed_constraints.append({
                'type': c_type,
                'w_vo_vec': (wa_vo.ravel(), wb_vo.ravel())
            })
        
        def h_op_new(x):
            hx = h_op_orig(x)
            
            xa_vec = x[:ndim_a]
            xb_vec = x[ndim_a:]
            
            for item in processed_constraints:
                c_type = item['type']
                wa_vo_vec, wb_vo_vec = item['w_vo_vec']
                
                # --- Term 1: Rank-1 Update ---
                # H1 * x = 8 * lambda * W_vo * (W_vo . x)
                if c_type == 'charge':
                    dot_val = cp.dot(wa_vo_vec, xa_vec) + cp.dot(wb_vo_vec, xb_vec)
                    scale_rank1 = 8.0 * penalty_weight * dot_val
                    
                    hx[:ndim_a] += scale_rank1 * wa_vo_vec
                    hx[ndim_a:] += scale_rank1 * wb_vo_vec
                    
                else: # spin
                    dot_val = cp.dot(wa_vo_vec, xa_vec) - cp.dot(wb_vo_vec, xb_vec)
                    scale_rank1 = 8.0 * penalty_weight * dot_val
                    
                    hx[:ndim_a] += scale_rank1 * wa_vo_vec
                    hx[ndim_a:] -= scale_rank1 * wb_vo_vec
                    
            return hx
        
        for item in processed_constraints:
            c_type = item['type']
            wa_vo_vec, wb_vo_vec = item['w_vo_vec']
            
            # 1. Rank-1 Diag
            factor_rank1 = 8.0 * penalty_weight
            h_diag[:ndim_a] += factor_rank1 * (wa_vo_vec**2)
            h_diag[ndim_a:] += factor_rank1 * (wb_vo_vec**2)

        return g, h_op_new, h_diag

def newton_cdft(mf):
    obj = CDFTSecondOrderUHF(mf)
    return lib.set_class(obj, (CDFTSecondOrderUHF, mf.__class__))