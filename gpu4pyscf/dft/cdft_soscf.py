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
         
      If this term is added, the convergence of SCF will be slower. So, delete it.
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

        # --- Part A: Prepare Matrices (OO, VV, VO) ---
        
        if self.atom_projectors is None:
            self.build_atom_projectors()

        # Slice orbitals
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
            
            # 1. Occupied-Occupied (for N_curr and Shift term)
            wa_oo = orb_oa.conj().T.dot(w_ao_sum).dot(orb_oa)
            wb_oo = orb_ob.conj().T.dot(w_ao_sum).dot(orb_ob)
            
            # 2. Virtual-Virtual (for Shift term)
            wa_vv = orb_va.conj().T.dot(w_ao_sum).dot(orb_va)
            wb_vv = orb_vb.conj().T.dot(w_ao_sum).dot(orb_vb)

            # wa_oo = None
            # wb_oo = None
            # wa_vv = None
            # wb_vv = None
            # 3. Virtual-Occupied (for Gradient/Rank-1 term)
            wa_vo = orb_va.conj().T.dot(w_ao_sum).dot(orb_oa)
            wb_vo = orb_vb.conj().T.dot(w_ao_sum).dot(orb_ob)
            
            return (wa_oo, wb_oo), (wa_vv, wb_vv), (wa_vo, wb_vo)

        # Pre-calculate data for all constraints
        constraint_data = []
        
        for i, group in enumerate(self.charge_groups):
            w_oo, w_vv, w_vo = get_w_blocks(group)
            target = self.charge_targets[i]
            constraint_data.append(('charge', target, w_oo, w_vv, w_vo))
            
        for i, group in enumerate(self.spin_groups):
            w_oo, w_vv, w_vo = get_w_blocks(group)
            target = self.spin_targets[i]
            constraint_data.append(('spin', target, w_oo, w_vv, w_vo))

        # --- Part B: Pre-calculate Delta N ---
        processed_constraints = []

        for c_type, target, (wa_oo, wb_oo), (wa_vv, wb_vv), (wa_vo, wb_vo) in constraint_data:
            val_a = cp.trace(wa_oo)
            val_b = cp.trace(wb_oo)
            
            if c_type == 'charge':
                curr_val = val_a + val_b
            else: 
                curr_val = val_a - val_b
                
            delta = curr_val - target
            
            processed_constraints.append({
                'type': c_type,
                'delta': delta,
                'w_vv': (wa_vv, wb_vv),
                'w_oo': (wa_oo, wb_oo),
                'w_vo_vec': (wa_vo.ravel(), wb_vo.ravel())
            })

        # --- Part C: Define Hessian-Product Function ---
        
        def h_op_new(x):
            hx = h_op_orig(x)
            
            xa_vec = x[:ndim_a]
            xb_vec = x[ndim_a:]
            xa_mat = xa_vec.reshape(nvira, nocca)
            xb_mat = xb_vec.reshape(nvirb, noccb)
            
            for item in processed_constraints:
                c_type = item['type']
                delta = item['delta']
                wa_vv, wb_vv = item['w_vv']
                wa_oo, wb_oo = item['w_oo']
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

                # --- Term 2: Shift Update (Optional but mathematically exact) ---
                # H2 * x = 4 * lambda * delta * (W_vv * x - x * W_oo)
                # WARNING: This term can introduce negative curvature if delta is large.
                # It is recommended to enable this only when close to convergence.
                # * Uncomment the following line to disable Term 2 for stability
                # continue 
                
                # scale_shift = 4.0 * penalty_weight * delta
                
                # # Alpha part: W_vv_a * xa - xa * W_oo_a
                # shift_a = wa_vv.dot(xa_mat) - xa_mat.dot(wa_oo)
                # hx[:ndim_a] += scale_shift * shift_a.ravel()
                
                # # Beta part
                # shift_b = wb_vv.dot(xb_mat) - xb_mat.dot(wb_oo)
                
                # if c_type == 'charge':
                #     hx[ndim_a:] += scale_shift * shift_b.ravel()
                # else: # spin (d2E/dNb2 part has +1 sign, cross term d2E/dNaNb has -1)
                #     # For spin constraint E = (Na - Nb - t)^2
                #     # The beta-beta block of Hessian involves (-1)*(-1) = +1
                #     # However, the Shift term derivation: d(2*lambda*Delta*(-2W_vo))/dx
                #     # leads to a negative sign relative to the charge case.
                #     hx[ndim_a:] -= scale_shift * shift_b.ravel()
                    
            return hx

        # --- Part D: Update Diagonal Preconditioner ---
        
        for item in processed_constraints:
            c_type = item['type']
            delta = item['delta']
            wa_vv, wb_vv = item['w_vv']
            wa_oo, wb_oo = item['w_oo']
            wa_vo_vec, wb_vo_vec = item['w_vo_vec']
            
            # 1. Rank-1 Diag
            factor_rank1 = 8.0 * penalty_weight
            h_diag[:ndim_a] += factor_rank1 * (wa_vo_vec**2)
            h_diag[ndim_a:] += factor_rank1 * (wb_vo_vec**2)
            
            # 2. Shift Diag
            # * Uncomment the following line to disable Term 2 diag
            # continue 
            
            # scale_shift = 4.0 * penalty_weight * delta
            
            # wa_vv_diag = cp.diagonal(wa_vv)
            # wa_oo_diag = cp.diagonal(wa_oo)
            # diag_shift_a = (wa_vv_diag[:, None] - wa_oo_diag[None, :]).ravel()
            # h_diag[:ndim_a] += scale_shift * diag_shift_a
            
            # wb_vv_diag = cp.diagonal(wb_vv)
            # wb_oo_diag = cp.diagonal(wb_oo)
            # diag_shift_b = (wb_vv_diag[:, None] - wb_oo_diag[None, :]).ravel()
            
            # if c_type == 'charge':
            #     h_diag[ndim_a:] += scale_shift * diag_shift_b
            # else:
            #     h_diag[ndim_a:] -= scale_shift * diag_shift_b

        return g, h_op_new, h_diag

def newton_cdft(mf):
    obj = CDFTSecondOrderUHF(mf)
    return lib.set_class(obj, (CDFTSecondOrderUHF, mf.__class__))