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
from cupyx.scipy.sparse.linalg import minres, LinearOperator
from gpu4pyscf.scf.soscf import _SecondOrderUHF
from gpu4pyscf.lib import logger
from pyscf import lib

class CDFTSecondOrderUHF(_SecondOrderUHF):
    '''
    Second Order SCF solver for Constrained DFT (UKS) optimizing both
    orbital parameters and constraint multipliers (weights) simultaneously.
    
    Algorithm:
        Coupled Newton-KKT method using MINRES solver.
        
    Objective Function (Lagrangian):
        L(k, v) = E(k) + sum_i v_i * (N_i(k) - N_target_i)
        
    Solving for the saddle point of L(k, v) ensures that:
    1. The energy is minimized with respect to orbitals (k).
    2. The constraints are satisfied (N_i = N_target_i).
    
    The Hessian of this system is symmetric but indefinite,
    requiring the MINRES solver instead of the standard Davidson algorithm.
    '''

    def __init__(self, mf, step_size=1.0):
        super().__init__(mf)
        self._scf = mf 
        # Damping factor: It may need to be adjusted based on the problem size.
        self.step_size = step_size
        self.constraint_tol = 1e-6
        assert mf.method == 'lagrange'

    def get_kkt_system(self, mo_coeff, mo_occ, fock_ao):
        '''
        Constructs the LinearOperator for the KKT matrix and the RHS vector
        for the system: Ax = b.
        
        System Structure:
        | H_orb   J.T | | d_k |   | -g_orb       |
        | J        0  | | d_v | = | -(N - N_t)   |
        J is the jacobian matrix of the constraints and kappa
        
        Args:
            mo_coeff: Molecular orbital coefficients
            mo_occ: Orbital occupation numbers
            fock_ao: Fock matrix in AO representation (includes V_eff)

        Returns:
            A_op (LinearOperator): The implicit KKT Hessian matrix.
            M_op (LinearOperator): The preconditioner.
            b_vec (cp.ndarray): The Right-Hand Side vector (gradients).
            constraints_val (cp.ndarray): Current values of the constraint errors.
        '''
        # Since 'fock_ao' already includes the constraint potential V_eff, 
        # g_orb corresponds to dL/dk.
        g_orb, h_op_orb, h_diag_orb = super().gen_g_hop(mo_coeff, mo_occ, fock_ao)
        h_diag_orb = cp.asarray(h_diag_orb)
        
        # Ensure projectors are built.
        # Updated to use constraint_projectors instead of atom_projectors
        if getattr(self._scf, 'constraint_projectors', None) is None:
            self._scf.build_projectors()
        
        occidxa = mo_occ[0] > 0
        viridxa = ~occidxa
        occidxb = mo_occ[1] > 0
        viridxb = ~occidxb
        orb_oa = mo_coeff[0][:, occidxa]
        orb_va = mo_coeff[0][:, viridxa]
        orb_ob = mo_coeff[1][:, occidxb]
        orb_vb = mo_coeff[1][:, viridxb]
        
        dm = self._scf.make_rdm1(mo_coeff, mo_occ)
        
        j_rows = []
        constraints_diff = [] 
        
        # Helper to compute W_vo (Row of Jacobian) and Value
        # Updated: Now takes the direct index into constraint_projectors
        def compute_constraint_data(projector_idx, target, is_spin):
            # Direct access to the pre-summed/calculated projector
            w_ao = self._scf.constraint_projectors[projector_idx]
            
            # Value: Trace(D * W) -> Population
            val_a = cp.trace(dm[0] @ w_ao)
            val_b = cp.trace(dm[1] @ w_ao)
            val = val_a - val_b if is_spin else val_a + val_b
            diff = val - target
            
            # Jacobian Vector: W_vo = C_vir.T * W * C_occ
            # This represents dN/d(kappa)
            wa_vo = orb_va.conj().T.dot(w_ao).dot(orb_oa).ravel()
            wb_vo = orb_vb.conj().T.dot(w_ao).dot(orb_ob).ravel()
            
            if is_spin:
                wb_vo *= -1.0
                
            return cp.hstack([wa_vo, wb_vo]), diff

        # We track the index of constraint_projectors globally across Charge and Spin loops
        projector_idx = 0

        # Process Charge Constraints
        for i, target in enumerate(self._scf.charge_targets):
            vec, val = compute_constraint_data(projector_idx, target, False)
            j_rows.append(vec) 
            constraints_diff.append(val)
            projector_idx += 1
            
        # Process Spin Constraints
        for i, target in enumerate(self._scf.spin_targets):
            vec, val = compute_constraint_data(projector_idx, target, True)
            j_rows.append(vec)
            constraints_diff.append(val)
            projector_idx += 1
            
        # J matrix (2 N_constraints x N_orbital_params)
        J = cp.vstack(j_rows)    
        residual = cp.asarray(constraints_diff) 

        # Construct Linear Operators for MINRES
        n_orb = g_orb.size
        n_con = len(constraints_diff)
        n_tot = n_orb + n_con
        
        # --- Matrix-Vector Product A * x ---
        def matvec(x):
            d_k = x[:n_orb] # Orbital step (kappa)
            d_v = x[n_orb:] # Multiplier step (v)
            
            # Top Block: H_kk * d_k + J.T * d_v
            # h_op_orb is the standard electronic Hessian product
            res_top = h_op_orb(d_k) 
            # Add coupling term: J.T * d_v
            res_top += cp.dot(d_v, J)
            
            # Bottom Block: J * d_k + 0 * d_v
            res_bot = cp.dot(J, d_k)
            
            return cp.hstack([res_top, res_bot])

        # --- Preconditioner M * x ---
        # Implement Schur Complement approximation for the right-bottom block.
        # Structure:
        # M = | diag(|H_kk|)    0 |
        #     | 0               S |
        # where S_kk = sum_i (J_ki^2 / |H_ii|)
        
        h_diag_abs = cp.abs(h_diag_orb)
        h_diag_abs[h_diag_abs < 1e-6] = 1e-6
        inv_h_diag = 1.0 / h_diag_abs
        
        # J (n_con, n_orb)
        m_22_diag = cp.dot(J**2, inv_h_diag)
        m_22_diag[m_22_diag < 1e-6] = 1.0

        def precond_matvec(x):
            d_k = x[:n_orb]
            d_v = x[n_orb:]
            
            out_k = d_k / h_diag_abs
            out_v = d_v / m_22_diag
            # out_v = d_v
            
            return cp.hstack([out_k, out_v])
            
        A_op = LinearOperator((n_tot, n_tot), matvec=matvec, dtype=cp.float64)
        M_op = LinearOperator((n_tot, n_tot), matvec=precond_matvec, dtype=cp.float64)
        
        # Construct RHS Vector b = - [g_orb, residual]
        # * 0.5 due to the SOSCF codes in gpu4pyscf
        # * where the 0.5 is originated in the codes.
        b_vec = -1.0 * cp.hstack([g_orb, 0.5 * residual])
        
        return A_op, M_op, b_vec, residual

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        '''
        Updates both orbital parameters (kappa) and Lagrange multipliers (v).
        dx is the full step vector [delta_kappa, delta_v].
        '''
        dx = cp.asarray(dx)
        
        occidxa = mo_occ[0] > 0
        viridxa = ~occidxa
        occidxb = mo_occ[1] > 0
        viridxb = ~occidxb
        n_orb_params = int(cp.sum(occidxa)*cp.sum(viridxa) + cp.sum(occidxb)*cp.sum(viridxb))
        
        # Split vector
        d_k = dx[:n_orb_params]
        d_v = dx[n_orb_params:]
        
        # Update Lagrange Multipliers (v)
        # Standard Newton update: v_new = v_old + delta_v
        if d_v.size > 0:
            d_v_cpu = cp.asnumpy(d_v)
            self._scf.v_lagrange += d_v_cpu
            # Log the update details
            logger.info(self, f"SOSCF: Updated Multipliers. Max dV = {np.max(np.abs(d_v_cpu)):.2e}")
            logger.debug(self, f"SOSCF: New V = {self._scf.v_lagrange}")

        # Update Orbitals (kappa)
        # Delegate to the parent class method which handles the exponential map (expm)
        return super().update_rotate_matrix(d_k, mo_occ, u0, mo_coeff)

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        '''
        Main loop for the Coupled Newton-KKT optimization.
        Replaces the standard minimization loop (davidson) with a saddle-point search (MINRES).
        '''
        log = logger.new_logger(self, self.verbose)
        cput0 = log.init_timer()
        self.dump_flags()
        
        # TODO: this calculation can be modified.
        if self.conv_tol_grad is None:
            self.conv_tol_grad = np.sqrt(self.conv_tol*0.1)
            log.info('Set conv_tol_grad to %g', self.conv_tol_grad)
        
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ

        # If mo_coeff is still None (First run / Cold start), generate initial guess
        if mo_coeff is None:
            log.debug('No orbitals found. Generating initial guess for CDFT Newton Solver.')
            
            if dm0 is None:
                dm0 = self._scf.get_init_guess(self.mol, self._scf.init_guess)
                
            h1e = self._scf.get_hcore(self.mol)
            s1e = self._scf.get_ovlp(self.mol)
            vhf = self._scf.get_veff(self.mol, dm0)
            fock = self._scf.get_fock(h1e, s1e, vhf, dm0)
            
            mo_energy, mo_coeff = self._scf.eig(fock, s1e)
            mo_occ = self._scf.get_occ(mo_energy, mo_coeff)
            
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
            self.mo_energy = mo_energy
            
        cput1 = log.timer('initializing second order scf', *cput0)
        dm = self._scf.make_rdm1(mo_coeff, mo_occ)
        vhf = self._scf.get_veff(self.mol, dm) 
        e_tot = self._scf.energy_tot(dm, h1e=None, vhf=vhf)
        
        log.info('Initial guess E= %.15g', e_tot)
        
        scf_conv = False
        
        # Macro Iterations (Newton Steps)
        for imacro in range(self.max_cycle):
            # 1. Build Fock Matrix
            # This must include the potential V_eff = sum v_i * W_i
            # NOTE: in this part, cycle=-1, thus the v_lagrange will not be updated via nested-iteration.
            fock = self._scf.get_fock(dm=dm)
            
            # 2. Build KKT System (Linear Operator)
            # This constructs the Jacobian J and the KKT matrix-vector product function
            # and precondition function.
            # TODO: There is redundant calculation
            A_op, M_op, b_vec, constraints_res = self.get_kkt_system(mo_coeff, mo_occ, fock)
            
            # 3. Check Convergence
            # Convergence requires both Gradient -> 0 (Optimality) and Residuals -> 0 (Primal Feasibility)
            # TODO: The convergence check can be more sophisticated.
            norm_g = cp.linalg.norm(b_vec)
            max_const_err = cp.max(cp.abs(constraints_res))
            
            log.info('Macro %d: E= %.15g  |KKT Res|= %.5g  |Max Constr|= %.5g', 
                     imacro, e_tot, norm_g, max_const_err)
            cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)
            # Use conv_tol_grad for the KKT residual norm
            if norm_g < self.conv_tol_grad and max_const_err < self.constraint_tol:
                scf_conv = True
                log.info('Coupled Newton-KKT Converged.')
                break
                
            # 4. Solve Linear System using MINRES
            dx, info = minres(A_op, b_vec, M=M_op, tol=1e-5, maxiter=200)
            
            if info != 0:
                log.warn(f'MINRES did not fully converge (info={info}), using best guess.')
            
            # Use a conservative step size to prevent oscillation around the saddle point
            scale = self.step_size
            
            # # Optional: Dynamic scaling (Simple Trust Region)
            # # If the step is huge, scale it down more
            norm_dx = np.linalg.norm(dx)
            # if norm_dx > 1.0:
            #      scale *= (1.0 / norm_dx)
            
            logger.debug(self, f"Applying step size: {scale:.4f} (Raw |dx|={norm_dx:.4f})")
            dx_scaled = dx * scale
            
            # 5. Update Parameters (Orbitals + Multipliers)
            # update_rotate_matrix handles the splitting of dx into d_k and d_v
            u = self.update_rotate_matrix(dx_scaled, mo_occ, mo_coeff=mo_coeff)
            mo_coeff = self.rotate_mo(mo_coeff, u)
            
            # 6. Update Density and Energy for the next iteration
            dm = self._scf.make_rdm1(mo_coeff, mo_occ)
            e_tot = self._scf.energy_tot(dm)

        fock = self._scf.get_fock(dm=dm, level_shift_factor=0)
        mo_energy, mo_coeff1 = self._scf.canonicalize(mo_coeff, mo_occ, fock)
        if self.canonicalization:
            log.info('Canonicalize SCF orbitals')
            mo_coeff = mo_coeff1

        self.e_tot = e_tot
        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy
        self.converged = scf_conv
        self.mo_occ = self._scf.get_occ(mo_energy, mo_coeff)
        if cp.any(mo_occ==0):
            homo = mo_energy[mo_occ>0].max()
            lumo = mo_energy[mo_occ==0].min()
            if homo > lumo:
                log.warn('canonicalized orbital HOMO %s > LUMO %s ', homo, lumo)
        logger.timer(self, 'Second order SCF', *cput0)
        self._finalize()
        return e_tot

def newton_cdft(mf):
    '''
    Returns a CDFT-SOSCF solver optimizing (Orbitals + Multipliers).
    '''
    obj = CDFTSecondOrderUHF(mf)
    return lib.set_class(obj, (CDFTSecondOrderUHF, mf.__class__))