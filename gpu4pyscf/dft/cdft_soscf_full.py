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
        Coupled Newton-KKT method using MINRES solver with Trust Region.
        
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
        # Initial step size for Newton-KKT iterations in TRM
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
            fock_ao: Fock matrix in AO representation (includes V_cons)

        Returns:
            A_op (LinearOperator): The implicit KKT Hessian matrix.
            M_op (LinearOperator): The preconditioner.
            b_vec (cp.ndarray): The Right-Hand Side vector (gradients).
            constraints_val (cp.ndarray): Current values of the constraint errors.
        '''
        # Since 'fock_ao' already includes the constraint potential V_cons, 
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

        # Track the index of constraint_projectors globally across Charge and Spin loops
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
            
        # J matrix (N_constraints x 2 N_orbital_params)
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
            
            return cp.hstack([out_k, out_v])
            
        A_op = LinearOperator((n_tot, n_tot), matvec=matvec, dtype=cp.float64)
        M_op = LinearOperator((n_tot, n_tot), matvec=precond_matvec, dtype=cp.float64)
        
        # Construct RHS Vector b = - [g_orb, residual]
        # * 0.5 due to the SOSCF codes in gpu4pyscf
        # * where the 0.5 is originated in the codes.
        b_vec = -1.0 * cp.hstack([g_orb, 0.5 * residual])
        
        return A_op, M_op, b_vec, residual

    def solve_trust_region_subproblem(self, A_op, b_vec, M_op, radius, shift=0.0, tol=1e-5):
        '''
        Solves the trust region subproblem: Minimize quadratic model s.t. ||p|| <= radius.
        
        1. Form implicit shifted operator (A + shift*I).
        2. Solve (A + shift*I) p = b using MINRES.
        3. If ||p|| > radius, scale p back to the trust region boundary.
        Shift: Adds a small constant to the diagonal of A. This maybe used in future.
        '''
        n_dim = A_op.shape[0]

        if abs(shift) > 1e-9:
            def matvec_shifted(x):
                return A_op.matvec(x) + shift * x
            
            A_shifted = LinearOperator((n_dim, n_dim), matvec=matvec_shifted, dtype=cp.float64)
        else:
            A_shifted = A_op

        # Solve linear system
        # Updated: Use the 'tol' passed from the kernel.
        dx, info = minres(A_shifted, b_vec, M=M_op, tol=tol, maxiter=200)
        
        if info != 0:
            logger.warn(self, f'MINRES did not fully converge (info={info}).')
        
        # Check Trust Region Boundary
        norm_dx = float(cp.linalg.norm(dx))
        scale_factor = 1.0
        
        if norm_dx > radius:
            scale_factor = radius / norm_dx
            dx *= scale_factor
            logger.debug(self, f"TRM: Step truncated. Radius={radius:.4f}, |dx|={norm_dx:.4f} -> {radius:.4f}")
        else:
            logger.debug(self, f"TRM: Step inside region. Radius={radius:.4f}, |dx|={norm_dx:.4f}")
            
        return dx, scale_factor, norm_dx

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        '''
        Main loop for the Coupled Newton-KKT optimization.
        Replaces the standard minimization loop (davidson) with a saddle-point search (MINRES).
        Uses a TRM for robustness.
        '''
        log = logger.new_logger(self, self.verbose)
        cput0 = log.init_timer()
        self.dump_flags()
        
        # TODO: gradient norm threshold can be modified.
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
        
        # --- Trust Region Initialization ---
        trust_radius = self.step_size  # Initial radius
        min_radius = 1e-4
        max_radius = 8.0
        
        # Pre-compute initial Fock and KKT system
        fock = self._scf.get_fock(dm=dm)
        A_op, M_op, b_vec, constraints_res = self.get_kkt_system(mo_coeff, mo_occ, fock)
        
        # Current Gradient Norm (for Merit Function)
        # Merit function for saddle point: 0.5 * ||g||^2 -> minimize ||g||
        grad_norm_current = float(cp.linalg.norm(b_vec))
        
        e_last = e_tot

        # Macro Iterations (Newton Steps)
        imacro = 0
        while imacro < self.max_cycle:
            
            max_const_err = cp.max(cp.abs(constraints_res))
            dE = abs(e_tot - e_last)
            log.info('Macro %d: E= %.15g  dE= %.5g  |KKT Grad|= %.5g  |Max Constr|= %.5g  TR_Rad= %.4g', 
                     imacro, e_tot, dE, grad_norm_current, max_const_err, trust_radius)
            
            cput1 = log.timer('cycle= %d'%(imacro+1), *cput1)
            
            if grad_norm_current < self.conv_tol_grad and max_const_err < self.constraint_tol and dE < self.conv_tol:
                scf_conv = True
                log.info('Coupled Newton-KKT Converged.')
                break
            
            # Save state before attempting step (for rejection rollback)
            mo_coeff_old = mo_coeff.copy()
            v_lagrange_old = self._scf.v_lagrange.copy()
            
            # TODO: Add shift if matrix is singular or step was previously bad
            # When gradient is small, MINRES needs tighter tolerance to provide accurate direction.
            # Otherwise, the residual error might exceed the predicted reduction.
            minres_tol = max(1e-9, min(1e-5, grad_norm_current * 0.1))

            dx, scale, dx_norm_raw = self.solve_trust_region_subproblem(
                A_op, b_vec, M_op, trust_radius, tol=minres_tol
            )
            
            # --- Trial Step Calculation (Start) ---
            n_con = len(constraints_res)
            n_orb_params = dx.size - n_con
            
            d_k = dx[:n_orb_params]
            d_v = dx[n_orb_params:]
            
            # Update Lagrange Multipliers (v) explicitly
            if d_v.size > 0:
                d_v_cpu = cp.asnumpy(d_v)
                self._scf.v_lagrange += d_v_cpu
                logger.info(self, f"SOSCF: Updated Multipliers. Max dV = {np.max(np.abs(d_v_cpu)):.2e}")

            # Update Orbitals (kappa) using the parent class method
            # super().update_rotate_matrix only computes the 'u' matrix for orbitals
            u = super().update_rotate_matrix(d_k, mo_occ, mo_coeff=mo_coeff)
            mo_coeff = self.rotate_mo(mo_coeff, u)
            # --- Trial Step Calculation (End) ---
            
            # Evaluate New State
            dm = self._scf.make_rdm1(mo_coeff, mo_occ)
            fock_new = self._scf.get_fock(dm=dm)
            
            A_op_new, M_op_new, b_vec_new, constraints_res_new = self.get_kkt_system(mo_coeff, mo_occ, fock_new)
            
            grad_norm_new = float(cp.linalg.norm(b_vec_new))
            actual_reduction = grad_norm_current - grad_norm_new
            
            # Previous logic (scale * grad_norm) assumes exact linear solution (A*dx = b).
            # Explicitly compute the model reduction: Pred = ||b|| - ||b - A*dx||.
            # This accounts for both TRM scaling and MINRES approximations.
            ax = A_op.matvec(dx)
            b_vec_pred = b_vec - ax # residule from minres
            grad_norm_pred = float(cp.linalg.norm(b_vec_pred))
            predicted_reduction = grad_norm_current - grad_norm_pred
            
            if predicted_reduction < 1e-15:
                 rho = 0.0 # Avoid division by zero
            else:
                 rho = actual_reduction / predicted_reduction
            
            log.info(f"TRM Check: rho={rho:.4f} (Act={actual_reduction:.2e}, Pred={predicted_reduction:.2e})")

            if rho < 0.25:
                # REJECT STEP
                # TODO: in the next iteration, there may be redundant calculations.
                log.info(f"Step Rejected (rho={rho:.4f}). Shrinking radius.")

                norm_dx = float(cp.linalg.norm(dx))
                if trust_radius > norm_dx:
                    trust_radius = norm_dx
                
                trust_radius *= 0.5
                if trust_radius < 1e-7 or norm_dx < 1e-9:
                    log.warn("Step size hit numerical precision limit. Forcing acceptance or stop.")

                # Restore previous state (Orbitals and Multipliers)
                mo_coeff = mo_coeff_old
                self._scf.v_lagrange = v_lagrange_old
                
                if trust_radius < min_radius:
                    log.warn("Trust radius hit minimum.")
                    # TODO: how to handle this?
                    # raise ValueError("Trust radius shrunk below minimum value. Convergence may be slow.")
                continue
                
            else:
                # ACCEPT STEP
                e_last = e_tot
                e_tot = self._scf.energy_tot(dm)
                
                A_op = A_op_new
                M_op = M_op_new
                b_vec = b_vec_new
                constraints_res = constraints_res_new
                grad_norm_current = grad_norm_new
                
                if rho > 0.75:
                    trust_radius = min(trust_radius * 2.0, max_radius)
                
                imacro += 1

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