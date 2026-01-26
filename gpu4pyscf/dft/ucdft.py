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
from scipy.optimize import root
from gpu4pyscf.scf.hf import damping, level_shift
from pyscf import lib as pyscf_lib
from pyscf import gto
from gpu4pyscf.dft.rkspu import reference_mol, _make_minao_lo
from gpu4pyscf import dft
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import radi

def normalize_constraints(constraints):
    '''
    Output: [[[group1_atom1, group1_atom2, ...], [group2_atom1, ...]], [target1, target2, ...]]
    This is not the same as DFT+U, may be modified in the future.
    '''
    if not constraints:
        return [], []
        
    groups = constraints[0]
    targets = constraints[1]
        
    if len(groups) != len(targets):
        raise ValueError("The number of groups must match the number of targets.")

    normalized_groups = []
    for constraint_indices in groups:
        if isinstance(constraint_indices, (int, np.integer, str)):
            normalized_groups.append([constraint_indices])
        else:
            normalized_groups.append(list(constraint_indices))
                
    return normalized_groups, targets

def _get_minao_basis_indices(minao_mol, identifier):
    """
    Convert atom ID or orbital label to AO indices.
    """
    minao_slices = minao_mol.aoslice_by_atom()
    
    if isinstance(identifier, (int, np.integer)):
        atom_id = int(identifier)
        p0 = minao_slices[atom_id, 2]
        p1 = minao_slices[atom_id, 3]
        return list(range(p0, p1))
    
    elif isinstance(identifier, str):
        return minao_mol.search_ao_label(identifier)
    
    else:
        raise ValueError(f"Unsupported identifier type {type(identifier)}: {identifier}")

class CDFTBaseMixin:

    _keys = {'charge_groups', 'charge_targets', 'spin_groups', 'spin_targets',
             'n_constraints', 'v_lagrange', 'micro_tol', 'micro_max_cycle',
             'constraint_projectors', 'method', 'penalty_weight', 'projection_method',
             'minao_ref'}

    def init_cdft_params(self, charge_constraints=None, spin_constraints=None, 
                         method='lagrange', penalty_weight=500.0,
                         projection_method='minao'):
        self.charge_groups, self.charge_targets = normalize_constraints(charge_constraints)
        self.spin_groups, self.spin_targets = normalize_constraints(spin_constraints)
        
        self.n_constraints = len(self.charge_targets) + len(self.spin_targets)
        
        # Initial guess for Lagrange multipliers V
        self.v_lagrange = np.zeros(self.n_constraints) + 0.01 # FIXME: remove hardcoded guess
        
        # Microiteration parameters
        # FIXME: modify hardcoded values.
        self.micro_tol = 1e-4
        self.micro_max_cycle = 50
        
        # List of matrices, one for each constraint (Charge first, then Spin)
        self.constraint_projectors = None

        self.method = method.lower() # 'lagrange' or 'penalty'
        self.penalty_weight = penalty_weight
        self.projection_method = projection_method.lower()
        self.minao_ref = 'MINAO'

    def build_projectors(self):
        if self.constraint_projectors is not None:
            return self.constraint_projectors

        if self.projection_method == 'minao':
            return self._build_minao_projectors()
        else:
            return self._build_becke_projectors()

    def get_constraint_potential(self, v_vec):
        '''
        Constructs V_const matrix.
        Compatible with both Molecular and PBC.
        '''
        if self.constraint_projectors is None:
            self.build_projectors()
        
        # If no projectors, return 0 (scalar broadcasts correctly in most operations)
        if not self.constraint_projectors:
            return 0.0, 0.0

        # Initialize with zeros matching the shape/dtype of the first projector
        vc_a = cp.zeros_like(self.constraint_projectors[0])
        vc_b = cp.zeros_like(self.constraint_projectors[0])
        
        n_charge = len(self.charge_targets)
        
        for i in range(n_charge):
            v = float(v_vec[i])
            w = self.constraint_projectors[i]
            
            vc_a += v * w
            vc_b += v * w
            
        for i in range(len(self.spin_targets)):
            idx = n_charge + i
            v = float(v_vec[idx])
            w = self.constraint_projectors[idx]
                
            vc_a += v * w
            vc_b -= v * w

        return cp.asarray([vc_a, vc_b])

    def update_fock_with_constraints(self, f, s1e, dm, cycle):
        if self.n_constraints == 0:
            return f

        run_micro = False
        if cycle != -1:
            run_micro = True
        
        if self.method == 'lagrange':
            if run_micro:
                res = root(
                    fun=self._micro_objective_func,
                    x0=self.v_lagrange,
                    args=(f[0], f[1], s1e),
                    method='hybr',
                    tol=self.micro_tol,
                    options={'maxfev': self.micro_max_cycle}
                )
                if not res.success:
                    logger.warn(self, f"Micro optimization did not converge: {res.message}")
                self.v_lagrange = res.x
                
            vc = self.get_constraint_potential(self.v_lagrange)
            logger.info(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
            f = cp.asarray(f) + vc
                          
        elif self.method == 'penalty':
            if not hasattr(self, 'get_penalty_potential'):
                raise NotImplementedError("Penalty method not implemented for this class.")
            
            vc = self.get_penalty_potential(dm, self.penalty_weight)
            f = cp.asarray(f) + vc
            logger.info(self, f"Cycle {cycle}: Applied Penalty (Weight={self.penalty_weight:.1f})")

        return f

class CDFT_UKS(CDFTBaseMixin, dft.UKS):
    '''
    Constrained DFT implementation for unrestricted Kohn-Sham.
    Implements spatial charge/spin constraints using either Becke weights or MINAO projection.
    Supports two modes: 'lagrange' (Lagrange Multipliers) and 'penalty' (Quadratic Penalty).
    Supports constraints on whole atoms or specific orbitals (MINAO only).

    Becke partition will be deprecated in the future.
    '''

    def __init__(self, mol, charge_constraints=None, spin_constraints=None, 
                 method='lagrange', penalty_weight=500.0,
                 projection_method='minao'):
        super().__init__(mol)
        self.init_cdft_params(charge_constraints, spin_constraints, 
                              method, penalty_weight, projection_method)

    def _build_minao_projectors(self):
        """
        Constructs constraint weight matrices 'W' using Projected Lowdin Orthogonalization.
        This follows the exact logic used in PySCF/GPU4PySCF DFT+U implementation.
        Supports specific orbital selection (e.g., 'Ni 0 3d') via search_ao_label.
        """
        mol = self.mol
        nao = mol.nao_nr()
        logger.info(self, "Building projectors using Orthogonalized MINAO reference (DFT+U style)...")

        ovlp = self.get_ovlp()
        if isinstance(self.minao_ref, str):
            minao_mol = reference_mol(mol, self.minao_ref)
        else:
            minao_mol = self.minao_ref
        C_minao = _make_minao_lo(mol, minao_mol)
        SC = ovlp @ C_minao
        
        projectors = []
        all_groups = self.charge_groups + self.spin_groups
        
        for group in all_groups:
            indices = []
            
            for label_or_id in group:
                idx_list = _get_minao_basis_indices(minao_mol, label_or_id)
                indices.extend(idx_list)
            
            if not indices:
                projectors.append(cp.zeros((nao, nao)))
                continue
            
            indices = sorted(list(set(indices)))
            indices = cp.asarray(indices)
            
            sc_subset = SC[:, indices]
            
            W = sc_subset @ sc_subset.conj().T
            
            projectors.append(W)
            
        self.constraint_projectors = projectors
        return self.constraint_projectors

    def _build_becke_projectors(self):
        r'''
        Constructs constraint weight matrices 'w' on GPU using Becke partitioning.
        
        Ported from gen_grid_partition (grids_response_cc)
        It only supports atom-based constraints.
        '''
        mol = self.mol
        nao = mol.nao_nr()
        logger.warn(self, "Building projectors using Becke partitioning (will be deprecated in future)...")
        
        all_groups = self.charge_groups + self.spin_groups
        for group in all_groups:
            for item in group:
                if isinstance(item, str):
                    raise ValueError(f"Becke projection does not support orbital-specific constraints ('{item}'). "
                                      "Please use 'minao' projection method or use atom indices.")

        atm_coords = np.asarray(mol.atom_coords(), order='C')
        atm_dist = gto.inter_distance(mol, atm_coords)
        atm_dist = cp.asarray(atm_dist)
        atm_coords = cp.asarray(atm_coords)

        grids = self.grids
        atomic_radii = grids.atomic_radii
        radii_adjust_method = grids.radii_adjust

        def _radii_adjust(mol, atomic_radii):
            charges = mol.atom_charges()
            if radii_adjust_method == radi.treutler_atomic_radii_adjust:
                rad = np.sqrt(atomic_radii[charges]) + 1e-200
            elif radii_adjust_method == radi.becke_atomic_radii_adjust:
                rad = atomic_radii[charges] + 1e-200
            else:
                fadjust = lambda i, j, g: g
                return fadjust

            rr = rad.reshape(-1,1) * (1./rad)
            a = .25 * (rr.T - rr)
            a[a<-.5] = -.5
            a[a>0.5] = 0.5
            
            a_gpu = cp.asarray(a)

            def fadjust(i, j, g):
                return g + a_gpu[i,j]*(1-g**2)
            return fadjust

        fadjust = _radii_adjust(mol, atomic_radii)

        atom_projectors_temp = [cp.zeros((nao, nao)) for _ in range(mol.natm)]

        if self.grids.coords is None:
            self.grids.build()
            
        coords_all = self.grids.coords
        weights_all = self.grids.weights
        ni = self._numint
        
        blksize = 4000
        n_grids_total = weights_all.shape[0]
        
        logger.info(self, f"Building Becke projectors on GPU using {n_grids_total} grid points...")
        
        for p0, p1 in pyscf_lib.prange(0, n_grids_total, blksize):
            coords_batch = cp.asarray(coords_all[p0:p1])
            weights_batch = cp.asarray(weights_all[p0:p1])
            ngrids_batch = coords_batch.shape[0]
            ao_batch = ni.eval_ao(mol, coords_batch, deriv=0)
            
            grid_dist = []
            
            for ia in range(mol.natm):
                v = (atm_coords[ia] - coords_batch).T
                normv = cp.linalg.norm(v, axis=0) + 1e-200
                grid_dist.append(normv)
            
            pbecke = cp.ones((mol.natm, ngrids_batch))
            
            for ia in range(mol.natm):
                for ib in range(ia):
                    g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                    p0 = fadjust(ia, ib, g)
                    p1 = (3 - p0**2) * p0 * .5
                    p2 = (3 - p1**2) * p1 * .5
                    p3 = (3 - p2**2) * p2 * .5

                    s_uab = .5 * (1 - p3 + 1e-200)
                    s_uba = .5 * (1 + p3 + 1e-200)

                    pbecke[ia] *= s_uab
                    pbecke[ib] *= s_uba
            
            z = 1./pbecke.sum(axis=0)

            for atom_id in range(mol.natm):
                w_atom = weights_batch * pbecke[atom_id] * z
                ao_weighted = ao_batch * w_atom[:, None]
                atom_projectors_temp[atom_id] += ao_batch.T @ ao_weighted
        
        projectors = []
        for group in all_groups:
            w_sum = cp.zeros((nao, nao))
            for atom_id in group:
                atom_id = int(atom_id)
                w_sum += atom_projectors_temp[atom_id]
            projectors.append(w_sum)
        
        self.constraint_projectors = projectors
        return self.constraint_projectors

    def get_penalty_potential(self, dm, penalty_value):
        '''
        Constructs V_const matrix based on the quadratic penalty method.
        V_shift = sum( 2 * lambda * (Pop_calc - Pop_target) * W )
        '''
        if self.constraint_projectors is None:
            self.build_projectors()
        
        nao = self.mol.nao_nr()
        vc_a = cp.zeros((nao, nao))
        vc_b = cp.zeros((nao, nao))
        
        dm_a, dm_b = dm[0], dm[1]

        log_msgs = []
        projector_idx = 0

        for i, target in enumerate(self.charge_targets):
            w = self.constraint_projectors[projector_idx]
            
            val = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
            n_val = float(val)
            
            diff = n_val - target
            log_msgs.append(f"Chg[{i}] err: {diff:.5f}")
            
            shift = 2.0 * penalty_value * diff
            
            vc_a += shift * w
            vc_b += shift * w
            projector_idx += 1

        for i, target in enumerate(self.spin_targets):
            w = self.constraint_projectors[projector_idx]
            
            val = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
            m_val = float(val)
                
            diff = m_val - target
            log_msgs.append(f"Spin[{i}] err: {diff:.5f}")

            shift = 2.0 * penalty_value * diff
            
            vc_a += shift * w
            vc_b -= shift * w
            projector_idx += 1

        if log_msgs:
            logger.info(self, f"CDFT Penalty (L={penalty_value:.1f}): " + ", ".join(log_msgs))

        return cp.asarray([vc_a, vc_b])

    def _micro_objective_func(self, v_vec, f_std_a, f_std_b, s):
        '''
        Objective function for scipy.optimize.root.
        Returns the error vector (calculated_val - target).
        '''
        vc_a, vc_b = self.get_constraint_potential(v_vec)
        f_tot_a = f_std_a + vc_a
        f_tot_b = f_std_b + vc_b
        
        mo_e, mo_c = self.eig((f_tot_a, f_tot_b), s)
        mo_c_a, mo_c_b = mo_c
        nocc_a = self.mol.nelec[0]
        nocc_b = self.mol.nelec[1]
        
        dm_a = mo_c_a[:, :nocc_a] @ mo_c_a[:, :nocc_a].T
        dm_b = mo_c_b[:, :nocc_b] @ mo_c_b[:, :nocc_b].T
        
        errors = []
        projector_idx = 0
        
        for target in self.charge_targets:
            w = self.constraint_projectors[projector_idx]
            val = cp.trace(dm_a @ w) + cp.trace(dm_b @ w)
            errors.append(float(val) - target)
            projector_idx += 1
            
        for target in self.spin_targets:
            w = self.constraint_projectors[projector_idx]
            val = cp.trace(dm_a @ w) - cp.trace(dm_b @ w)
            errors.append(float(val) - target)
            projector_idx += 1

        return np.array(errors)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if s1e is None: s1e = self.get_ovlp()
        h1e = cp.asarray(h1e)
        vhf = cp.asarray(vhf)
        f = h1e + vhf
        if f.ndim == 2:
            f = (f, f)
        
        if dm is None: 
            dm = self.make_rdm1()
        dm = cp.asarray(dm)

        f = self.update_fock_with_constraints(f, s1e, dm, cycle)
        
        if isinstance(f, tuple):
            f = cp.stack(f)

        if cycle < 0 and diis is None:
            return f

        s1e = cp.asarray(s1e)

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if damp_factor is None:
            damp_factor = self.damp
        if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
            if isinstance(damp_factor, (tuple, list, np.ndarray)):
                dampa, dampb = damp_factor
            else:
                dampa = dampb = damp_factor
            f = cp.asarray((damping(f[0], fock_last[0], dampa),
                            damping(f[1], fock_last[1], dampb)))
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm, f)

        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if level_shift_factor is not None:
            if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
                shifta, shiftb = level_shift_factor
            else:
                shifta = shiftb = level_shift_factor
            f = (level_shift(s1e, dm[0], f[0], shifta),
                level_shift(s1e, dm[1], f[1], shiftb))
        return f

    def get_canonical_mo(self, dm=None):
        '''
        Diagonalize the standard Fock matrix (without constraint potentials) 
        using the converged density matrix to obtain canonical orbitals and energies.
        '''
        if dm is None:
            dm = self.make_rdm1()
            
        dm = cp.asarray(dm)
        s1e = self.get_ovlp()
        h1e = self.get_hcore()
        vhf = self.get_veff(self.mol, dm)
        f_std = cp.asarray(h1e) + cp.asarray(vhf)
        mo_energy, mo_coeff = self.eig(f_std, s1e)
        
        return mo_energy.get(), mo_coeff.get()

    def newton(self):
        from gpu4pyscf.dft.cdft_soscf_full import newton_cdft
        return newton_cdft(self)

    def newton_penalty(self):
        from gpu4pyscf.dft.cdft_soscf import newton_cdft
        return newton_cdft(self)
    
    def Gradients(self):
        from gpu4pyscf.grad.ucdft import Gradients
        return Gradients(self)

    def nuc_grad_method(self):
        return self.Gradients()
    
    def reset(self, mol=None):
        raise NotImplementedError('reset method is not implemented for CDFT_UKS')