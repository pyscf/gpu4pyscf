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
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.dft import krks
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf.hf import damping
from gpu4pyscf.dft.rkspu import reference_mol
from gpu4pyscf.dft.ucdft import _get_minao_basis_indices
from gpu4pyscf.pbc.dft.krkspu import _make_minao_lo
from gpu4pyscf.dft.ucdft import normalize_constraints


class CDFT_KUKS(dft.KUKS):
    '''
    Constrained DFT implementation for PBC Unrestricted Kohn-Sham with K-points.
    '''

    def __init__(self, cell, kpts, charge_constraints=None, spin_constraints=None, 
                 method='lagrange', penalty_weight=500.0,
                 projection_method='minao'):
        super().__init__(cell, kpts)
        
        self.charge_groups, self.charge_targets = normalize_constraints(charge_constraints)
        self.spin_groups, self.spin_targets = normalize_constraints(spin_constraints)
        
        self.n_constraints = len(self.charge_targets) + len(self.spin_targets)
        
        self.v_lagrange = np.zeros(self.n_constraints) + 0.01
        
        self.micro_tol = 1e-4
        self.micro_max_cycle = 50
        
        self.constraint_projectors = None

        self.method = method.lower()
        self.penalty_weight = penalty_weight
        self.projection_method = projection_method.lower()
        self.minao_ref = 'MINAO'

    def build_projectors(self):
        if self.constraint_projectors is not None:
            return self.constraint_projectors

        if self.projection_method == 'minao':
            return self._build_minao_projectors()
        else:
            raise NotImplementedError(f"Projection method {self.projection_method} not supported in PBC.")

    def _build_minao_projectors(self):
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        
        logger.info(self, "CDFT: Building projectors using Orthogonalized MINAO reference (PBC)...")

        ovlp = int1e.int1e_ovlp(cell, kpts)
        C_minao = _make_minao_lo(cell, self.minao_ref, kpts)
        pcell = reference_mol(cell, self.minao_ref)
        SC = contract('kpq,kqr->kpr', ovlp, C_minao)
        
        projectors = []
        all_groups = self.charge_groups + self.spin_groups
        
        for group in all_groups:
            indices = []
            
            for identifier in group:
                idx_list = _get_minao_basis_indices(pcell, identifier)
                indices.extend(idx_list)
            
            if not indices:
                projectors.append(cp.zeros((nkpts, nao, nao)))
                continue
            
            indices = sorted(list(set(indices)))
            indices_gpu = cp.asarray(indices)
            sc_subset = SC[:, :, indices_gpu]
            
            w_matrix = contract('kpi,kqi->kpq', sc_subset, sc_subset.conj())
            
            projectors.append(w_matrix)
            
        self.constraint_projectors = projectors
        return self.constraint_projectors

    def get_constraint_potential(self, v_vec):
        '''
        Constructs V_const (nkpts, nao, nao).
        '''
        if self.constraint_projectors is None:
            self.build_projectors()
            
        nkpts = len(self.kpts)
        nao = self.cell.nao_nr()
        vc_a = cp.zeros((nkpts, nao, nao), dtype=cp.complex128)
        vc_b = cp.zeros((nkpts, nao, nao), dtype=cp.complex128)
        
        n_charge = len(self.charge_targets)
        
        for i in range(n_charge):
            v = float(v_vec[i])
            w = self.constraint_projectors[i] # (nkpts, nao, nao)
            
            vc_a += v * w
            vc_b += v * w
            
        for i in range(len(self.spin_targets)):
            idx = n_charge + i
            v = float(v_vec[idx])
            w = self.constraint_projectors[idx]
                
            vc_a += v * w
            vc_b -= v * w

        return (vc_a, vc_b)

    def _micro_objective_func(self, v_vec, f_std_a, f_std_b, s):
        '''
        Objective function for root finding.
        '''
        vc_a, vc_b = self.get_constraint_potential(v_vec)
        f_tot_a = f_std_a + vc_a
        f_tot_b = f_std_b + vc_b
        
        f_tot = (f_tot_a, f_tot_b) 
        
        mo_e, mo_c = self.eig(f_tot, s)
        
        mo_occ = self.get_occ(mo_e, mo_c)
        
        nkpts = len(self.kpts)
        
        dm_a, dm_b = self.make_rdm1(mo_c, mo_occ)
        
        if hasattr(self.kpts, "weights_ibz"):
            k_weights = cp.asarray(self.kpts.weights_ibz)
        else:
            k_weights = cp.full(nkpts, 1.0/nkpts)

        def compute_pop(d, w):
            t = contract('kij,kji->k', d, w)
            return cp.sum(t * k_weights).real

        errors = []
        projector_idx = 0
        
        for target in self.charge_targets:
            w = self.constraint_projectors[projector_idx]
            val = compute_pop(dm_a, w) + compute_pop(dm_b, w)
            errors.append(float(val) - target)
            projector_idx += 1
            
        for target in self.spin_targets:
            w = self.constraint_projectors[projector_idx]
            val = compute_pop(dm_a, w) - compute_pop(dm_b, w)
            errors.append(float(val) - target)
            projector_idx += 1

        return np.array(errors)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
        
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.cell, dm)
        if s1e is None: s1e = self.get_ovlp()
        if dm is None: dm = self.make_rdm1()
        
        h1e = cp.asarray(h1e)
        vhf = cp.asarray(vhf)
        
        if vhf.ndim == 4 and vhf.shape[1] == 2: 
            vhf = (vhf[:,0], vhf[:,1])
        elif vhf.ndim == 5: 
             vhf = (vhf[0], vhf[1])
             
        f = (h1e + vhf[0], h1e + vhf[1])
        
        dm = cp.asarray(dm)

        if self.n_constraints != 0:
            run_micro = False
            if cycle != -1:
                run_micro = True
            
            if self.method == 'lagrange':
                if run_micro:
                    f_a, f_b = f
                    res = root(
                        fun=self._micro_objective_func,
                        x0=self.v_lagrange,
                        args=(f_a, f_b, s1e),
                        method='hybr',
                        tol=self.micro_tol,
                        options={'maxfev': self.micro_max_cycle}
                    )
                    if not res.success:
                        logger.warn(self, f"Micro optimization did not converge: {res.message}")
                    self.v_lagrange = res.x
                    logger.info(self, f"Cycle {cycle}: Optimized V = {self.v_lagrange}")
                    
                vc_a, vc_b = self.get_constraint_potential(self.v_lagrange)
                f = (f[0] + vc_a, f[1] + vc_b)
                              
            elif self.method == 'penalty':                
                raise NotImplementedError("Penalty method not implemented for PBC CDFT")

        if cycle < 0 and diis is None:
            return f

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if damp_factor is None:
            damp_factor = self.damp
            
        if damp_factor is not None and 0 <= cycle < diis_start_cycle-1 and fock_last is not None:
             if isinstance(damp_factor, (tuple, list, np.ndarray)):
                dampa, dampb = damp_factor
             else:
                dampa = dampb = damp_factor
             f = (damping(f[0], fock_last[0], dampa),
                  damping(f[1], fock_last[1], dampb))

        if diis and cycle >= diis_start_cycle:
            f_stack = cp.stack(f)
            f_stack = diis.update(s1e, dm, f_stack)
            f = (f_stack[0], f_stack[1])

        return f