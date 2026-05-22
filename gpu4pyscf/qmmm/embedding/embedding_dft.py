# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import pyscf.ao2mo
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.qmmm.embedding.embedding import DMET, lowdin_orth, _as_cupy


class SingleFragmentEmbedding(DMET):
    """
    Single-Fragment ONIOM-like embedding for DFT.
    
    This class performs a single-shot,
    single-fragment delta-method energy evaluation WITHOUT macroscopic iterations.
    """
    
    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-5, verbose=None):
        """
        Parameters
       -------
        mf_outer : SCF object
            Low-level mean-field on the full system (e.g., PBE).
        mf_inner : SCF/DFT/post-HF object
            High-level template applied to the embedded cluster (e.g., B3LYP).
        fragment : list of int
            A single list of atom indices defining the QM region.
        threshold : float
            Eigenvalue cutoff used to classify environment orbitals.
        """
        fragments = [fragment]
        
        super().__init__(mf_outer, mf_inner, fragments,
                         threshold=threshold, max_macro_iter=1, verbose=verbose)
        
        self.fragment = self.fragments[0]
        
    def _evaluate_embedded_energy(self, mf_obj, dm_emb, h_eval_bare, B, dm_core):
        e_h = cp.sum(dm_emb * h_eval_bare)
        
        # Full density reconstruction
        dm_full_ao = dm_core + B @ dm_emb @ B.T
        v_eff_full = mf_obj.get_veff(self.full_mol, dm_full_ao)
        
        # Coulomb J interaction traced over active space
        vj_full = getattr(v_eff_full, 'vj', None)
        if vj_full is None:
            vj_full = mf_obj.get_j(self.full_mol, dm_full_ao)
        vj_emb = B.T @ _as_cupy(vj_full) @ B
        e_J = 0.5 * cp.sum(dm_emb * vj_emb)
        
        # Exact Exchange interaction traced over active space + Grid XC extraction
        exc_tot = getattr(v_eff_full, 'exc', 0.0)
        vk_full = getattr(v_eff_full, 'vk', None)
        
        e_K = 0.0
        grid_exc_tot = exc_tot
        if vk_full is not None:
            vk_full = _as_cupy(vk_full)
            vk_emb = B.T @ vk_full @ B
            e_K = -0.5 * cp.sum(dm_emb * vk_emb)
            e_K_global = -0.5 * cp.sum(dm_full_ao * vk_full)
            # Isolate the pure non-linear grid integration part
            grid_exc_tot = exc_tot - e_K_global
            
        # Core evaluation for pure Grid XC subtraction
        v_eff_core = mf_obj.get_veff(self.full_mol, dm_core)
        exc_core = getattr(v_eff_core, 'exc', 0.0)
        vk_core = getattr(v_eff_core, 'vk', None)
        
        grid_exc_core = exc_core
        if vk_core is not None:
            vk_core = _as_cupy(vk_core)
            e_K_global_core = -0.5 * cp.sum(dm_core * vk_core)
            grid_exc_core = exc_core - e_K_global_core
        
        return e_h + e_J + e_K + grid_exc_tot - grid_exc_core

    def kernel(self):
        if not self.mf_outer.converged:
            self.mf_outer.kernel()
            
        e_global_low = self.mf_outer.e_tot
        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        dm_full_ao_low = _as_cupy(self.mf_outer.make_rdm1())
        
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)

        ifrag = 0
        
        self.build_bath(ifrag, mo_coeff, mo_occ, X_inv, X)
        self.build_embedded_hamiltonian(ifrag, hcore_orig)
        
        # Build and Run Inner embedded solver
        mf_inner = self._build_inner_mf(ifrag, dm_full_ao_low)
        self.log.info("Running high-level inner solver...")
        self.solve_embedded(ifrag)
        
        dm_emb_high = _as_cupy(mf_inner.make_rdm1())
        dm_emb_low = self.dm_emb_init[ifrag]
        
        B = self.B[ifrag]
        dm_core = self.dm_core[ifrag]
        is_mean_field = hasattr(self.mf_inner_template, 'get_veff')
        
        if is_mean_field:
            h_eval_bare = B.T @ hcore_orig @ B
            
            # Evaluate High-Level energy
            e_high = self._evaluate_embedded_energy(
                self.mf_inner_template, dm_emb_high, h_eval_bare, B, dm_core
            )
            
            # Evaluate Low-Level energy
            e_low = self._evaluate_embedded_energy(
                self.mf_outer, dm_emb_low, h_eval_bare, B, dm_core
            )
        else:
            raise NotImplementedError("WFT evaluation is not implemented for this class.")
        
        delta_e = float(e_high - e_low)
        self.log.note(f"Global Low-Level E : {e_global_low:.8f}")
        self.log.note(f"Active Space dE    : {delta_e:.8f}")
        
        self.e_tot = e_global_low + delta_e
        self.log.note(f"Total Embedded E   : {self.e_tot:.8f}")
        
        return self.e_tot