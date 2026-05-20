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
    Single-Fragment ONIOM-like embedding.
    
    This class performs a single-shot,
    single-fragment delta-method energy evaluation WITHOUT macroscopic iterations.
    It rigorously traces over the entire active space (Fragment + Bath) to capture
    full polarization correlation, eliminating the 0.5 double-counting factor.
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
                         threshold=threshold, max_macro_iter=1, 
                         energy_method='delta', verbose=verbose)
        
        # Expose the single fragment directly for user convenience
        self.fragment = self.fragments[0]
        
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
        is_mean_field = hasattr(self.mf_inner_template, 'get_veff')
        
        # Evaluate High-Level trace
        if is_mean_field:
            # Bare one-electron Hamiltonian trace
            h_eval_bare = B.T @ hcore_orig @ B
            e_high_h = cp.sum(dm_emb_high * h_eval_bare)
            
            # Full density reconstruction
            dm_full_ao_high = self.dm_core[ifrag] + B @ dm_emb_high @ B.T
            v_eff_full_high = self.mf_inner_template.get_veff(self.full_mol, dm_full_ao_high)
            
            # Coulomb J interaction traced over active space
            vj_full_high = getattr(v_eff_full_high, 'vj', None)
            vj_emb_high = B.T @ _as_cupy(vj_full_high) @ B
            e_high_J = 0.5 * cp.sum(dm_emb_high * vj_emb_high)
            
            # Exact Exchange interaction traced over active space + Grid XC extraction
            exc_tot_high = getattr(v_eff_full_high, 'exc', 0.0)
            vk_full_high = getattr(v_eff_full_high, 'vk', None)
            
            e_high_K = 0.0
            grid_exc_tot_high = exc_tot_high
            if vk_full_high is not None:
                vk_full_high = _as_cupy(vk_full_high)
                vk_emb_high = B.T @ vk_full_high @ B
                e_high_K = -0.5 * cp.sum(dm_emb_high * vk_emb_high)
                e_K_global_high = -0.5 * cp.sum(dm_full_ao_high * vk_full_high)
                # Isolate the pure non-linear grid integration part
                grid_exc_tot_high = exc_tot_high - e_K_global_high
                
            # Core evaluation for pure Grid XC subtraction
            v_eff_core_high = self.mf_inner_template.get_veff(self.full_mol, self.dm_core[ifrag])
            exc_core_high = getattr(v_eff_core_high, 'exc', 0.0)
            vk_core_high = getattr(v_eff_core_high, 'vk', None)
            
            grid_exc_core_high = exc_core_high
            if vk_core_high is not None:
                vk_core_high = _as_cupy(vk_core_high)
                e_K_global_core_high = -0.25 * cp.sum(self.dm_core[ifrag] * vk_core_high)
                grid_exc_core_high = exc_core_high - e_K_global_core_high
            
            e_high = e_high_h + e_high_J + e_high_K + grid_exc_tot_high - grid_exc_core_high
        else:
            raise NotImplementedError("WFT evaluation is not implemented for this class.")
            
        # Evaluate Low-Level trace (Exact Real-Space XC Integration)
        if is_mean_field:
            # 1. Bare one-electron Hamiltonian trace
            e_low_h = cp.sum(dm_emb_low * h_eval_bare)
            
            # Reconstruct full low-level density strictly from embedded projection
            dm_full_ao_low_reconstructed = self.dm_core[ifrag] + B @ dm_emb_low @ B.T
            v_eff_full_low = self.mf_outer.get_veff(self.full_mol, dm_full_ao_low_reconstructed)
            
            # 2. Coulomb (J) interaction traced over active space
            vj_full_low = getattr(v_eff_full_low, 'vj', None)
            if vj_full_low is None:
                vj_full_low = self.mf_outer.get_j(self.full_mol, dm_full_ao_low_reconstructed)
            vj_emb_low = B.T @ _as_cupy(vj_full_low) @ B
            e_low_J = 0.5 * cp.sum(dm_emb_low * vj_emb_low)
            
            # 3. Exact Exchange (K) interaction traced over active space + Grid XC extraction
            exc_tot_low = getattr(v_eff_full_low, 'exc', 0.0)
            vk_full_low = getattr(v_eff_full_low, 'vk', None)
            
            e_low_K = 0.0
            grid_exc_tot_low = exc_tot_low
            if vk_full_low is not None:
                vk_full_low = _as_cupy(vk_full_low)
                vk_emb_low = B.T @ vk_full_low @ B
                e_low_K = -0.5 * cp.sum(dm_emb_low * vk_emb_low)
                e_K_global_low = -0.5 * cp.sum(dm_full_ao_low_reconstructed * vk_full_low)
                # Isolate the pure non-linear grid integration part
                grid_exc_tot_low = exc_tot_low - e_K_global_low
                
            # Core evaluation for pure Grid XC subtraction
            v_eff_core_low = self.mf_outer.get_veff(self.full_mol, self.dm_core[ifrag])
            exc_core_low = getattr(v_eff_core_low, 'exc', 0.0)
            vk_core_low = getattr(v_eff_core_low, 'vk', None)
            
            grid_exc_core_low = exc_core_low
            if vk_core_low is not None:
                vk_core_low = _as_cupy(vk_core_low)
                e_K_global_core_low = -0.25 * cp.sum(self.dm_core[ifrag] * vk_core_low)
                grid_exc_core_low = exc_core_low - e_K_global_core_low
                
            e_low = e_low_h + e_low_J + e_low_K + grid_exc_tot_low - grid_exc_core_low
        else:
            raise NotImplementedError("WFT evaluation is not implemented for this class.")
        
        # Assembly
        delta_e = float(e_high - e_low)
        self.log.note(f"Global Low-Level E : {e_global_low:.8f}")
        self.log.note(f"Active Space dE    : {delta_e:.8f}")
        
        self.e_tot = e_global_low + delta_e
        self.log.note(f"Total Embedded E   : {self.e_tot:.8f}")

        return self.e_tot