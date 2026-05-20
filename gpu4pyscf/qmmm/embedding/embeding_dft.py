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

# Import your original DMET base class and helper functions
# from dmet import DMET, lowdin_orth, _as_cupy
from .dmet import DMET, lowdin_orth, _as_cupy


class SingleFragmentEmbedding(DMET):
    """
    Single-Fragment ONIOM-like Embedding driver inheriting from the DMET base class.
    
    This class overrides the initialization and kernel to perform a single-shot,
    single-fragment delta-method energy evaluation without macroscopic iterations.
    It rigorously traces over the entire active space (Fragment + Bath) to capture
    full polarization correlation, eliminating the 0.5 double-counting factor.
    """
    
    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-5, verbose=None):
        """
        Parameters
        ----------
        mf_outer : SCF object
            Low-level mean-field on the full system (e.g., PBE).
        mf_inner : SCF/DFT/post-HF object
            High-level template applied to the embedded cluster (e.g., B3LYP).
        fragment : list of int
            A single list of atom indices defining the QM region.
        threshold : float
            Eigenvalue cutoff used to classify environment orbitals.
        """
        # Wrap the single fragment into a list of lists to satisfy parent DMET __init__
        fragments = [fragment]
        
        # Initialize parent class. 
        # Force max_macro_iter=1 and energy_method='delta' strictly
        super().__init__(mf_outer, mf_inner, fragments,
                         threshold=threshold, max_macro_iter=1, 
                         energy_method='delta', verbose=verbose)
        
        # Expose the single fragment directly for user convenience
        self.fragment = self.fragments[0]
        
    def kernel(self):
        """
        Executes the single-shot embedding workflow.
        """
        # 1. Run Outer Mean-Field (if not already converged)
        if not self.mf_outer.converged:
            self.mf_outer.kernel()
            
        e_global_low = self.mf_outer.e_tot
        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        dm_full_ao_low = _as_cupy(self.mf_outer.make_rdm1())
        
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)

        ifrag = 0 # Strictly single fragment at index 0
        
        # 2. Schmidt Decomposition & Bath Construction using parent methods
        self.build_bath(ifrag, mo_coeff, mo_occ, X_inv, X)
        self.build_embedded_hamiltonian(ifrag, hcore_orig)
        
        # 3. Build and Run Inner embedded solver
        # _build_inner_mf already encapsulates the rigorous dual-functional core potential logic
        mf_inner = self._build_inner_mf(ifrag, dm_full_ao_low)
        self.log.info("Running high-level inner solver...")
        self.solve_embedded(ifrag)
        
        dm_emb_high = _as_cupy(mf_inner.make_rdm1())
        dm_emb_low = self.dm_emb_init[ifrag]
        
        B = self.B[ifrag]
        nemb = B.shape[1]
        is_mean_field = hasattr(self.mf_inner_template, 'get_veff')
        
        # 4. Evaluate Energy using strict Delta Method
        
        # --- Evaluate High-Level trace ---
        # Note: Trace is implicitly over the FULL active space (dm_emb_high * h_eval_high).
        # No 0.5 reduction factor is applied for the core potential since there are no other fragments.
        if is_mean_field:
            v_core_inner_ao = _as_cupy(self.mf_inner_template.get_veff(self.full_mol, self.dm_core[ifrag]))
            h_eval_high = B.T @ (hcore_orig + v_core_inner_ao) @ B
        else:
            h_eval_high = self.h_emb[ifrag]
            
        e_high = cp.sum(dm_emb_high * h_eval_high) 
        
        if is_mean_field:
            v_eff_emb_high = mf_inner.get_veff(dm=dm_emb_high)
            e_high += 0.5 * cp.sum(dm_emb_high * _as_cupy(v_eff_emb_high))
        else:
            # WFT evaluation over full active space (kept for future CCSD/MP2 extensions)
            B_cpu = cp.asnumpy(B)
            eri_emb_cpu = pyscf.ao2mo.kernel(self.full_mol, B_cpu)
            eri_emb_cpu = pyscf.ao2mo.restore(1, eri_emb_cpu, nemb)
            eri_emb = _as_cupy(eri_emb_cpu)
            
            if hasattr(mf_inner, 'make_rdm2'):
                dm2_emb_high = _as_cupy(mf_inner.make_rdm2())
            else:
                dm2_emb_high = (cp.einsum('ij,kl->ijkl', dm_emb_high, dm_emb_high) 
                           - 0.5 * cp.einsum('il,jk->ijkl', dm_emb_high, dm_emb_high))
            e_high += 0.5 * cp.sum(dm2_emb_high * eri_emb)
            
        # --- Evaluate Low-Level trace ---
        # self.h_emb strictly contains 1.0 * v_core_outer_ao natively
        h_eval_low = self.h_emb[ifrag] 
        e_low = cp.sum(dm_emb_low * h_eval_low)
        
        if is_mean_field:
            # Reconstruct full low-level density strictly from embedded projection
            dm_full_ao_low_reconstructed = self.dm_core[ifrag] + B @ dm_emb_low @ B.T
            v_eff_full_low = self.mf_outer.get_veff(self.full_mol, dm_full_ao_low_reconstructed)
            v_eff_active_low = _as_cupy(v_eff_full_low) - self.v_core_ao[ifrag]
            v_eff_emb_low = B.T @ v_eff_active_low @ B
            
            e_low += 0.5 * cp.sum(dm_emb_low * v_eff_emb_low)
        else:
            dm2_emb_low = (cp.einsum('ij,kl->ijkl', dm_emb_low, dm_emb_low) 
                       - 0.5 * cp.einsum('il,jk->ijkl', dm_emb_low, dm_emb_low))
            e_low += 0.5 * cp.sum(dm2_emb_low * eri_emb)
        
        # --- Assembly ---
        delta_e = float(e_high - e_low)
        self.log.note(f"Global Low-Level E : {e_global_low:.8f}")
        self.log.note(f"Active Space dE    : {delta_e:.8f}")
        
        self.e_tot = e_global_low + delta_e
        self.log.note(f"Total Embedded E   : {self.e_tot:.8f}")

        return self.e_tot