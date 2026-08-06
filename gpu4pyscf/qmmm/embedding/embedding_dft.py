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
        e_h_active = float(cp.sum(dm_emb * h_eval_bare))
        
        dm_full_ao = dm_core + B @ dm_emb @ B.T
        
        v_eff_full = mf_obj.get_veff(self.full_mol, dm_full_ao)
        e_2e_full = float(getattr(v_eff_full, 'ecoul', 0.0) + getattr(v_eff_full, 'exc', 0.0))
        
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        e_1e_core = float(cp.sum(dm_core * hcore_orig))
        
        e_nuc = float(self.full_mol.energy_nuc())
        return e_nuc + e_1e_core + e_h_active + e_2e_full

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
        
        B_mat = self.B[ifrag]
        dm_core_mat = self.dm_core[ifrag]
        h_eval_bare_mat = B_mat.T @ hcore_orig @ B_mat

        # Add the missing core 1-electron energy (kinetic + nuclear attraction from the frozen core)
        e1_core = float(cp.sum(dm_core_mat * hcore_orig))
        
        # Precompute the frozen core's 2-electron energy (constant during inner SCF)
        v_eff_core_high = self.mf_inner_template.get_veff(self.full_mol, dm_core_mat)
        e_coul_core = float(getattr(v_eff_core_high, 'ecoul', 0.0))
        e_xc_core = float(getattr(v_eff_core_high, 'exc', 0.0))
        
        e_nuc_full = float(self.full_mol.energy_nuc())
        mf_inner.energy_nuc = lambda *args, **kwargs: e_nuc_full
        
        # Override energy_elec to print the true ONIOM energy difference
        def custom_energy_elec(dm=None, h1e=None, vhf=None):
            if dm is None: dm = mf_inner.make_rdm1()
            if vhf is None: vhf = mf_inner.get_veff(mf_inner.mol, dm)
            
            dm_cp = _as_cupy(dm)
            
            # e1: Active space single-electron energy + Core single-electron energy
            e1_active = float(cp.sum(dm_cp * h_eval_bare_mat))
            e1 = e1_active + e1_core
            
            # e2: Full system 2e energy minus core 2e energy
            ecoul_full = float(getattr(vhf, 'ecoul', 0.0))
            exc_full = float(getattr(vhf, 'exc', 0.0))
            e2 = ecoul_full + exc_full
            
            # Update scf_summary for meaningful PySCF debugging output
            mf_inner.scf_summary['e1'] = e1
            mf_inner.scf_summary['coul'] = ecoul_full - e_coul_core
            mf_inner.scf_summary['exc'] = exc_full - e_xc_core
            
            return e1 + e2, e2
            
        mf_inner.energy_elec = custom_energy_elec
        
        self.log.info("Running high-level inner solver...")
        self.solve_embedded(ifrag)
        if not self.mf_inner[ifrag].converged:
            raise RuntimeError(
                f"Embedded high-level SCF did not converge for fragment {ifrag}; "
                "do not use this density for delta energy."
            )
        
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