# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from gpu4pyscf.dft import rks
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.qmmm.embedding.embedding import DMET, lowdin_orth, _as_cupy
from gpu4pyscf.qmmm.embedding.embedding_dft import SingleFragmentEmbedding


class HarrisRKS(rks.RKS):
    """
    Harris RKS class based on machine learning (ML) predicted density.
    
    This class bypasses traditional SCF iterations. Instead, it relies entirely 
    on an external ML density evaluation function to construct the global effective 
    potential and calculate the double counting energy.
    """
    def __init__(self, mol, eval_density_func, xc='LDA,VWN'):
        super().__init__(mol)
        self.xc = xc
        self.max_cycle = 1  
        
        # eval_density_func is the external ML interface.
        # Signature: def func(mol, xc, grids)
        # Returns 7 elements:
        #   1. vj: Coulomb potential matrix (AO basis)
        #   2. vk: Exact exchange potential matrix (AO basis, can be None for pure DFT)
        #   3. vxc: Exchange-correlation potential matrix (AO basis)
        #   4. e_j: Coulomb energy (scalar)
        #   5. e_k: Exact exchange energy (scalar, can be 0.0 for pure DFT)
        #   6. e_xc: Exchange-correlation energy (scalar)
        #   7. int_rho_vxc: Integral of rho * V_xc (scalar)
        self.eval_density_func = eval_density_func
        
        self._v_eff_global = None
        self._e_dc_global = None
        self._use_harris_veff = False

    def _get_harris_veff(self, mol=None):
        if mol is None: 
            mol = self.mol
        
        if self._v_eff_global is not None:
            return self._v_eff_global
            
        if self.grids.coords is None:
            self.grids.build()
            
        vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc = self.eval_density_func(
            mol, self.xc, self.grids)
        
        v_eff_ao = _as_cupy(vj) + _as_cupy(vxc)
        if vk is not None:
            v_eff_ao -= _as_cupy(vk)
            e_k = float(e_k)
        else:
            e_k = 0.0
            
        # double counting energy
        e_dc = float(e_j) - e_k + float(int_rho_vxc) - float(e_xc)
        
        vk_array = _as_cupy(vk) if vk is not None else cp.zeros_like(v_eff_ao)
        v_eff_ao = tag_array(v_eff_ao, ecoul=float(e_j) - e_k, exc=float(e_xc), vj=_as_cupy(vj), vk=vk_array)
        
        self._v_eff_global = v_eff_ao
        self._e_dc_global = e_dc
        return self._v_eff_global

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # Use ML evaluation ONLY during the global SCF step.
        # For standard embedding steps, fallback to the native exact DFT evaluation.
        if getattr(self, '_use_harris_veff', False):
            return self._get_harris_veff(mol)
        return rks.RKS.get_veff(self, mol, dm, dm_last, vhf_last, hermi)

    def kernel(self, dm0=None, **kwargs):

        if self.max_cycle != 1:
            lib.logger.warn(self, "HarrisRKS is a non-iterative method. "
                                  f"Overriding max_cycle from {self.max_cycle} to 1.")
            self.max_cycle = 1

        # Temporarily enable Harris ML potential for the global 1-step evaluation
        self._use_harris_veff = True
        try:
            e_tot = rks.RKS.kernel(self, dm0=dm0, **kwargs)
        finally:
            self._use_harris_veff = False
            
        self.converged = True
        return e_tot

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """
        E_elec = Tr[D * (h + Veff)] - E_DC
        """
        if getattr(self, '_use_harris_veff', False):
            if dm is None: dm = self.make_rdm1()
            if h1e is None: h1e = self.get_hcore()
            if vhf is None: vhf = self._get_harris_veff(self.mol)
            
            dm_cp = _as_cupy(dm)
            h1e_cp = _as_cupy(h1e)
            vhf_cp = _as_cupy(vhf)
            
            fock = h1e_cp + vhf_cp
            e_band = float(cp.sum(dm_cp * fock))
            
            e_elec = e_band - self._e_dc_global
            return e_elec, self._e_dc_global
        else:
            # Fallback to standard energy evaluation during embedding steps
            return rks.RKS.energy_elec(self, dm, h1e, vhf)


class SingleFragmentEmbedding_ML(SingleFragmentEmbedding):
    """
    Single-Fragment ONIOM-like embedding utilizing ML density for the global low-level.
    
    This class performs DMET bond-breaking via SVD, and evaluates the local embedded 
    energies using rigorous standard SCF evaluations to guarantee exact error cancellation 
    between the high-level and low-level local calculations.
    """
    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-5, verbose=None):
        """
        Parameters
        ----------
        mf_outer : HarrisRKS object
            The global low-level solver driven by ML density.
        mf_inner : SCF/DFT/post-HF object
            The high-level solver applied to the embedded fragment+bath cluster.
        fragment : list of int
            List of atom indices defining the core QM region.
        threshold : float
            Eigenvalue cutoff for the Schmidt decomposition to classify bath orbitals.
        """
        super().__init__(mf_outer, mf_inner, fragment,
                         threshold=threshold, verbose=verbose)
        self.fragment = self.fragments[0]

    def kernel(self):

        if not self.mf_outer.converged:
            self.mf_outer.kernel()
            
        e_global_low = self.mf_outer.e_tot
        self.log.note(f"Global Low-Level E (Harris) = {e_global_low:.8f}")
        
        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        dm_full_ao_low = _as_cupy(self.mf_outer.make_rdm1())
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)

        ifrag = 0

        self.build_bath(ifrag, mo_coeff, mo_occ, X_inv, X)
        self.build_embedded_hamiltonian(ifrag, hcore_orig)
        
        self.log.info("Running high-level inner SCF in embedding space...")
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
        
        # Override energy_elec to print the true full system energy
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
            
            # Update scf_summary for meaningful debugging output
            mf_inner.scf_summary['e1'] = e1
            mf_inner.scf_summary['coul'] = ecoul_full - e_coul_core
            mf_inner.scf_summary['exc'] = exc_full - e_xc_core
            
            return e1 + e2, e2
            
        mf_inner.energy_elec = custom_energy_elec

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
            
            e_high = self._evaluate_embedded_energy(
                self.mf_inner_template, dm_emb_high, h_eval_bare, B, dm_core
            )
            self.log.note(f"High-Level E : {e_high:.8f}")
            
            # Evaluate Low-Level energy (mf_outer will automatically use exact get_veff for xc here)
            e_low = self._evaluate_embedded_energy(
                self.mf_outer, dm_emb_low, h_eval_bare, B, dm_core
            )
            self.log.note(f"Low-Level E : {e_low:.8f}")
        else:
            raise NotImplementedError("WFT evaluation is not implemented for this class.")
        
        delta_e = float(e_high - e_low)
        self.log.note(f"Global Low-Level E : {e_global_low:.8f}")
        self.log.note(f"Active Space dE    : {delta_e:.8f}")
        
        self.e_tot = e_global_low + delta_e
        self.log.note(f"Total Embedded E   : {self.e_tot:.8f}")
        
        return self.e_tot