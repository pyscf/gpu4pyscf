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
from gpu4pyscf.qmmm.embedding.embedding import DMET, lowdin_orth, _as_cupy

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
        # Signature: def func(mol, xc, grids, atomic_weights=None, grid_weights=None)
        # Returns 7 elements:
        #   1. vj: Coulomb potential matrix (AO basis)
        #   2. vk: Exact exchange potential matrix (AO basis, can be None for pure DFT)
        #   3. vxc: Exchange-correlation potential matrix (AO basis)
        #   4. e_j: Coulomb energy (scalar)
        #   5. e_k: Exact exchange energy (scalar, can be 0.0 for pure DFT)
        #   6. e_xc: Exchange-correlation energy (scalar)
        #   7. int_rho_vxc: Integral of rho * V_xc (scalar)
        self.eval_density_func = eval_density_func
        
        # Cache for global evaluation results to avoid redundant ML inferences
        self._v_eff_global = None
        self._e_dc_global = None

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):

        if mol is None: 
            mol = self.mol
        
        if self._v_eff_global is not None:
            return self._v_eff_global
            
        if self.grids.coords is None:
            self.grids.build()
            
        # Global evaluation uses no weights
        vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc = self.eval_density_func(
            mol, self.xc, self.grids, atomic_weights=None, grid_weights=None
        )
        
        v_eff_ao = _as_cupy(vj) + _as_cupy(vxc)
        if vk is not None:
            v_eff_ao -= _as_cupy(vk)
            e_k = float(e_k)
        else:
            e_k = 0.0
            
        # Assemble double counting energy
        e_dc = float(e_j) - e_k + float(int_rho_vxc) - float(e_xc)
        
        self._v_eff_global = v_eff_ao
        self._e_dc_global = e_dc
        return self._v_eff_global

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """
        Overrides electronic energy evaluation using the Harris energy formula:
        E_elec = Tr[D * (h + Veff)] - E_DC
        """
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        
        dm_cp = _as_cupy(dm)
        h1e_cp = _as_cupy(h1e)
        vhf_cp = _as_cupy(vhf)
        
        fock = h1e_cp + vhf_cp
        e_band = float(cp.sum(dm_cp * fock))
        
        e_elec = e_band - self._e_dc_global
        return e_elec, self._e_dc_global

    def get_local_veff_and_dc(self, atomic_weights=None, grid_weights=None):
        # Pass both weight options to the external ML interface. 
        # The ML function should apply the provided one appropriately.
        if self.grids.coords is None:
            self.grids.build()
            
        vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc = self.eval_density_func(
            self.mol, self.xc, self.grids, 
            atomic_weights=atomic_weights, 
            grid_weights=grid_weights
        )
        
        v_eff_ao_local = _as_cupy(vj) + _as_cupy(vxc)
        if vk is not None:
            v_eff_ao_local -= _as_cupy(vk)
            e_k = float(e_k)
        else:
            e_k = 0.0
            
        e_dc_local = float(e_j) - e_k + float(int_rho_vxc) - float(e_xc)
        
        return v_eff_ao_local, e_dc_local


class SingleFragmentEmbedding_ML(DMET):
    """
    Single-Fragment ONIOM-like embedding utilizing ML density scaling.
    
    This class performs DMET bond-breaking via SVD, maps the DMET orbital
    population to atomic weights, extracts a perfectly matched local ML density, 
    and evaluates the total energy using ONIOM error cancellation.
    """
    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-5, partition_type='atom', verbose=None):
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
        partition_type : str
            'atom' for Mulliken population-based atomic weights.
            'grid' for real-space density-based grid weights w(r) = rho_local(r) / rho_global(r).
        """
        fragments = [fragment]
        super().__init__(mf_outer, mf_inner, fragments,
                         threshold=threshold, max_macro_iter=1, verbose=verbose)
        self.fragment = self.fragments[0]
        self.partition_type = partition_type

    def _get_atomic_weights(self, dm_active_ao, dm_full_ao, s_ao, mol):
        pop_active = cp.einsum('ij,ji->i', dm_active_ao, s_ao)
        pop_full = cp.einsum('ij,ji->i', dm_full_ao, s_ao)
        
        aoslice = mol.aoslice_by_atom()
        weights = np.zeros(mol.natm)
        
        for ia in range(mol.natm):
            p0, p1 = aoslice[ia, 2], aoslice[ia, 3]
            if p1 > p0:
                n_active = float(cp.sum(pop_active[p0:p1]))
                n_full = float(cp.sum(pop_full[p0:p1]))
                
                if n_full > 1e-12:
                    w = n_active / n_full
                    weights[ia] = max(0.0, min(1.0, w))
                else:
                    weights[ia] = 0.0
                    
        return weights

    def _get_grid_weights(self, dm_active_ao, dm_full_ao, mol, grids):

        ni = self.mf_outer._numint
        
        rho_active = ni.get_rho(mol, dm_active_ao, grids)
        rho_full   = ni.get_rho(mol, dm_full_ao, grids)
        
        weights = rho_active / cp.maximum(rho_full, 1e-12)
        
        weights = cp.clip(weights, 0.0, 1.0)
        
        return weights

    def kernel(self):

        if not self.mf_outer.converged:
            self.mf_outer.kernel()
            
        e_global_low = self.mf_outer.e_tot
        self.log.note(f"Step 1: Global Low-Level E (Harris) = {e_global_low:.8f}")
        
        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        dm_full_ao_low = _as_cupy(self.mf_outer.make_rdm1())
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)

        ifrag = 0

        self.build_bath(ifrag, mo_coeff, mo_occ, X_inv, X)
        B = self.B[ifrag]
        
        # Project density to active space and back to AO for population analysis
        dm_emb_low = B.T @ dm_full_ao_low @ B
        dm_active_ao = B @ dm_emb_low @ B.T
        
        # Calculate mapping weights and extract local ML components based on partition_type
        if self.partition_type == 'atom':
            self.log.info("Step 2 & 3: DMET SVD and calculating Atomic Weights...")
            w_active = self._get_atomic_weights(dm_active_ao, dm_full_ao_low, s_ao, self.full_mol)
            w_core = 1.0 - w_active
            
            self.log.info("Step 4a: Extracting pure CORE potential using (1-w)...")
            v_core_ao, _ = self.mf_outer.get_local_veff_and_dc(atomic_weights=w_core)
            
            self.log.info("Step 4b: Extracting ACTIVE components for Double Counting...")
            v_eff_ao_local, e_dc_local = self.mf_outer.get_local_veff_and_dc(atomic_weights=w_active)
            
        elif self.partition_type == 'grid':
            self.log.info("Step 2 & 3: DMET SVD and calculating Grid Weights w(r)...")
            if self.mf_outer.grids.coords is None:
                self.mf_outer.grids.build()
            w_active = self._get_grid_weights(dm_active_ao, dm_full_ao_low, self.full_mol, self.mf_outer.grids)
            w_core = 1.0 - w_active
            
            self.log.info("Step 4a: Extracting pure CORE potential using (1-w)...")
            v_core_ao, _ = self.mf_outer.get_local_veff_and_dc(grid_weights=w_core)
            
            self.log.info("Step 4b: Extracting ACTIVE components for Double Counting...")
            v_eff_ao_local, e_dc_local = self.mf_outer.get_local_veff_and_dc(grid_weights=w_active)
            
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}. Use 'atom' or 'grid'.")

        e_nuc_constant = self.full_mol.energy_nuc()

        # Construct exact embedded Hamiltonian: h_emb = B^T (h_core^AO + V_core) B
        fock_core_ao = hcore_orig + v_core_ao
        h_core_fb_eff = B.T @ fock_core_ao @ B
        
        self.h_emb[ifrag] = h_core_fb_eff  
        self.e_core[ifrag] = 0.0  # ONIOM framework implies E_core shift is 0

        fock_fb_local = h_core_fb_eff + (B.T @ v_eff_ao_local @ B)
        e_band_local = float(cp.sum(dm_emb_low * fock_fb_local))
        e_local_low = e_band_local - e_dc_local + e_nuc_constant
        self.log.note(f"Step 5: Matched Local Low-Level E   = {e_local_low:.8f}")

        self.dm_core[ifrag] = cp.zeros_like(dm_full_ao_low)
        self.v_core_ao[ifrag] = cp.zeros_like(dm_full_ao_low)
        
        self.log.info("Step 6: Running high-level inner SCF in embedding space...")
        self._build_inner_mf(ifrag, dm_full_ao_low)
        self.solve_embedded(ifrag)
        
        e_local_high = self.e_inner[ifrag]
        self.log.note(f"Step 6: Local High-Level E (SCF)    = {e_local_high:.8f}")

        self.e_tot = e_global_low - e_local_low + e_local_high
        
        self.log.note("="*50)
        self.log.note(f"FINAL ONIOM TOTAL ENERGY = {self.e_tot:.8f}")
        self.log.note("="*50)
        
        return self.e_tot