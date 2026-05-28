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

    def _get_harris_veff(self, mol=None):

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

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        
        dm_cp = _as_cupy(dm)
        s_ao = _as_cupy(self.get_ovlp())
        
        # Calculate the actual number of electrons represented by the density matrix in AO basis
        nelec_dm = float(cp.sum(dm_cp * s_ao))
        
        # Handle zero density matrix under full-system inclusion limit safely
        if nelec_dm < 1e-4:
            v_eff_ao = cp.zeros_like(dm_cp)
            return tag_array(v_eff_ao, ecoul=0.0, exc=0.0, vj=cp.zeros_like(dm_cp), vk=cp.zeros_like(dm_cp))
            
        # Rigorous electron count inspection instead of the non-orthogonal matrix trace
        if nelec_dm > self.mol.nelectron - 0.5:
            v_eff_ao = self._get_harris_veff(mol)
            e_2e = float(cp.sum(dm_cp * v_eff_ao)) - self._e_dc_global
            return tag_array(v_eff_ao, ecoul=e_2e, exc=0.0, vj=v_eff_ao, vk=cp.zeros_like(v_eff_ao))
        else:
            # Core evaluation using the pre-stored complementary weights
            if self.grids.coords is None:
                self.grids.build()
            if isinstance(self.current_w_core, cp.ndarray) and self.current_w_core.ndim == 1:
                vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc = self.eval_density_func(
                    mol, self.xc, self.grids, atomic_weights=None, grid_weights=self.current_w_core
                )
            else:
                vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc = self.eval_density_func(
                    mol, self.xc, self.grids, atomic_weights=self.current_w_core, grid_weights=None
                )
            v_eff_ao = _as_cupy(vj) + _as_cupy(vxc)
            if vk is not None: v_eff_ao -= _as_cupy(vk)
            e_k = float(e_k) if vk is not None else 0.0
            e_dc = float(e_j) - e_k + float(int_rho_vxc) - float(e_xc)
            e_2e = float(cp.sum(dm_cp * v_eff_ao)) - e_dc
            return tag_array(v_eff_ao, ecoul=e_2e, exc=0.0, vj=_as_cupy(vj), vk=_as_cupy(vk) if vk is not None else cp.zeros_like(v_eff_ao))

    def kernel(self, dm0=None, **kwargs):
        # Pass through to the standard solver, get_veff handles everything natively via electron counting
        e_tot = rks.RKS.kernel(self, dm0=dm0, **kwargs)
        self.converged = True
        return e_tot

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """
        Overrides electronic energy evaluation using the Harris energy formula:
        E_elec = Tr[D * (h + Veff)] - E_DC
        """
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

    def _evaluate_embedded_energy(self, mf_obj, dm_emb, h_eval_bare, B, dm_core):
        e_h_active = cp.sum(dm_emb * h_eval_bare)
        
        dm_full_ao = dm_core + B @ dm_emb @ B.T
        
        v_eff_full = mf_obj.get_veff(self.full_mol, dm_full_ao)
        v_eff_core = mf_obj.get_veff(self.full_mol, dm_core)
        
        e_2e_full = getattr(v_eff_full, 'ecoul', 0.0) + getattr(v_eff_full, 'exc', 0.0)
        e_2e_core = getattr(v_eff_core, 'ecoul', 0.0) + getattr(v_eff_core, 'exc', 0.0)
        # E_active = E_1e(Active) + [E_2e(Full) - E_2e(Core)]
        return e_h_active + e_2e_full - e_2e_core

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
        
        # Rigorous density matrix projection incorporating the non-orthogonal overlap metric S
        dm_emb_low = B.T @ s_ao @ dm_full_ao_low @ s_ao @ B
        dm_active_ao = B @ dm_emb_low @ B.T
        
        # Calculate mapping weights and extract local ML components based on partition_type
        if self.partition_type == 'atom':
            self.log.info("Step 2 & 3: DMET SVD and calculating Atomic Weights...")
            w_active = self._get_atomic_weights(dm_active_ao, dm_full_ao_low, s_ao, self.full_mol)
            w_core = 1.0 - w_active
            
        elif self.partition_type == 'grid':
            self.log.info("Step 2 & 3: DMET SVD and calculating Grid Weights w(r)...")
            if self.mf_outer.grids.coords is None:
                self.mf_outer.grids.build()
            w_active = self._get_grid_weights(dm_active_ao, dm_full_ao_low, self.full_mol, self.mf_outer.grids)
            w_core = 1.0 - w_active
            
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}. Use 'atom' or 'grid'.")
        print("debug w_core:", w_core)

        # Store w_core into mf_outer for automated core potential evaluation via trace inspection
        self.mf_outer.current_w_core = w_core

        # Standard DMET embedded Hamiltonian and core potentials construction
        self.build_embedded_hamiltonian(ifrag, hcore_orig)
        
        self.log.info("Step 6: Running high-level inner SCF in embedding space...")
        mf_inner = self._build_inner_mf(ifrag, dm_full_ao_low)
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