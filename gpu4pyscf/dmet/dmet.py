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


import copy
import numpy as np
import cupy as cp
from pyscf import gto


def _as_cupy(x):
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(x)


def lowdin_orth(s):
    """
    Loewdin symmetric orthogonalization.
    """
    s = _as_cupy(s)
    s = 0.5 * (s + s.T)
    eigvals, eigvecs = cp.linalg.eigh(s)
    keep = eigvals > 1e-12
    if not cp.all(keep):
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]
    inv_sqrt = 1.0 / cp.sqrt(eigvals)
    sqrt = cp.sqrt(eigvals)
    X = (eigvecs * inv_sqrt) @ eigvecs.T          # S^{-1/2}
    X_inv = (eigvecs * sqrt) @ eigvecs.T          # S^{+1/2}
    return X, X_inv


def get_fragment_ao_indices(mol, frag_atoms):
    """
    Return the atomic-orbital indices that belong to the listed atoms.
    """
    aoslice = mol.aoslice_by_atom()
    indices = []
    for ia in frag_atoms:
        ia = int(ia)
        if ia < 0 or ia >= mol.natm:
            raise ValueError(f"Atom index {ia} is out of range [0, {mol.natm}).")
        p0, p1 = int(aoslice[ia, 2]), int(aoslice[ia, 3])
        indices.extend(range(p0, p1))
    indices = cp.asarray(sorted(indices), dtype=cp.int32)
    if indices.size == 0:
        raise ValueError("Fragment is empty: no atomic orbitals were selected.")
    return indices


def schmidt_decompose(dm_full, env_idx, threshold=1e-5):
    """
    Schmidt decomposition.
    """
    dm = _as_cupy(dm_full)
    env_idx = _as_cupy(env_idx)
    if env_idx.size == 0:
        return (cp.zeros((0, 0)),
                cp.zeros((0, 0)),
                {'core': cp.zeros(0), 'bath': cp.zeros(0), 'virtual': cp.zeros(0), 'n_core_electrons': 0})

    D_env = dm[env_idx[:, None], env_idx[None, :]]
    D_env = 0.5 * (D_env + D_env.T)

    eigvals, eigvecs = cp.linalg.eigh(D_env)

    is_core = eigvals > (2.0 - threshold)
    is_virt = eigvals < threshold
    is_bath = ~(is_core | is_virt)

    bath_orb = eigvecs[:, is_bath]
    core_orb = eigvecs[:, is_core]

    info = {
        'core':    eigvals[is_core],
        'bath':    eigvals[is_bath],
        'virtual': eigvals[is_virt],
        'n_core_electrons': 2 * int(is_core.sum()),
    }
    return bath_orb, core_orb, info


def build_embedding_basis(nao, frag_idx, env_idx, bath_orb):
    """
    Construct the AO -> embedded transformation matrix B.
    """
    frag_idx = _as_cupy(frag_idx)
    env_idx = _as_cupy(env_idx)
    n_frag = frag_idx.size
    n_bath = bath_orb.shape[1] if bath_orb.size else 0

    B = cp.zeros((nao, n_frag + n_bath), dtype=float)
    B[frag_idx, cp.arange(n_frag)] = 1.0
    if n_bath > 0:
        B[env_idx[:, None], cp.arange(n_bath)[None, :] + n_frag] = bath_orb
    return B


def build_core_dm(env_idx, core_orb, nao):
    """
    Build the core 1-RDM in the full AO basis.
    """
    env_idx = _as_cupy(env_idx)
    if core_orb.size == 0:
        return cp.zeros((nao, nao), dtype=float)
    C_core = cp.zeros((nao, core_orb.shape[1]), dtype=float)
    C_core[env_idx, :] = core_orb
    return 2.0 * (C_core @ C_core.T)


def transform_h1(h_ao, B):
    """
    Project a 1-electron operator from the full AO basis to the embedded basis.
    """
    return B.T @ h_ao @ B


def _build_embedded_mole(nemb, n_emb_electrons, spin=0, verbose=0, max_memory=4000):
    if n_emb_electrons < 0 or n_emb_electrons > 2 * nemb:
        raise ValueError(f"Invalid embedded electron count: {n_emb_electrons}")

    mol = gto.Mole()
    mol.verbose = verbose
    mol.max_memory = max_memory
    mol.atom = []
    mol.basis = {}
    mol.unit = 'Bohr'
    mol.spin = spin
    mol.nelectron = int(n_emb_electrons)
    mol.charge = 0
    mol.build(parse_arg=False, dump_input=False)

    nemb_int = int(nemb)
    def _nao_nr(self=mol, _n=nemb_int):
        return _n

    mol.nao_nr = _nao_nr
    mol.nao = nemb_int
    return mol


def _instantiate_inner_mf(mf_template, embedded_mol):
    cls = type(mf_template)
    try:
        new_mf = cls(embedded_mol)
    except TypeError:
        new_mf = copy.copy(mf_template)
        new_mf.mol = embedded_mol
        new_mf.mo_coeff = None
        new_mf.mo_energy = None
        new_mf.mo_occ = None
        new_mf.converged = False

    for attr in ('xc', 'conv_tol', 'conv_tol_grad', 'max_cycle',
                 'level_shift', 'damp', 'diis', 'verbose'):
        if hasattr(mf_template, attr):
            try:
                setattr(new_mf, attr, getattr(mf_template, attr))
            except Exception:
                pass

    if hasattr(mf_template, 'grids') and hasattr(new_mf, 'grids'):
        for g_attr in ('level', 'prune', 'atom_grid'):
            if hasattr(mf_template.grids, g_attr):
                try:
                    setattr(new_mf.grids, g_attr, getattr(mf_template.grids, g_attr))
                except Exception:
                    pass

    return new_mf


class DMET:
    """
    Density Matrix Embedding Theory driver with macroscopic iteration.

    Parameters
    ----------
    mf_outer : SCF object (gpu4pyscf)
        Low-level mean-field on the full system.
    mf_inner : SCF/DFT object (gpu4pyscf)
        High-level mean-field template applied to the embedded cluster.
    fragments : list of lists of int
        List of fragments, where each fragment is a list of atom indices.
    threshold : float
        Eigenvalue cutoff used to classify environment orbitals.
    max_macro_iter : int
        Maximum number of macroscopic iterations for correlation potential (u).
    macro_tol : float
        Convergence tolerance for the difference in fragment 1-RDMs.
    """

    def __init__(self, mf_outer, mf_inner, fragments,
                 threshold=1e-5, max_macro_iter=20, macro_tol=1e-4):
        if mf_outer is None or mf_inner is None:
            raise ValueError("mf_outer and mf_inner are both required.")
        if not fragments:
            raise ValueError("Provide a list of fragments to define the DMET regions.")

        self.mf_outer = mf_outer
        self.mf_inner_template = mf_inner
        self.full_mol = mf_outer.mol
        self.threshold = float(threshold)
        self.max_macro_iter = max_macro_iter
        self.macro_tol = macro_tol

        self.fragments = [list(int(a) for a in frag) for frag in fragments]
        self.nfrags = len(self.fragments)
        
        nao = int(self.full_mol.nao_nr())
        all_idx = cp.arange(nao, dtype=cp.int32)
        
        self.frag_idx = []
        self.env_idx = []
        for frag_atoms in self.fragments:
            f_idx = get_fragment_ao_indices(self.full_mol, frag_atoms)
            self.frag_idx.append(f_idx)
            env_mask = cp.ones(nao, dtype=bool)
            env_mask[f_idx] = False
            self.env_idx.append(all_idx[env_mask])

        # ---- intermediate / output caches (lists for multiple fragments) ----
        self.bath_orb = [None] * self.nfrags
        self.core_orb = [None] * self.nfrags
        self.eig_info = [None] * self.nfrags
        self.B_oao = [None] * self.nfrags
        self.B = [None] * self.nfrags
        self.dm_core = [None] * self.nfrags
        self.h_emb = [None] * self.nfrags
        self.e_core = [None] * self.nfrags
        self.mf_inner = [None] * self.nfrags
        self.dm_emb_init = [None] * self.nfrags
        self.e_inner = [None] * self.nfrags
        self.e_tot = None            
        self.u = cp.zeros((nao, nao))  # Global correlation potential

    def build_bath(self, ifrag, dm_full_oao, X):
        """
        Run the Schmidt decomposition for a specific fragment.
        """
        bath_orb, core_orb, info = schmidt_decompose(
            dm_full_oao, self.frag_idx[ifrag], self.env_idx[ifrag], self.threshold)

        nao_oao = X.shape[1]
        B_oao = build_embedding_basis(nao_oao, self.frag_idx[ifrag], self.env_idx[ifrag], bath_orb)
        B_ao = X @ B_oao

        if core_orb.size > 0:
            C_core_oao = cp.zeros((nao_oao, core_orb.shape[1]), dtype=float)
            C_core_oao[self.env_idx[ifrag], :] = core_orb
            C_core_ao = X @ C_core_oao
            dm_core_ao = 2.0 * (C_core_ao @ C_core_ao.T)
        else:
            dm_core_ao = cp.zeros((X.shape[0], X.shape[0]), dtype=float)

        self.bath_orb[ifrag] = bath_orb
        self.core_orb[ifrag] = core_orb
        self.eig_info[ifrag] = info
        self.B_oao[ifrag] = B_oao        
        self.B[ifrag] = B_ao             
        self.dm_core[ifrag] = dm_core_ao
        return self

    def build_embedded_hamiltonian(self, ifrag, hcore_orig):
        """
        Construct h^A in the embedded basis A.
        Uses bare hcore_orig (without the correlation potential 'u').
        """
        mol = self.full_mol
        h_ao = _as_cupy(hcore_orig)

        if self.eig_info[ifrag]['n_core_electrons'] > 0:
            vj_core, vk_core = self.mf_outer.get_jk(mol, self.dm_core[ifrag])
            v_core_ao = _as_cupy(vj_core) - 0.5 * _as_cupy(vk_core)
        else:
            v_core_ao = cp.zeros_like(h_ao)

        h_emb = transform_h1(h_ao + v_core_ao, self.B[ifrag])

        if self.eig_info[ifrag]['n_core_electrons'] > 0:
            e_core = (cp.einsum('ij,ji->', self.dm_core[ifrag], h_ao)
                      + 0.5 * cp.einsum('ij,ji->', self.dm_core[ifrag], v_core_ao))
        else:
            e_core = 0.0

        self.h_emb[ifrag] = h_emb
        self.e_core[ifrag] = float(e_core)
        return self

    def _build_inner_mf(self, ifrag, dm_full_ao):
        """Instantiate the inner SCF on the embedded mole."""
        nemb = self.B[ifrag].shape[1]
        n_total_electrons = int(self.full_mol.nelectron)
        n_emb_electrons = n_total_electrons - int(self.eig_info[ifrag]['n_core_electrons'])

        emb_mol = _build_embedded_mole(
            nemb=nemb,
            n_emb_electrons=n_emb_electrons,
            spin=int(getattr(self.full_mol, 'spin', 0)),
            verbose=0,
            max_memory=int(getattr(self.full_mol, 'max_memory', 4000)),
        )

        mf_inner = _instantiate_inner_mf(self.mf_inner_template, emb_mol)

        h_emb = self.h_emb[ifrag]
        ovlp = cp.eye(nemb)

        # Base energy offset for debugging per fragment
        e_nuc = float(self.full_mol.energy_nuc())
        mf_inner.get_hcore = lambda *args, **kwargs: h_emb
        mf_inner.get_ovlp = lambda *args, **kwargs: ovlp
        mf_inner.energy_nuc = lambda *args, **kwargs: e_nuc + self.e_core[ifrag]

        # Overwrite get_jk to compute J and K on-the-fly using the outer MF
        # without computing or storing 4-index ERIs.
        def _get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
            if dm is None:
                dm = mf_inner.make_rdm1()
            dm_cp = _as_cupy(dm)
            B_mat = self.B[ifrag]
            
            # Project embedded dm to full AO basis
            if dm_cp.ndim == 2:
                dm_ao = B_mat @ dm_cp @ B_mat.T
            else:
                dm_ao = cp.einsum('pi,xij,qj->xpq', B_mat, dm_cp, B_mat)
                
            # Compute J and K in full AO basis using outer SCF's optimized routine
            vj_ao, vk_ao = self.mf_outer.get_jk(self.full_mol, dm_ao, hermi, with_j, with_k, omega)
            
            # Project J and K back to embedded basis
            vj_emb = vk_emb = None
            if vj_ao is not None:
                if dm_cp.ndim == 2:
                    vj_emb = B_mat.T @ vj_ao @ B_mat
                else:
                    vj_emb = cp.einsum('pi,xpq,qj->xij', B_mat, vj_ao, B_mat)
            if vk_ao is not None:
                if dm_cp.ndim == 2:
                    vk_emb = B_mat.T @ vk_ao @ B_mat
                else:
                    vk_emb = cp.einsum('pi,xpq,qj->xij', B_mat, vk_ao, B_mat)
                    
            return vj_emb, vk_emb

        mf_inner.get_jk = _get_jk

        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        sB = s_ao @ self.B[ifrag]
        dm_emb_init = sB.T @ dm_full_ao @ sB
        
        trace = float(cp.trace(dm_emb_init))
        if trace > 0:
            dm_emb_init = dm_emb_init * (n_emb_electrons / trace)
        self.dm_emb_init[ifrag] = dm_emb_init

        self.mf_inner[ifrag] = mf_inner
        return mf_inner

    def solve_embedded(self, ifrag):
        """Run the high-level embedded SCF for a specific fragment."""
        e_inner = self.mf_inner[ifrag].kernel(dm0=self.dm_emb_init[ifrag])
        if isinstance(e_inner, tuple):
            e_inner = float(self.mf_inner[ifrag].e_tot)
        else:
            e_inner = float(e_inner)
        self.e_inner[ifrag] = e_inner
        return e_inner

    def kernel(self):
        """
        Drive the macroscopic-iterating DMET workflow.
        Returns the DMET total energy.
        """
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)

        for macro_iter in range(self.max_macro_iter):
            # 1. Run low-level SCF with current correlation potential 'u'
            self.mf_outer.get_hcore = lambda *args, **kwargs: cp.asnumpy(hcore_orig + self.u)
            self.mf_outer.mo_coeff = None # Force re-run
            self.mf_outer.kernel()
            
            dm_full_ao = _as_cupy(self.mf_outer.make_rdm1())
            dm_full_oao = X_inv @ dm_full_ao @ X_inv

            e_tot = 0.0
            dm_inners = []

            # 2. Loop over all fragments
            for ifrag in range(self.nfrags):
                self.build_bath(ifrag, dm_full_oao, X)
                self.build_embedded_hamiltonian(ifrag, hcore_orig)
                mf_inner = self._build_inner_mf(ifrag, dm_full_ao)
                self.solve_embedded(ifrag)

                dm_emb = _as_cupy(mf_inner.make_rdm1())
                
                # Transform inner DM back to full AO basis
                B = self.B[ifrag]
                dm_inner_active_ao = B @ dm_emb @ B.T
                
                dm_inner_full_ao = self.dm_core[ifrag] + dm_inner_active_ao
                dm_inners.append(dm_inner_full_ao)

                # 2.1 Reconstruct the full effective Fock matrix for the embedded system in AO
                vj, vk = self.mf_outer.get_jk(self.full_mol, dm_inner_full_ao)
                fock_full_ao = hcore_orig + _as_cupy(vj) - 0.5 * _as_cupy(vk)
                
                # 2.2 Transform D, H, and F to the Lowdin orthogonalized (OAO) basis
                dm_full_oao_inner = X_inv @ dm_inner_full_ao @ X_inv
                hcore_oao = X.T @ hcore_orig @ X
                fock_oao = X.T @ fock_full_ao @ X
                
                # 2.3 Extract Fragment Energy: 1/2 \sum_{i \in A, j} D_{ij}^{OAO} (H_{ij}^{OAO} + F_{ij}^{OAO})
                # In symmetric orthogonalization, AO index mapping is perfectly preserved in OAO.
                idx = self.frag_idx[ifrag]
                e_frag_elec = 0.5 * cp.sum(
                    dm_full_oao_inner[idx, :] * (hcore_oao[idx, :] + fock_oao[idx, :])
                )
                
                # Extract Fragment Nuclear Energy
                e_frag_nuc = 0.0
                coords = self.full_mol.atom_coords()
                charges = self.full_mol.atom_charges()
                frag_atoms = self.fragments[ifrag]
                for i in frag_atoms:
                    for j in range(self.full_mol.natm):
                        if i == j: continue
                        r = np.linalg.norm(coords[i] - coords[j])
                        factor = 0.5 if j in frag_atoms else 1.0
                        e_frag_nuc += factor * charges[i] * charges[j] / r
                
                e_tot += float(e_frag_elec) + e_frag_nuc

            # 3. Macroscopic iteration: update correlation potential 'u'
            error = 0.0
            for ifrag in range(self.nfrags):
                idx = self.frag_idx[ifrag]
                idx_mesh = cp.ix_(idx, idx)
                
                # Cost function: \Delta D = D_inner_full - D_outer_full over fragment blocks
                diff = dm_inners[ifrag][idx_mesh] - dm_full_ao[idx_mesh]
                error += float(cp.linalg.norm(diff))
                
                # Simple gradient descent step
                # Note: 0.5 is a hyperparameter. If it oscillates, reduce it (e.g. to 0.1).
                self.u[idx_mesh] -= 0.5 * diff
            
            print(f"Macro Iter {macro_iter + 1:2d} | E_DMET = {e_tot:.8f} | max(dD) = {error:.6e}")
            self.e_tot = e_tot
            if error < self.macro_tol:
                print("DMET macroscopic iterations converged.")
                break

        return self.e_tot

    def __call__(self):
        return self.kernel()