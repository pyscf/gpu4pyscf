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
import pyscf
from pyscf import gto, ao2mo
import gpu4pyscf
from gpu4pyscf.scf import hf as gpu_hf


def _as_cupy(x):
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(x)


# TODO: use already implemented lowdin_orth
def lowdin_orth(s):
    """
    Loewdin symmetric orthogonalization.

    Given an AO overlap matrix ``S``, return ``X = S^{-1/2}`` and
    ``X_inv = S^{1/2}``. Eigenvalues of ``S`` smaller than 1e-12 are
    treated as linearly dependent and dropped.

    Returns
    -------
    X : cp.ndarray, shape (nao, nao_orth)
        AO -> orthonormal AO transformation. Columns of ``X`` are the
        coefficients of the orthonormal AOs in the AO basis.
    X_inv : cp.ndarray, shape (nao_orth, nao)
        Inverse transformation: ``X_inv = X^T S``.
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

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The full system molecule.
    frag_atoms : sequence of int
        Atom indices that constitute the fragment.

    Returns
    -------
    ao_indices : cp.ndarray of int
        Sorted AO indices (in the AO ordering of ``mol``) that belong
        to ``frag_atoms``.
    """
    aoslice = mol.aoslice_by_atom()
    indices = []
    for ia in frag_atoms:
        ia = int(ia)
        if ia < 0 or ia >= mol.natm:
            raise ValueError(
                f"Atom index {ia} is out of range [0, {mol.natm})."
            )
        p0, p1 = int(aoslice[ia, 2]), int(aoslice[ia, 3])
        indices.extend(range(p0, p1))
    indices = cp.asarray(sorted(indices), dtype=cp.int32)
    if indices.size == 0:
        raise ValueError(
            "Fragment is empty: no atomic orbitals were selected."
        )
    return indices


def schmidt_decompose(dm_full, frag_idx, env_idx, threshold=1e-5):
    """
    Schmidt decomposition.

    Parameters
    ----------
    dm_full : array_like, shape (nao, nao)
        Spin-summed 1-RDM in the full AO basis. The trace equals the
        number of electrons.
    frag_idx, env_idx : cp.ndarray
        AO indices of fragment and environment, respectively.
        ``frag_idx`` and ``env_idx`` together must form a partition of
        ``range(nao)``.
    threshold : float
        Eigenvalue cutoff used to classify the environment orbitals.

    Returns
    -------
    bath_orb : cp.ndarray, shape (n_env, n_bath)
        Eigenvectors of D^E whose eigenvalues are within
        (threshold, 2 - threshold).
    core_orb : cp.ndarray, shape (n_env, n_core)
        Eigenvectors of D^E whose eigenvalues exceed 2 - threshold.
        These define the unentangled occupied (core) orbitals.
    info : dict
        Dictionary with eigenvalue arrays for each category and the
        electron count of the core space.
    """
    dm = _as_cupy(dm_full)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("dm_full must be a square 2D matrix.")

    env_idx = _as_cupy(env_idx)
    if env_idx.size == 0:
        # Pure fragment, no environment to entangle with.
        return (cp.zeros((0, 0)),
                cp.zeros((0, 0)),
                {'core': cp.zeros(0),
                 'bath': cp.zeros(0),
                 'virtual': cp.zeros(0),
                 'n_core_electrons': 0})

    # Symmetrize to suppress numerical asymmetry from the SCF solver
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
        # Each unentangled-occupied orbital is doubly occupied in the
        # spin-restricted formulation.
        'n_core_electrons': 2 * int(is_core.sum()),
    }
    return bath_orb, core_orb, info


def build_embedding_basis(nao, frag_idx, env_idx, bath_orb):
    """
    Construct the AO -> embedded transformation matrix B.

    Columns of B are arranged as
        [ fragment-AO basis (identity columns),
          bath orbitals (eigenvectors lifted into the env block) ].

    Parameters
    ----------
    nao : int
        Number of atomic orbitals in the full system.
    frag_idx : cp.ndarray of int
        AO indices of the fragment.
    env_idx : cp.ndarray of int
        AO indices of the environment.
    bath_orb : cp.ndarray, shape (n_env, n_bath)
        Bath orbitals expressed in the environment AO subspace.

    Returns
    -------
    B : cp.ndarray, shape (nao, n_frag + n_bath)
        Transformation matrix whose columns span the embedded space A.
    """
    frag_idx = _as_cupy(frag_idx)
    env_idx = _as_cupy(env_idx)
    n_frag = frag_idx.size
    n_bath = bath_orb.shape[1] if bath_orb.size else 0

    B = cp.zeros((nao, n_frag + n_bath), dtype=float)
    # Fragment columns: identity on fragment AOs
    B[frag_idx, cp.arange(n_frag)] = 1.0
    # Bath columns: embed env eigenvectors into the env rows
    if n_bath > 0:
        B[env_idx[:, None], cp.arange(n_bath)[None, :] + n_frag] = bath_orb
    return B


def build_core_dm(env_idx, core_orb, nao):
    """
    Build the spin-summed core 1-RDM in the full AO basis.

    Each unentangled-occupied orbital is doubly occupied:

        D_core = 2 * C_core C_core^T,

    where C_core is the matrix of core orbitals lifted into the full
    AO basis (the rows corresponding to fragment AOs are zero).
    """
    env_idx = _as_cupy(env_idx)
    if core_orb.size == 0:
        return cp.zeros((nao, nao), dtype=float)
    C_core = cp.zeros((nao, core_orb.shape[1]), dtype=float)
    C_core[env_idx, :] = core_orb
    return 2.0 * (C_core @ C_core.T)


# ---------------------------------------------------------------------------
# Hamiltonian transformations
# ---------------------------------------------------------------------------
def transform_h1(h_ao, B):
    """
    Project a 1-electron operator from the full AO basis to the
    embedded basis: ``h_emb = B^T h_ao B``.
    """
    h_emb = B.T @ h_ao @ B
    return h_emb


def transform_eri(mol, B):
    """
    Transform the four-index two-electron repulsion integrals from the
    full AO basis to the embedded basis using ``pyscf.ao2mo``:

        V^A_{xy,zw} = sum_{rstu} B^r_x B^s_y V^{rs}_{tu} B^t_z B^u_w.

    The result is returned in 4-fold symmetric packed form, suitable
    for assignment to ``mf._eri`` of an SCF object.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Full-system molecule providing the AO integrals.
    B : cp.ndarray, shape (nao, nemb)
        AO -> embedded transformation matrix.

    Returns
    -------
    eri_emb : cp.ndarray
        ERIs in the embedded basis (4-fold symmetric, packed).
    """
    nemb = B.shape[1]
    # pyscf.ao2mo requires CPU numpy arrays
    B_cpu = cp.asnumpy(B)
    eri_emb = ao2mo.kernel(mol, B_cpu, compact=True)
    # ``ao2mo.kernel`` already returns the 4-fold packed form for
    # real, equal-MOs inputs; ensure consistent shape.
    eri_emb = ao2mo.restore(4, eri_emb, nemb)
    return cp.asarray(eri_emb)


# ---------------------------------------------------------------------------
# Embedded Mole helper
# ---------------------------------------------------------------------------
def _build_embedded_mole(nemb, n_emb_electrons, spin=0,
                         verbose=0, max_memory=4000):
    """
    Build a placeholder ``pyscf.gto.Mole`` whose only role is to carry
    the bookkeeping needed by a PySCF SCF driver: the number of
    electrons, the number of orbitals, and the ``incore_anyway`` flag
    (so that the driver consumes ``mf._eri`` directly instead of
    rebuilding integrals from atomic basis functions).
    """
    if n_emb_electrons < 0:
        raise ValueError(
            f"Embedded electron count {n_emb_electrons} is negative; "
            "check the fragment definition and the Schmidt threshold."
        )
    if n_emb_electrons > 2 * nemb:
        raise ValueError(
            f"Embedded electron count {n_emb_electrons} exceeds "
            f"2 * nemb = {2 * nemb}; the embedded space is too small."
        )

    mol = gto.Mole()
    mol.verbose = verbose
    mol.max_memory = max_memory
    mol.atom = []
    mol.basis = {}
    mol.unit = 'Bohr'
    mol.spin = spin
    mol.nelectron = int(n_emb_electrons)
    mol.charge = 0
    mol.incore_anyway = True
    mol.build(parse_arg=False, dump_input=False)

    # Override the basis-counting helpers so PySCF treats the molecule
    # as having exactly nemb orbitals.
    nemb_int = int(nemb)

    def _nao_nr(self=mol, _n=nemb_int):
        return _n

    mol.nao_nr = _nao_nr
    mol.nao = nemb_int
    return mol


def _instantiate_inner_mf(mf_template, embedded_mol):
    """
    Create an SCF/DFT object on ``embedded_mol`` that mirrors
    the type/configuration of ``mf_template``. 
    """
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

    # Propagate selected configuration parameters
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
                    setattr(new_mf.grids, g_attr,
                            getattr(mf_template.grids, g_attr))
                except Exception:
                    pass

    return new_mf


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
class DMET:
    """
    Single-shot Density Matrix Embedding Theory driver.

    Parameters
    ----------
    mf_outer : SCF object (gpu4pyscf)
        Low-level mean-field on the full system. Must be (or be made)
        converged before its 1-RDM is consumed. If ``mf_outer`` does
        not yet hold a converged MO set, ``kernel()`` will run it.
    mf_inner : SCF/DFT object (gpu4pyscf)
        High-level mean-field template applied to the embedded cluster.
        A fresh PySCF object of the same class is instantiated on
        the embedded "mole" and patched with the embedded Hamiltonian
        (h^A, V^A). The user-supplied object is left untouched.
    frag_atoms : sequence of int, optional
        Atom indices that define the fragment region A. Mutually
        exclusive with ``frag_orbs``.
    frag_orbs : sequence of int, optional
        Explicit AO indices defining the fragment region.
    threshold : float
        Eigenvalue cutoff used to classify environment orbitals into
        core / bath / virtual. Defaults to 1e-5.
    """

    def __init__(self, mf_outer, mf_inner,
                 frag_atoms=None, frag_orbs=None,
                 threshold=1e-5):
        if mf_outer is None or mf_inner is None:
            raise ValueError("mf_outer and mf_inner are both required.")
        if frag_atoms is None and frag_orbs is None:
            raise ValueError(
                "Provide either 'frag_atoms' or 'frag_orbs' to define "
                "the DMET fragment."
            )
        if frag_atoms is not None and frag_orbs is not None:
            raise ValueError(
                "Specify only one of 'frag_atoms' or 'frag_orbs'."
            )
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold must lie in (0, 1); got {threshold}."
            )

        self.mf_outer = mf_outer
        self.mf_inner_template = mf_inner
        self.full_mol = mf_outer.mol
        self.threshold = float(threshold)

        nao = int(self.full_mol.nao_nr())
        if frag_atoms is not None:
            self.frag_atoms = list(int(a) for a in frag_atoms)
            self.frag_idx = get_fragment_ao_indices(
                self.full_mol, self.frag_atoms)
        else:
            self.frag_atoms = None
            self.frag_idx = cp.asarray(sorted(int(i) for i in frag_orbs),
                                       dtype=cp.int32)

        all_idx = cp.arange(nao, dtype=cp.int32)
        env_mask = cp.ones(nao, dtype=bool)
        env_mask[self.frag_idx] = False
        self.env_idx = all_idx[env_mask]

        # ---- intermediate / output caches ----
        self.bath_orb = None         # (n_env, n_bath)
        self.core_orb = None         # (n_env, n_core)
        self.eig_info = None         # dict from schmidt_decompose
        self.B = None                # AO -> embedded basis transform
        self.dm_core = None          # full-AO core density matrix
        self.h_emb = None            # embedded 1e Hamiltonian (cupy)
        self.eri_emb = None          # embedded 2e Hamiltonian (cupy)
        self.e_core = None           # core energy contribution
        self.e_nuc = None            # nuclear repulsion energy
        self.mf_inner = None         # patched inner SCF object
        self.dm_emb_init = None      # initial embedded density matrix
        self.e_inner = None          # inner SCF total energy w/ overrides
        self.e_tot = None            # final DMET total energy

    # ------------------------------------------------------------------
    # Step 1: ensure low-level mean-field is converged
    # ------------------------------------------------------------------
    def _ensure_outer_converged(self):
        if getattr(self.mf_outer, 'mo_coeff', None) is None or not getattr(self.mf_outer, 'converged', True):
            self.mf_outer.kernel()

    # ------------------------------------------------------------------
    # Step 2: bath construction
    # ------------------------------------------------------------------
    def build_bath(self):
        """
        Run the Schmidt decomposition on the environment block of the
        outer-SCF density matrix expressed in the Loewdin orthonormal
        AO (OAO) basis. Populates ``self.bath_orb``, ``self.core_orb``,
        ``self.eig_info``, ``self.B_oao``, ``self.X``, and ``self.B``
        (the AO coefficients of the embedded orbitals).
        """
        self._ensure_outer_converged()
        dm_full_ao = _as_cupy(self.mf_outer.make_rdm1())

        # Loewdin orthogonalization of the AO basis
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)
        # 1-RDM in the OAO basis: D' = S^{1/2} D S^{1/2}
        dm_full_oao = X_inv @ dm_full_ao @ X_inv

        bath_orb, core_orb, info = schmidt_decompose(
            dm_full_oao, self.frag_idx, self.env_idx, self.threshold)

        nao_oao = X.shape[1]
        # OAO -> embedded transformation
        B_oao = build_embedding_basis(nao_oao, self.frag_idx, self.env_idx,
                                      bath_orb)
        # AO coefficients of the embedded orbitals: C_emb = X B'
        B_ao = X @ B_oao

        # Core orbitals lifted from OAO env subspace into the AO basis.
        if core_orb.size > 0:
            C_core_oao = cp.zeros((nao_oao, core_orb.shape[1]), dtype=float)
            C_core_oao[self.env_idx, :] = core_orb
            C_core_ao = X @ C_core_oao
            dm_core_ao = 2.0 * (C_core_ao @ C_core_ao.T)
        else:
            dm_core_ao = cp.zeros_like(dm_full_ao)

        self.X = X
        self.X_inv = X_inv
        self.bath_orb = bath_orb
        self.core_orb = core_orb
        self.eig_info = info
        self.B_oao = B_oao        # OAO -> embedded
        self.B = B_ao             # AO  -> embedded (orthonormal columns)
        self.dm_core = dm_core_ao
        return self

    # ------------------------------------------------------------------
    # Step 3: build the embedded Hamiltonian
    # ------------------------------------------------------------------
    def build_embedded_hamiltonian(self):
        """
        Construct h^A and V^A in the embedded basis A and the
        constant core energy.
        """
        if self.B is None:
            self.build_bath()

        mol = self.full_mol
        # Bare 1e Hamiltonian on the full AO basis. Use the outer-mf
        # implementation to inherit any custom modifications (ECPs,
        # external charges, etc.).
        h_ao = _as_cupy(self.mf_outer.get_hcore())

        # Mean-field potential generated by the unentangled-occupied
        # core orbitals in the full AO basis.
        if self.eig_info['n_core_electrons'] > 0:
            vj_core, vk_core = self.mf_outer.get_jk(mol, self.dm_core)
            v_core_ao = _as_cupy(vj_core) - 0.5 * _as_cupy(vk_core)
        else:
            v_core_ao = cp.zeros_like(h_ao)

        # 1-electron Hamiltonian in the embedded basis
        h_emb = transform_h1(h_ao + v_core_ao, self.B)

        # 2-electron Hamiltonian in the embedded basis
        eri_emb = transform_eri(mol, self.B)

        # Constant core energy: 1/2 Tr[D_core (h + (h + v_core))]
        # = Tr[D_core h] + 1/2 Tr[D_core v_core]
        if self.eig_info['n_core_electrons'] > 0:
            e_core = (cp.einsum('ij,ji->', self.dm_core, h_ao)
                      + 0.5 * cp.einsum('ij,ji->', self.dm_core, v_core_ao))
        else:
            e_core = 0.0

        self.h_emb = h_emb
        self.eri_emb = eri_emb
        self.e_core = float(e_core)
        self.e_nuc = float(mol.energy_nuc())
        return self

    # ------------------------------------------------------------------
    # Step 4: build / patch the inner SCF object and solve
    # ------------------------------------------------------------------
    def _build_inner_mf(self):
        """Instantiate the inner SCF on the embedded mole."""
        if self.h_emb is None:
            self.build_embedded_hamiltonian()

        nemb = self.B.shape[1]
        n_total_electrons = int(self.full_mol.nelectron)
        n_emb_electrons = n_total_electrons \
            - int(self.eig_info['n_core_electrons'])

        emb_mol = _build_embedded_mole(
            nemb=nemb,
            n_emb_electrons=n_emb_electrons,
            spin=int(getattr(self.full_mol, 'spin', 0)),
            verbose=int(getattr(self.full_mol, 'verbose', 0)),
            max_memory=int(getattr(self.full_mol, 'max_memory', 4000)),
        )

        mf_inner = _instantiate_inner_mf(self.mf_inner_template, emb_mol)

        # ----- Patch the underlying Hamiltonian -----
        h_emb = self.h_emb
        ovlp = cp.eye(nemb)

        mf_inner.get_hcore = lambda *args, **kwargs: h_emb
        mf_inner.get_ovlp = lambda *args, **kwargs: ovlp
        mf_inner.energy_nuc = lambda *args, **kwargs: self.e_nuc + self.e_core

        # Use ao2mo's 8-fold packed format for the in-core ERIs so
        # PySCF's optimized JK routines can be reused.
        eri_emb_cpu = cp.asnumpy(self.eri_emb)
        eri_8fold = ao2mo.restore(8, eri_emb_cpu, nemb)
        mf_inner._eri = cp.asarray(eri_8fold)

        # Initial guess: project the outer 1-RDM into the embedded
        # basis. With C_emb expressed in AO coefficients, the projector
        # is C_emb^T S D_AO S C_emb (which equals B_oao^T D_OAO B_oao).
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        dm_full_ao = _as_cupy(self.mf_outer.make_rdm1())
        sB = s_ao @ self.B
        dm_emb_init = sB.T @ dm_full_ao @ sB
        
        # Ensure exact electron count consistency
        trace = float(cp.trace(dm_emb_init))
        if trace > 0:
            dm_emb_init = dm_emb_init * (n_emb_electrons / trace)
        self.dm_emb_init = dm_emb_init

        self.mf_inner = mf_inner
        return mf_inner

    def solve_embedded(self):
        """Run the high-level embedded SCF and return its total energy."""
        if self.mf_inner is None:
            self._build_inner_mf()

        e_inner = self.mf_inner.kernel(dm0=self.dm_emb_init)
        if isinstance(e_inner, tuple):
            e_inner = float(self.mf_inner.e_tot)
        else:
            e_inner = float(e_inner)
        self.e_inner = e_inner
        return e_inner

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def kernel(self):
        """
        Drive the full single-shot DMET workflow and return the total
        energy.

        E_DMET = E_inner_total

        Note: the inner SCF's ``energy_nuc`` is set to (E_nuc + E_core),
        so the energy returned by the inner solver already accounts for
        the nuclear repulsion of the full system and the mean-field
        contribution of the unentangled-occupied core orbitals.
        """
        self.build_bath()
        self.build_embedded_hamiltonian()
        self._build_inner_mf()
        e_inner = self.solve_embedded()
        self.e_tot = float(e_inner)
        return self.e_tot

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def energy_decomposition(self):
        """
        Return a dictionary describing the various energy contributions
        gathered during the DMET calculation.
        """
        if self.e_tot is None:
            self.kernel()
        return {
            'E_nuc':   self.e_nuc,
            'E_core':  self.e_core,
            'E_inner': self.e_inner,
            'E_DMET':  self.e_tot,
        }

    def bath_summary(self):
        """
        Return a brief description of the Schmidt decomposition
        outcome: the sizes of the fragment, bath, core and virtual
        spaces, and the eigenvalue arrays of each environment block.
        """
        if self.eig_info is None:
            self.build_bath()
        return {
            'n_fragment_aos': int(self.frag_idx.size),
            'n_bath':         int(self.bath_orb.shape[1]),
            'n_core':         int(self.core_orb.shape[1]),
            'n_virtual':      int(self.eig_info['virtual'].size),
            'core_eigvals':   self.eig_info['core'],
            'bath_eigvals':   self.eig_info['bath'],
            'virt_eigvals':   self.eig_info['virtual'],
            'n_core_electrons': int(self.eig_info['n_core_electrons']),
        }

    def __call__(self):
        """Allow ``DMET(...)()`` invocation in the PySCF mf style."""
        return self.kernel()
