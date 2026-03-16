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
import cupyx
from gpu4pyscf.sem.integral import hcore2c1e

# Pre-compute the local 2D indices for the 45-element packed array
# This maps a 1D index (0..44) to its (row, col) in a 9x9 lower triangular block
_LOCAL_ROW_IDX = cp.zeros(45, dtype=cp.int32)
_LOCAL_COL_IDX = cp.zeros(45, dtype=cp.int32)

_idx = 0
for i in range(9):
    for j in range(i + 1):
        _LOCAL_ROW_IDX[_idx] = i
        _LOCAL_COL_IDX[_idx] = j
        _idx += 1

def build_hcore_matrix(mol, h1elec_mat, e1b, e2a):
    """
    Constructs the full dense Core Hamiltonian matrix H_core (nao, nao).
    
    This function completely replaces MOPAC's packed format assembly logic. 
    It assembles H_core by combining:
    1. One-center one-electron energies (USPD) on the main diagonal.
    2. Two-center resonance integrals (Overlap * Beta) across off-diagonal blocks.
    3. Core-electron attraction integrals (e1b, e2a) on the diagonal blocks.
    
    Args:
        mol: PM6Mole instance.
        h1elec_mat: (nao, nao) CuPy array - Two-center resonance integrals from h1elec().
        e1b: (n_pairs, 45) CuPy array - Attraction of Atom A's electrons by Atom B's core.
        e2a: (n_pairs, 45) CuPy array - Attraction of Atom B's electrons by Atom A's core.
        
    Returns:
        H_core: (nao, nao) CuPy array - The dense Core Hamiltonian matrix.
    """
    nao = mol.nao
    n_pairs = mol.npairs
    
    H_core = cp.copy(h1elec_mat)
    
    # Add One-Center energies (USPD) to the main diagonal
    uspd_gpu = cp.asarray(mol.uspd, dtype=cp.float64)
    diag_indices = cp.arange(nao)
    H_core[diag_indices, diag_indices] = uspd_gpu
    
    if n_pairs == 0:
        return H_core

    # Process Core-Electron Attraction Integrals (e1b and e2a)
    # We need to map the 45-element 1D arrays back to the dense 2D diagonal blocks of H_core.
    pair_i = cp.asarray(mol.pair_i, dtype=cp.int32)  # Atom A
    pair_j = cp.asarray(mol.pair_j, dtype=cp.int32)  # Atom B
    
    natorb_i = cp.asarray(mol.norbitals_per_atom[pair_i], dtype=cp.int32)
    natorb_j = cp.asarray(mol.norbitals_per_atom[pair_j], dtype=cp.int32)
    
    # Extract global starting orbital indices for each atom
    aoslice = cp.asarray(mol._aoslice, dtype=cp.int32)
    offset_i = aoslice[pair_i, 0]
    offset_j = aoslice[pair_j, 0]
    
    row_A = offset_i[:, None] + _LOCAL_ROW_IDX[None, :]
    col_A = offset_i[:, None] + _LOCAL_COL_IDX[None, :]
    
    mask_A = (_LOCAL_ROW_IDX[None, :] < natorb_i[:, None]) & \
             (_LOCAL_COL_IDX[None, :] < natorb_i[:, None])
    
    valid_rows_A = row_A[mask_A]
    valid_cols_A = col_A[mask_A]
    valid_e1b = e1b[mask_A]
    
    # Add to Lower Triangle
    # scatter_add should be used, due to the non-unique indices in valid_rows_A and valid_cols_A
    cupyx.scatter_add(H_core, (valid_rows_A, valid_cols_A), valid_e1b)
    
    # Add to Upper Triangle
    off_diag_mask_A = valid_rows_A != valid_cols_A
    cupyx.scatter_add(H_core, (valid_cols_A[off_diag_mask_A], valid_rows_A[off_diag_mask_A]), valid_e1b[off_diag_mask_A])

    # --- Process e2a (Attraction of Atom B's electrons by Atom A's core) ---
    # Goes to block H_BB
    
    row_B = offset_j[:, None] + _LOCAL_ROW_IDX[None, :]
    col_B = offset_j[:, None] + _LOCAL_COL_IDX[None, :]
    
    mask_B = (_LOCAL_ROW_IDX[None, :] < natorb_j[:, None]) & \
             (_LOCAL_COL_IDX[None, :] < natorb_j[:, None])
    
    valid_rows_B = row_B[mask_B]
    valid_cols_B = col_B[mask_B]
    valid_e2a = e2a[mask_B]
    
    # Add to Lower Triangle
    cupyx.scatter_add(H_core, (valid_rows_B, valid_cols_B), valid_e2a)
    
    # Add to Upper Triangle (Symmetric matrix)
    off_diag_mask_B = valid_rows_B != valid_cols_B
    cupyx.scatter_add(H_core, (valid_cols_B[off_diag_mask_B], valid_rows_B[off_diag_mask_B]), valid_e2a[off_diag_mask_B])

    return H_core


def get_hcore(mol):
    """
    Computes the full core Hamiltonian matrix for the given molecule.
    
    This function computes the two-center one-electron integrals (resonance) 
    and combines them with the pre-computed core-electron attraction integrals 
    (e1b, e2a) stored in the mol object to assemble the final dense H_core matrix.
    
    Args:
        mol: PM6Mole instance. Must have undergone _compute_integrals().
        
    Returns:
        H_core: (nao, nao) CuPy array - The dense Core Hamiltonian matrix.
    """
    if not hasattr(mol, 'e1b') or not hasattr(mol, 'e2a'):
        mol._compute_integrals()

    h1elec_mat = hcore2c1e.h1elec(
        mol.principal_quantum_numbers, 
        mol.eta_1e, 
        mol._coords, 
        mol.norbitals_per_atom, 
        mol.beta, 
        cutoff=mol.cutoff, 
        BOHR=mol.BOHR
    )

    H_core = build_hcore_matrix(mol, h1elec_mat, mol.e1b, mol.e2a)
    
    return H_core


def unpack_eri_4d(mol):
    """
    [DEBUG ONLY] 
    Expands the highly compressed 1D two-electron integral array (w_1d) and the 
    one-center integrals into a dense 4-index tensor ERI (nao, nao, nao, nao).
    This strictly maintains the native MOPAC basis order.
    
    Under the NDDO approximation:
    ERI[mu, nu, lam, sig] = (mu nu | lam sig)
    This value is non-zero ONLY IF (mu, nu) belong to the same Atom A, 
    and (lam, sig) belong to the same Atom B.
    
    Args:
        mol: PM6Mole instance.
        w_1d: (total_w_size,) CuPy array - 2c2e integrals computed from the GPU.
        
    Returns:
        eri_4d: (nao, nao, nao, nao) CuPy array.
    """
    nao = mol.nao
    n_pairs = mol.npairs
    w_1d = mol.w
    
    # Initialize a full dense 4D tensor with zeros
    eri_4d = cp.zeros((nao, nao, nao, nao), dtype=cp.float64)

    natorb = mol.norbitals_per_atom
    aoslice = mol._aoslice

    # =========================================================================
    # PART 1: Unpack Two-Center Integrals (A != B) from the GPU w_1d array
    # =========================================================================
    if n_pairs > 0:
        pair_i = mol.pair_i
        pair_j = mol.pair_j
        
        # Calculate kr_offsets for correct slicing of the contiguous w_1d array
        ii_arr = natorb[pair_i]
        kk_arr = natorb[pair_j]
        limij_arr = ii_arr * (ii_arr + 1) // 2
        limkl_arr = kk_arr * (kk_arr + 1) // 2
        block_sizes = limij_arr * limkl_arr
        
        kr_offsets = np.zeros(n_pairs + 1, dtype=np.int32)
        kr_offsets[1:] = np.cumsum(block_sizes)

        # Loop to unpack the integral block for each atom pair
        for p in range(n_pairs):
            A = int(pair_i[p])
            B = int(pair_j[p])
            
            # Starting orbital index for each atom in the global matrix
            offset_A = int(aoslice[A, 0])
            offset_B = int(aoslice[B, 0])
            
            limij = int(limij_arr[p])
            limkl = int(limkl_arr[p])
            
            # Extract the 1D block for this atom pair and reshape it to 2D
            start = int(kr_offsets[p])
            end = int(kr_offsets[p+1])
            w_block = w_1d[start:end].reshape((limij, limkl))
            
            # Iterate over the block and map to the global 4D tensor
            for IJ in range(limij):
                # Directly use the native triangular index mapping
                mu_loc = int(_LOCAL_ROW_IDX[IJ])
                nu_loc = int(_LOCAL_COL_IDX[IJ])
                
                # Add the offset to get global coordinates
                mu = offset_A + mu_loc
                nu = offset_A + nu_loc
                
                for KL in range(limkl):
                    lam_loc = int(_LOCAL_ROW_IDX[KL])
                    sig_loc = int(_LOCAL_COL_IDX[KL])
                    
                    lam = offset_B + lam_loc
                    sig = offset_B + sig_loc
                    
                    val = w_block[IJ, KL]
                    if val == 0.0: 
                        continue
                    
                    # Apply the 8-fold permutation symmetry of two-electron integrals
                    eri_4d[mu, nu, lam, sig] = val
                    eri_4d[nu, mu, lam, sig] = val
                    eri_4d[mu, nu, sig, lam] = val
                    eri_4d[nu, mu, sig, lam] = val
                    
                    eri_4d[lam, sig, mu, nu] = val
                    eri_4d[lam, sig, nu, mu] = val
                    eri_4d[sig, lam, mu, nu] = val
                    eri_4d[sig, lam, nu, mu] = val

    # =========================================================================
    # PART 2: Unpack One-Center Integrals (A == B) from the MOPAC Environment
    # =========================================================================
    env = mol.PM6env
    atom_ids = mol._atom_ids  # 1-based atomic numbers
    
    for A in range(mol.natm):
        offset = int(aoslice[A, 0])
        nao_A = int(natorb[A])
        ni = int(atom_ids[A]) - 1  # 0-based element index for accessing env arrays
        
        # 1. S orbital (Gss) - Local index 0
        gss = float(env.gss6[ni])
        eri_4d[offset, offset, offset, offset] = gss
        
        # 2. P orbitals (Gsp, Gpp, Gp2, Hsp, Hpp)
        if nao_A >= 4:
            gsp = float(env.gsp6[ni])
            gpp = float(env.gpp6[ni])
            gp2 = float(env.gp26[ni])
            hsp = float(env.hsp6[ni])
            hpp = 0.5 * (gpp - gp2)
            
            # Native MOPAC order: px=1, py=2, pz=3
            for i in range(1, 4):
                mu = offset + i
                
                # Gsp = (s s | pi pi)
                eri_4d[offset, offset, mu, mu] = gsp
                eri_4d[mu, mu, offset, offset] = gsp
                
                # Gpp = (pi pi | pi pi)
                eri_4d[mu, mu, mu, mu] = gpp
                
                # Hsp = (s pi | s pi)
                eri_4d[offset, mu, offset, mu] = hsp
                eri_4d[mu, offset, offset, mu] = hsp
                eri_4d[offset, mu, mu, offset] = hsp
                eri_4d[mu, offset, mu, offset] = hsp
                
                for j in range(1, i):
                    nu = offset + j
                    
                    # Gp2 = (pi pi | pj pj)
                    eri_4d[mu, mu, nu, nu] = gp2
                    eri_4d[nu, nu, mu, mu] = gp2
                    
                    # Hpp = (pi pj | pi pj)
                    eri_4d[mu, nu, mu, nu] = hpp
                    eri_4d[nu, mu, mu, nu] = hpp
                    eri_4d[mu, nu, nu, mu] = hpp
                    eri_4d[nu, mu, nu, mu] = hpp
                    
        # 3. D orbitals (Using MOPAC's precomputed intij/intkl mapping arrays)
        if nao_A == 9 and hasattr(env, 'intij'):
            ij_arr = env.intij - 1
            kl_arr = env.intkl - 1
            rp_arr = env.intrep - 1
            
            for k_idx in range(len(ij_arr)):
                IJ = int(ij_arr[k_idx])
                KL = int(kl_arr[k_idx])
                rp = int(rp_arr[k_idx])
                
                val = float(env.repd[rp, ni])
                if val == 0.0: 
                    continue
                
                # Extract local orbital indices using the native lower-triangular map
                mu_loc = int(_LOCAL_ROW_IDX[IJ])
                nu_loc = int(_LOCAL_COL_IDX[IJ])
                lam_loc = int(_LOCAL_ROW_IDX[KL])
                sig_loc = int(_LOCAL_COL_IDX[KL])
                
                mu = offset + mu_loc
                nu = offset + nu_loc
                lam = offset + lam_loc
                sig = offset + sig_loc
                
                # Apply 8-fold symmetry
                eri_4d[mu, nu, lam, sig] = val
                eri_4d[nu, mu, lam, sig] = val
                eri_4d[mu, nu, sig, lam] = val
                eri_4d[nu, mu, sig, lam] = val
                
                eri_4d[lam, sig, mu, nu] = val
                eri_4d[lam, sig, nu, mu] = val
                eri_4d[sig, lam, mu, nu] = val
                eri_4d[sig, lam, nu, mu] = val
                
    return eri_4d

def get_jk_debug(mol, dm, w_1d):
    """
    [DEBUG ONLY] 
    Calculate full J and K matrices explicitly using Einsum for benchmark.
    This includes both one-center and two-center integral contributions, 
    serving as the absolute ground truth for our GPU JK-builders.
    
    J_mu,nu = sum_lam,sig P_lam,sig (mu nu | lam sig)
    K_mu,nu = sum_lam,sig P_lam,sig (mu lam | nu sig)
    """
    eri_4d = unpack_eri_4d(mol, w_1d)
    
    # Calculate complete Coulomb (J) and Exchange (K) matrices
    J = cp.einsum('ls, mnls -> mn', dm, eri_4d)
    K = cp.einsum('ls, mlns -> mn', dm, eri_4d)
    
    return J, K