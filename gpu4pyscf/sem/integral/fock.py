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


import cupy as cp

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

def build_hcore_matrix(mol, h1elec_mat, e1b_out, e2a_out):
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
        e1b_out: (n_pairs, 45) CuPy array - Attraction of Atom A's electrons by Atom B's core.
        e2a_out: (n_pairs, 45) CuPy array - Attraction of Atom B's electrons by Atom A's core.
        
    Returns:
        H_core: (nao, nao) CuPy array - The dense Core Hamiltonian matrix.
    """
    nao = mol.nao
    n_pairs = mol.npairs
    
    # 1. Initialize H_core with the Two-Center Resonance matrix
    # Since h1elec_mat is already a full (nao, nao) matrix with 0s on the diagonal blocks,
    # we can use it as the base.
    H_core = cp.copy(h1elec_mat)
    
    # 2. Add One-Center energies (USPD) to the main diagonal
    uspd_gpu = cp.asarray(mol.uspd, dtype=cp.float64)
    diag_indices = cp.arange(nao)
    H_core[diag_indices, diag_indices] = uspd_gpu
    
    if n_pairs == 0:
        return H_core

    # 3. Process Core-Electron Attraction Integrals (e1b and e2a)
    # We need to map the 45-element 1D arrays back to the dense 2D diagonal blocks of H_core.
    
    pair_i = cp.asarray(mol.pair_i, dtype=cp.int32)  # Atom A
    pair_j = cp.asarray(mol.pair_j, dtype=cp.int32)  # Atom B
    
    natorb_i = cp.asarray(mol.norbitals_per_atom[pair_i], dtype=cp.int32)
    natorb_j = cp.asarray(mol.norbitals_per_atom[pair_j], dtype=cp.int32)
    
    # Extract global starting orbital indices for each atom
    # mol._aoslice shape is (natm, 2). Column 0 is the start index.
    aoslice_gpu = cp.asarray(mol._aoslice, dtype=cp.int32)
    offset_i = aoslice_gpu[pair_i, 0] # (n_pairs,)
    offset_j = aoslice_gpu[pair_j, 0] # (n_pairs,)
    
    # We use cupyx.scatter_add for atomic additions to the same memory locations,
    # because multiple atoms (B) will exert attraction on the same atom (A).
    import cupyx
    
    # --- Process e1b (Attraction of Atom A's electrons by Atom B's core) ---
    # Goes to block H_AA
    
    # Broadcast local coordinates to global matrix coordinates: shape (n_pairs, 45)
    row_A = offset_i[:, None] + _LOCAL_ROW_IDX[None, :]
    col_A = offset_i[:, None] + _LOCAL_COL_IDX[None, :]
    
    # Create a mask to filter out dummy orbitals (e.g., if atom only has s,p orbitals)
    mask_A = (_LOCAL_ROW_IDX[None, :] < natorb_i[:, None]) & \
             (_LOCAL_COL_IDX[None, :] < natorb_i[:, None])
    
    valid_rows_A = row_A[mask_A]
    valid_cols_A = col_A[mask_A]
    valid_e1b = e1b_out[mask_A]
    
    # Add to Lower Triangle
    cupyx.scatter_add(H_core, (valid_rows_A, valid_cols_A), valid_e1b)
    
    # Add to Upper Triangle (Symmetric matrix)
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
    valid_e2a = e2a_out[mask_B]
    
    # Add to Lower Triangle
    cupyx.scatter_add(H_core, (valid_rows_B, valid_cols_B), valid_e2a)
    
    # Add to Upper Triangle (Symmetric matrix)
    off_diag_mask_B = valid_rows_B != valid_cols_B
    cupyx.scatter_add(H_core, (valid_cols_B[off_diag_mask_B], valid_rows_B[off_diag_mask_B]), valid_e2a[off_diag_mask_B])

    return H_core