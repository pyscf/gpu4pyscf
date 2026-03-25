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
import ctypes
import os

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

# totally 45*(45+1)/2 = 1035 integrals, considering symmetry, there are 243 non-zero (52 unique) integrals.
# INTIJ maps the pure index to the orbital index, and the INTREP maps the pure index to the integral idex.
INTIJ = np.array([
            1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
            5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10,
            10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13,
            13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16,
            16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 19,
            20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22,
            22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24,
            24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28,
            28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 31,
            31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34,
            35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
            37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 41,
            42, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45,
            45, 45, 45, 45, 45, 45, 45
        ], dtype=np.int32) - 1

INTKL = np.array([
            15, 21, 28, 36, 45, 12, 19, 23, 39, 11, 15, 21, 22, 26, 28,
            36, 45, 13, 24, 32, 38, 34, 37, 43, 11, 15, 21, 22, 26, 28, 36, 45, 17,
            25, 31, 16, 20, 27, 44, 29, 33, 35, 42, 15, 21, 22, 28, 36, 45, 3, 6,
            11, 21, 26, 36, 2, 12, 19, 23, 39, 4, 13, 24, 32, 38, 14, 17, 31, 1,
            3, 6, 10, 15, 21, 22, 28, 36, 45, 8, 16, 20, 27, 44, 7, 14, 17, 25, 31,
            18, 30, 40, 2, 12, 19, 23, 39, 8, 16, 20, 27, 44, 1, 3, 6, 10, 11, 15,
            21, 22, 26, 28, 36, 45, 3, 6, 10, 15, 21, 22, 28, 36, 45, 2, 12, 19,
            23, 39, 4, 13, 24, 32, 38, 7, 17, 25, 31, 3, 6, 11, 21, 26, 36, 8, 16,
            20, 27, 44, 1, 3, 6, 10, 15, 21, 22, 28, 36, 45, 9, 29, 33, 35, 42, 18,
            30, 40, 7, 14, 17, 25, 31, 4, 13, 24, 32, 38, 9, 29, 33, 35, 42, 5,
            34, 37, 43, 9, 29, 33, 35, 42, 1, 3, 6, 10, 11, 15, 21, 22, 26, 28, 36,
            45, 5, 34, 37, 43, 4, 13, 24, 32, 38, 2, 12, 19, 23, 39, 18, 30, 40,
            41, 9, 29, 33, 35, 42, 5, 34, 37, 43, 8, 16, 20, 27, 44, 1, 3, 6, 10,
            15, 21, 22, 28, 36, 45
        ], dtype=np.int32) - 1

INTREP = np.array([
            1, 1, 1, 1, 1, 3, 3, 8, 3, 9, 6, 6, 12, 14, 13, 7, 6, 15, 8,
            3, 3, 11, 9, 14, 17, 6, 7, 12, 18, 13, 6, 6, 3, 2, 3, 9, 11, 10, 11,
            9, 16, 10, 11, 7, 6, 4, 5, 6, 7, 9, 17, 19, 32, 22, 40, 3, 33, 34, 27,
            46, 15, 33, 28, 41, 47, 35, 35, 42, 1, 6, 6, 7, 29, 38, 22, 31, 38, 51,
            9, 19, 32, 21, 32, 3, 35, 33, 24, 34, 35, 35, 35, 3, 34, 33, 26, 34,
            11, 32, 44, 37, 49, 1, 6, 7, 6, 32, 38, 29, 21, 39, 30, 38, 38, 12, 12,
            4, 22, 21, 19, 20, 21, 22, 8, 27, 26, 25, 27, 8, 28, 25, 26, 27, 2,
            24, 23, 24, 14, 18, 22, 39, 48, 45, 10, 21, 37, 36, 37, 1, 13, 13, 5,
            31, 30, 20, 29, 30, 31, 9, 19, 40, 21, 32, 35, 35, 35, 3, 42, 34, 24,
            33, 3, 41, 26, 33, 34, 16, 40, 44, 43, 50, 11, 44, 32, 39, 10, 21, 43,
            36, 37, 1, 7, 6, 6, 40, 38, 38, 21, 45, 30, 29, 38, 9, 32, 19, 22, 3,
            47, 27, 34, 33, 3, 46, 34, 27, 33, 35, 35, 35, 52, 11, 32, 50, 37, 44,
            14, 39, 22, 48, 11, 32, 49, 37, 44, 1, 6, 6, 7, 51, 38, 22, 31, 38, 29
        ], dtype=np.int32) - 1


def _load_jk_cuda_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'libfock.so'
    
    lib_path = os.path.join(curr_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}. Please compile fock_jk.cu first.")
    
    lib = ctypes.CDLL(lib_path)

    # Define the argument types for the C host function
    lib.launch_build_jk_2c2e.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,  # w_1d, P
        ctypes.c_void_p, ctypes.c_void_p,  # J, K
        ctypes.c_void_p, ctypes.c_void_p,  # pair_i, pair_j
        ctypes.c_void_p, ctypes.c_void_p,  # kr_offsets, aoslice
        ctypes.c_void_p,                   # natorb
        ctypes.c_void_p, ctypes.c_void_p,  # loc_row, loc_col
        ctypes.c_int, ctypes.c_int         # npairs, nao
    ]

    lib.launch_build_jk_1c2e.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # P, J, K
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gss, gsp, hsp
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gpp, gp2, repd
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # intij, intkl, intrep
        ctypes.c_void_p, ctypes.c_void_p,                   # aoslice, natorb
        ctypes.c_void_p, ctypes.c_void_p,                   # loc_row, loc_col
        ctypes.c_int, ctypes.c_int, ctypes.c_int            # natm, nao, num_d_pairs
    ]

    return lib
    

_jk_module = _load_jk_cuda_library()


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
    uspd_gpu = cp.asarray(mol.one_center_integrals.uspd, dtype=cp.float64)
    diag_indices = cp.arange(nao)
    H_core[diag_indices, diag_indices] = uspd_gpu
    
    if n_pairs == 0:
        return H_core

    # Process Core-Electron Attraction Integrals (e1b and e2a)
    # We need to map the 45-element 1D arrays back to the dense 2D diagonal blocks of H_core.
    pair_i = cp.asarray(mol.pair_i, dtype=cp.int32)  # Atom A
    pair_j = cp.asarray(mol.pair_j, dtype=cp.int32)  # Atom B
    
    natorb_i = cp.asarray(mol.topology.norbitals_per_atom[pair_i], dtype=cp.int32)
    natorb_j = cp.asarray(mol.topology.norbitals_per_atom[pair_j], dtype=cp.int32)
    
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
    cp.add.at(H_core, (valid_rows_A, valid_cols_A), valid_e1b)
    
    # Add to Upper Triangle
    off_diag_mask_A = valid_rows_A != valid_cols_A
    cp.add.at(H_core, (valid_cols_A[off_diag_mask_A], valid_rows_A[off_diag_mask_A]), valid_e1b[off_diag_mask_A])

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
    cp.add.at(H_core, (valid_rows_B, valid_cols_B), valid_e2a)
    
    # Add to Upper Triangle (Symmetric matrix)
    off_diag_mask_B = valid_rows_B != valid_cols_B
    cp.add.at(H_core, (valid_cols_B[off_diag_mask_B], valid_rows_B[off_diag_mask_B]), valid_e2a[off_diag_mask_B])

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
        mol.topology.principal_quantum_numbers, 
        mol.topology.eta_1e, 
        mol._coords, 
        mol.topology.norbitals_per_atom, 
        mol.beta, 
        cutoff=mol.cutoff, 
        BOHR=mol.BOHR
    )

    H_core = build_hcore_matrix(mol, h1elec_mat, mol.two_center_integrals.e1b, mol.two_center_integrals.e2a)
    
    return H_core


def unpack_eri_4d(mol):
    """
    Expands the compressed 1D two-electron integral array (w) and the 
    one-center integrals into a dense 4-index tensor ERI (nao, nao, nao, nao).
    This strictly maintains the native MOPAC basis order.
    
    Under the NDDO approximation:
    ERI[mu, nu, lam, sig] = (mu nu | lam sig)
    This value is non-zero ONLY IF (mu, nu) belong to the same Atom A, 
    and (lam, sig) belong to the same Atom B.
    
    Args:
        mol: PM6Mole instance.
        
    Returns:
        eri_4d: (nao, nao, nao, nao) CuPy array.
    """
    nao = mol.nao
    n_pairs = mol.npairs
    w_1d = mol.two_center_integrals.w
    
    eri_4d = cp.zeros((nao, nao, nao, nao), dtype=cp.float64)

    natorb = mol.topology.norbitals_per_atom
    aoslice = mol._aoslice

    # Two-Center Integrals
    if n_pairs > 0:
        pair_i = mol.pair_i
        pair_j = mol.pair_j
        
        ii_arr = natorb[pair_i]
        kk_arr = natorb[pair_j]
        limij_arr = ii_arr * (ii_arr + 1) // 2
        limkl_arr = kk_arr * (kk_arr + 1) // 2
        block_sizes = limij_arr * limkl_arr
        
        kr_offsets = cp.zeros(n_pairs + 1, dtype=cp.int32)
        kr_offsets[1:] = cp.cumsum(block_sizes)

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
            
            for IJ in range(limij):
                # Directly use the native triangular index mapping
                mu_loc = int(_LOCAL_ROW_IDX[IJ])
                nu_loc = int(_LOCAL_COL_IDX[IJ])
                
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
                    
                    eri_4d[mu, nu, lam, sig] = val
                    eri_4d[nu, mu, lam, sig] = val
                    eri_4d[mu, nu, sig, lam] = val
                    eri_4d[nu, mu, sig, lam] = val
                    
                    eri_4d[lam, sig, mu, nu] = val
                    eri_4d[lam, sig, nu, mu] = val
                    eri_4d[sig, lam, mu, nu] = val
                    eri_4d[sig, lam, nu, mu] = val

    gss = mol.one_center_integrals.gss
    gsp = mol.one_center_integrals.gsp
    hsp = mol.one_center_integrals.hsp
    gpp = mol.one_center_integrals.gpp
    gp2 = mol.one_center_integrals.gp2
    repd = mol.one_center_integrals.repd
    
    for A in range(mol.natm):
        offset = int(aoslice[A, 0])
        nao_A = int(natorb[A])
        
        # S orbital (Gss) - Local index 0
        val_gss = float(gss[A])
        eri_4d[offset, offset, offset, offset] = val_gss
        
        # P orbitals (Gsp, Gpp, Gp2, Hsp, Hpp)
        if nao_A >= 4:
            val_gsp = float(gsp[A])
            val_gpp = float(gpp[A])
            val_gp2 = float(gp2[A])
            val_hsp = float(hsp[A])
            val_hpp = 0.5 * (val_gpp - val_gp2)
            
            # Native MOPAC order: px=1, py=2, pz=3
            for i in range(1, 4):
                mu = offset + i
                
                # Gsp = (s s | pi pi)
                eri_4d[offset, offset, mu, mu] = val_gsp
                eri_4d[mu, mu, offset, offset] = val_gsp
                
                # Gpp = (pi pi | pi pi)
                eri_4d[mu, mu, mu, mu] = val_gpp
                
                # Hsp = (s pi | s pi)
                eri_4d[offset, mu, offset, mu] = val_hsp
                eri_4d[mu, offset, offset, mu] = val_hsp
                eri_4d[offset, mu, mu, offset] = val_hsp
                eri_4d[mu, offset, mu, offset] = val_hsp
                
                for j in range(1, i):
                    nu = offset + j
                    
                    # Gp2 = (pi pi | pj pj)
                    eri_4d[mu, mu, nu, nu] = val_gp2
                    eri_4d[nu, nu, mu, mu] = val_gp2
                    
                    # Hpp = (pi pj | pi pj)
                    eri_4d[mu, nu, mu, nu] = val_hpp
                    eri_4d[nu, mu, mu, nu] = val_hpp
                    eri_4d[mu, nu, nu, mu] = val_hpp
                    eri_4d[nu, mu, nu, mu] = val_hpp
                    
        if nao_A == 9:
            ij_arr = INTIJ
            kl_arr = INTKL
            rp_arr = INTREP
            
            for k_idx in range(len(ij_arr)):
                IJ = int(ij_arr[k_idx])
                KL = int(kl_arr[k_idx])
                rp = int(rp_arr[k_idx])
                
                val = float(repd[rp, A])
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

def get_jk_debug(mol, dm, hermi=1):
    """
    Calculate full J and K matrices explicitly using Einsum for benchmark.
    This includes both one-center and two-center integral contributions, 
    serving as the absolute ground truth for our GPU JK-builders.
    
    J_mu,nu = sum_lam,sig P_lam,sig (mu nu | lam sig)
    K_mu,nu = sum_lam,sig P_lam,sig (mu lam | nu sig)
    """
    assert hermi == 1, "Only hermitian matrices are supported."
    eri_4d = unpack_eri_4d(mol)
    
    J = cp.einsum('ls, mnls -> mn', dm, eri_4d)
    K = cp.einsum('ls, mlns -> mn', dm, eri_4d)
    
    return J, K


def get_jk(mol, dm):

    w_1d = mol.two_center_integrals.w
    nao = mol.nao

    if isinstance(dm, np.ndarray):
        dm = cp.asarray(dm, dtype=cp.float64)

    J = cp.zeros((nao, nao), dtype=cp.float64)
    K = cp.zeros((nao, nao), dtype=cp.float64)
    
    dm_c = cp.ascontiguousarray(dm, dtype=cp.float64)
    aoslice_c = cp.ascontiguousarray(cp.asarray(mol._aoslice), dtype=cp.int32)
    natorb_c = cp.ascontiguousarray(mol.topology.norbitals_per_atom, dtype=cp.int32)
    
    if mol.npairs > 0:
        w_1d_c = cp.ascontiguousarray(w_1d, dtype=cp.float64)
        pair_i_c = cp.ascontiguousarray(cp.asarray(mol.pair_i), dtype=cp.int32)
        pair_j_c = cp.ascontiguousarray(cp.asarray(mol.pair_j), dtype=cp.int32)
        
        ii_arr = mol.topology.norbitals_per_atom[mol.pair_i]
        kk_arr = mol.topology.norbitals_per_atom[mol.pair_j]
        block_sizes = (ii_arr * (ii_arr + 1) // 2) * (kk_arr * (kk_arr + 1) // 2)
        
        kr_offsets = cp.zeros(mol.npairs + 1, dtype=np.int32)
        kr_offsets[1:] = cp.cumsum(block_sizes)
        kr_offsets_c = cp.asarray(kr_offsets, dtype=cp.int32)
        
        _jk_module.launch_build_jk_2c2e(
            ctypes.c_void_p(w_1d_c.data.ptr),
            ctypes.c_void_p(dm_c.data.ptr),
            ctypes.c_void_p(J.data.ptr),
            ctypes.c_void_p(K.data.ptr),
            ctypes.c_void_p(pair_i_c.data.ptr),
            ctypes.c_void_p(pair_j_c.data.ptr),
            ctypes.c_void_p(kr_offsets_c.data.ptr),
            ctypes.c_void_p(aoslice_c.data.ptr),
            ctypes.c_void_p(natorb_c.data.ptr),
            ctypes.c_void_p(_LOCAL_ROW_IDX.data.ptr), 
            ctypes.c_void_p(_LOCAL_COL_IDX.data.ptr),
            ctypes.c_int(int(mol.npairs)),
            ctypes.c_int(int(nao))
        )

    if mol.natm > 0:
        gss = mol.one_center_integrals.gss
        gsp = mol.one_center_integrals.gsp
        hsp = mol.one_center_integrals.hsp
        gpp = mol.one_center_integrals.gpp
        gp2 = mol.one_center_integrals.gp2
        repd = mol.one_center_integrals.repd
        
        if cp.any(mol.topology.has_d_orbitals):
            intij = cp.asarray(INTIJ, dtype=cp.int32)
            intkl = cp.asarray(INTKL, dtype=cp.int32)
            intrep = cp.asarray(INTREP, dtype=cp.int32)
            num_d_pairs = len(intij)
        else:
            intij = cp.zeros(1, dtype=cp.int32)
            intkl = cp.zeros(1, dtype=cp.int32)
            intrep = cp.zeros(1, dtype=cp.int32)
            num_d_pairs = 0
            
        gss_c = cp.ascontiguousarray(gss, dtype=cp.float64)
        gsp_c = cp.ascontiguousarray(gsp, dtype=cp.float64)
        hsp_c = cp.ascontiguousarray(hsp, dtype=cp.float64)
        gpp_c = cp.ascontiguousarray(gpp, dtype=cp.float64)
        gp2_c = cp.ascontiguousarray(gp2, dtype=cp.float64)
        repd_c = cp.ascontiguousarray(repd, dtype=cp.float64)
        
        intij_c = cp.ascontiguousarray(intij, dtype=cp.int32)
        intkl_c = cp.ascontiguousarray(intkl, dtype=cp.int32)
        intrep_c = cp.ascontiguousarray(intrep, dtype=cp.int32)
        
        _jk_module.launch_build_jk_1c2e(
            ctypes.c_void_p(dm_c.data.ptr),
            ctypes.c_void_p(J.data.ptr),
            ctypes.c_void_p(K.data.ptr),
            ctypes.c_void_p(gss_c.data.ptr),
            ctypes.c_void_p(gsp_c.data.ptr),
            ctypes.c_void_p(hsp_c.data.ptr),
            ctypes.c_void_p(gpp_c.data.ptr),
            ctypes.c_void_p(gp2_c.data.ptr),
            ctypes.c_void_p(repd_c.data.ptr),
            ctypes.c_void_p(intij_c.data.ptr),
            ctypes.c_void_p(intkl_c.data.ptr),
            ctypes.c_void_p(intrep_c.data.ptr),
            ctypes.c_void_p(aoslice_c.data.ptr),
            ctypes.c_void_p(natorb_c.data.ptr),
            ctypes.c_void_p(_LOCAL_ROW_IDX.data.ptr),
            ctypes.c_void_p(_LOCAL_COL_IDX.data.ptr),
            ctypes.c_int(int(mol.natm)),
            ctypes.c_int(int(nao)),
            ctypes.c_int(int(num_d_pairs))
        )
        
    return J, K