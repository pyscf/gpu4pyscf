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

import ctypes
import os
import numpy as np
import cupy as cp
from scipy.special import comb

_MAX_N = 17
_FACTORIALS_CPU = np.ones(_MAX_N + 1, dtype=np.float64)
_FACTORIALS_CPU[1:] = np.cumprod(np.arange(1, _MAX_N + 1, dtype=np.float64))
FACTORIALS_GPU = cp.asarray(_FACTORIALS_CPU)

_n_grid = np.arange(13).reshape(-1, 1)
_k_grid = np.arange(13).reshape(1, -1)
_BINOMIALS_CPU = comb(_n_grid, _k_grid)
BINOMIALS_GPU = cp.asarray(_BINOMIALS_CPU)
BINOMIALS_GPU_FLAT = cp.asarray(_BINOMIALS_CPU.ravel(), dtype=np.float64)

# Angular factors
_AFF_CPU = np.zeros((3, 3, 3), dtype=np.float64)
_AFF_CPU[0, 0, 0] = 1.0
_AFF_CPU[1, 0, 0] = 1.0
_AFF_CPU[1, 1, 0] = np.sqrt(0.5)
_AFF_CPU[2, 0, 0] = 1.5
_AFF_CPU[2, 1, 0] = np.sqrt(1.5)
_AFF_CPU[2, 2, 0] = np.sqrt(0.375)
_AFF_CPU[2, 0, 2] = -0.5
AFF_GPU = cp.asarray(_AFF_CPU)


def _precompute_taylor_coeffs():
    """
    Generate the coefficient matrix C[m, i] for the Taylor expansion using vectorization.
    
    Dimensions:
        Rows (m): 0 to 15 (power terms of x)
        Cols (i): 0 to 12 (index of B_n)
    
    Formula: 
        C[m, i] = 2 / (m! * (m + i + 1))   if (m + i + 1) is odd
        C[m, i] = 0                        otherwise
    
    Returns:
        np.ndarray: Coefficients matrix of shape (16, 13).
    """
    last = 15
    k_max = 12
    
    m_idx = np.arange(last + 1, dtype=np.float64)[:, None]
    i_idx = np.arange(k_max + 1, dtype=np.float64)[None, :]
    term_sum = m_idx + i_idx + 1.0
    
    # Determine parity mask (Odd sum -> Non-zero coefficient)
    mask = (term_sum % 2) == 1
    fact_m = _FACTORIALS_CPU[:last + 1][:, None]
    
    coeffs = np.zeros((last + 1, k_max + 1), dtype=np.float64)
    np.divide(2.0, fact_m * term_sum, out=coeffs, where=mask)
    
    return coeffs


_TAYLOR_COEFFS_CPU = _precompute_taylor_coeffs()
TAYLOR_COEFFS_GPU = cp.asarray(_TAYLOR_COEFFS_CPU)
TAYLOR_COEFFS_GPU_T = cp.asarray(_TAYLOR_COEFFS_CPU.T.ravel(), dtype=np.float64)


def _load_cuda_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'libss_kernel.so'
    
    lib_path = os.path.join(curr_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)
    
    lib.launch_ss_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p
    ]

    lib.launch_afn_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p
    ]

    lib.launch_bfn_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    lib.launch_rotation_transform_kernel.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    return lib

_SS_MODULE = _load_cuda_library()


def bfn(x):
    original_shape = x.shape
    x_flat = x.ravel()
    n_data = x_flat.size

    bf = cp.empty((n_data, 13), dtype=np.float64)
    
    _SS_MODULE.launch_bfn_kernel_c(
        ctypes.c_int(n_data),
        ctypes.c_void_p(x_flat.data.ptr),
        ctypes.c_void_p(TAYLOR_COEFFS_GPU_T.data.ptr),
        ctypes.c_void_p(bf.data.ptr)
    )

    if x.ndim != 1:
        return bf.reshape(original_shape + (13,))
    return bf

def afn(p):
    n_data = p.size
    p_flat = p.ravel()
    af = cp.empty((n_data, 20), dtype=np.float64)
    
    _SS_MODULE.launch_afn_kernel_c(
        ctypes.c_int(n_data),
        ctypes.c_void_p(p_flat.data.ptr),
        ctypes.c_void_p(af.data.ptr)
    )
    return af


def rotation_transform(S_local, C_tensor):
    """
    Assembly of global 9x9 overlap matrix.
    """
    n_pairs = S_local.shape[0]
    di = cp.zeros((n_pairs, 9, 9), dtype=np.float64)
    
    _SS_MODULE.launch_rotation_transform_kernel(
        ctypes.c_int(n_pairs),
        ctypes.c_void_p(S_local.data.ptr),
        ctypes.c_void_p(C_tensor.data.ptr),
        ctypes.c_void_p(di.data.ptr)
    )
    return di


def ovlp_in_2c1e(na, nb, la, lb, m, ua, ub, r):
    """
    Compute two-center overlap integrals (resonance) on GPU.
    
    Args:
        na, nb (cp.ndarray): Principal Quantum Numbers (N,). ! from 1
        la, lb (cp.ndarray): Angular Momentum Quantum Numbers (N,). ! from 0
        m      (cp.ndarray): Magnetic Quantum Number (N,). ! from 0
        ua, ub (cp.ndarray): Orbital Exponents (N,).
        r      (cp.ndarray): Interatomic Distance in **Bohr** (N,).
    
    Returns:
        cp.ndarray: Overlap integral values (N,).
    """
    n_pairs = r.size
    
    p = (ua + ub) * r * 0.5
    b = (ua - ub) * r * 0.5
    
    af = afn(p)  # (N, 20)
    bf = bfn(b)  # (N, 13)
    
    lam1 = la - m
    lbm1 = lb - m
    
    # Flatten the (i_val, j_val) combinations into a batch dimension (4, 1). 
    # Non-zero overlap integrals for s/p/d strictly require i_val and j_val to be 0 or 2 (parity constraints).
    i_vals = cp.array([0, 0, 2, 2], dtype=cp.int32)[:, None]
    j_vals = cp.array([0, 2, 0, 2], dtype=cp.int32)[:, None]
    
    # Expand masks and data arrays for batching
    mask_i = i_vals <= lam1[None, :]
    mask_j = (j_vals <= lbm1[None, :]) & mask_i
    
    ia = na[None, :] + i_vals - la[None, :]
    ic = la[None, :] - i_vals - m[None, :]
    ib = nb[None, :] + j_vals - lb[None, :]
    id_ = lb[None, :] - j_vals - m[None, :]
    iab = ia + ib
    
    aff_a = AFF_GPU[la[None, :], m[None, :], i_vals]
    aff_b = AFF_GPU[lb[None, :], m[None, :], j_vals]
    pre_factor = aff_a * aff_b * mask_j
    
    total_tasks = 4 * n_pairs
    
    ia_in = cp.where(mask_j, ia, -1).astype(cp.int32).ravel()
    ib_in = cp.where(mask_j, ib, 0).astype(cp.int32).ravel()
    ic_in = cp.where(mask_j, ic, 0).astype(cp.int32).ravel()
    id_in = cp.where(mask_j, id_, 0).astype(cp.int32).ravel()
    m_in  = cp.tile(m, 4).astype(cp.int32)
    iab_in = cp.where(mask_j, iab, 0).astype(cp.int32).ravel()
    
    af_in = cp.tile(af, (4, 1))
    bf_in = cp.tile(bf, (4, 1))
    
    kernel_out = cp.zeros(total_tasks, dtype=cp.float64)
    
    # Launch CUDA Kernel only ONCE
    _SS_MODULE.launch_ss_kernel_c(
        ctypes.c_int(total_tasks),
        ctypes.c_void_p(ia_in.data.ptr),
        ctypes.c_void_p(ib_in.data.ptr),
        ctypes.c_void_p(ic_in.data.ptr),
        ctypes.c_void_p(id_in.data.ptr),
        ctypes.c_void_p(m_in.data.ptr),
        ctypes.c_void_p(iab_in.data.ptr),
        ctypes.c_void_p(af_in.data.ptr),
        ctypes.c_void_p(bf_in.data.ptr),
        ctypes.c_void_p(BINOMIALS_GPU_FLAT.data.ptr),
        ctypes.c_void_p(kernel_out.data.ptr)
    )
    
    # Reshape the 1D output back to (4, N) and sum over the 4 combinations
    kernel_out_reshaped = kernel_out.reshape(4, n_pairs)
    total_val = cp.sum(kernel_out_reshaped * pre_factor, axis=0)
    
    fact_2na = FACTORIALS_GPU[2 * na]
    fact_2nb = FACTORIALS_GPU[2 * nb]
    
    term_sqrt = cp.sqrt(
        (ua * ub) / (fact_2na * fact_2nb) * ((2*la + 1) * (2*lb + 1))
    )
    
    val = (
        total_val * (r**(na + nb + 1)) * (ua**na) * (ub**nb) * 0.5 * term_sqrt
    )
    
    return val


def get_direction_cosines(rij_vec):
    """
    Calculation of the direction cosine tensor.
    
    Args:
        rij_vec (cp.ndarray): Shape (N_pairs, 3). Vectors R_j - R_i

    Returns:
        C (cp.ndarray): Shape (N_pairs, 3, 5, 5). 
                        The rotation coefficient tensor.
                        Dim 1: Shell index (0=S, 1=P, 2=D).
                        Dim 2: global m.
                        Dim 3: local m. 2-> sigma, 1,3-> pi, 0,5-> delta
    """
    rij_vec = cp.asarray(rij_vec, dtype=cp.float64)
    
    x = rij_vec[:, 0]
    y = rij_vec[:, 1]
    z = rij_vec[:, 2]
    
    r2 = x*x + y*y + z*z
    r = cp.sqrt(r2)
    xy2 = x*x + y*y
    xy = cp.sqrt(xy2)
    
    n_pairs = len(x)
    
    ca = cp.ones(n_pairs, dtype=cp.float64)
    sa = cp.zeros(n_pairs, dtype=cp.float64)
    cb = cp.ones(n_pairs, dtype=cp.float64)
    sb = cp.zeros(n_pairs, dtype=cp.float64)
    
    eps = 1.0e-10
    mask_nonzero_xy = xy >= eps
    mask_zero_xy = ~mask_nonzero_xy
    mask_nonzero_r = r > 0.0
    
    inv_xy = cp.zeros_like(xy)
    inv_xy[mask_nonzero_xy] = 1.0 / xy[mask_nonzero_xy]
    
    inv_r = cp.zeros_like(r)
    inv_r[mask_nonzero_r] = 1.0 / r[mask_nonzero_r]
    
    ca[mask_nonzero_xy] = x[mask_nonzero_xy] * inv_xy[mask_nonzero_xy]
    sa[mask_nonzero_xy] = y[mask_nonzero_xy] * inv_xy[mask_nonzero_xy]
    
    cb[mask_nonzero_r] = z[mask_nonzero_r] * inv_r[mask_nonzero_r]
    sb[mask_nonzero_r] = xy[mask_nonzero_r] * inv_r[mask_nonzero_r]
    
    mask_neg_z = mask_zero_xy & (z < 0.0)
    ca[mask_neg_z] = -1.0
    cb[mask_neg_z] = -1.0

    mask_zero_z = mask_zero_xy & (z == 0.0)
    ca[mask_zero_z] = 0.0
    cb[mask_zero_z] = 0.0
    
    c2a = 2.0 * ca * ca - 1.0
    c2b = 2.0 * cb * cb - 1.0
    s2a = 2.0 * sa * ca
    s2b = 2.0 * sb * cb
    
    rt34 = 0.86602540378444  # sqrt(3)/2
    rt13 = 0.57735026918963  # 1/sqrt(3)
    
    C = cp.zeros((n_pairs, 3, 5, 5), dtype=cp.float64)
    
    # Shell S (index 0)
    C[:, 0, 2, 2] = 1.0 #37
    
    # Shell P (index 1)
    cacb = ca * cb
    casb = ca * sb
    sacb = sa * cb
    sasb = sa * sb
    
    # Shell P (index 1)
    C[:, 1, 3, 3] = cacb                        # 56 (original index)
    C[:, 1, 3, 2] = casb                        # 41
    C[:, 1, 3, 1] = -sa                         # 26
    C[:, 1, 2, 3] = -sb                         # 53
    C[:, 1, 2, 2] = cb                          # 38
    # C[:, 1, 2, 1] = 0.0                       # 23
    C[:, 1, 1, 3] = sacb                        # 50
    C[:, 1, 1, 2] = sasb                        # 35
    C[:, 1, 1, 1] = ca                          # 20
    
    # Shell D (index 2)
    C[:, 2, 4, 4] = c2a*cb*cb + 0.5*c2a*sb*sb   # 75
    C[:, 2, 4, 3] = 0.5*c2a*s2b                 # 60
    C[:, 2, 4, 2] = rt34*c2a*sb*sb              # 45
    C[:, 2, 4, 1] = -s2a*sb                     # 30
    C[:, 2, 4, 0] = -s2a*cb                     # 15
    C[:, 2, 3, 4] = -0.5*ca*s2b                 # 72
    C[:, 2, 3, 3] =  ca*c2b                     # 57
    C[:, 2, 3, 2] =  rt34*ca*s2b                # 42
    C[:, 2, 3, 1] = -sa*cb                      # 27
    C[:, 2, 3, 0] =  sa*sb                      # 12
    C[:, 2, 2, 4] = rt13*1.5*sb*sb              # 69
    C[:, 2, 2, 3] = -rt34*s2b                   # 54
    C[:, 2, 2, 2] = cb*cb - 0.5*sb*sb           # 39
    C[:, 2, 1, 4] = -0.5*sa*s2b                 # 66
    C[:, 2, 1, 3] =  sa*c2b                     # 51  
    C[:, 2, 1, 2] =  rt34*sa*s2b                # 36
    C[:, 2, 1, 1] =  ca*cb                      # 21
    C[:, 2, 1, 0] = -ca*sb                      # 6
    C[:, 2, 0, 4] = s2a*cb*cb + 0.5*s2a*sb*sb   # 63
    C[:, 2, 0, 3] = 0.5*s2a*s2b                 # 48
    C[:, 2, 0, 2] = rt34*s2a*sb*sb              # 33
    C[:, 2, 0, 1] = c2a*sb                      # 18
    C[:, 2, 0, 0] = c2a*cb                      # 3
    
    return C


def calc_local_overlap(na_mat, nb_mat, za_exps, zb_exps, r_dist):
    """
    Args:
        na_mat, nb_mat (cp.ndarray): (N, 3) matrix of principal quantum numbers [ns, np, nd].
        za_exps, zb_exps (tuple): Tuple of (N,) tuples for exponents (zs, zp, zd).
        r_dist (cp.ndarray): (N,) Interatomic distance.

        In this function, no d-orbital is added for defensive purpose. Because
        we use the mask.
        
    Returns:
        cp.ndarray: S_local (N, 3, 3, 3)
    """
    n_pairs = len(r_dist)
    za_exps = cp.asarray(za_exps, dtype=cp.float64)
    zb_exps = cp.asarray(zb_exps, dtype=cp.float64)
    
    # Generate 3D grid representing combinations of (Shell A, Shell B, Symmetry)
    # Shell: 0=S, 1=P, 2=D. Symmetry (m): 0=Sigma, 1=Pi, 2=Delta.
    # Total combinations = 3 * 3 * 3 = 27
    grids = cp.mgrid[0:3, 0:3, 0:3]
    # Shape (27,)
    ia_indices = grids[0].ravel().astype(cp.int32) 
    ib_indices = grids[1].ravel().astype(cp.int32)
    m_indices  = grids[2].ravel().astype(cp.int32)
    
    r_flat = cp.broadcast_to(r_dist[:, None], (n_pairs, 27)).ravel()
    
    la_flat = cp.broadcast_to(ia_indices[None, :], (n_pairs, 27)).ravel()
    lb_flat = cp.broadcast_to(ib_indices[None, :], (n_pairs, 27)).ravel()
    m_flat  = cp.broadcast_to(m_indices[None, :],  (n_pairs, 27)).ravel()
    
    idx_a = cp.broadcast_to(ia_indices[None, :], (n_pairs, 27))
    idx_b = cp.broadcast_to(ib_indices[None, :], (n_pairs, 27))
    
    ua_flat = cp.take_along_axis(za_exps, idx_a, axis=1).ravel()
    ub_flat = cp.take_along_axis(zb_exps, idx_b, axis=1).ravel()
    
    na_flat = cp.take_along_axis(na_mat, idx_a, axis=1).ravel().astype(cp.int32)
    nb_flat = cp.take_along_axis(nb_mat, idx_b, axis=1).ravel().astype(cp.int32)

    # mask uncontributed terms
    # TODO: this may be more strict
    mask = (ua_flat > 1.0e-8) & \
           (ub_flat > 1.0e-8) & \
           (m_flat <= la_flat) & \
           (m_flat <= lb_flat) # nk1 = min(i, j) + 1 
           
    val_flat = cp.zeros_like(r_flat)
    
    if cp.any(mask):
        val_computed = ovlp_in_2c1e(
            na_flat[mask], nb_flat[mask], 
            la_flat[mask], lb_flat[mask], m_flat[mask], 
            ua_flat[mask], ub_flat[mask], r_flat[mask]
        )
        
        val_flat[mask] = val_computed
   
    # [Pair Index, Atom A Shell, Atom B Shell, Symmetry m]
    S_local = val_flat.reshape(n_pairs, 3, 3, 3)
    
    return S_local


# TODO: this can be fused with above calculations into 1 kernel
# def rotation_transform(S_local, C_tensor):
#     ...


def h1elec(principal_quantum_numbers, eta_1e, coords, natorb, beta, cutoff=10.0, BOHR=0.529177210903):
    """
    Main entry point for calculating 2c1e matrices (H-core) on GPU.
    
    Args:
        principal_quantum_numbers (cp.ndarray): Principal quantum numbers (N, 3).
        eta_1e (cp.ndarray): Orbital exponents for 1e integrals (N, 3).
        coords (cp.ndarray): Coordinates (N, 3) in Bohr.
        natorb (cp.ndarray): Number of orbitals per atom (N,).
        beta (cp.ndarray): Beta parameters (s, p, d) for all atoms (N, 3).
        cutoff (float): Distance cutoff (Angstrom).
        BOHR (float): Conversion factor from Bohr to Angstrom.
        
    Returns:
        cp.ndarray: (NAO, NAO) H-core integrals (Overlap * Beta_avg).
    """
    if not isinstance(principal_quantum_numbers, cp.ndarray):
        principal_quantum_numbers = cp.asarray(principal_quantum_numbers)
    if not isinstance(eta_1e, cp.ndarray):
        eta_1e = cp.asarray(eta_1e)
    if not isinstance(coords, cp.ndarray):
        coords = cp.asarray(coords)
    if not isinstance(natorb, cp.ndarray):
        natorb = cp.asarray(natorb)
    if not isinstance(beta, cp.ndarray):
        beta = cp.asarray(beta)
    cutoff_bohr = cutoff / BOHR
    n_atoms = coords.shape[0]
    
    offsets = cp.zeros(n_atoms + 1, dtype=cp.int32)
    cp.cumsum(natorb, out=offsets[1:])
    nao = int(offsets[-1])
    
    rij_all = coords[None, :, :] - coords[:, None, :]
    dist_sq = cp.sum(rij_all**2, axis=2)
    dist = cp.sqrt(dist_sq) # (N, N)

    mask_pairs = (dist < cutoff_bohr) & (dist > 1e-6)
    mask_triu = cp.triu(cp.ones((n_atoms, n_atoms), dtype=bool), k=1) # diagonal excluded
    valid_mask = mask_pairs & mask_triu
    
    idx_i, idx_j = cp.where(valid_mask)
    n_pairs = len(idx_i)
    
    if n_pairs == 0:
        return cp.zeros((nao, nao), dtype=cp.float64)

    rij_vec = rij_all[idx_i, idx_j] # (N_pairs, 3)
    r_dist = dist[idx_i, idx_j]     # (N_pairs,)
    
    na_pairs = principal_quantum_numbers[idx_i]
    nb_pairs = principal_quantum_numbers[idx_j]
    za_pairs = eta_1e[idx_i]
    zb_pairs = eta_1e[idx_j]
    
    # Local Overlap Matrix (N_pairs, 3, 3, 3)
    S_local = calc_local_overlap(na_pairs, nb_pairs, za_pairs, zb_pairs, r_dist)
    # Rotation Tensor (N_pairs, 3, 5, 5)
    C_tensor = get_direction_cosines(rij_vec)
    # Global Overlap Blocks (N_pairs, 9, 9)
    S_global = rotation_transform(S_local, C_tensor)
    
    beta_expanded = cp.zeros((n_atoms, 9), dtype=cp.float64)
    beta_expanded[:, 0]   = beta[:, 0]        # s
    beta_expanded[:, 1:4] = beta[:, 1][:, None] # p
    beta_expanded[:, 4:9] = beta[:, 2][:, None] # d
    
    b_i = beta_expanded[idx_i]
    b_j = beta_expanded[idx_j]
    
    beta_sum = 0.5 * (b_i[:, :, None] + b_j[:, None, :])
    H_blocks = S_global * beta_sum
    
    H_core = cp.zeros((nao, nao), dtype=cp.float64)
    orb_range = cp.arange(9, dtype=cp.int32)
    
    grid_r = orb_range[None, :, None] # (1, 9, 1)
    grid_c = orb_range[None, None, :] # (1, 1, 9)
    
    # Calculate global matrix indices for mapping 9x9 local blocks back to full H_core matrix.
    # Note: adding grid_c * 0 / grid_r * 0 forces broadcasting to shape (N_pairs, 9, 9)
    global_row_indices = offsets[idx_i][:, None, None] + grid_r + grid_c * 0
    global_col_indices = offsets[idx_j][:, None, None] + grid_c + grid_r * 0
    
    # Create mask for valid orbitals (handling natorb 1 vs 4 vs 9)
    n_i = natorb[idx_i][:, None, None]
    n_j = natorb[idx_j][:, None, None]
    
    # Valid if local_row < n_i AND local_col < n_j
    block_mask = (grid_r < n_i) & (grid_c < n_j) # (N_pairs, 9, 9)
    
    valid_rows = global_row_indices[block_mask]
    valid_cols = global_col_indices[block_mask]
    valid_data = H_blocks[block_mask]
    
    H_core[valid_rows, valid_cols] = valid_data
    H_core[valid_cols, valid_rows] = valid_data
    
    return H_core