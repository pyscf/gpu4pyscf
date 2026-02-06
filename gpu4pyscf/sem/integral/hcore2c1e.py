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


def bfn(x):
    """
    Compute auxiliary function B_n(x) for n=0..12 on GPU.
    
    This function calculates the integrals appearing in semi-empirical overlaps.
    It splits the input domain into three regions for numerical stability:
    1. Tiny (|x| <= 1e-6): Analytic limit.
    2. Small (1e-6 < |x| <= 3.0): Taylor series expansion (via Matrix Multiplication).
    3. Large (|x| > 3.0): Recursive relation.

    Args:
        x (cp.ndarray): Input array of values. Shape (N,).
    
    Returns:
        cp.ndarray: Result array B_n(x). Shape (N, 13).
    """
    original_shape = x.shape
    x_flat = x.ravel()
    n_data = x_flat.size
    
    bf = cp.zeros((n_data, 13), dtype=np.float64)
    absx = cp.abs(x_flat)
    
    mask_tiny  = absx <= 1.0e-6
    mask_small1 = (absx > 1.0e-6) & (absx <= 0.5)
    mask_small2 = (absx > 0.5) & (absx <= 1.0)
    mask_small3 = (absx > 1.0) & (absx <= 2.0)
    mask_small4 = (absx > 2.0) & (absx <= 3.0)
    mask_large = absx > 3.0
    
    # Limit: B_i(0) = 2/(i+1) if i is even, else 0
    if cp.any(mask_tiny):
        indices = cp.arange(13, dtype=np.float64)
        tiny_vals = (2.0 * ((indices + 1) % 2)) / (indices + 1.0)
        bf[mask_tiny, :] = tiny_vals[None, :]

    # B_i(x) = Sum_m [ (-x)^m * C_{m,i} ]
    if cp.any(mask_small1):
        x_s = x_flat[mask_small1]
        norder_cut_off = 6 + 1
        m_range = cp.arange(norder_cut_off, dtype=np.float64)
        pow_minus_x = cp.power(-x_s[:, None], m_range[None, :])
        bf[mask_small1, :] = cp.dot(pow_minus_x, TAYLOR_COEFFS_GPU[:norder_cut_off, :])

    if cp.any(mask_small2):
        x_s = x_flat[mask_small2]
        norder_cut_off = 7 + 1
        m_range = cp.arange(norder_cut_off, dtype=np.float64)
        pow_minus_x = cp.power(-x_s[:, None], m_range[None, :])
        bf[mask_small2, :] = cp.dot(pow_minus_x, TAYLOR_COEFFS_GPU[:norder_cut_off, :])

    if cp.any(mask_small3):
        x_s = x_flat[mask_small3]
        norder_cut_off = 12 + 1
        m_range = cp.arange(norder_cut_off, dtype=np.float64)
        pow_minus_x = cp.power(-x_s[:, None], m_range[None, :])
        bf[mask_small3, :] = cp.dot(pow_minus_x, TAYLOR_COEFFS_GPU[:norder_cut_off, :])

    if cp.any(mask_small4):
        x_s = x_flat[mask_small4]
        norder_cut_off = 15 + 1
        m_range = cp.arange(norder_cut_off, dtype=np.float64)
        pow_minus_x = cp.power(-x_s[:, None], m_range[None, :])
        bf[mask_small4, :] = cp.dot(pow_minus_x, TAYLOR_COEFFS_GPU[:norder_cut_off, :])

    # Recursion: B_i = (i * B_{i-1} + (-1)^i * e^x - e^{-x}) / x
    if cp.any(mask_large):
        x_l = x_flat[mask_large]
        inv_x = 1.0 / x_l
        expx = cp.exp(x_l)
        expmx = 1.0 / expx  # exp(-x)
        
        val_curr = (expx - expmx) * inv_x
        bf[mask_large, 0] = val_curr
        
        for i in range(1, 13):
            # Term: (-1)^i * e^x - e^{-x}
            if i % 2 == 1:
                term = -expx - expmx
            else:
                term = expx - expmx
            
            val_next = (i * val_curr + term) * inv_x
            bf[mask_large, i] = val_next
            val_curr = val_next

    if x.ndim != 1:
        return bf.reshape(original_shape + (13,))
        
    return bf


def afn(p):
    n_data = p.size
    af = cp.zeros((n_data, 20), dtype=np.float64)
    p_safe = p + 1e-16
    inv_p = 1.0 / p_safe
    term0 = inv_p * cp.exp(-p)
    af[:, 0] = term0
    for n in range(1, 20):
        af[:, n] = (n * inv_p * af[:, n-1]) + term0
    return af


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
    return lib

_SS_MODULE = _load_cuda_library()

def ovlp_in_2c1e(na, nb, la, lb, m, ua, ub, r):
    """
    Compute Two-Center Overlap Integrals (SS) on GPU.
    
    Implementation Strategy:
    1. Outer loops (i, j): Handled in Python (since they only iterate 0 and 2).
    2. Inner loops (k1..k6): Offloaded to optimized CUDA C++ Kernel.
    
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
    
    total_val = cp.zeros_like(r)
    
    # i_val, j_val can only be 0 or 2.
    for i_val in [0, 2]:

        mask_i = i_val <= lam1
        if not cp.any(mask_i): 
            continue
        
        ia = na + i_val - la
        ic = la - i_val - m
        aff_a = AFF_GPU[la, m, i_val]
        
        for j_val in [0, 2]:

            mask_j = (j_val <= lbm1) & mask_i
            if not cp.any(mask_j): 
                continue

            ib = nb + j_val - lb
            id_ = lb - j_val - m
            iab = ia + ib
            aff_b = AFF_GPU[lb, m, j_val]
            
            pre_factor = aff_a * aff_b * mask_j
            
            # We filter inputs using mask_j to ensure we don't compute garbage for inactive pairs.
            # For inactive threads (where mask_j is False), we set 'ia' to -1.
            # The CUDA kernel checks: if (ia < 0) return 0.0;
            
            ia_in = cp.where(mask_j, ia, -1).astype(cp.int32)
            ib_in = cp.where(mask_j, ib, 0).astype(cp.int32)
            ic_in = cp.where(mask_j, ic, 0).astype(cp.int32)
            id_in = cp.where(mask_j, id_, 0).astype(cp.int32)
            m_in  = cp.where(mask_j, m, 0).astype(cp.int32)
            iab_in = cp.where(mask_j, iab, 0).astype(cp.int32)
            
            kernel_out = cp.zeros_like(r)
            
            _SS_MODULE.launch_ss_kernel_c(
                ctypes.c_int(n_pairs),
                ctypes.c_void_p(ia_in.data.ptr),
                ctypes.c_void_p(ib_in.data.ptr),
                ctypes.c_void_p(ic_in.data.ptr),
                ctypes.c_void_p(id_in.data.ptr),
                ctypes.c_void_p(m_in.data.ptr),
                ctypes.c_void_p(iab_in.data.ptr),
                ctypes.c_void_p(af.data.ptr),
                ctypes.c_void_p(bf.data.ptr),
                ctypes.c_void_p(BINOMIALS_GPU_FLAT.data.ptr),
                ctypes.c_void_p(kernel_out.data.ptr)
            )
            
            total_val += kernel_out * pre_factor
    
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
def rotation_transform(S_local, C_tensor):
    """
    Assembly of global 9x9 overlap matrix.
    """
    n_pairs = S_local.shape[0]
    di = cp.zeros((n_pairs, 9, 9), dtype=cp.float64)
    
    c1 = C_tensor[..., 0] # delta
    c2 = C_tensor[..., 1] # pi
    c3 = C_tensor[..., 2] # sigma
    c4 = C_tensor[..., 3] # pi
    c5 = C_tensor[..., 4] # delta
    
    # (N, 3, 3)
    s_sig = S_local[..., 0]
    s_pi  = S_local[..., 1]
    s_del = S_local[..., 2]
    
    # Define the IVAL mapping as a small lookup (keeping it simple logic-wise)
    # Structure: ival[shell][k_index] -> AO_index (0-based here for Python)
    # -1 indicates invalid
    ival = [
        [0, 0, 0, 0, -1],
        [-1, 2, 3, 1, -1],
        [8, 7, 6, 5, 4]
    ]
    
    for i in range(3): # Shell A
        # i=0 (s): k in [2] (val=1 in ival table above at idx 2) -> Range 2..3
        # i=1 (p): k in [1, 2, 3] -> Range 1..4
        # i=2 (d): k in [0, 1, 2, 3, 4] -> Range 0..5
        k_start = 2 - i
        k_end = 3 + i
        
        for j in range(3): # Shell B
            l_start = 2 - j
            l_end = 3 + j
            
            # Phase factors
            # aa = -1.0 if (j == 1) else 1.0
            # bb = -1.0 if (j == 2) else (1.0 if (j != 1) else 1.0)
            aa = -1.0 if j == 1 else 1.0
            bb = -1.0 if j == 2 else 1.0
            
            val_sigma = s_sig[:, i, j]
            val_pi  = s_pi[:, i, j]
            val_delta = s_del[:, i, j]
            
            for k in range(k_start, k_end): # global index for shell A
                idx_a = ival[i][k]
                if idx_a < 0: 
                    continue
                
                for l in range(l_start, l_end): # global index for shell B
                    idx_b = ival[j][l]
                    if idx_b < 0: 
                        continue
                    
                    term = val_sigma * (c3[:, i, k] * c3[:, j, l]) * aa
                    
                    if i > 0 and j > 0:
                        term += val_pi * (c4[:, i, k] * c4[:, j, l] + 
                                          c2[:, i, k] * c2[:, j, l]) * bb
                        
                        if i > 1 and j > 1:
                            term += val_delta * (c5[:, i, k] * c5[:, j, l] + 
                                               c1[:, i, k] * c1[:, j, l])
                    
                    di[:, idx_a, idx_b] += term
                    
    return di


