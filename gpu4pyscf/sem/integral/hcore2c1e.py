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
from scipy.special import comb

_MAX_N = 17
_FACTORIALS_CPU = np.ones(_MAX_N + 1, dtype=np.float64)
_FACTORIALS_CPU[1:] = np.cumprod(np.arange(1, _MAX_N + 1, dtype=np.float64))
FACTORIALS_GPU = cp.asarray(_FACTORIALS_CPU)

_n_grid = np.arange(21).reshape(-1, 1)
_k_grid = np.arange(21).reshape(1, -1)
_BINOMIALS_CPU = comb(_n_grid, _k_grid)
BINOMIALS_GPU = cp.asarray(_BINOMIALS_CPU)

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
    # mask_small = (absx > 1.0e-6) & (absx <= 3.0)
    mask_large = absx > 3.0
    
    # Limit: B_i(0) = 2/(i+1) if i is even, else 0
    if cp.any(mask_tiny):
        indices = cp.arange(13, dtype=np.float64)
        tiny_vals = (2.0 * ((indices + 1) % 2)) / (indices + 1.0)
        bf[mask_tiny, :] = tiny_vals[None, :]

    # B_i(x) = Sum_m [ (-x)^m * C_{m,i} ]
    # if cp.any(mask_small):
    #     x_s = x_flat[mask_small]
    #     norder_cut_off = 16
    #     m_range = cp.arange(norder_cut_off, dtype=np.float64)
    #     pow_minus_x = cp.power(-x_s[:, None], m_range[None, :])
    #     bf[mask_small, :] = cp.dot(pow_minus_x, TAYLOR_COEFFS_GPU[:norder_cut_off, :])

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



