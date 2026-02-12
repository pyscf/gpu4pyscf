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

import ctypes
import os
import numpy as np
import cupy as cp

_MAX_FAC = 30
_FACT_CPU = np.ones(_MAX_FAC, dtype=np.float64)
_FACT_CPU[1:] = np.cumprod(np.arange(1, _MAX_FAC, dtype=np.float64))
_FACT_GPU = cp.asarray(_FACT_CPU)

def _load_cuda_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'liberi_2c2e_kernel.so'
    
    lib_path = os.path.join(curr_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)
    
    lib.launch_multipole_eval_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p
    ]

    lib.launch_solve_poij_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_double
    ]

    return lib

_eri2c2e_MODULE = _load_cuda_library()

def multipole_eval(r, l1, l2, m, da, db, add):
    """
    Compute multipole interaction potential V(r) on GPU.
    All inputs must be CuPy arrays of shape (N,).
    
    Args:
        r: Distance (Bohr)
        l1, l2: Angular momentum (0=s, 1=p, 2=d)
        m: Magnetic quantum number
        da, db: Multipole lengths
        add: Klopman-Ohno squared term
        
    Returns:
        out: (N,) Interaction potential
    """

    r = cp.ascontiguousarray(r, dtype=cp.float64)
    l1 = cp.ascontiguousarray(l1, dtype=cp.int32)
    l2 = cp.ascontiguousarray(l2, dtype=cp.int32)
    m = cp.ascontiguousarray(m, dtype=cp.int32)
    da = cp.ascontiguousarray(da, dtype=cp.float64)
    db = cp.ascontiguousarray(db, dtype=cp.float64)
    add = cp.ascontiguousarray(add, dtype=cp.float64)

    n_pair = r.shape[0]
    out = cp.zeros(n_pair, dtype=cp.float64)
    _eri2c2e_MODULE.launch_multipole_eval_kernel_c(
        ctypes.c_int(n_pair),
        ctypes.c_void_p(r.data.ptr),
        ctypes.c_void_p(l1.data.ptr),
        ctypes.c_void_p(l2.data.ptr),
        ctypes.c_void_p(m.data.ptr),
        ctypes.c_void_p(da.data.ptr),
        ctypes.c_void_p(db.data.ptr),
        ctypes.c_void_p(add.data.ptr),
        ctypes.c_void_p(out.data.ptr)
    )

    return out


def a_function_ijl(z1, z2, n1, n2, l):
    """
    This function calculate the function defined in S3 of 10.1002/qua.25799
    Which will be used in determining the distance of the multipole interaction.
    
    **WARNING**
    The MOPAC code is as follows:
    double precision function aijl (z1, z2, n1, n2, l)
      double precision, intent(in) :: z1
      double precision, intent(in) :: z2
      integer, intent(in) :: n1
      integer, intent(in) :: n2
      integer, intent(in) :: l
      double precision :: zz
      zz = z1 + z2 + 1.d-20
      aijl = fx(n1+n2+l+1)/sqrt(fx(2*n1+1)*fx(2*n2+1))*(2*z1/zz)**n1*sqrt(2&
        *z1/zz)*(2*z2/zz)**n2*sqrt(2*z2/zz)*2**l/zz**l
      return
    end function aijl

    ** INCONSISTENT WITH THE PAPER **
    """

    zz = z1 + z2 + 1e-20
    
    idx1 = (n1 + n2 + l).astype(cp.int32)
    idx2 = (2 * n1).astype(cp.int32)
    idx3 = (2 * n2).astype(cp.int32)
    
    t1 = _FACT_GPU[idx1] / cp.sqrt(_FACT_GPU[idx2] * _FACT_GPU[idx3])
    t2 = cp.power(2.0 * z1 / zz, n1 + 0.5) * cp.power(2.0 * z2 / zz, n2 + 0.5)
    t3 = cp.power(2.0 / zz, l)
    
    return t1 * t2 * t3


def calc_aij_tensor(zs, zp, zd, ns, nd, dorbs, element_ids):
    """
    Compute the AIJ tensor (multipole interaction distances) for all atoms.
    
    Args:
        zs          : (N,) CuPy array (float64) - Slater exponent for s orbital
        zp          : (N,) CuPy array (float64) - Slater exponent for p orbital
        zd          : (N,) CuPy array (float64) - Slater exponent for d orbital
        ns          : (N,) CuPy array (int32)   - Principal quantum number for s (formerly 'iii')
        nd          : (N,) CuPy array (int32)   - Principal quantum number for d (formerly 'iiid')
        dorbs       : (N,) CuPy array (bool)    - Mask indicating if d orbitals exist
        element_ids : (N,) CuPy array (int32)   - **0-based element index** (H=0, He=1, Li=2...)
                                                  Corresponds to 'ni' in the original code.
             
    Returns:
        aij_tensor : (3, 3, N) CuPy array (float64)
                     Indices: [0=s, 1=p, 2=d]
                     Example: aij_tensor[0, 1, :] is the SP distance parameter.
    """
    n_atom = zs.shape[0]
    
    aij_tensor = cp.zeros((3, 3, n_atom), dtype=cp.float64)
    
    mask_heavy = element_ids >= 2
    mask_p = mask_heavy & (zp > 1e-4)
    mask_d = dorbs & mask_heavy

    # L=1 (Dipole-like) SP Interaction
    if cp.any(mask_p):
        val_sp = a_function_ijl(zs, zp, ns, ns, 1)
        val_sp = cp.where(mask_p, val_sp, 0.0)
        
        aij_tensor[0, 1, :] = val_sp
        aij_tensor[1, 0, :] = val_sp

    # L=2 (Quadrupole-like) PP Interaction
    if cp.any(mask_p):
        val_pp = a_function_ijl(zp, zp, ns, ns, 2)
        val_pp = cp.where(mask_p, val_pp, 0.0)
        
        aij_tensor[1, 1, :] = val_pp

    # D-Orbital Interactions
    if cp.any(mask_d):
        val_sd = a_function_ijl(zs, zd, ns, nd, 2)
        val_sd = cp.where(mask_d, val_sd, 0.0)
        aij_tensor[0, 2, :] = val_sd
        aij_tensor[2, 0, :] = val_sd
        
        val_pd = a_function_ijl(zp, zd, ns, nd, 1)
        val_pd = cp.where(mask_d, val_pd, 0.0)
        aij_tensor[1, 2, :] = val_pd
        aij_tensor[2, 1, :] = val_pd
        
        val_dd = a_function_ijl(zd, zd, nd, nd, 2)
        val_dd = cp.where(mask_d, val_dd, 0.0)
        aij_tensor[2, 2, :] = val_dd
    
    return aij_tensor

# def solve_poij(l_vec, d_vec, fg_vec, HARTREE2EV=27.211386245988):
#     """
#     Vectorized Golden Section Search to find 'rho' for all atoms simultaneously.
#     Handles mixed L values (0, 1, 2) in a single batch.
    
#     Args:
#         mol: Molecule object (containing constants like HARTREE2EV)
#         l_vec (cp.ndarray): Multipole order array (N_atom,). Values must be 0, 1, or 2.
#         d_vec (cp.ndarray): Distance array (N_atom,).
#         fg_vec (cp.ndarray): Target integral array (N_atom,).
        
#     Returns:
#         rho_vec (cp.ndarray): Optimized rho values (N_atom,)
#     """
    
#     d_vec = d_vec.astype(cp.float64)
#     fg_vec = fg_vec.astype(cp.float64)
    
#     safe_fg = cp.where(cp.abs(fg_vec) < 1e-9, 1.0, fg_vec)
#     rho_analytical = 0.5 * HARTREE2EV / safe_fg
    
#     rho_analytical = cp.where(cp.abs(fg_vec) < 1e-9, 1e6, rho_analytical)

#     dsq = d_vec * d_vec
#     ev4 = HARTREE2EV / 4.0
#     ev8 = HARTREE2EV / 8.0
    
#     def evaluate_obj(y):
#         val_l1 = ev4 * (1.0/y - 1.0/cp.sqrt(y*y + dsq))
#         val_l2 = ev8 * (1.0/y - 2.0/cp.sqrt(y*y + 0.5*dsq) + 1.0/cp.sqrt(y*y + dsq))
        
#         val_calc = cp.where(l_vec == 1, val_l1, val_l2)
        
#         diff = val_calc - fg_vec
#         return diff * diff

#     n_atom = d_vec.shape[0]
#     a = cp.full(n_atom, 0.1, dtype=cp.float64)
#     b = cp.full(n_atom, 5.0, dtype=cp.float64)
    
#     invphi = (np.sqrt(5) - 1) / 2  # ~0.618
#     invphi2 = (3 - np.sqrt(5)) / 2 # ~0.382
    
#     c = a + invphi2 * (b - a)
#     d = a + invphi * (b - a)
    
#     fc = evaluate_obj(c)
#     fd = evaluate_obj(d)
    
#     for _ in range(40):
#         mask = fc < fd  # True: min in [a, d], False: min in [c, b]
        
#         b = cp.where(mask, d, b)
#         a = cp.where(~mask, c, a)
        
#         # TODO: Only one new point is needed per atom, but we compute vectors for simplicity
#         c_fresh = a + invphi2 * (b - a)
#         d_fresh = a + invphi * (b - a)
        
#         # If mask (left):  we need to eval c_fresh. (d becomes old c)
#         # If ~mask (right): we need to eval d_fresh. (c becomes old d)
#         x_eval = cp.where(mask, c_fresh, d_fresh)
#         f_new = evaluate_obj(x_eval)
        
#         next_c = cp.where(mask, c_fresh, d)
#         next_d = cp.where(mask, c, d_fresh)
        
#         next_fc = cp.where(mask, f_new, fd)
#         next_fd = cp.where(mask, fc, f_new)
        
#         c = next_c
#         d = next_d
#         fc = next_fc
#         fd = next_fd

#     # This is different from the original algorithm where the a or b is used
#     rho_numerical = (a + b) / 2.0
    
#     final_rho = cp.where(l_vec == 0, rho_analytical, rho_numerical)
    
#     return final_rho


def solve_poij(l_vec, d_vec, fg_vec, HARTREE2EV=27.211386245988):
    """
    Calls the CUDA kernel 'solve_poij_kernel' which strictly mimics the original Python code.
    This function is less sophisticated than the upper code, and only for benchmark with yunze.
    It should be changed to the upper code.
    """
    n_atom = d_vec.shape[0]
    rho_out = cp.zeros(n_atom, dtype=cp.float64)
    
    l_vec = cp.ascontiguousarray(l_vec, dtype=cp.int32)
    d_vec = cp.ascontiguousarray(d_vec, dtype=cp.float64)
    fg_vec = cp.ascontiguousarray(fg_vec, dtype=cp.float64)

    _eri2c2e_MODULE.launch_solve_poij_kernel_c(
        ctypes.c_int(n_atom),
        ctypes.c_void_p(l_vec.data.ptr),
        ctypes.c_void_p(d_vec.data.ptr),
        ctypes.c_void_p(fg_vec.data.ptr),
        ctypes.c_void_p(rho_out.data.ptr),
        ctypes.c_double(HARTREE2EV)
    )
    
    return rho_out