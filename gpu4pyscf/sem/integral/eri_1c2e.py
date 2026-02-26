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


_MAX_FAC = 30
_FACT_CPU = np.ones(_MAX_FAC, dtype=np.float64)
_FACT_CPU[1:] = np.cumprod(np.arange(1, _MAX_FAC, dtype=np.float64))
_FACT_GPU = cp.asarray(_FACT_CPU)


_MAX_GRID = 30
_n_grid = np.arange(_MAX_GRID).reshape(-1, 1)
_k_grid = np.arange(_MAX_GRID).reshape(1, -1)
_BINOMIALS_CPU = comb(_n_grid, _k_grid)
BINOMIALS_GPU = cp.asarray(_BINOMIALS_CPU)
BINOMIALS_GPU_FLAT = cp.asarray(_BINOMIALS_CPU.ravel(), dtype=np.float64)


def _load_cuda_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = 'liberi_1c2e_kernel.so'
    
    lib_path = os.path.join(curr_dir, lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")
    
    lib = ctypes.CDLL(lib_path)

    lib.launch_rsc_kernel_c.argtypes = [
        ctypes.c_int, ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p
    ]

    return lib


_eri1c2e_MODULE = _load_cuda_library()


def rsc(k, na, ea, nb, eb, nc, ec, nd, ed, HARTREE2EV=27.211386245988):
    """
    Compute Reduced Single Center (RSC) one-center two-electron integrals.
    
    This function calculates one-center two-electron integrals for semi-empirical methods
    based on Slater-type orbitals and Klopman-Ohno approximation. It uses CUDA kernels
    for GPU-accelerated computation.
    
    Args:
        k (int or cupy.ndarray): Angular momentum quantum number or array of angular momenta
        na, nb, nc, nd (cupy.ndarray): Quantum numbers for the four orbitals
        ea, eb, ec, ed (cupy.ndarray): Orbital exponents for the four orbitals
        HARTREE2EV (float): Conversion factor from Hartree to electron volts, default 27.211386245988
        
    Returns:
        cupy.ndarray: Array of computed one-center two-electron integral values, shape (n_size,)
        
    Note:
        - All input arrays must have the same length n_size
        - Scalar k will be converted to an array of length n_size internally
        - Uses CUDA kernel for parallel computation, suitable for large-scale integral calculations
        - Results are in Hartree units, can be converted to eV using HARTREE2EV parameter
    """
    if isinstance(k, int):
        n_size = na.shape[0]
        k_arr = cp.full(n_size, k, dtype=cp.int32)
    else:
        k_arr = cp.ascontiguousarray(k, dtype=cp.int32)
        n_size = k_arr.shape[0]

    na = cp.ascontiguousarray(na, dtype=cp.int32)
    nb = cp.ascontiguousarray(nb, dtype=cp.int32)
    nc = cp.ascontiguousarray(nc, dtype=cp.int32)
    nd = cp.ascontiguousarray(nd, dtype=cp.int32)
    
    ea = cp.ascontiguousarray(ea, dtype=cp.float64)
    eb = cp.ascontiguousarray(eb, dtype=cp.float64)
    ec = cp.ascontiguousarray(ec, dtype=cp.float64)
    ed = cp.ascontiguousarray(ed, dtype=cp.float64)
    
    out = cp.zeros(n_size, dtype=cp.float64)
    
    _eri1c2e_MODULE.launch_rsc_kernel_c(
        ctypes.c_int(n_size),
        ctypes.c_double(HARTREE2EV),
        ctypes.c_void_p(k_arr.data.ptr),
        ctypes.c_void_p(na.data.ptr), ctypes.c_void_p(ea.data.ptr),
        ctypes.c_void_p(nb.data.ptr), ctypes.c_void_p(eb.data.ptr),
        ctypes.c_void_p(nc.data.ptr), ctypes.c_void_p(ec.data.ptr),
        ctypes.c_void_p(nd.data.ptr), ctypes.c_void_p(ed.data.ptr),
        ctypes.c_void_p(_FACT_GPU.data.ptr),
        ctypes.c_void_p(BINOMIALS_GPU_FLAT.data.ptr),
        ctypes.c_void_p(out.data.ptr)
    )
    
    return out


# TODO: This function should be optimized.
# TODO: I think, all this integrals can be parameterized, just leave an interface.
def calc_sp_two_electron(env_params, ns, es, ep, main_group, hartree2ev=27.211386245988):
    """
    Vectorized calculation of one-center two-electron integrals for s and p orbitals.
    Replaces the original 'sp_two_electron' function.

    Args:
        env_params: A tuple of 5 CuPy arrays (gss, gsp, hsp, gpp, gp2) containing existing parameters.
        ns: (N,) CuPy array (int32) - Principal quantum number for s/p orbitals (env.iii).
        es: (N,) CuPy array (float64) - s-orbital Slater exponent (env.zsn6).
        ep: (N,) CuPy array (float64) - p-orbital Slater exponent (env.zpn6).
        main_group: (N,) CuPy array (bool) - True if the element is main-group.
        hartree2ev: float - Unit conversion factor.

    Returns:
        A tuple of 5 CuPy arrays (float64), each of shape (N,):
        (gss, gsp, hsp, gpp, gp2) with theoretical values overwriting empirical ones where appropriate.
    """
    gss_in, gsp_in, hsp_in, gpp_in, gp2_in = env_params

    ns = cp.ascontiguousarray(ns, dtype=cp.int32)
    es = cp.ascontiguousarray(es, dtype=cp.float64)
    ep = cp.ascontiguousarray(ep, dtype=cp.float64)
    main_group = cp.ascontiguousarray(main_group, dtype=cp.bool_)
    
    mask_valid = (es >= 1e-4) & (ep >= 1e-4) & (~main_group)
    
    es_safe = cp.where(es < 1e-4, 1.0, es)
    ep_safe = cp.where(ep < 1e-4, 1.0, ep)
    
    # GSS = <ss|ss> (k=0)
    gss_calc = rsc(0, ns, es_safe, ns, es_safe, ns, es_safe, ns, es_safe, HARTREE2EV=hartree2ev)
    
    # GSP = <ss|pp> (k=0)
    gsp_calc = rsc(0, ns, es_safe, ns, es_safe, ns, ep_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    
    # HSP = <sp|sp> (k=1)
    hsp_raw = rsc(1, ns, es_safe, ns, ep_safe, ns, es_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    hsp_calc = hsp_raw / 3.0
    
    # R033 and R233 for p-p interactions
    r033 = rsc(0, ns, ep_safe, ns, ep_safe, ns, ep_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    r233 = rsc(2, ns, ep_safe, ns, ep_safe, ns, ep_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    
    # Construct GPP and GP2
    gpp_calc = r033 + 0.16 * r233
    gp2_calc = r033 - 0.08 * r233

    gss_out = cp.where(mask_valid, gss_calc, gss_in)
    gsp_out = cp.where(mask_valid, gsp_calc, gsp_in)
    hsp_out = cp.where(mask_valid, hsp_calc, hsp_in)
    gpp_out = cp.where(mask_valid, gpp_calc, gpp_in)
    gp2_out = cp.where(mask_valid, gp2_calc, gp2_in)

    return gss_out, gsp_out, hsp_out, gpp_out, gp2_out


# TODO: This function should be optimized.
# TODO: I think, all this integrals can be parameterized, just leave an interface.
def calc_scprm(ns, nd, es, ep, ed, dorbs, hartree2ev=27.211386245988):
    """
    Vectorized calculation of radial integrals for the MNDO/d model.
    Calculates 12 specific integral types for atoms with d-orbitals.
    Replaces the original 'scprm' function.

    Args:
        ns: (N,) CuPy array (int32) - Principal quantum number for s/p orbitals (env.iii).
        nd: (N,) CuPy array (int32) - Principal quantum number for d orbitals (env.iiid).
        es: (N,) CuPy array (float64) - s-orbital Slater exponent (env.zsn6).
        ep: (N,) CuPy array (float64) - p-orbital Slater exponent (env.zpn6).
        ed: (N,) CuPy array (float64) - d-orbital Slater exponent (env.zdn6).
        dorbs: (N,) CuPy array (bool) - Mask indicating if d orbitals exist.
        hartree2ev: float - Unit conversion factor.

    Returns:
        A tuple of 12 CuPy arrays (float64), each of shape (N,):
        (r016, r036, r066, r155, r125, r244, r236, r266, r234, r246, r355, r466)
    """
    ns = cp.ascontiguousarray(ns, dtype=cp.int32)
    nd = cp.ascontiguousarray(nd, dtype=cp.int32)
    es = cp.ascontiguousarray(es, dtype=cp.float64)
    ep = cp.ascontiguousarray(ep, dtype=cp.float64)
    ed = cp.ascontiguousarray(ed, dtype=cp.float64)
    dorbs = cp.ascontiguousarray(dorbs, dtype=cp.bool_)

    es_safe = cp.where(es < 1e-4, 1.0, es)
    ep_safe = cp.where(ep < 1e-4, 1.0, ep)
    ed_safe = cp.where(ed < 1e-4, 1.0, ed)
    
    ns_safe = cp.maximum(ns, 1)
    nd_safe = cp.maximum(nd, 1)
    
    # R0ssdd
    r016 = rsc(0, ns_safe, es_safe, ns_safe, es_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R0ppdd
    r036 = rsc(0, ns_safe, ep_safe, ns_safe, ep_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R0dddd = F0dd
    r066 = rsc(0, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    
    # R1pdpd = G1pd
    r155 = rsc(1, ns_safe, ep_safe, nd_safe, ed_safe, ns_safe, ep_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R1sppd
    r125 = rsc(1, ns_safe, es_safe, ns_safe, ep_safe, ns_safe, ep_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    
    # R2sdsd
    r244 = rsc(2, ns_safe, es_safe, nd_safe, ed_safe, ns_safe, es_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R2ppdd
    r236 = rsc(2, ns_safe, ep_safe, ns_safe, ep_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R2dd = F2dd
    r266 = rsc(2, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R2ppsd
    r234 = rsc(2, ns_safe, ep_safe, ns_safe, ep_safe, ns_safe, es_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R2sddd
    r246 = rsc(2, ns_safe, es_safe, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R3pdpd = G3pd
    r355 = rsc(3, ns_safe, ep_safe, nd_safe, ed_safe, ns_safe, ep_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)
    # R4dddd = F4dd
    r466 = rsc(4, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, nd_safe, ed_safe, HARTREE2EV=hartree2ev)

    r016 = cp.where(dorbs, r016, 0.0)
    r036 = cp.where(dorbs, r036, 0.0)
    r066 = cp.where(dorbs, r066, 0.0)
    r155 = cp.where(dorbs, r155, 0.0)
    r125 = cp.where(dorbs, r125, 0.0)
    r244 = cp.where(dorbs, r244, 0.0)
    r236 = cp.where(dorbs, r236, 0.0)
    r266 = cp.where(dorbs, r266, 0.0)
    r234 = cp.where(dorbs, r234, 0.0)
    r246 = cp.where(dorbs, r246, 0.0)
    r355 = cp.where(dorbs, r355, 0.0)
    r466 = cp.where(dorbs, r466, 0.0)

    return r016, r036, r066, r155, r125, r244, r236, r266, r234, r246, r355, r466
    
