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
    Calculation of one-center two-electron integrals for s and p orbitals.
    Replaces the original 'sp_two_electron' function.
    
    This function contracts the radial part and hardcoded angular parts, 
    giving the 5 sp integrals. (Total 6 integrals, 1 is omited, 
    because it can be derived from other 2 integrals.)

    In the comments of this function, we provide the equations used in the 
    supportin information of 10.1002/qua.25799

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
        h_{pp} = \frac{1}{2} (G_{pp} - G_{p2}), i.e. Eq. (S21) is omitted, 
        because it can be calculated from other 2 integrals.
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
    # Eq. (S16)
    gss_calc = rsc(0, ns, es_safe, ns, es_safe, ns, es_safe, ns, es_safe, HARTREE2EV=hartree2ev)
    
    # GSP = <ss|pp> (k=0)
    # Eq. (S17)
    gsp_calc = rsc(0, ns, es_safe, ns, es_safe, ns, ep_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    
    # HSP = <sp|sp> (k=1)
    # Eq. (S18)
    hsp_raw = rsc(1, ns, es_safe, ns, ep_safe, ns, es_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    hsp_calc = hsp_raw / 3.0
    
    # R033 and R233 for p-p interactions
    r033 = rsc(0, ns, ep_safe, ns, ep_safe, ns, ep_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    r233 = rsc(2, ns, ep_safe, ns, ep_safe, ns, ep_safe, ns, ep_safe, HARTREE2EV=hartree2ev)
    
    # Construct GPP and GP2
    # Eq. (S19)
    gpp_calc = r033 + 0.16 * r233
    # Eq. (S20)
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
    Calculation of radial integrals for the MNDO/d model.
    Calculates 12 specific integral types for atoms with d-orbitals.
    Replaces the original 'scprm' function.

    This function gives all the temporary radial parts for eri1c2e including d orbitals.

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
    

def _init_eiscor_tables():
    """
    Initialize the isolated atom energy correction tables
    These arrays map Atomic Number (Z) to correction coefficients.
    """
    ir016 = np.zeros(101, dtype=np.int32)
    ir066 = np.zeros(101, dtype=np.int32)
    ir244 = np.zeros(101, dtype=np.int32)
    ir266 = np.zeros(101, dtype=np.int32)
    ir466 = np.zeros(101, dtype=np.int32)

    # 20..28: Sc..Cu
    ir016[20:29] = [ 2,  4,  6,  5, 10, 12, 14, 16, 10]
    ir066[20:29] = [ 0,  1,  3, 10, 10, 15, 21, 28, 45]
    ir244[20:29] = [ 1,  2,  3,  5,  5,  6,  7,  8,  5]
    ir266[20:29] = [ 0,  8, 15, 35, 35, 35, 43, 50, 70]
    ir466[20:29] = [ 0,  1,  8, 35, 35, 35, 36, 43, 70]

    # 38..46: Y..Ag
    ir016[38:47] = [ 2,  4,  4,  5, 10,  7,  8,  0, 10]
    ir066[38:47] = [ 0,  1,  6, 10, 10, 21, 28, 45, 45]
    ir244[38:47] = [ 1,  2,  4,  5,  5,  5,  5,  0,  5]
    ir266[38:47] = [ 0,  8, 21, 35, 35, 43, 50, 70, 70]
    ir466[38:47] = [ 0,  1, 21, 35, 35, 36, 43, 70, 70]

    # 57: La
    ir016[56] = 2
    ir066[56] = 0
    ir244[56] = 1
    ir266[56] = 0
    ir466[56] = 0

    # 70: Lu
    ir016[70] = 2
    ir066[70] = 0
    ir244[70] = 1
    ir266[70] = 0
    ir466[70] = 0

    # 72..80: Hf..Hg
    ir016[71:80] = [ 4,  6,  5, 10, 12, 14,  9, 10, 0]
    ir066[71:80] = [ 1,  3, 10, 10, 15, 21, 36, 45, 0]
    ir244[71:80] = [ 2,  3,  5,  5,  6,  7,  5,  5, 0]
    ir266[71:80] = [ 8, 15, 35, 35, 35, 43, 56, 70, 0]
    ir466[71:80] = [ 1,  8, 35, 35, 35, 36, 56, 70, 0]

    return (
        cp.asarray(ir016, dtype=cp.float64),
        cp.asarray(ir066, dtype=cp.float64),
        cp.asarray(ir244, dtype=cp.float64),
        cp.asarray(ir266, dtype=cp.float64),
        cp.asarray(ir466, dtype=cp.float64)
    )

_IR016, _IR066, _IR244, _IR266, _IR466 = _init_eiscor_tables()

def calc_repd_and_eiscor(
    atomic_numbers, f0sd_params, g2sd_params, dorbs, integrals_tuple):
    """
    Construction of the REPD matrix and isolated atom energy corrections.
    Replaces original 'inighd' and 'eiscor' functions.

    In the comments of this function, we provide the equations used in the 
    supportin information of 10.1002/qua.25799

    Args:
        atomic_numbers: (N,) CuPy array (int32) - True atomic numbers (Z) for lookup.
        f0sd_params:    (N,) CuPy array (float64) - Empirical F0sd (env.f0sd6).
        g2sd_params:    (N,) CuPy array (float64) - Empirical G2sd (env.g2sd6).
        dorbs:          (N,) CuPy array (bool) - Mask indicating if d orbitals exist.
        integrals_tuple (r016 ... r466):  a tuple of (N,) CuPy arrays - Output from calc_scprm.

    Returns:
        repd:       (52, N) CuPy array (float64) - The full d-orbital interaction matrix.
        eisol_corr: (N,) CuPy array (float64) - Correction term to be added to env.eisol.
        params_dict: dict - Contains the finalized empirical parameters to update 'env'.
    """
    r016, r036, r066, r155, r125, r244, r236, r266, r234, r246, r355, r466 = integrals_tuple
    n_atom = r016.shape[0]

    # TODO: future should use more accurate values
    # s3 = 1.7320508100147274      # sqrt(3)
    # s5 = 2.23606797749979        # sqrt(5)
    # s15 = 3.872983346207417      # sqrt(15)
    s3 = 1.7320508
    s5 = 2.23606797
    s15 = 3.87298334

    # TODO: In future, I should generate the confidential parameters, combined
    # TODO: with analytical values and parameters to create a big paremeter data.
    r016 = cp.where(f0sd_params > 0.001, f0sd_params, r016)
    r244 = cp.where(g2sd_params > 0.001, g2sd_params, r244)

    # atomic_numbers is 1-based
    Z = cp.ascontiguousarray(atomic_numbers-1, dtype=cp.int32)
    
    Z_safe = cp.clip(Z, 0, 100) 

    eisol_corr = (
          _IR016[Z_safe] * r016 
        + _IR066[Z_safe] * r066 
        - _IR244[Z_safe] * r244 / 5.0 
        - _IR266[Z_safe] * r266 / 49.0 
        - _IR466[Z_safe] * r466 / 49.0
    )
    
    eisol_corr = cp.where(dorbs, eisol_corr, 0.0)

    repd = cp.zeros((52, n_atom), dtype=cp.float64)
    
    repd[0] = r016                          # Eq. (S45)
    repd[1] = (2.0 / (3.0 * s5)) * r125     # Eq. (S69)
    repd[2] = (1.0 / s15) * r125            # Eq. (S70)
    repd[3] = (2.0 / (5.0 * s5)) * r234     # Eq. (S71)
    
    repd[4] = r036 + (4.0 / 35.0) * r236    # Eq. (S37)
    repd[5] = r036 + (2.0 / 35.0) * r236    # Eq. (S42)
    repd[6] = r036 - (4.0 / 35.0) * r236    # Eq. (S39)
    
    repd[7] = -(1.0 / (3.0 * s5)) * r125    # Eq. (S73)
    repd[8] = np.sqrt(3.0 / 125.0) * r234   # Eq. (S72)
    repd[9] = (s3 / 35.0) * r236            # Eq. (S38)
    repd[10]= (3.0 / 35.0) * r236           # Eq. (S40)
    repd[11]= -(1.0 / (5.0 * s5)) * r234    # Eq. (S66)
    
    repd[12]= r036 - (2.0 / 35.0) * r236    # Eq. (S26)
    repd[13]= -(2.0 * s3 / 35.0) * r236     # Eq. (S27)
    
    repd[14]= -repd[2]                      # Eq. (S67)
    repd[15]= -repd[10]                     # Eq. (S41)
    repd[16]= -repd[8]                      # Eq. (S68)
    repd[17]= -repd[13]                     # Eq. (S43)     
    
    repd[18]= (1.0 / 5.0) * r244            # Eq. (S44)
    repd[19]= (2.0 / (7.0 * s5)) * r246     # Eq. (S61)
    repd[20]= repd[19] / 2.0                # Eq. (S63)
    repd[21]= -repd[19]                     # Eq. (S62)
    
    repd[22]= (4.0 / 15.0) * r155 + (27.0 / 245.0) * r355           # Eq. (S29)
    repd[23]= (2.0 * s3 / 15.0) * r155 - (9.0 * s3 / 245.0) * r355  # Eq. (S32)
    repd[24]= (1.0 / 15.0) * r155 + (18.0 / 245.0) * r355           # Eq. (S22)
    repd[25]= -(s3 / 15.0) * r155 + (12.0 * s3 / 245.0) * r355      # Eq. (S35)
    repd[26]= -(s3 / 15.0) * r155 - (3.0 * s3 / 245.0) * r355       # Eq. (S23)  
    repd[27]= -repd[26]                                             # Eq. (S28)
    
    repd[28]= r066 + (4.0 / 49.0) * r266 + (4.0 / 49.0) * r466      # Eq. (S46)
    repd[29]= r066 + (2.0 / 49.0) * r266 - (24.0 / 441.0) * r466    # Eq. (S52)
    repd[30]= r066 - (4.0 / 49.0) * r266 + (6.0 / 441.0) * r466     # Eq. (S49)
    
    repd[31]= np.sqrt(3.0 / 245.0) * r246                           # Eq. (S64)
    repd[32]= (1.0 / 5.0) * r155 + (24.0 / 245.0) * r355            # Eq. (S31)
    repd[33]= (1.0 / 5.0) * r155 - (6.0 / 245.0) * r355             # Eq. (S33)
    repd[34]= (3.0 / 49.0) * r355                                   # Eq. (S30)
    
    repd[35]= (1.0 / 49.0) * r266 + (30.0 / 441.0) * r466               # Eq. (S48)
    repd[36]= (s3 / 49.0) * r266 - (5.0 * s3 / 441.0) * r466            # Eq. (S50)
    repd[37]= r066 - (2.0 / 49.0) * r266 - (4.0 / 441.0) * r466         # Eq. (S60)
    repd[38]= -(2.0 * s3 / 49.0) * r266 + (10.0 * s3 / 441.0) * r466    # Eq. (S53)
    
    repd[39]= -repd[31]                                                 # Eq. (S65)
    repd[40]= -repd[33]                                                 # Eq. (S36)
    repd[41]= -repd[34]                                                 # Eq. (S34)
    repd[42]= -repd[36]                                                 # Eq. (S51)
    
    repd[43]= (3.0 / 49.0) * r266 + (20.0 / 441.0) * r466               # Eq. (S56)
    repd[44]= -repd[38]                                                 # Eq. (S54)
    repd[45]= (1.0 / 5.0) * r155 - (3.0 / 35.0) * r355                  # Eq. (S24)
    repd[46]= -repd[45]                                                 # Eq. (S25)
    
    repd[47]= (4.0 / 49.0) * r266 + (15.0 / 441.0) * r466               # Eq. (S47)
    repd[48]= (3.0 / 49.0) * r266 - (5.0 / 147.0) * r466                # Eq. (S58)
    repd[49]= -repd[48]                                                 # Eq. (S59)
    
    repd[50]= r066 + (4.0 / 49.0) * r266 - (34.0 / 441.0) * r466        # Eq. (S57)
    repd[51]= (35.0 / 441.0) * r466                                     # Eq. (S55)       

    repd = cp.where(dorbs, repd, 0.0)

    params_dict = {
        'f0dd': cp.where(dorbs, r066, 0.0),
        'f2dd': cp.where(dorbs, r266, 0.0),
        'f4dd': cp.where(dorbs, r466, 0.0),
        'f0sd': cp.where(dorbs, r016, 0.0),
        'g2sd6':cp.where(dorbs, r244, 0.0),
        'f0pd': cp.where(dorbs, r036, 0.0),
        'f2pd': cp.where(dorbs, r236, 0.0),
        'g1pd': cp.where(dorbs, r155, 0.0),
        'g3pd': cp.where(dorbs, r355, 0.0)
    }

    return repd, eisol_corr, params_dict