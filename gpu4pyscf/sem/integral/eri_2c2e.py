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
from gpu4pyscf.sem.gto.params import build_gpu_task_instructions

_MAX_FAC = 30
_FACT_CPU = np.ones(_MAX_FAC, dtype=np.float64)
_FACT_CPU[1:] = np.cumprod(np.arange(1, _MAX_FAC, dtype=np.float64))
_FACT_GPU = cp.asarray(_FACT_CPU)

TASK_ACTION, TASK_TARGET, TASK_IJ, TASK_KL, TASK_LI, TASK_LJ, TASK_LK, TASK_LL, IND2 = build_gpu_task_instructions()
TASK_ACTION_GPU = cp.asarray(TASK_ACTION)
TASK_TARGET_GPU = cp.asarray(TASK_TARGET)
TASK_IJ_GPU = cp.asarray(TASK_IJ)
TASK_KL_GPU = cp.asarray(TASK_KL)
TASK_LI_GPU = cp.asarray(TASK_LI)
TASK_LJ_GPU = cp.asarray(TASK_LJ)
TASK_LK_GPU = cp.asarray(TASK_LK)
TASK_LL_GPU = cp.asarray(TASK_LL)
IND2 = cp.asarray(IND2)

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

    lib.launch_test_rijkl_kernel_c.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p
    ]

    lib.launch_calc_local_rep_core_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    lib.launch_global_transform_kernel_c.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
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
    t3 = cp.power(2.0 / zz, l)  # ! this is different from 10.1002/qua.25799, but same as the codes.
    # ! the 2^l is multiplied in purpose, because in other places, this will be divided.
    # ! For dipole, the final D will be saved. But for quadrupole, the sqrt2 D is saved.
    # TODO: This is very ugly and not easy to read, in the future the corrsponding places should be removed.
    
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


# This function only for debug purpose
def test_rijkl(ni, nj, ij, kl, li, lj, lk, ll, ic, r, 
                   po_tensor, ddp_tensor, core_rho, ch):
    """
    Vectorized evaluation of the rijkl integration using CUDA.
    Used strictly to verify the correctness of the tensor-based parameter fetching.
    
    Args:
        ni, nj    : (N_tasks,) 1D CuPy array of int32. Atom indices (0-based).
        ij, kl    : (N_tasks,) 1D CuPy array of int32. Orbital combination indices.
        li, lj    : (N_tasks,) 1D CuPy array of int32. Angular momentum.
        lk, ll    : (N_tasks,) 1D CuPy array of int32. Angular momentum.
        ic        : (N_tasks,) 1D CuPy array of int32. Core flag. 
                     1=left is core, 2=right is core, 0=normal
        r         : (N_tasks,) 1D CuPy array of float64. Interatomic distance in Bohr.
        po_tensor : (3, 3, 3, N_atoms) CuPy array. 
        ddp_tensor: (3, 3, N_atoms) CuPy array.
        core_rho  : (N_atoms,) CuPy array.
        ch        : (45, 3, 5) CuPy array. Angular factors.
        
    Returns:
        out_val   : (N_tasks,) 1D CuPy array of float64. Computed integrals.
    """
    n_tasks = ni.shape[0]
    n_atom = po_tensor.shape[3]
    
    # Ensure inputs are contiguous GPU arrays with correct types
    ni = cp.ascontiguousarray(ni, dtype=cp.int32)
    nj = cp.ascontiguousarray(nj, dtype=cp.int32)
    ij = cp.ascontiguousarray(ij, dtype=cp.int32)
    kl = cp.ascontiguousarray(kl, dtype=cp.int32)
    li = cp.ascontiguousarray(li, dtype=cp.int32)
    lj = cp.ascontiguousarray(lj, dtype=cp.int32)
    lk = cp.ascontiguousarray(lk, dtype=cp.int32)
    ll = cp.ascontiguousarray(ll, dtype=cp.int32)
    ic = cp.ascontiguousarray(ic, dtype=cp.int32)
    r  = cp.ascontiguousarray(r, dtype=cp.float64)
    
    # Ensure tensors are contiguous
    po_tensor  = cp.ascontiguousarray(po_tensor, dtype=cp.float64)
    ddp_tensor = cp.ascontiguousarray(ddp_tensor, dtype=cp.float64)
    core_rho   = cp.ascontiguousarray(core_rho, dtype=cp.float64)
    ch         = cp.ascontiguousarray(ch, dtype=cp.float64)
    
    out_val = cp.zeros(n_tasks, dtype=cp.float64)
    
    _eri2c2e_MODULE.launch_test_rijkl_kernel_c(
        n_tasks, n_atom,
        ctypes.c_void_p(ni.data.ptr), ctypes.c_void_p(nj.data.ptr),
        ctypes.c_void_p(ij.data.ptr), ctypes.c_void_p(kl.data.ptr),
        ctypes.c_void_p(li.data.ptr), ctypes.c_void_p(lj.data.ptr),
        ctypes.c_void_p(lk.data.ptr), ctypes.c_void_p(ll.data.ptr),
        ctypes.c_void_p(ic.data.ptr), ctypes.c_void_p(r.data.ptr),
        ctypes.c_void_p(po_tensor.data.ptr),
        ctypes.c_void_p(ddp_tensor.data.ptr),
        ctypes.c_void_p(core_rho.data.ptr),
        ctypes.c_void_p(ch.data.ptr),
        ctypes.c_void_p(out_val.data.ptr)
    )
    
    return out_val


# TODO: This needs to be simplified.
# TODO: The output of this funcions should be parameterized.
def calc_multipole_params(
    aij_tensor, 
    gss, hsp, gpp, gp2, repd, 
    dorbs, element_ids, pocord,
    natorb, main_group,
    am, ad, aq, dd, qq
):
    """
    Original ddpo and inid
    Calculate multipole distances (D) and Klopman-Ohno additive terms (Rho).
    Includes logic for both Transition Metals (MNDO/d) and Main Group elements.
    
    Args:
        aij_tensor  : (3, 3, N) CuPy array - From calc_aij_tensor.
        gss, hsp    : (N,) CuPy arrays - One-center 2-electron integrals.
        gpp, gp2    : (N,) CuPy arrays - One-center 2-electron integrals.
        repd        : (52, N) CuPy array - Monatomic d-orbital interactions.
        dorbs       : (N,) CuPy array (bool) - D-orbital existence mask.
        element_ids : (N,) CuPy array (int32) - 0-based atomic numbers (H=0, He=1...).
        pocord      : (N,) CuPy array - Core interaction parameters.
        natorb      : (N,) CuPy array (int32) - Number of atomic orbitals.
        main_group  : (N,) CuPy array (bool)  - Is main group element?
        am, ad, aq  : (N,) CuPy arrays (float64) - Scaling parameters (Monopole/Dipole/Quad).
        dd, qq      : (N,) CuPy arrays (float64) - Distance parameters.

    Returns:
        po_tensor   : (3, 3, 3, N) CuPy array (float64). 
                      Indices: [shell_i, shell_j, L, atom].
                      (shell: 0=s, 1=p, 2=d; L: 0=Monopole, 1=Dipole, 2=Quadrupole)
                      the mapping can be seen in the following comments.
        ddp_tensor  : (3, 3, N) CuPy array (float64).
                      Indices: [shell_i, shell_j, atom].
                      the mapping can be seen in the following comments.
        core_rho    : (N,) CuPy array (float64). Additive terms for core. (original po[8])
    """
    n_atom = aij_tensor.shape[2]
    
    po_tensor = cp.zeros((3, 3, 3, n_atom), dtype=cp.float64)
    ddp_tensor = cp.zeros((3, 3, n_atom), dtype=cp.float64)
    
    # Masks
    mask_heavy = element_ids >= 2
    mask_d = dorbs & mask_heavy
    
    # Helper arrays
    l0 = cp.zeros(n_atom, dtype=cp.int32)
    l1 = cp.ones(n_atom, dtype=cp.int32)
    l2 = cp.full(n_atom, 2, dtype=cp.int32)
    d_ones = cp.ones(n_atom, dtype=cp.float64)
    
    # --- SS (L=0) ---
    po_ss = solve_poij(l0, d_ones, gss)
    po_ss = cp.where(gss > 0.1, po_ss, 0.0)
    
    # --- SP (L=1) ---
    d_sp = aij_tensor[0, 1, :] / np.sqrt(12.0)
    po_sp = solve_poij(l1, d_sp, hsp)
    po_sp = cp.where(mask_heavy, po_sp, 0.0)
    d_sp  = cp.where(mask_heavy, d_sp, 0.0)
    
    # --- PP (L=2) ---
    d_pp = cp.sqrt(aij_tensor[1, 1, :] * 0.1)
    po_pp2 = solve_poij(l2, d_pp, 0.5 * (gpp - gp2))
    po_pp2 = cp.where(mask_heavy, po_pp2, 0.0)
    d_pp   = cp.where(mask_heavy, d_pp, 0.0)

    # --- D-Orbitals (Standard Logic) ---
    if cp.any(mask_d):
        # SD
        d_sd = cp.sqrt(aij_tensor[0, 2, :] / 60.0)
        po_sd = solve_poij(l2, d_sd, repd[18, :])
        po_sd = cp.where(mask_d, po_sd, 0.0)
        d_sd = cp.where(mask_d, d_sd, 0.0)
        po_tensor[0, 2, 2, :] = po_sd    # po[3] in original codes
        po_tensor[2, 0, 2, :] = po_sd    # po[3] in original codes
        ddp_tensor[0, 2, :] = d_sd       # ddp[3] in original codes
        ddp_tensor[2, 0, :] = d_sd       # ddp[3] in original codes
        
        # PD
        d_pd = aij_tensor[1, 2, :] / np.sqrt(20.0)
        fg_pd = repd[22, :] - 1.8 * repd[34, :]
        po_pd = solve_poij(l1, d_pd, fg_pd)
        po_pd = cp.where(mask_d, po_pd, 0.0)
        d_pd = cp.where(mask_d, d_pd, 0.0)

        # L=1 dipole
        po_tensor[1, 2, 1, :] = po_pd   # po[4] in original codes
        po_tensor[2, 1, 1, :] = po_pd   # po[4] in original codes
        ddp_tensor[1, 2, :] = d_pd      # ddp[4] in original codes
        ddp_tensor[2, 1, :] = d_pd      # ddp[4] in original codes

        # L=2 quadrupole 
        # the po for l=2 is the same as l=1, this is the setting!
        po_tensor[1, 2, 2, :] = po_pd   # po[4] in original codes
        po_tensor[2, 1, 2, :] = po_pd   # po[4] in original codes
        
        # DD (L=0)
        fg_dd0 = 0.2 * (repd[28, :] + 2.0*repd[29, :] + 2.0*repd[30, :])
        po_dd0 = solve_poij(l0, d_ones, fg_dd0)
        po_dd0 = cp.where(mask_d & (fg_dd0 > 1e-5), po_dd0, 1e5)
        po_tensor[2, 2, 0, :] = po_dd0  # po[7] in original codes
        
        # DD (L=2)
        d_dd2 = cp.sqrt(aij_tensor[2, 2, :] / 14.0)
        fg_dd2 = repd[43, :] - (20.0/35.0) * repd[51, :]
        po_dd2 = solve_poij(l2, d_dd2, fg_dd2)
        po_dd2 = cp.where(mask_d, po_dd2, 0.0)
        d_dd2 = cp.where(mask_d, d_dd2, 0.0)
        po_tensor[2, 2, 2, :] = po_dd2  # po[5] in original codes
        ddp_tensor[2, 2, :] = d_dd2     # ddp[5] in original codes
    
    mask_mg = (natorb < 6) | main_group
    
    am_safe = cp.where(am < 1e-4, 1.0, am)
    po_ss_mg = 0.5 / am_safe
    po_ss = cp.where(mask_mg, po_ss_mg, po_ss)
    
    mask_ad = mask_mg & (ad > 1e-5)
    po_sp_mg = 0.5 / ad 
    
    po_sp = cp.where(mask_ad, po_sp_mg, po_sp)
    d_sp  = cp.where(mask_mg, dd, d_sp)
    
    mask_aq = mask_mg & (aq > 1e-5)
    po_pp2_mg = 0.5 / aq
    d_pp_mg   = qq * 1.4142135623730951
    
    po_pp2 = cp.where(mask_aq, po_pp2_mg, po_pp2)
    d_pp   = cp.where(mask_mg, d_pp_mg, d_pp)
    
    # SS (L=0)
    po_tensor[0, 0, 0, :] = po_ss   # po[0] in original codes
    
    # SP (L=1)
    po_tensor[0, 1, 1, :] = po_sp   # po[1] in original codes
    po_tensor[1, 0, 1, :] = po_sp   # po[1] in original codes
    ddp_tensor[0, 1, :]   = d_sp    # ddp[1] in original codes
    ddp_tensor[1, 0, :]   = d_sp    # ddp[1] in original codes
    
    # PP (L=0) -> Inherits po[0] (po_ss)
    # Note: Only apply where valid (mask_heavy).
    po_tensor[1, 1, 0, :] = cp.where(mask_heavy, po_ss, 0.0)    # po[6] in original codes
    
    # PP (L=2)
    po_tensor[1, 1, 2, :] = po_pp2  # po[2] in original codes
    ddp_tensor[1, 1, :]   = d_pp    # ddp[2] in original codes
    
    core_rho = cp.where(pocord > 1e-5, pocord, po_ss)   # po[8] in original codes

    return po_tensor, ddp_tensor, core_rho


# TODO: The output of this funcions should be parameterized.
def calc_multipole_scaling_params(
    gss, hsp, gpp, gp2, 
    zs, zp, element_ids,
    HATREE2EV=27.211386245988
):
    """
    Calculation of the scaling parameters (am, ad, aq) and 
    distance parameters (dd, qq) for all elements in the molecule.
    
    Args:
        gss  : (N,) CuPy array - One-center SS integral
        hsp  : (N,) CuPy array - One-center SP exchange integral
        gpp  : (N,) CuPy array - One-center PP integral (diag)
        gp2  : (N,) CuPy array - One-center PP integral (off-diag)
        zs   : (N,) CuPy array - Slater exponent for s
        zp   : (N,) CuPy array - Slater exponent for p
        element_ids : (N,) CuPy array (int32) - 0-based atomic numbers (H=0, He=1...).
        HATREE2EV : float - Hartree to eV conversion factor
        
    Returns:
        am : (N,) CuPy array - Dipole Distance
        ad : (N,) CuPy array - Quadrupole Distance
        aq : (N,) CuPy array - Additive term for Monopole
        dd : (N,) CuPy array - Additive term for Dipole
        qq : (N,) CuPy array - Additive term for Quadrupole
    """
    n_atom = gss.shape[0]
    
    nspqn_global = cp.array([1]*2+[2]*8+[3]*8+[4]*18+[5]*18+[6]*32+[0]*16, dtype=cp.float64)
    qn = nspqn_global[element_ids]
    
    am = cp.zeros(n_atom, dtype=cp.float64)
    ad = cp.zeros(n_atom, dtype=cp.float64)
    aq = cp.zeros(n_atom, dtype=cp.float64)
    dd = cp.zeros(n_atom, dtype=cp.float64)
    qq = cp.zeros(n_atom, dtype=cp.float64)
    
    valid_mask = (zp >= 1e-4) | (zs >= 1e-4)
    zp_safe = cp.where(zp < 0.3, 0.3, zp)
    
    hpp = 0.5 * (gpp - gp2)
    hpp = cp.where(hpp < 0.1, 0.1, hpp)
    
    t1 = (2.0 * qn + 1.0)
    t2 = cp.power(4.0 * zs * zp_safe, qn + 0.5)
    t3 = cp.power(zs + zp_safe, 2.0 * qn + 2.0) * cp.sqrt(3.0)
    
    dd = cp.where(valid_mask & (t3 > 1e-20), t1 * t2 / t3, 0.0)
    
    q_num = 4.0 * qn * qn + 6.0 * qn + 2.0
    qq = cp.where(valid_mask, cp.sqrt(q_num / 20.0) / zp_safe, 0.0)
    
    mask_ad = valid_mask & (dd > 1e-8) & (hsp > 1e-8)
    dd_safe = cp.where(mask_ad, dd, 1.0)
    hsp_safe = cp.where(mask_ad, hsp, 1.0)
    
    gdd1 = cp.power(hsp_safe / (HATREE2EV * dd_safe**2), 1.0 / 3.0)
    d1 = gdd1
    d2 = gdd1 + 0.04
    
    for _ in range(5):
        df = d2 - d1
        hsp1 = 0.50 * d1 - 0.50 / cp.sqrt(4.0 * dd_safe**2 + 1.0 / (d1**2))
        hsp2 = 0.50 * d2 - 0.50 / cp.sqrt(4.0 * dd_safe**2 + 1.0 / (d2**2))
        
        diff = hsp2 - hsp1
        diff = cp.where(cp.abs(diff) < 1e-25, 1e-25, diff)
        
        d3 = d1 + df * (hsp_safe / HATREE2EV - hsp1) / diff
        d1 = cp.where(mask_ad, d2, d1)
        d2 = cp.where(mask_ad, d3, d2)
        
    ad = cp.where(mask_ad, d2, 0.0)
    
    mask_aq = valid_mask & (qq > 1e-8) & (hpp > 1e-8)
    qq_safe = cp.where(mask_aq, qq, 1.0)
    hpp_safe = cp.where(mask_aq, hpp, 1.0)
    
    p4 = 16.0 # p = 2.0, p^4 = 16.0
    gqq = cp.power(p4 * hpp_safe / (HATREE2EV * 48.0 * qq_safe**4), 0.2)
    q1 = gqq
    q2 = gqq + 0.04
    
    for _ in range(5):
        qf = q2 - q1
        
        term1_1 = 0.25 * q1
        term2_1 = 0.5 / cp.sqrt(4.0 * qq_safe**2 + 1.0 / (q1**2))
        term3_1 = 0.25 / cp.sqrt(8.0 * qq_safe**2 + 1.0 / (q1**2))
        hpp1 = term1_1 - term2_1 + term3_1
        
        term1_2 = 0.25 * q2
        term2_2 = 0.5 / cp.sqrt(4.0 * qq_safe**2 + 1.0 / (q2**2))
        term3_2 = 0.25 / cp.sqrt(8.0 * qq_safe**2 + 1.0 / (q2**2))
        hpp2 = term1_2 - term2_2 + term3_2
        
        diff = hpp2 - hpp1
        diff = cp.where(cp.abs(diff) < 1e-25, 1e-25, diff)
        
        q3 = q1 + qf * (hpp_safe / HATREE2EV - hpp1) / diff
        q1 = cp.where(mask_aq, q2, q1)
        q2 = cp.where(mask_aq, q3, q2)
        
    aq = cp.where(mask_aq, q2, 0.0)
    am = cp.where(valid_mask, gss / HATREE2EV, 0.0)

    mask_am_small = am < 1e-20
    am_fallback = cp.where(gss > 1e-20, gss / HATREE2EV, 1.0)
    am = cp.where(mask_am_small, am_fallback, am)
    
    mask_H = (element_ids == 0)
    am = cp.where(mask_H, gss / HATREE2EV, am)
    ad = cp.where(mask_H, am, ad)
    aq = cp.where(mask_H, am, aq)
    dd = cp.where(mask_H, 0.0, dd)
    qq = cp.where(mask_H, 0.0, qq)
    
    mask_97 = (element_ids >= 97)
    qq = cp.where(mask_97, 0.0, qq)
    
    mask_101 = (element_ids == 101)
    am = cp.where(mask_101, 1e-10, am)

    return am, ad, aq, dd, qq


def calc_local_rep_core(
    pair_i, pair_j, ele_id, r_vec, 
    am, ad, aq, dd, qq, 
    po_tensor, ddp_tensor, core_rho, ch, 
    tore, natorb, dorbs, 
    task_arrays, 
    HATREE2EV=27.211386245988
):
    """
    Launch the global scheduling kernel to compute the full 491 representation terms,
    the 10x2 electron-core interaction matrix, and the gab monopole term for all atom pairs.
    
    Args:
        pair_i, pair_j : (n_pairs,) CuPy array (int32) - 0-based atom indices for the pairs.
        ele_id         : (n_atom,) CuPy array (int32) - Global element indices for checking heavy_atom logic.
        r_vec          : (n_pairs,) CuPy array (float64) - Interatomic distances in Bohr.
        am, ad, aq     : (n_atom,) CuPy arrays (float64) - Scaling parameters (now dimensioned to molecule).
        dd, qq         : (n_atom,) CuPy arrays (float64) - Distance parameters (now dimensioned to molecule).
        po_tensor      : (3, 3, 3, n_atom) CuPy array (float64) - Klopman-Ohno parameters.
        ddp_tensor     : (3, 3, n_atom) CuPy array (float64) - Multipole distances.
        core_rho       : (n_atom,) CuPy array (float64) - Core-core interaction terms.
        ch             : (45, 3, 5) CuPy array (float64) - Angular factors.
        tore           : (n_atom,) CuPy array (float64) - Core charges.
        natorb         : (n_atom,) CuPy array (int32) - Number of AOs per atom.
        dorbs          : (n_atom,) CuPy array (bool) - D-orbital mask.
        task_arrays    : tuple of 8 CuPy arrays (1D, int32) - Task instructions.
        HATREE2EV      : float - Conversion constant.
        
    Returns:
        rep_out  : (n_pairs, 491) CuPy array (float64)
        core_out : (n_pairs, 10, 2) CuPy array (float64)
        gab_out  : (n_pairs,) CuPy array (float64)
    """
    n_pairs = len(pair_i)
    n_atom = am.shape[0] 
    
    rep_out = cp.zeros((n_pairs, 491), dtype=cp.float64)
    core_out = cp.zeros((n_pairs, 10, 2), dtype=cp.float64)
    gab_out = cp.zeros(n_pairs, dtype=cp.float64)
    
    action, target, t_ij, t_kl, t_li, t_lj, t_lk, t_ll = task_arrays
    
    pair_i = cp.ascontiguousarray(pair_i, dtype=cp.int32)
    pair_j = cp.ascontiguousarray(pair_j, dtype=cp.int32)
    ele_id = cp.ascontiguousarray(ele_id, dtype=cp.int32)
    r_vec = cp.ascontiguousarray(r_vec, dtype=cp.float64)
    
    am = cp.ascontiguousarray(am, dtype=cp.float64)
    ad = cp.ascontiguousarray(ad, dtype=cp.float64)
    aq = cp.ascontiguousarray(aq, dtype=cp.float64)
    dd = cp.ascontiguousarray(dd, dtype=cp.float64)
    qq = cp.ascontiguousarray(qq, dtype=cp.float64)
    
    po_tensor = cp.ascontiguousarray(po_tensor, dtype=cp.float64)
    ddp_tensor = cp.ascontiguousarray(ddp_tensor, dtype=cp.float64)
    core_rho = cp.ascontiguousarray(core_rho, dtype=cp.float64)
    ch = cp.ascontiguousarray(ch, dtype=cp.float64)
    
    natorb = cp.ascontiguousarray(natorb, dtype=cp.int32)
    dorbs = cp.ascontiguousarray(dorbs, dtype=cp.bool_)
    tore = cp.ascontiguousarray(tore, dtype=cp.float64)

    _eri2c2e_MODULE.launch_calc_local_rep_core_kernel_c(
        ctypes.c_int(n_pairs),
        ctypes.c_void_p(pair_i.data.ptr), ctypes.c_void_p(pair_j.data.ptr), 
        ctypes.c_void_p(ele_id.data.ptr), ctypes.c_void_p(r_vec.data.ptr),
        ctypes.c_int(n_atom),
        ctypes.c_void_p(am.data.ptr), ctypes.c_void_p(ad.data.ptr), ctypes.c_void_p(aq.data.ptr),
        ctypes.c_void_p(dd.data.ptr), ctypes.c_void_p(qq.data.ptr),
        ctypes.c_void_p(po_tensor.data.ptr), ctypes.c_void_p(ddp_tensor.data.ptr), 
        ctypes.c_void_p(core_rho.data.ptr), ctypes.c_void_p(ch.data.ptr),
        ctypes.c_void_p(tore.data.ptr), ctypes.c_void_p(natorb.data.ptr), ctypes.c_void_p(dorbs.data.ptr),
        ctypes.c_void_p(action.data.ptr), ctypes.c_void_p(target.data.ptr),
        ctypes.c_void_p(t_ij.data.ptr), ctypes.c_void_p(t_kl.data.ptr),
        ctypes.c_void_p(t_li.data.ptr), ctypes.c_void_p(t_lj.data.ptr),
        ctypes.c_void_p(t_lk.data.ptr), ctypes.c_void_p(t_ll.data.ptr),
        ctypes.c_double(HATREE2EV),
        ctypes.c_void_p(rep_out.data.ptr), ctypes.c_void_p(core_out.data.ptr), ctypes.c_void_p(gab_out.data.ptr)
    )
    
    return rep_out, core_out, gab_out


def global_transform_gpu(
    pair_i, pair_j, ele_id, coords_bohr, 
    rep_in, core_in, gab_in,
    natorb, tore, xfac, alpb, guess1, guess2, guess3, v_par6, 
    BOHR=0.529177210903
    ):
    """
    Launch the GPU global transformation kernel.
    Replaces rotmat, tx, w2mat, elenuc, and ccrep.
    
    Args:
        pair_i, pair_j : (n_pairs,) CuPy array (int32) - 0-based atom indices.
        ele_id         : (n_atoms,) CuPy array (int32) - Global element indices.
        coords_bohr    : (n_atoms, 3) CuPy array (float64) - Coordinates in Bohr.
        rep_in         : (n_pairs, 491) CuPy array (float64) - Local 2c2e integrals.
        core_in        : (n_pairs, 10, 2) CuPy array (float64) - Local core integrals.
        gab_in         : (n_pairs,) CuPy array (float64) - Monopole core-core terms.
        natorb         : (107,) CuPy array (int32) - Number of AOs per element.
        tore...v_par6  : (107,) / (107, 4) / (4,) CuPy arrays - PM6 empirical parameters.
        
    Returns:
        w_out    : (total_w_size,) CuPy array - Flattened globally rotated 2c2e integrals.
        e1b_out  : (n_pairs, 45) CuPy array - Rotated elenuc integrals (Atom A core effect).
        e2a_out  : (n_pairs, 45) CuPy array - Rotated elenuc integrals (Atom B core effect).
        enuc_out : (n_pairs,) CuPy array - Final core-core repulsion energy.
    """
    n_pairs = len(pair_i)
    n_elements = tore.shape[0]  # Usually 107
    
    # Pre-calculate kr_offsets for the W vector allocation
    # The size of w block for pair (i,j) is limij * limkl
    ii_arr = natorb[ele_id[pair_i]]
    kk_arr = natorb[ele_id[pair_j]]
    limij_arr = ii_arr * (ii_arr + 1) // 2
    limkl_arr = kk_arr * (kk_arr + 1) // 2
    block_sizes = limij_arr * limkl_arr
    
    kr_offsets = cp.zeros(n_pairs + 1, dtype=cp.int32)
    kr_offsets[1:] = cp.cumsum(block_sizes)
    total_w_size = int(kr_offsets[-1].get())
    kr_offsets = kr_offsets[:-1] # We only need the start indices
    
    # Initialize output arrays
    w_out = cp.zeros(total_w_size, dtype=cp.float64)
    e1b_out = cp.zeros((n_pairs, 45), dtype=cp.float64)
    e2a_out = cp.zeros((n_pairs, 45), dtype=cp.float64)
    enuc_out = cp.zeros(n_pairs, dtype=cp.float64)
    
    # Format the ind2 mapping matrix explicitly (as a contiguous 1D block)
    # Note: ind2 is expected to be 0-based indexing. -1 means unmapped/zero.
    ind2_arr = cp.ascontiguousarray(IND2.flatten(), dtype=cp.int32)
    
    # Ensure memory continuity
    pair_i = cp.ascontiguousarray(pair_i, dtype=cp.int32)
    pair_j = cp.ascontiguousarray(pair_j, dtype=cp.int32)
    ele_id = cp.ascontiguousarray(ele_id, dtype=cp.int32)
    coords_bohr = cp.ascontiguousarray(coords_bohr, dtype=cp.float64)
    
    tore = cp.ascontiguousarray(tore, dtype=cp.float64)
    xfac = cp.ascontiguousarray(xfac, dtype=cp.float64)
    alpb = cp.ascontiguousarray(alpb, dtype=cp.float64)
    guess1 = cp.ascontiguousarray(guess1, dtype=cp.float64)
    guess2 = cp.ascontiguousarray(guess2, dtype=cp.float64)
    guess3 = cp.ascontiguousarray(guess3, dtype=cp.float64)
    v_par6 = cp.ascontiguousarray(v_par6, dtype=cp.float64)
    natorb = cp.ascontiguousarray(natorb, dtype=cp.int32)
    
    # Launch Kernel
    _eri2c2e_MODULE.launch_global_transform_kernel_c(
        ctypes.c_int(n_pairs),
        ctypes.c_void_p(pair_i.data.ptr), ctypes.c_void_p(pair_j.data.ptr), ctypes.c_void_p(ele_id.data.ptr),
        ctypes.c_void_p(coords_bohr.data.ptr), 
        ctypes.c_void_p(rep_in.data.ptr), ctypes.c_void_p(core_in.data.ptr), ctypes.c_void_p(gab_in.data.ptr),
        ctypes.c_void_p(ind2_arr.data.ptr), ctypes.c_void_p(natorb.data.ptr), ctypes.c_void_p(kr_offsets.data.ptr),
        ctypes.c_void_p(tore.data.ptr), ctypes.c_void_p(xfac.data.ptr), ctypes.c_void_p(alpb.data.ptr),
        ctypes.c_void_p(guess1.data.ptr), ctypes.c_void_p(guess2.data.ptr), ctypes.c_void_p(guess3.data.ptr),
        ctypes.c_void_p(v_par6.data.ptr), ctypes.c_int(n_elements), ctypes.c_double(BOHR),
        ctypes.c_void_p(w_out.data.ptr), ctypes.c_void_p(e1b_out.data.ptr), 
        ctypes.c_void_p(e2a_out.data.ptr), ctypes.c_void_p(enuc_out.data.ptr))
   
    return w_out, e1b_out, e2a_out, enuc_out