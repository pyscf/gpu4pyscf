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

import numpy as np
import cupy as cp
import time
import cupyx.scipy.linalg as cpx_linalg

from pyscf import gto, lib
from gpu4pyscf.df.int3c2e import VHFOpt, get_int3c2e_slice
from gpu4pyscf.lib.cupy_helper import cart2sph, contract, get_avail_mem
from gpu4pyscf.tdscf import parameter, math_helper, spectralib, _lr_eig
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.lib import logger


CITATION_INFO = """
Please cite the TDDFT-ris method:

    1.  Zhou, Zehao, Fabio Della Sala, and Shane M. Parker.
        Minimal auxiliary basis set approach for the electronic excitation spectra
        of organic molecules. The Journal of Physical Chemistry Letters
        14, no. 7 (2023): 1968-1976.
        (must cite)

    2.  Zhou, Zehao, and Shane M. Parker.
        Converging Time-Dependent Density Functional Theory Calculations in Five Iterations
        with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation
        20, no. 15 (2024): 6738-6746.
        (for efficient orbital truncation technique)

    2.  Giannone, Giulia, and Fabio Della Sala.
        Minimal auxiliary basis set for time-dependent density functional theory and
        comparison with tight-binding approximations: Application to silver nanoparticles.
        The Journal of Chemical Physics 153, no. 8 (2020).
        (TDDFT-ris is for hybrid functionals, originates from TDDFT-as with pure functional)
"""

LINEAR_EPSILON = 1e-8

def get_memory_info(words):
    cp.cuda.PinnedMemoryPool().free_all_blocks()
    cp.get_default_memory_pool().free_all_blocks()
    device = cp.cuda.Device()
    free_mem, total_mem = device.mem_info
    used_mem = total_mem - free_mem
    memory_info = f"{words} memory usage: {used_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB"
    return memory_info


def release_memory():
    '''Releases the GPU memory using Cupy.'''
    cp.cuda.PinnedMemoryPool().free_all_blocks()
    cp.get_default_memory_pool().free_all_blocks()

def get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis):
    '''
    Args:
        auxmol_basis_keys: (['C1', 'H2', 'O3', 'H4', 'H5', 'H6'])
        theta: float 0.2
        fitting_basis: str ('s','sp','spd')

    return:
        aux_basis:
        C1 [[0, [0.1320292535005648, 1.0]]]
        H2 [[0, [0.1999828038466018, 1.0]]]
        O3 [[0, [0.2587932305664396, 1.0]]]
        H4 [[0, [0.1999828038466018, 1.0]]]
        H5 [[0, [0.1999828038466018, 1.0]]]
        H6 [[0, [0.1999828038466018, 1.0]]]
    '''
    aux_basis = {}

    for atom_index in auxmol_basis_keys:
        atom = ''.join([char for char in atom_index if char.isalpha()])
        '''
        exponent_alpha = theta/R^2
        '''
        exp_alpha = parameter.ris_exp[atom] * theta

        if 's' in fitting_basis:
            aux_basis[atom_index] = [[0, [exp_alpha, 1.0]]]

        if atom != 'H':
            if 'p' in fitting_basis:
                aux_basis[atom_index].append([1, [exp_alpha, 1.0]])
            if 'd' in fitting_basis:
                aux_basis[atom_index].append([2, [exp_alpha, 1.0]])
    return aux_basis

def get_auxmol(mol, theta=0.2, fitting_basis='s'):
    """
    Assigns a minimal auxiliary basis set to the molecule.

    Args:
        mol: The input molecule object.
        theta: The scaling factor for the exponents.
        fitting_basis: Basis set type ('s', 'sp', 'spd').

    Returns:
        auxmol: The molecule object with assigned auxiliary basis.
    """


    '''
    parse_arg = False
    turns off PySCF built-in parsing function
    '''
    auxmol = gto.M(atom=mol.atom,
                    basis=mol.basis,
                    parse_arg=False,
                    spin=mol.spin,
                    charge=mol.charge,
                    cart=mol.cart)

    auxmol_basis_keys = mol._basis.keys()
    auxmol.basis = get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis)
    auxmol.build(dump_input=False)
    return auxmol


'''
            n_occ          n_vir
       -|-------------||-------------|
        |             ||             |
  n_occ |   3c2e_ij   ||  3c2e_ia    |
        |             ||             |
        |             ||             |
       =|=============||=============|
        |             ||             |
  n_vir |             ||  3c2e_ab    |
        |             ||             |
        |             ||             |
       -|-------------||-------------|
'''

def get_Ppq_to_Tpq(Ppq: cp.ndarray, lower_inv_eri2c: cp.ndarray):
    ''' Ppq  (nauxao, n_p, n_q) -> (nauxao, n_p*n_q)
        lower_inv_eri2c  (nauxao, nauxao)
        >> Ppq (nauxao, n_p*n_q) -> (nauxao, n_p, n_q)'''
    nauxao, n_p, n_q = Ppq.shape

    Ppq = Ppq.reshape(nauxao, n_p*n_q)

    T_pq = cp.dot(lower_inv_eri2c.T, Ppq)
    T_pq = T_pq.reshape(nauxao, n_p, n_q)

    return T_pq

def get_PuvCupCvq_to_Ppq(eri3c: cp.ndarray, C_p: cp.ndarray, C_q: cp.ndarray):
    # # '''
    # # eri3c : (P|pq) , P = auxnao or 3
    # # C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    # # Ppq = einsum("Puv,up,vq->Ppq", eri3c, Cp, C_q)

    # # manually reshape and transpose is faster than einsum

    # # '''

    # # '''eri3c in shape (nauxao, nao, nao)'''
    # nao = eri3c.shape[1]
    # nauxao = eri3c.shape[0]

    # n_p = C_p.shape[1]
    # n_q = C_q.shape[1]


    # # '''eri3c (nauxao, nao, nao) -> (nauxao*nao, nao)
    # #    C_p (nao, n_p)
    # #    >> eri3c_C_p (nauxao*nao, n_p)'''
    # eri3c = eri3c.reshape(nauxao*nao, nao)
    # eri3c_C_p = cp.dot(eri3c, C_p)

    # # ''' eri3c_C_p (nauxao*nao, n_p)
    # #     -> (nauxao, nao, n_p)
    # #     -> (nauxao, n_p, nao) '''
    # eri3c_C_p = eri3c_C_p.reshape(nauxao, nao, n_p)
    # eri3c_C_p = eri3c_C_p.transpose(0,2,1)

    # # ''' eri3c_C_p  (nauxao, n_p, nao) -> (nauxao*n_p, nao)
    # #     C_q  (nao, n_q)
    # #     >> Ppq (nauxao*n_p, n_q) >  (nauxao, n_p, n_q)  '''
    # eri3c_C_p = eri3c_C_p.reshape(nauxao*n_p, nao)
    # Ppq = cp.dot(eri3c_C_p, C_q)
    # Ppq = Ppq.reshape(nauxao, n_p, n_q)

    tmp = contract('Puv,up->Ppv', eri3c, C_p)
    Ppq = contract('Ppv,vq->Ppq', tmp, C_q)

    return Ppq


BLKSIZE = 10000
AUXBLKSIZE = 256

# for debug purpose
def get_int3c2e(mol, auxmol, aosym=True, omega=None):
    '''
    Generate full int3c2e tensor on GPU
    '''
    nao = mol.nao
    naux = auxmol.nao
    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(diag_block_with_triu=True, aosym=aosym, group_size=BLKSIZE, group_size_aux=BLKSIZE)
    int3c = cp.empty([naux, nao, nao], order='C')
    for cp_ij_id, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]

        int3c_slice = cp.empty([naux, j1-j0, i1-i0], order='C')
        for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
            k0, k1 = intopt.aux_ao_loc[cp_kl_id], intopt.aux_ao_loc[cp_kl_id+1]
            get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, out=int3c_slice[k0:k1], omega=omega)

        if not mol.cart:
            int3c_slice = cart2sph(int3c_slice, axis=1, ang=lj)
            int3c_slice = cart2sph(int3c_slice, axis=2, ang=li)

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        int3c[:, j0:j1, i0:i1] = int3c_slice
    if aosym:
        row, col = np.tril_indices(nao)
        int3c[:, row, col] = int3c[:, col, row]
    int3c = intopt.unsort_orbitals(int3c, aux_axis=[0], axis=[1,2])
    return int3c

def compute_Tpq_on_gpu_general(mol, auxmol, C_p, C_q, lower_inv_eri2c,
                                calc='JK', aosym=True, omega=None, alpha=None, beta=None,
                                group_size=BLKSIZE, group_size_aux=AUXBLKSIZE):
    """
    (3c2e_{Puv}, C_{up}, C_{vq} -> Ppq)。

    Parameters:
        mol: pyscf.gto.Mole
        auxmol: pyscf.gto.Mole
        C_p: cupy.ndarray (nao, p)
        C_q: cupy.ndarray  (nao, q)

    Returns:
        Tpq: cupy.ndarray (naux, nao, nao)
    """

    intopt = VHFOpt(mol, auxmol, 'int2e')
    intopt.build(aosym=aosym, group_size=group_size, group_size_aux=group_size_aux)

    nao = mol.nao
    naux = auxmol.nao

    siz_p = C_p.shape[1]
    siz_q = C_q.shape[1]

    if 'J' in calc:
        Ppq = cp.empty((naux, siz_p, siz_q), dtype=cp.float32)

    if 'K' in calc:
        Ppp = cp.empty((naux, siz_p, siz_p), dtype=cp.float32)
        Pqq = cp.empty((naux, siz_q, siz_q), dtype=cp.float32)

    for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
        k0, k1 = intopt.aux_ao_loc[cp_kl_id], intopt.aux_ao_loc[cp_kl_id+1]

        int3c_slice = cp.empty((k1 - k0, nao, nao), dtype=cp.float32, order='C')

        for cp_ij_id, _ in enumerate(intopt.log_qs):
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            int3c_slice_blk = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, omega=0)

            if not mol.cart:
                int3c_slice_blk = cart2sph(int3c_slice_blk, axis=1, ang=lj)
                int3c_slice_blk = cart2sph(int3c_slice_blk, axis=2, ang=li)


            if omega and omega != 0:
                int3c_slice_blk_omega = get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, omega=omega)

                if not mol.cart:
                    int3c_slice_blk_omega = cart2sph(int3c_slice_blk_omega, axis=1, ang=lj)
                    int3c_slice_blk_omega = cart2sph(int3c_slice_blk_omega, axis=2, ang=li)
                int3c_slice_blk = alpha * int3c_slice_blk +  beta * int3c_slice_blk_omega

            int3c_slice_blk = cp.asarray(int3c_slice_blk, dtype=cp.float32, order='C')
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]

            assert int3c_slice[:,j0:j1, i0:i1].shape == int3c_slice_blk.shape
            int3c_slice[:,j0:j1, i0:i1] = int3c_slice_blk

        if aosym:
            row, col = cp.tril_indices(nao)
            int3c_slice[:, row, col] = int3c_slice[:, col, row]


        unsorted_ao_index = cp.argsort(intopt._ao_idx)
        int3c_slice = int3c_slice[:, unsorted_ao_index, :]
        int3c_slice = int3c_slice[:, :, unsorted_ao_index]

        if 'J' in calc:
            Ppq[k0:k1,:,:] = get_PuvCupCvq_to_Ppq(int3c_slice,C_p,C_q)

        if 'K' in calc:

            Ppp[k0:k1,:,:] = get_PuvCupCvq_to_Ppq(int3c_slice,C_p,C_p)
            Pqq[k0:k1,:,:] = get_PuvCupCvq_to_Ppq(int3c_slice,C_q,C_q)


    unsorted_aux_ao_index = cp.argsort(intopt._aux_ao_idx)


    # DEBUG = False
    # if DEBUG:
    #     eri_3c2e = get_int3c2e(mol, auxmol, omega=0)
    #     if omega and omega != 0:
    #         eri_3c2e_erf = get_int3c2e(mol, auxmol, omega=omega)
    #         eri_3c2e = alpha * eri_3c2e + beta * eri_3c2e_erf
    #     tmp = cp.einsum('Puv,up->Ppv', eri_3c2e, C_p)
    #     Ppq = cp.einsum('Ppv,vq->Ppq', tmp, C_q)


    if calc == 'J':
        Tpq = get_Ppq_to_Tpq(Ppq[unsorted_aux_ao_index,:,:], lower_inv_eri2c)
        return Tpq

    if calc == 'K':
        Tpp = get_Ppq_to_Tpq(Ppp[unsorted_aux_ao_index,:,:], lower_inv_eri2c)
        Tqq = get_Ppq_to_Tpq(Pqq[unsorted_aux_ao_index,:,:], lower_inv_eri2c)
        return Tpp, Tqq

    if calc == 'JK':
        Tpq = get_Ppq_to_Tpq(Ppq[unsorted_aux_ao_index,:,:], lower_inv_eri2c)
        Tpp = get_Ppq_to_Tpq(Ppp[unsorted_aux_ao_index,:,:], lower_inv_eri2c)
        Tqq = get_Ppq_to_Tpq(Pqq[unsorted_aux_ao_index,:,:], lower_inv_eri2c)
        return Tpq, Tpp, Tqq

def get_eri2c_inv_lower(auxmol, omega=0, alpha=None, beta=None):

    eri2c = auxmol.intor('int2c2e')

    if omega and omega != 0:

        with auxmol.with_range_coulomb(omega):
            eri2c_erf = auxmol.intor('int2c2e')

        eri2c = alpha * eri2c + beta * eri2c_erf

    eri2c = cp.asarray(eri2c, dtype=cp.float64, order='C')

    try:
        ''' eri2c=L L.T
            LX = I
            lower_inv_eri2c = X = L^-1
        '''
        L = cp.linalg.cholesky(eri2c)
        L_inv = cpx_linalg.solve_triangular(L, cp.eye(L.shape[0]), lower=True)
        lower_inv_eri2c = L_inv.T

    except cp.linalg.LinAlgError:
        ''' lower_inv_eri2c = eri2c ** -0.5
            LINEAR_EPSILON = 1e-8 to remove the linear dependency, sometimes the aux eri2c is not full rank.
        '''
        lower_inv_eri2c = math_helper.matrix_power(eri2c,-0.5,epsilon=LINEAR_EPSILON)


    lower_inv_eri2c = cp.asarray(lower_inv_eri2c, dtype=cp.float32, order='C')
    return lower_inv_eri2c

def get_inter_contract_C(int_tensor, C_occ, C_vir):

    P = get_PuvCupCvq_to_Ppq(int_tensor, C_occ, C_vir)

    ''' 3 for xyz three directions.
        reshape is helpful when calculating oscillator strength and polarizability.
    '''
    P = cp.asarray(P.reshape(3,-1))
    return P

def gen_hdiag_MVP(hdiag, n_occ, n_vir):
    def hdiag_MVP(V):
        m = V.shape[0]
        V = V.reshape(m, n_occ*n_vir)
        hdiag_v = hdiag * V
        hdiag_v = hdiag_v.reshape(m, n_occ, n_vir)
        return hdiag_v

    return hdiag_MVP


def gen_iajb_MVP(T_left, T_right):
    '''
    (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
            = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    if T_left == T_right, then it is either
        (1) (ia|jb) in RKS
        or
        (2)(ia_α|jb_α) or (ia_β|jb_β) in UKS,
    elif T_left != T_right
        it is (ia_α|jb_β) or (ia_β|jb_α) in UKS

    V in shape (m, n_occ * n_vir)
    '''

    # def iajb_MVP(V):
    #     T_right_jb_V = einsum("Pjb,mjb->Pm", T_right, V)
    #     iajb_V = einsum("Pia,Pm->mia", T_left, T_right_jb_V)
    #     return iajb_V

    def iajb_MVP(V):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (m, n_occ * n_vir).

        Returns:
            iajb_V (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        # get_memory_info('before iajb_MVP')
        # Get the shape of the tensors
        nauxao, n_occ_l, n_vir_l = T_left.shape
        nauxao, n_occ_r, n_vir_r = T_right.shape
        n_state, n_occ_r, n_vir_r = V.shape
        # Initialize result tensor
        iajb_V = cp.zeros((n_state, n_occ_l, n_vir_l), dtype=T_left.dtype)

        # Estimate the memory size for one chunk
        estimated_chunk_size_bytes = n_occ_r * n_vir_r * T_right.itemsize * 4  # 4 for each element (complex or float64)

        # Get available GPU memory in bytes
        available_gpu_memory = get_avail_mem()

        # Estimate the optimal chunk size based on available GPU memory
        aux_chunk_size = int(available_gpu_memory * 0.8 // estimated_chunk_size_bytes)

        # Ensure the chunk size is at least 1 and doesn't exceed the total number of auxao
        aux_chunk_size = max(1, min(nauxao, aux_chunk_size))

        # Iterate over chunks of the auxao dimension
        for aux_start in range(0, nauxao, aux_chunk_size):
            aux_end = min(aux_start + aux_chunk_size, nauxao)

            T_left_chunk = T_left[aux_start:aux_end, :, :]  # Shape: (aux_range, n_occ, n_vir)
            T_right_chunk = T_right[aux_start:aux_end, :, :]   # Shape: (aux_range, n_occ * n_vir)


            T_right_jb_V_chunk = contract("Pjb,mjb->Pm", T_right_chunk, V)

            iajb_V_chunk = contract("Pia,Pm->mia", T_left_chunk, T_right_jb_V_chunk)
            del T_right_jb_V_chunk

            iajb_V += iajb_V_chunk  # Accumulate the result

            del iajb_V_chunk
            release_memory()

        # get_memory_info('after iajb_MVP')
        return iajb_V


    return iajb_MVP

def gen_ijab_MVP(T_ij, T_ab):
    '''
    (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
            = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    # def ijab_MVP(V):
    #     T_ab_V = einsum("Pab,mjb->Pamj", T_ab, V)
    #     ijab_V = einsum("Pij,Pamj->mia", T_ij, T_ab_V)
    #     return ijab_V

    def ijab_MVP(V):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
        by chunking along the n_vir dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        nauxao, n_vir, n_vir = T_ab.shape  # Dimensions of T_ab
        n_state, n_occ, n_vir = V.shape      # Dimensions of V

        # Initialize result tensor
        ijab_V = cp.empty((n_state, n_occ, n_vir), dtype=T_ab.dtype)

        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_mem()
        bytes_per_vir = nauxao * n_occ * n_state * 4  # Assuming float32 (4 bytes per element)
        vir_chunk_size = max(1, int(available_gpu_memory * 0.2 // bytes_per_vir))  # Ensure at least 1

        # Iterate over chunks of the n_vir dimension
        for vir_start in range(0, n_vir, vir_chunk_size):
            vir_end = min(vir_start + vir_chunk_size, n_vir)
            # vir_range = vir_end - vir_start

            # Extract the current chunk of V
            V_chunk = V[:, :, vir_start:vir_end]  # Shape: (n_state, n_occ, vir_range)

            # Extract the corresponding chunk of T_ab
            T_ab_chunk = T_ab[:, vir_start:vir_end, vir_start:vir_end]  # Shape: (nauxao, vir_range, n_vir)

            # Compute T_ab_V for the current chunk
            T_ab_V_chunk = contract("Pab,mjb->Pamj", T_ab_chunk, V_chunk)

            # Compute ijab_V for the current chunk
            ijab_V[:, :, vir_start:vir_end] = contract("Pij,Pamj->mia", T_ij, T_ab_V_chunk)

            # Release intermediate variables and clean up memory
            # del V_chunk, T_ab_V_chunk
            # cp.get_default_memory_pool().free_all_blocks()

        return ijab_V


    return ijab_MVP

def get_ibja_MVP(T_ia):
    '''
    the exchange (ib|ja) in B matrix
    (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
            = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]
    '''
    # def ibja_MVP(V):
    #     T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
    #     ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)
    #     return ibja_V

    def ibja_MVP(V, occ_chunk_size=100):
        '''
        Optimized calculation of (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
        by chunking along the n_occ dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            occ_chunk_size (int): Chunk size for splitting the n_occ dimension.

        Returns:
            ibja_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        nauxao, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape
        # assert n_occ == n_occ_v and n_vir == n_vir_v, "Shapes of V and T_ia must match"

        # Initialize result tensor
        ibja_V = cp.empty((n_state, n_occ, n_vir), dtype=T_ia.dtype)

        # Iterate over chunks of the n_occ dimension
        for occ_start in range(0, n_occ, occ_chunk_size):
            occ_end = min(occ_start + occ_chunk_size, n_occ)
            #occ_range = occ_end - occ_start

            # Extract the current chunk of V
            V_chunk = V[:, occ_start:occ_end, :]  # Shape: (n_state, occ_range, n_vir)

            # Extract the corresponding chunk of T_ia
            T_ia_chunk = T_ia[:, occ_start:occ_end, :]  # Shape: (nauxao, occ_range, n_vir)

            # Compute T_ib_V for the current chunk
            T_ib_V_chunk = contract("Pib,mjb->Pimj", T_ia_chunk, V_chunk)

            # Compute ibja_V for the current chunk
            ibja_V[:, occ_start:occ_end, :] = contract("Pja,Pimj->mia", T_ia_chunk, T_ib_V_chunk)

            # Release intermediate variables and clean up memory
            # del V_chunk, T_ia_chunk, T_ib_V_chunk
            # cp.get_default_memory_pool().free_all_blocks()

        return ibja_V

    return ibja_MVP

class RisBase(lib.StreamObject):
    def __init__(self,
                mf,
                theta: float = 0.2,
                J_fit: str = 'sp',
                K_fit: str = 's',
                Ktrunc: float = 40.0,
                a_x: float = None,
                omega: float = None,
                alpha: float = None,
                beta: float = None,
                conv_tol: float = 1e-3,
                nstates: int = 5,
                max_iter: int = 25,
                spectra: bool = False,
                out_name: str = '',
                print_threshold: float = 0.05,
                GS: bool = False,
                single: bool = True,
                group_size: int = 256,
                group_size_aux: int = 256):

        self.single = single

        if single:
            mf = mf.copy()
            mf.mo_coeff = cp.asarray(mf.mo_coeff, dtype=cp.float32)

        self.mf = mf
        self.theta = theta
        self.J_fit = J_fit
        self.K_fit = K_fit

        self.Ktrunc = Ktrunc
        self.a_x = a_x
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conv_tol = conv_tol
        self.nstates = nstates
        self.max_iter = max_iter
        self.mol = mf.mol
        self.spectra = spectra
        self.out_name = out_name
        self.print_threshold = print_threshold
        self.GS = GS
        self.group_size = group_size
        self.group_size_aux = group_size_aux

        self.verbose = mf.verbose
        self.device = mf.device

        logger.TIMER_LEVEL = 4
        self.log = logger.new_logger(self)

    def build(self):
        log = self.log
        log.info(f'nstates: {self.nstates}')
        log.info(f'conv_tol: {self.conv_tol}')
        log.info(f'max_iter: {self.max_iter}')
        log.info(f'Ktrunc: {self.Ktrunc}')

        if self.a_x or self.omega or self.alpha or self.beta:
            ''' user wants to define some XC parameters '''
            if self.a_x:
                if self.a_x == 0:
                    log.info('use pure XC functional')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info('use hybrid XC functional')
                elif self.a_x == 1:
                    log.info('use HF')
                else:
                    log.info('a_x > 1, weird')

            elif self.omega and self.alpha and self.beta:
                log.info('use range-separated hybrid XC functional')
            else:
                raise ValueError('Please dounble check the XC functional parameters')
        else:
            ''' use default XC parameters
                note: the definition of a_x, α and β is kind of weird in pyscf/libxc
            '''

            omega, alpha_libxc, hyb_libxc = self.mf._numint.rsh_and_hybrid_coeff(self.mf.xc, spin=self.mf.mol.spin)
            log.info(f'omega, alpha_libxc, hyb_libxc: {omega}, {alpha_libxc}, {hyb_libxc}')

            if omega > 0:
                log.info('use range-separated hybrid XC functional')
                self.a_x = 1
                self.omega = omega
                self.alpha = hyb_libxc
                self.beta = alpha_libxc - hyb_libxc

            elif omega == 0:
                self.a_x = alpha_libxc
                if self.a_x == 0:
                    log.info('use pure XC functional')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info('use hybrid XC functional')
                elif self.a_x == 1:
                    log.info('use HF')
                else:
                    log.info('a_x > 1, weird')

        log.info(f'omega: {self.omega}')
        log.info(f'alpha: {self.alpha}')
        log.info(f'beta: {self.beta}')
        log.info(f'a_x: {self.a_x}')
        log.info(f'GS: {self.GS}')
        log.info(f'single: {self.single}')
        log.info(f'group_size: {self.group_size}')

        if self.J_fit == self.K_fit:
            log.info(f'use same J and K fitting basis: {self.J_fit}')
        else:
            log.info(f'use different J and K fitting basis: J with {self.J_fit} and K with {self.K_fit}')

        if self.mol.cart:
            self.eri_tag = '_cart'
        else:
            self.eri_tag = '_sph'
        log.info(f'cartesian or spherical electron integral = {self.eri_tag}')

        if self.mf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            n_occ = int(sum(self.mf.mo_occ>0))
            n_vir = int(sum(self.mf.mo_occ==0))
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = cp.asfortranarray(self.mf.mo_coeff[:,:n_occ])
            self.C_vir_notrunc = cp.asfortranarray(self.mf.mo_coeff[:,n_occ:])
            mo_energy = self.mf.mo_energy
            log.info(f'mo_energy.shape: {mo_energy.shape}')
            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)
            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
            delta_hdiag = cp.repeat(vir_ene, n_occ, axis=0) - cp.repeat(occ_ene, n_vir, axis=1)
            if self.single:
                delta_hdiag = cp.asarray(delta_hdiag, dtype=cp.float32)

            self.delta_hdiag = delta_hdiag

            log.info(f'n_occ = {n_occ}')
            log.info(f'n_vir = {n_vir}')

            if self.Ktrunc > 0:
                log.info(f' MO truncation in K with threshold {self.Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = self.Ktrunc/HARTREE2EV

                homo_vir_delta_ene = delta_hdiag[-1,:]
                occ_lumo_delta_ene = delta_hdiag[:,0]

                rest_occ = cp.sum(occ_lumo_delta_ene <= trunc_tol_au)
                rest_vir = cp.sum(homo_vir_delta_ene <= trunc_tol_au)

            elif self.Ktrunc == 0:
                log.info('no MO truncation in K')
                rest_occ = n_occ
                rest_vir = n_vir

            log.info(f'rest_occ = {rest_occ}')
            log.info(f'rest_vir = {rest_vir}')

            self.C_occ_Ktrunc = cp.asfortranarray(self.mf.mo_coeff[:,n_occ-rest_occ:n_occ])
            self.C_vir_Ktrunc = cp.asfortranarray(self.mf.mo_coeff[:,n_occ:n_occ+rest_vir])

            self.rest_occ = rest_occ
            self.rest_vir = rest_vir

        elif self.mf.mo_coeff.ndim == 3:
            ''' TODO UKS method '''
            self.RKS = False
            self.UKS = True
            self.n_occ_a = sum(self.mf.mo_occ[0]>0)
            self.n_vir_a = sum(self.mf.mo_occ[0]==0)
            self.n_occ_b = sum(self.mf.mo_occ[1]>0)
            self.n_vir_b = sum(self.mf.mo_occ[1]==0)
            log.info('n_occ for alpha spin = {self.n_occ_a}')
            log.info('n_vir for alpha spin = {self.n_vir_a}')
            log.info('n_occ for beta spin = {self.n_occ_b}')
            log.info('n_vir for beta spin = {self.n_vir_b}')

        self.log = log

    def get_P(self):
        '''
        transition dipole u
        '''
        int_r = self.mol.intor_symmetric('int1e_r' + self.eri_tag)
        int_r = cp.asarray(int_r, dtype=cp.float32 if self.single else cp.float64)
        if self.RKS:
            P = get_inter_contract_C(int_tensor=int_r, C_occ=self.C_occ_notrunc, C_vir=self.C_vir_notrunc)
        else:
            ''' TODO '''
            P_alpha = get_inter_contract_C(int_tensor=int_r, C_occ=self.C_occ[0], C_vir=self.C_vir[0])
            P_beta = get_inter_contract_C(int_tensor=int_r, C_occ=self.C_occ[1], C_vir=self.C_vir[1])
            P = cp.vstack((P_alpha, P_beta))
        return P

    def get_mdpol(self):
        '''
        magnatic dipole m
        '''
        int_rxp = self.mol.intor('int1e_cg_irxp' + self.eri_tag, comp=3, hermi=2)
        int_rxp = cp.asarray(int_rxp, dtype=cp.float32 if self.single else cp.float64)

        if self.RKS:
            mdpol = get_inter_contract_C(int_tensor=int_rxp, C_occ=self.C_occ_notrunc, C_vir=self.C_vir_notrunc)
        else:
            ''' TODO '''
            mdpol_alpha = get_inter_contract_C(int_tensor=int_rxp, C_occ=self.C_occ[0], C_vir=self.C_vir[0])
            mdpol_beta = get_inter_contract_C(int_tensor=int_rxp, C_occ=self.C_occ[1], C_vir=self.C_vir[1])
            mdpol = cp.vstack((mdpol_alpha, mdpol_beta))
        return mdpol

class TDA(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.warn("TDA-ris is still in the experimental stage, and its APIs are subject to change in future releases.")
        log.info('TDA-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_RKS_TDA_hybrid_MVP(self):
        ''' TDA RKS hybrid '''
        log = self.log

        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir

        single = self.single

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        C_occ_Ktrunc = self.C_occ_Ktrunc
        C_vir_Ktrunc  = self.C_vir_Ktrunc

        rest_occ = self.rest_occ
        rest_vir = self.rest_vir

        hdiag = cp.asarray(self.delta_hdiag.reshape(-1))

        mol = self.mol
        theta = self.theta

        J_fit = self.J_fit
        K_fit = self.K_fit

        omega = self.omega
        alpha = self.alpha
        beta = self.beta

        group_size = self.group_size
        group_size_aux = self.group_size_aux

        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer()

        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
        log.info(f'n_bf in auxmol_J = {auxmol_J.nao_nr()}')
        unit = 4 if single else 8
        log.info(f'T_ia_J will take { auxmol_J.nao_nr() * n_occ * n_vir * unit / (1024 ** 2):.0f} MB memory')


        lower_inv_eri2c_J = get_eri2c_inv_lower(auxmol_J, omega=0)

        T_ia_J = compute_Tpq_on_gpu_general(mol, auxmol_J,
                                            C_p=C_occ_notrunc,
                                            C_q=C_vir_notrunc,
                                            lower_inv_eri2c=lower_inv_eri2c_J,
                                            calc="J",
                                            omega=0,
                                            group_size=group_size,
                                            group_size_aux=group_size_aux)

        log.timer('T_ia_J', *cpu0)


        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()

        if K_fit == J_fit and (omega == 0 or omega is None):
            log.info('K uese exactly same basis as J, and they share same set of Tensors')
            auxmol_K = auxmol_J
            lower_inv_eri2c_K = lower_inv_eri2c_J

        else:
            log.info('K uese different basis as J')
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
            lower_inv_eri2c_K = get_eri2c_inv_lower(auxmol_K, omega=omega, alpha=alpha, beta=beta)

        log.info(f'n_bf in auxmol_K = {auxmol_K.nao_nr()}')
        unit = 4 if single else 8
        log.info(f'T_ij_K will take {auxmol_K.nao_nr() * rest_occ * rest_occ * unit / (1024 ** 2):.0f} MB memory')
        log.info(f'T_ab_K will take {auxmol_K.nao_nr() * rest_vir * rest_vir * unit / (1024 ** 2):.0f} MB memory')

        T_ij_K, T_ab_K = compute_Tpq_on_gpu_general(mol, auxmol_K,
                                                    C_p=C_occ_Ktrunc,
                                                    C_q=C_vir_Ktrunc,
                                                    lower_inv_eri2c=lower_inv_eri2c_K,
                                                    calc='K',
                                                    omega=omega,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    group_size = group_size,
                                                    group_size_aux = group_size_aux)

        log.timer('T_ij_K T_ab_K', *cpu1)


        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)

        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K,   T_ab=T_ab_K)

        def RKS_TDA_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V) - a_x*ijab_MVP(V)
                for RSH, a_x = 1

                if not MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)
            AX = hdiag_MVP(X)
            AX += 2 * iajb_MVP(X)

            AX[:,n_occ-rest_occ:,:rest_vir] -= a_x * ijab_MVP(X[:,n_occ-rest_occ:,:rest_vir])
            AX = AX.reshape(nstates, n_occ*n_vir)

            return AX

        return RKS_TDA_hybrid_MVP, hdiag


    ''' ===========  RKS pure =========== '''
    def get_RKS_TDA_pure_MVP(self):
        '''hybrid RKS TDA'''
        log = self.log
        n_occ = self.n_occ
        n_vir = self.n_vir

        single = self.single

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc


        hdiag = self.delta_hdiag.reshape(-1)

        mol = self.mol
        theta = self.theta

        J_fit = self.J_fit

        group_size = self.group_size
        group_size_aux = self.group_size_aux

        log.info('==================== RIJ ====================')
        tt = time.time()

        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)

        unit = 4 if single else 8
        log.info(f'T_ia_J will take { auxmol_J.nao_nr() * n_occ * n_vir * unit / (1024 ** 2):.0f} MB memory')

        lower_inv_eri2c_J = get_eri2c_inv_lower(auxmol_J, omega=0)

        T_ia_J = compute_Tpq_on_gpu_general(mol, auxmol_J,
                                            C_p=C_occ_notrunc,
                                            C_q=C_vir_notrunc,
                                            lower_inv_eri2c=lower_inv_eri2c_J,
                                            calc="J",
                                            omega=0,
                                            group_size = group_size,
                                            group_size_aux = group_size_aux,)
        log.info(f'T_ia_J time {time.time() - tt:.1f} seconds')


        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)
        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)
            AX = hdiag_MVP(X)
            AX += 2 * iajb_MVP(X)
            AX = AX.reshape(nstates, n_occ*n_vir)
            return AX

        return RKS_TDA_pure_MVP, hdiag

    #  TODO ===========  UKS ===========
    def get_UKS_TDA_MVP(self):
        a_x = self.a_x

        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        mol = self.mol
        auxmol = self.get_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.get_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvP_withL = self.get_uvP_withL(eri2c=eri2c, eri3c=eri3c)

        hdiag_a_MVP, hdiag_a = self.get_hdiag_MVP(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_MVP, hdiag_b = self.get_hdiag_MVP(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = cp.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        if a_x != 0:
            ''' UKS TDA hybrid '''
            T_ia_J_alpha, _, T_ij_K_alpha, T_ab_K_alpha = self.get_T_J_T_K(mol=mol,
                                                                                auxmol=auxmol,
                                                                                uvP_withL=uvP_withL,
                                                                                eri3c=eri3c,
                                                                                eri2c=eri2c,
                                                                                n_occ=n_occ_a,
                                                                                mo_coeff=mo_coeff[0])

            T_ia_J_beta, _, T_ij_K_beta, T_ab_K_beta  = self.get_T_J_T_K(mol=mol,
                                                                              auxmol=auxmol,
                                                                              uvP_withL=uvP_withL,
                                                                              eri3c=eri3c,
                                                                              eri2c=eri2c,
                                                                              n_occ=n_occ_b,
                                                                              mo_coeff=mo_coeff[1])

            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_beta)

            ijab_aa_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_alpha, T_ab=T_ab_K_alpha)
            ijab_bb_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_beta,  T_ab=T_ab_K_beta)

            def UKS_TDA_hybrid_MVP(X):
                '''
                UKS
                return AX
                A have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]
                    [ Aβα Aββ ]

                X = [ Xα ]
                    [ Xβ ]
                AX = [ Aαα Xα + Aαβ Xβ ]
                     [ Aβα Xα + Aββ Xβ ]

                Aαα Xα = hdiag_MVP(Xα) + iajb_aa_MVP(Xα) - a_x * ijab_aa_MVP(Xα)
                Aββ Xβ = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ) - a_x * ijab_bb_MVP(Xβ)
                Aαβ Xβ = iajb_ab_MVP(Xβ)
                Aβα Xα = iajb_ba_MVP(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_MVP(X_a) + iajb_aa_MVP(X_a) - a_x * ijab_aa_MVP(X_a)
                Aab_Xb = iajb_ab_MVP(X_b)

                Aba_Xa = iajb_ba_MVP(X_a)
                Abb_Xb = hdiag_b_MVP(X_b) + iajb_bb_MVP(X_b) - a_x * ijab_bb_MVP(X_b)

                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = cp.vstack((U_a, U_b))
                return U
            return UKS_TDA_hybrid_MVP, hdiag

        elif a_x == 0:
            ''' UKS TDA pure '''
            T_ia_alpha = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_a,
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            T_ia_beta = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_b,
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')

            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_beta)

            def UKS_TDA_pure_MVP(X):
                '''
                Aαα Xα = hdiag_MVP(Xα) + iajb_aa_MVP(Xα)
                Aββ Xβ = hdiag_MVP(Xβ) + iajb_bb_MVP(Xβ)
                Aαβ Xβ = iajb_ab_MVP(Xβ)
                Aβα Xα = iajb_ba_MVP(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_MVP(X_a) + iajb_aa_MVP(X_a)
                Aab_Xb = iajb_ab_MVP(X_b)

                Aba_Xa = iajb_ba_MVP(X_a)
                Abb_Xb = hdiag_b_MVP(X_b) + iajb_bb_MVP(X_b)

                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = cp.vstack((U_a, U_b))
                return U
            return UKS_TDA_pure_MVP, hdiag

    def kernel(self):

        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            pure TDA is not using MZ=Zw^2 form
        '''
        self.build()
        log = self.log
        if self.RKS:

            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_pure_MVP()


        elif self.UKS:
            TDA_MVP, hdiag = self.get_UKS_TDA_MVP()


        energies, X = _lr_eig.Davidson(matrix_vector_product=TDA_MVP,
                                            hdiag=hdiag,
                                            N_states=self.nstates,
                                            conv_tol=self.conv_tol,
                                            max_iter=self.max_iter,
                                            GS=self.GS,
                                            single=self.single,
                                            verbose=log)

        log.debug(f'check orthonormal of X: {cp.linalg.norm(cp.dot(X, X.T) - cp.eye(X.shape[0])):.2e}')

        P = self.get_P()
        mdpol = self.get_mdpol()

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies,
                                                       X=X/(2**0.5),
                                                       Y=None,
                                                       P=P,
                                                       mdpol=mdpol,
                                                       name=self.out_name+'_TDA_ris',
                                                       RKS=self.RKS,
                                                       spectra=self.spectra,
                                                       print_threshold = self.print_threshold,
                                                       n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                       n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))
        energies = energies*HARTREE2EV
        log.info(f'energies: {energies}')
        log.info(f'oscillator strength: {oscillator_strength}')
        log.info(CITATION_INFO)

        self.energies = energies
        self.X = X
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        return energies, X, oscillator_strength, rotatory_strength


class TDDFT(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.warn("TDDFT-ris is still in the experimental stage, and its APIs are subject to change in future releases.")
        log.info('TDDFT-ris is initialized')

    ''' ===========  RKS hybrid =========== '''
    def gen_RKS_TDDFT_hybrid_MVP(self):
        '''hybrid RKS TDDFT'''
        log = self.log
        a_x = self.a_x
        n_occ = self.n_occ
        n_vir = self.n_vir

        single = self.single

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        C_occ_Ktrunc = self.C_occ_Ktrunc
        C_vir_Ktrunc  = self.C_vir_Ktrunc

        rest_occ = self.rest_occ
        rest_vir = self.rest_vir

        hdiag = cp.asarray(self.delta_hdiag.reshape(-1))

        mol = self.mol
        theta = self.theta

        J_fit = self.J_fit
        K_fit = self.K_fit

        omega = self.omega
        alpha = self.alpha
        beta = self.beta

        group_size = self.group_size
        group_size_aux = self.group_size_aux

        log.info(get_memory_info('before T_ia_J'))

        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer()

        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)

        unit = 4 if single else 8
        log.info(f'T_ia_J will take { auxmol_J.nao_nr() * n_occ * n_vir * unit / (1024 ** 2):.0f} MB memory')


        lower_inv_eri2c_J = get_eri2c_inv_lower(auxmol_J, omega=0)

        T_ia_J = compute_Tpq_on_gpu_general(mol, auxmol_J,
                                            C_p=C_occ_notrunc,
                                            C_q=C_vir_notrunc,
                                            lower_inv_eri2c=lower_inv_eri2c_J,
                                            calc="J",
                                            omega=0,
                                            group_size = group_size,
                                            group_size_aux = group_size_aux)
        log.info(f'T_ia_J MEM: {T_ia_J.nbytes / (1024 ** 2):.0f} MB')
        log.timer('T_ia_J', *cpu0)
        log.info(get_memory_info('after T_ia_J'))

        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()
        if K_fit == J_fit and (omega == 0 or omega is None):
            log.info('K uese exactly same basis as J, and they share same set of Tensors')
            auxmol_K = auxmol_J
            lower_inv_eri2c_K = lower_inv_eri2c_J

        else:
            log.info('K uese different basis as J')
            auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)
            lower_inv_eri2c_K = get_eri2c_inv_lower(auxmol_K, omega=omega, alpha=alpha, beta=beta)

        unit = 4 if single else 8
        log.info(f'T_ia_K will take {auxmol_K.nao_nr() * rest_occ * rest_vir * unit / (1024 ** 2):.0f} MB memory')
        log.info(f'T_ij_K will take {auxmol_K.nao_nr() * rest_occ * rest_occ * unit / (1024 ** 2):.0f} MB memory')
        log.info(f'T_ab_K will take {auxmol_K.nao_nr() * rest_vir * rest_vir * unit / (1024 ** 2):.0f} MB memory')

        T_ia_K, T_ij_K, T_ab_K = compute_Tpq_on_gpu_general(mol, auxmol_K,
                                                            C_p=C_occ_Ktrunc,
                                                            C_q=C_vir_Ktrunc,
                                                            lower_inv_eri2c=lower_inv_eri2c_K,
                                                            calc='JK',
                                                            omega=omega,
                                                            alpha=alpha,
                                                            beta=beta,
                                                            group_size = group_size,
                                                            group_size_aux = group_size_aux)

        log.timer('T_ia_K T_ij_K T_ab_K', *cpu1)
        log.info(get_memory_info('after T_ia_K T_ij_K T_ab_K'))
        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)

        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K,   T_ab=T_ab_K)
        ibja_MVP = get_ibja_MVP(T_ia=T_ia_K)

        def RKS_TDDFT_hybrid_MVP(X, Y):
            '''
            RKS
            [A B][X] = [AX+BY] = [U1]
            [B A][Y]   [AY+BX]   [U2]
            we want AX+BY and AY+BX
            instead of directly computing AX+BY and AY+BX
            we compute (A+B)(X+Y) and (A-B)(X-Y)
            it can save one (ia|jb)V tensor contraction compared to directly computing AX+BY and AY+BX

            (A+B)V = hdiag_MVP(V) + 4*iajb_MVP(V) - a_x * [ ijab_MVP(V) + ibja_MVP(V) ]
            (A-B)V = hdiag_MVP(V) - a_x * [ ijab_MVP(V) - ibja_MVP(V) ]
            for RSH, a_x = 1, because the exchange component is defined by alpha+beta (alpha+beta not awlways == 1)

            # X Y in shape (m, n_occ*n_vir)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, n_occ, n_vir)
            Y = Y.reshape(nstates, n_occ, n_vir)

            XpY = X + Y
            XmY = X - Y
            ApB_XpY = hdiag_MVP(XpY)

            ApB_XpY += 4*iajb_MVP(XpY)

            ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= a_x*ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])

            ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= a_x*ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])

            AmB_XmY = hdiag_MVP(XmY)
            AmB_XmY[:,n_occ-rest_occ:,:rest_vir] -= a_x*ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])

            AmB_XmY[:,n_occ-rest_occ:,:rest_vir] += a_x*ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])

            ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                (A-B)(X-Y) = AX + BY - AY - BX   (2)
                (1) + (1) /2 = AX + BY = U1
                (1) - (2) /2 = AY + BX = U2
            '''
            U1 = (ApB_XpY + AmB_XmY)/2
            U2 = (ApB_XpY - AmB_XmY)/2

            U1 = U1.reshape(nstates, n_occ*n_vir)
            U2 = U2.reshape(nstates, n_occ*n_vir)

            return U1, U2
        return RKS_TDDFT_hybrid_MVP, hdiag

    ''' ===========  RKS pure =========== '''
    def gen_RKS_TDDFT_pure_MVP(self):
        log = self.log
        n_occ = self.n_occ
        n_vir = self.n_vir

        single = self.single

        C_occ_notrunc = self.C_occ_notrunc
        C_vir_notrunc = self.C_vir_notrunc

        hdiag = self.delta_hdiag.reshape(-1)

        mol = self.mol
        theta = self.theta

        J_fit = self.J_fit

        group_size = self.group_size
        group_size_aux = self.group_size_aux

        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer()

        auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)

        unit = 4 if single else 8
        log.info(f'T_ia_J will take { auxmol_J.nao_nr() * n_occ * n_vir * unit / (1024 ** 2):.0f} MB memory')

        lower_inv_eri2c_J = get_eri2c_inv_lower(auxmol_J, omega=0)

        T_ia_J = compute_Tpq_on_gpu_general(mol, auxmol_J,
                                            C_p=C_occ_notrunc,
                                            C_q=C_vir_notrunc,
                                            lower_inv_eri2c=lower_inv_eri2c_J,
                                            calc="J",
                                            omega=0,
                                            group_size = group_size,
                                            group_size_aux = group_size_aux)
        log.timer('T_ia_J', *cpu0)

        hdiag_sqrt_MVP = gen_hdiag_MVP(hdiag=hdiag**0.5, n_occ=n_occ, n_vir=n_vir)
        hdiag_MVP = gen_hdiag_MVP(hdiag=hdiag, n_occ=n_occ, n_vir=n_vir)
        iajb_MVP = gen_iajb_MVP(T_left=T_ia_J, T_right=T_ia_J)
        hdiag_sq = hdiag**2
        def RKS_TDDFT_pure_MVP(Z):
            '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                    MZ = Z w^2
                M = (A-B)^1/2 (A+B) (A-B)^1/2
                X+Y = (A-B)^1/2 Z

                (A+B)(V) = hdiag_MVP(V) + 4*iajb_MVP(V)
                (A-B)^1/2(V) = hdiag_sqrt_MVP(V)
            '''
            nstates = Z.shape[0]
            Z = Z.reshape(nstates, n_occ, n_vir)
            AmB_sqrt_V = hdiag_sqrt_MVP(Z)
            ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V) + 4*iajb_MVP(AmB_sqrt_V)
            MZ = hdiag_sqrt_MVP(ApB_AmB_sqrt_V)
            MZ = MZ.reshape(nstates, n_occ*n_vir)
            return MZ

        return RKS_TDDFT_pure_MVP, hdiag_sq

    #  TODO ===========  UKS ===========
    def get_UKS_TDDFT_MVP(self):

        a_x = self.a_x

        n_occ_a = self.n_occ_a
        n_vir_a = self.n_vir_a
        n_occ_b = self.n_occ_b
        n_vir_b = self.n_vir_b

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        mo_coeff = self.mf.mo_coeff
        mo_energy = self.mf.mo_energy

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab)  + (ij|alpha + beta*erf(omega)/r|ab)
        short-range part (ij|1-(alpha + beta*erf(omega))/r|ab) is treated by the DFT XC functional, thus not considered here
        long-range part  (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
        '''
        mol = self.mol
        auxmol = self.get_auxmol(theta=self.theta, add_p=self.add_p)
        eri2c, eri3c = self.get_eri2c_eri3c(mol=self.mol, auxmol=auxmol, omega=0)
        uvP_withL = self.get_uvP_withL(eri2c=eri2c, eri3c=eri3c)
        '''
        _aa_MVP means alpha-alpha spin
        _ab_MVP means alpha-beta spin
        T_ia_alpha means T_ia matrix for alpha spin
        T_ia_beta means T_ia matrix for beta spin
        '''

        hdiag_a_MVP, hdiag_a = self.get_hdiag_MVP(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_MVP, hdiag_b = self.get_hdiag_MVP(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = cp.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        if a_x != 0:
            T_ia_J_alpha, T_ia_K_alpha, T_ij_K_alpha, T_ab_K_alpha = self.get_T_J_T_K(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvP_withL=uvP_withL,
                                                                                            eri3c=eri3c,
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_a,
                                                                                            mo_coeff=mo_coeff[0])

            T_ia_J_beta,  T_ia_K_beta,  T_ij_K_beta,  T_ab_K_beta  = self.get_T_J_T_K(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvP_withL=uvP_withL,
                                                                                            eri3c=eri3c,
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_b,
                                                                                            mo_coeff=mo_coeff[1])

            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_J_alpha, T_right=T_ia_J_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_J_beta,  T_right=T_ia_J_beta)

            ijab_aa_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_alpha, T_ab=T_ab_K_alpha)
            ijab_bb_MVP = self.gen_ijab_MVP(T_ij=T_ij_K_beta,  T_ab=T_ab_K_beta)

            ibja_aa_MVP = self.get_ibja_MVP(T_ia=T_ia_K_alpha)
            ibja_bb_MVP = self.get_ibja_MVP(T_ia=T_ia_K_beta)

            def UKS_TDDFT_hybrid_MVP(X,Y):
                '''
                UKS
                [A B][X] = [AX+BY] = [U1]
                [B A][Y]   [AY+BX]   [U2]
                A B have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]   B = [ Bαα Bαβ ]
                    [ Aβα Aββ ]       [ Bβα Bββ ]

                X = [ Xα ]        Y = [ Yα ]
                    [ Xβ ]            [ Yβ ]

                (A+B)αα, (A+B)αβ is shown below

                βα, ββ can be obtained by change α to β
                we compute (A+B)(X+Y) and (A-B)(X-Y)

                V:= X+Y
                (A+B)αα Vα = hdiag_MVP(Vα) + 2*iaαjbα_MVP(Vα) - a_x*[ijαabα_MVP(Vα) + ibαjaα_MVP(Vα)]
                (A+B)αβ Vβ = 2*iaαjbβ_MVP(Vβ)

                V:= X-Y
                (A-B)αα Vα = hdiag_MVP(Vα) - a_x*[ijαabα_MVP(Vα) - ibαjaα_MVP(Vα)]
                (A-B)αβ Vβ = 0

                A+B = [ Cαα Cαβ ]   x+y = [ Vα ]
                      [ Cβα Cββ ]         [ Vβ ]
                (A+B)(x+y) =   [ Cαα Vα + Cαβ Vβ ]  = ApB_XpY
                               [ Cβα Vα + Cββ Vβ ]

                A-B = [ Cαα  0  ]   x-y = [ Vα ]
                      [  0  Cββ ]         [ Vβ ]
                (A-B)(x-y) =   [ Cαα Vα ]    = AmB_XmY
                               [ Cββ Vβ ]
                '''

                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)
                Y_a = Y[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Y_b = Y[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                XpY_a = X_a + Y_a
                XpY_b = X_b + Y_b

                XmY_a = X_a - Y_a
                XmY_b = X_b - Y_b

                '''============== (A+B) (X+Y) ================'''
                '''(A+B)aa(X+Y)a'''
                ApB_XpY_aa = hdiag_a_MVP(XpY_a) + 2*iajb_aa_MVP(XpY_a) - a_x*(ijab_aa_MVP(XpY_a) + ibja_aa_MVP(XpY_a))
                '''(A+B)bb(X+Y)b'''
                ApB_XpY_bb = hdiag_b_MVP(XpY_b) + 2*iajb_bb_MVP(XpY_b) - a_x*(ijab_bb_MVP(XpY_b) + ibja_bb_MVP(XpY_b))
                '''(A+B)ab(X+Y)b'''
                ApB_XpY_ab = 2*iajb_ab_MVP(XpY_b)
                '''(A+B)ba(X+Y)a'''
                ApB_XpY_ba = 2*iajb_ba_MVP(XpY_a)

                '''============== (A-B) (X-Y) ================'''
                '''(A-B)aa(X-Y)a'''
                AmB_XmY_aa = hdiag_a_MVP(XmY_a) - a_x*(ijab_aa_MVP(XmY_a) - ibja_aa_MVP(XmY_a))
                '''(A-B)bb(X-Y)b'''
                AmB_XmY_bb = hdiag_b_MVP(XmY_b) - a_x*(ijab_bb_MVP(XmY_b) - ibja_bb_MVP(XmY_b))

                ''' (A-B)ab(X-Y)b
                    AmB_XmY_ab = 0
                    (A-B)ba(X-Y)a
                    AmB_XmY_ba = 0
                '''

                ''' (A+B)(X+Y) = AX + BY + AY + BX   (1) ApB_XpY
                    (A-B)(X-Y) = AX + BY - AY - BX   (2) AmB_XmY
                    (1) + (1) /2 = AX + BY = U1
                    (1) - (2) /2 = AY + BX = U2
                '''
                ApB_XpY_alpha = (ApB_XpY_aa + ApB_XpY_ab).reshape(A_aa_size,-1)
                ApB_XpY_beta  = (ApB_XpY_ba + ApB_XpY_bb).reshape(A_bb_size,-1)
                ApB_XpY = cp.vstack((ApB_XpY_alpha, ApB_XpY_beta))

                AmB_XmY_alpha = AmB_XmY_aa.reshape(A_aa_size,-1)
                AmB_XmY_beta  = AmB_XmY_bb.reshape(A_bb_size,-1)
                AmB_XmY = cp.vstack((AmB_XmY_alpha, AmB_XmY_beta))

                U1 = (ApB_XpY + AmB_XmY)/2
                U2 = (ApB_XpY - AmB_XmY)/2

                return U1, U2

            return UKS_TDDFT_hybrid_MVP, hdiag

        elif a_x == 0:
            ''' UKS TDDFT pure '''

            hdiag_a_sqrt_MVP, hdiag_a_sq = self.get_hdiag_MVP(mo_energy=mo_energy[0],
                                                          n_occ=n_occ_a,
                                                          n_vir=n_vir_a,
                                                          sqrt=True)
            hdiag_b_sqrt_MVP, hdiag_b_sq = self.get_hdiag_MVP(mo_energy=mo_energy[1],
                                                          n_occ=n_occ_b,
                                                          n_vir=n_vir_b,
                                                          sqrt=True)
            '''hdiag_sq: preconditioner'''
            hdiag_sq = cp.vstack((hdiag_a_sq.reshape(-1,1), hdiag_b_sq.reshape(-1,1))).reshape(-1)

            T_ia_alpha = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_a,
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            T_ia_beta = self.get_T(uvP_withL=uvP_withL,
                                    n_occ=n_occ_b,
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')

            iajb_aa_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_alpha)
            iajb_ab_MVP = self.gen_iajb_MVP(T_left=T_ia_alpha, T_right=T_ia_beta)
            iajb_ba_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_alpha)
            iajb_bb_MVP = self.gen_iajb_MVP(T_left=T_ia_beta,  T_right=T_ia_beta)

            def UKS_TDDFT_pure_MVP(Z):
                '''       MZ = Z w^2
                    M = (A-B)^1/2(A+B)(A-B)^1/2
                    Z = (A-B)^1/2(X-Y)

                    X+Y = (A-B)^1/2 Z * 1/w
                    A+B = hdiag_MVP(V) + 4*iajb_MVP(V)
                    (A-B)^1/2 = hdiag_sqrt_MVP(V)


                    M =  [ (A-B)^1/2αα    0   ] [ (A+B)αα (A+B)αβ ] [ (A-B)^1/2αα    0   ]            Z = [ Zα ]
                         [    0   (A-B)^1/2ββ ] [ (A+B)βα (A+B)ββ ] [    0   (A-B)^1/2ββ ]                [ Zβ ]
                '''
                Z_a = Z[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Z_b = Z[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                AmB_aa_sqrt_Z_a = hdiag_a_sqrt_MVP(Z_a)
                AmB_bb_sqrt_Z_b = hdiag_b_sqrt_MVP(Z_b)

                ApB_aa_sqrt_V = hdiag_a_MVP(AmB_aa_sqrt_Z_a) + 2*iajb_aa_MVP(AmB_aa_sqrt_Z_a)
                ApT_ab_sqrt_V = 2*iajb_ab_MVP(AmB_bb_sqrt_Z_b)
                ApB_ba_sqrt_V = 2*iajb_ba_MVP(AmB_aa_sqrt_Z_a)
                ApB_bb_sqrt_V = hdiag_b_MVP(AmB_bb_sqrt_Z_b) + 2*iajb_bb_MVP(AmB_bb_sqrt_Z_b)

                MZ_a = hdiag_a_sqrt_MVP(ApB_aa_sqrt_V + ApT_ab_sqrt_V).reshape(A_aa_size, -1)
                MZ_b = hdiag_b_sqrt_MVP(ApB_ba_sqrt_V + ApB_bb_sqrt_V).reshape(A_bb_size, -1)

                MZ = cp.vstack((MZ_a, MZ_b))

                return MZ

            return UKS_TDDFT_pure_MVP, hdiag_sq

        # def TDDFT_spolar_MVP(X):

        #     ''' for RSH, a_x=1
        #         (A+B)X = hdiag_MVP(V) + 4*iajb_MVP(V) - a_x*[ijab_MVP(V) + ibja_MVP(V)]
        #     '''
        #     X = X.reshape(n_occ, n_vir, -1)

        #     ABX = hdiag_MVP(X) + 4*iajb_MVP(X) - a_x* (ibja_MVP(X) + ijab_MVP(X))
        #     ABX = ABX.reshape(n_occ*n_vir, -1)

        #     return ABX

    def kernel(self):
        self.build()
        log = self.log
        if self.a_x != 0:
            '''hybrid TDDFT'''
            if self.RKS:
                TDDFT_hybrid_MVP, hdiag = self.gen_RKS_TDDFT_hybrid_MVP()

            elif self.UKS:
                TDDFT_hybrid_MVP, hdiag = self.get_UKS_TDDFT_MVP()

            energies, X, Y = _lr_eig.Davidson_Casida(matrix_vector_product=TDDFT_hybrid_MVP,
                                                            hdiag=hdiag,
                                                            N_states=self.nstates,
                                                            conv_tol=self.conv_tol,
                                                            max_iter=self.max_iter,
                                                            GS=self.GS,
                                                            single=self.single,
                                                            verbose=self.verbose)

        elif self.a_x == 0:
            '''pure TDDFT'''
            if self.RKS:
                TDDFT_pure_MVP, hdiag_sq = self.gen_RKS_TDDFT_pure_MVP()

            elif self.UKS:
                TDDFT_pure_MVP, hdiag_sq = self.get_UKS_TDDFT_pure_MVP()
            energies_sq, Z = _lr_eig.Davidson(matrix_vector_product=TDDFT_pure_MVP,
                                                hdiag=hdiag_sq,
                                                N_states=self.nstates,
                                                conv_tol=self.conv_tol,
                                                max_iter=self.max_iter,
                                                GS=self.GS,
                                                single=self.single,
                                                verbose=self.verbose)

            energies = energies_sq**0.5
            Z = (energies**0.5).reshape(-1,1) * Z

            X, Y = math_helper.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)

        log.debug(f'check norm of X^TX - Y^YY - I = {cp.linalg.norm( (cp.dot(X, X.T) - cp.dot(Y, Y.T)) - cp.eye(self.nstates) ):.2e}')

        P = self.get_P()
        mdpol = self.get_mdpol()

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies,
                                                    X=X/(2**0.5),
                                                    Y=Y/(2**0.5),
                                                    P=P,
                                                    mdpol=mdpol,
                                                    name=self.out_name+'_TDDFT_ris',
                                                    spectra=self.spectra,
                                                    RKS=self.RKS,
                                                    print_threshold = self.print_threshold,
                                                    n_occ=self.n_occ if self.RKS else (self.n_occ_a, self.n_occ_b),
                                                    n_vir=self.n_vir if self.RKS else (self.n_vir_a, self.n_vir_b))
        energies = energies*HARTREE2EV
        log.info(f'energies: {energies}')
        log.info(f'oscillator strength: {oscillator_strength}')
        log.info(CITATION_INFO)
        self.energies = energies
        self.X = X
        self.Y = Y
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        return energies, X, Y, oscillator_strength, rotatory_strength

