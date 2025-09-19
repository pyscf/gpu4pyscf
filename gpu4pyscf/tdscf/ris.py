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
import cupyx.scipy.linalg as cpx_linalg
import gc

from pyscf import gto, lib
from gpu4pyscf import scf
from gpu4pyscf.df.int3c2e import VHFOpt, get_int3c2e_slice
from gpu4pyscf.lib.cupy_helper import cart2sph, contract, get_avail_mem, cholesky
from gpu4pyscf.tdscf import parameter, math_helper, spectralib, _lr_eig, _krylov_tools
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df import int3c2e_bdiv
from cupyx import scatter_add

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

    3.  Giannone, Giulia, and Fabio Della Sala.
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
    auxmol = mol.copy()
    auxmol_basis_keys = mol._basis.keys()
    auxmol.basis = get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis)
    auxmol.build(dump_input=False, parse_arg=False)
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

def get_PuvCupCvq_to_Ppq(eri3c: cp.ndarray, C_p: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''    
    eri3c : (P|pq) , P = auxnao or 3
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("Puv,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    tmp = contract('Puv,up->Ppv', eri3c, C_p)
    Ppq = contract('Ppv,vq->Ppq', tmp, C_q)

    if in_ram:
        Ppq = Ppq.get()
    return Ppq

def get_uvPCupCvq_to_Ppq(eri3c: cp.ndarray, C_p: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''    
    eri3c : (uv|P) , P = nauxao
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    tmp = contract('uvP,up->Ppv', eri3c, C_p)
    Ppq = contract('Ppv,vq->Ppq', tmp, C_q)

    if in_ram:
        Ppq = Ppq.get()
    return Ppq


def get_uvPCupCvq_to_Ppq_symmetry(eri3c: cp.ndarray, C_p: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''    
    eri3c : (uv|P) , P = nauxao
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    tmp = contract('uvP,up->Ppv', eri3c, C_p)
    Ppq = contract('Ppv,vq->Ppq', tmp, C_q)

    P, p, q = Ppq.shape
    tril_indices = cp.tril_indices(p)

    Ppq_lower = Ppq[:, tril_indices[0], tril_indices[1]]

    Ppq_flat = Ppq_lower.reshape(P, -1)

    if in_ram:
        Ppq_flat = Ppq_flat.get()
    return Ppq_flat

def einsum2dot(_,a,b):
    P, Q = a.shape
    Q, p, q = b.shape

    b = b.reshape(Q, p*q)

    out = np.dot(a, b)

    out = out.reshape(P, p, q)
    return out

BLKSIZE = 256
AUXBLKSIZE = 256

def get_Tpq_backup(mol, auxmol, lower_inv_eri2c, C_p, C_q, 
           calc='JK', aosym=True, omega=None, alpha=None, beta=None,
           group_size=BLKSIZE, group_size_aux=AUXBLKSIZE, log=None, 
           in_ram=True, single=True):
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

    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    nao, nao_orig = int3c2e_opt.coeff.shape
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
    rows, cols = divmod(ao_pair_mapping, nao_orig)
    print('ao_pair_mapping.shape', ao_pair_mapping.shape)
    # upper_inv_eri2c = lower_inv_eri2c.T

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])

    siz_p = C_p.shape[1]
    siz_q = C_q.shape[1]


    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff
    naux = aux_coeff.shape[0]
    # print('aux_coeff.shape', aux_coeff.shape)

    xp = np if in_ram else cp
    log.info(f'xp {xp}')

    P_dtype = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    print('cp_int3c_dtype', cp_int3c_dtype)
    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = cp.asarray(aux_coeff_lower_inv_eri2c, dtype=cp_int3c_dtype)
    upper_inv_eri2c = aux_coeff_lower_inv_eri2c.T

    if in_ram:
        aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
        upper_inv_eri2c = upper_inv_eri2c.get()
    

    if 'J' in calc:
        Pia = xp.zeros((naux, siz_p, siz_q), dtype=P_dtype)

    if 'K' in calc:
        Pij = xp.zeros((naux, siz_p, siz_p), dtype=P_dtype)
        Pab = xp.zeros((naux, siz_q, siz_q), dtype=P_dtype)

    byte_eri3c = nao_orig * nao_orig * cp_int3c_dtype.itemsize
    log.info(f'eri3c per aux dimension will take {byte_eri3c / (1024 ** 2):.0f} MB memory')


    aux_offset = 0
    batch_size = 16

    for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size):
        # print('eri3c_batch.shape', eri3c_batch.shape)
        eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
        eri3c_batch = cp.asarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
        # if in_ram:
        #     eri3c_batch = eri3c_batch.get()
        # print('eri3c_orbital_pair_cart2sph', eri3c_batch.shape)
        aopair, aux_batch_size = eri3c_batch.shape

        eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, aux_batch_size), order='F')

        eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
        eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch

        eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)
        
        '''Puv -> Ppq, AO->MO transform '''
        if 'J' in calc:
            Pia[aux_offset:aux_offset+aux_batch_size,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_p,C_q, in_ram=in_ram)

        if 'K' in calc:
            Pij[aux_offset:aux_offset+aux_batch_size,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_p,C_p, in_ram=in_ram)
            Pab[aux_offset:aux_offset+aux_batch_size,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_q,C_q, in_ram=in_ram)
        
        log.info(f'batch {aux_offset}-{aux_offset+aux_batch_size} / {naux} done')

        aux_offset += aux_batch_size

    if calc == 'J':
        if in_ram:
            tmp_einsum = einsum2dot 
            # upper_inv_eri2c = upper_inv_eri2c.get()
        else:
            tmp_einsum = contract
            
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        return Tia

    if calc == 'K':
        Tij = (aux_coeff_lower_inv_eri2c.dot(upper_inv_eri2c)).dot(Pij)
        # Tab = upper_inv_eri2c.dot(Pab)
        return Tij, Pab

    if calc == 'JK':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        Tij = (aux_coeff_lower_inv_eri2c.dot(upper_inv_eri2c)).dot(Pij)
        # Tab = upper_inv_eri2c.dot(Pab)
        return Tia, Tij, Pab
   
def get_Tpq_toomuch_ram(mol, auxmol, lower_inv_eri2c, C_p, C_q, 
           calc='JK', aosym=True, omega=None, alpha=None, beta=None,
           group_size=BLKSIZE, group_size_aux=AUXBLKSIZE, log=None, 
           in_ram=True, single=True):
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

    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    nao, nao_orig = int3c2e_opt.coeff.shape
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
    rows, cols = divmod(ao_pair_mapping, nao_orig)
    print('ao_pair_mapping.shape', ao_pair_mapping.shape)
    # upper_inv_eri2c = lower_inv_eri2c.T

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])

    siz_p = C_p.shape[1]
    siz_q = C_q.shape[1]


    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff
    naux = aux_coeff.shape[0]
    # print('aux_coeff.shape', aux_coeff.shape)

    xp = np if in_ram else cp
    log.info(f'xp {xp}')

    P_dtype = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    print('cp_int3c_dtype', cp_int3c_dtype)
    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = cp.asarray(aux_coeff_lower_inv_eri2c, dtype=cp_int3c_dtype)
    upper_inv_eri2c = aux_coeff_lower_inv_eri2c.T

    if in_ram:
        aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()

    if 'J' in calc:
        Pia = xp.zeros((naux, siz_p, siz_q), dtype=P_dtype)

    if 'K' in calc:
        Pij = xp.zeros((naux, siz_p, siz_p), dtype=P_dtype)
        Pab = xp.zeros((naux, siz_q, siz_q), dtype=P_dtype)

    byte_eri3c = nao_orig * nao_orig * cp_int3c_dtype.itemsize
    log.info(f'eri3c per aux dimension will take {byte_eri3c / (1024 ** 2):.0f} MB memory')


    aux_offset = 0
    batch_size = 16

    # off_diag = (rows > cols)
    p1 = 0
    pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()
    # off_diag = set(ao_pair_mapping) - set(pair_diag)

    for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size):
        # print('eri3c_batch.shape', eri3c_batch.shape)
        eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
        eri3c_batch = cp.asarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
        # if in_ram:
        #     eri3c_batch = eri3c_batch.get()
        # print('eri3c_orbital_pair_cart2sph', eri3c_batch.shape)
        # aopair, aux_batch_size = eri3c_batch.shape
        p0, p1 = p1, p1 + eri3c_batch.shape[1]
        eri3c_batch[pair_diag,:] *= 0.5
        # eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, aux_batch_size), order='F')

        # eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
        # eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch

        # eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)
        
        '''Puv -> Ppq, AO->MO transform '''
        if 'J' in calc:
            # Pia[aux_offset:aux_offset+aux_batch_size,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_p,C_q, in_ram=in_ram)
            C_p_rows = C_p[rows, :]  # (naopair, siz_p)
            C_p_cols = C_p[cols, :]  # (naopair, siz_p)
            C_q_rows = C_q[rows, :]  # (naopair, siz_q)
            C_q_cols = C_q[cols, :]  # (naopair, siz_q)

            Pia[p0:p1,:,:] = cp.einsum('kP,kp,kq->Ppq', eri3c_batch, C_p_rows, C_q_cols, optimize=True)
            Pia[p0:p1,:,:] += cp.einsum('kP,kp,kq->Ppq', eri3c_batch, C_p_cols, C_q_rows, optimize=True)

        if 'K' in calc:
            # Similar for Pij with C_p and C_p
            C_p_rows = C_p[rows, :]  # Reuse if already computed
            C_p_cols = C_p[cols, :]
            Pij_batch  = cp.einsum('kP,kp,kq->Ppq', eri3c_batch, C_p_rows, C_p_cols, optimize=True)
            Pij_batch += cp.einsum('kP,kp,kq->Ppq', eri3c_batch, C_p_cols, C_p_rows, optimize=True)
            Pij[p0:p1, :, :] = Pij_batch

            # For Pab with C_q and C_q
            C_q_rows = C_q[rows, :]
            C_q_cols = C_q[cols, :]
            Pab_batch = cp.einsum('kP,kp,kq->Ppq', eri3c_batch, C_q_rows, C_q_cols, optimize=True)
            Pab_batch += cp.einsum('kP,kp,kq->Ppq', eri3c_batch, C_q_cols, C_q_rows, optimize=True)
            Pab[p0:p1, :, :] = Pab_batch

            log.info(f'batch {p0}-{p1} / {naux} done')


    if in_ram:
        tmp_einsum = einsum2dot 
        upper_inv_eri2c = upper_inv_eri2c.get()

    else:
        tmp_einsum = contract

    if calc == 'J':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        # Tia = intopt.unsort_orbitals(Tia,aux_axis=[0])
        return Tia

    if calc == 'K':
        Tij = tmp_einsum('PQ,Qij->Pij', upper_inv_eri2c, Pij)
        Tab = tmp_einsum('PQ,Qab->Pab', upper_inv_eri2c, Pab)
        return Tij, Tab

    if calc == 'JK':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        Tij = tmp_einsum('PQ,Qij->Pij', upper_inv_eri2c, Pij)
        Tab = tmp_einsum('PQ,Qab->Pab', upper_inv_eri2c, Pab)
        return Tia, Tij, Tab

def get_Tpq(mol, auxmol, lower_inv_eri2c, C_p, C_q, 
           calc='JK',omega=None, alpha=None, beta=None,
           log=None, in_ram=True, single=True):
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

    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    nao, nao_orig = int3c2e_opt.coeff.shape
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
    rows, cols = divmod(ao_pair_mapping, nao_orig)
    print('ao_pair_mapping.shape', ao_pair_mapping.shape)


    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])

    siz_p = C_p.shape[1]
    siz_q = C_q.shape[1]


    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff
    naux = aux_coeff.shape[0]
    # print('aux_coeff.shape', aux_coeff.shape)

    xp = np if in_ram else cp
    log.info(f'xp {xp}')

    P_dtype = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    print('cp_int3c_dtype', cp_int3c_dtype)
    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.astype(cp_int3c_dtype, copy=False)
    upper_inv_eri2c = aux_coeff_lower_inv_eri2c.T

    if in_ram:
        aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
        upper_inv_eri2c           = upper_inv_eri2c.get()

    if 'J' in calc:
        Pia = xp.empty((naux, siz_p, siz_q), dtype=P_dtype)

    if 'K' in calc:
        '''only store lower triangle of Tij and Tab'''
        n_tri_p = (siz_p * (siz_p + 1)) // 2
        n_tri_q = (siz_q * (siz_q + 1)) // 2
        Pij = xp.empty((naux, n_tri_p), dtype=P_dtype)  
        Pab = xp.empty((naux, n_tri_q), dtype=P_dtype)  

    byte_eri3c = nao_orig * nao_orig * cp_int3c_dtype.itemsize
    pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()

    batch_size = 16
    log.info(f'eri3c per aux dimension will take {byte_eri3c / (1024 ** 2):.0f} MB memory')
    log.info(f'eri3c per aux batch will take {byte_eri3c * batch_size / (1024 ** 2):.0f} MB memory')
    p1 = 0

    log.info(get_memory_info('before int3c2e_bdiv_generator'))

    for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size):
        # print('eri3c_batch.shape', eri3c_batch.shape)
        # eri3c_batch = cp.asarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')

        eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
        eri3c_batch = eri3c_batch.astype(cp_int3c_dtype, copy=False)
        eri3c_batch = cp.asfortranarray(eri3c_batch)
        # if in_ram:
        #     eri3c_batch = eri3c_batch.get()
        # print('eri3c_orbital_pair_cart2sph', eri3c_batch.shape)
        aopair, aux_batch_size = eri3c_batch.shape
        p0, p1 = p1, p1 + eri3c_batch.shape[1]
        release_memory()
        log.info(get_memory_info('eri3c_unzip_batch'))
        eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, aux_batch_size), dtype=cp_int3c_dtype, order='F')

        eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
        eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch
        eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)

        '''Puv -> Ppq, AO->MO transform '''
        if 'J' in calc:
            # eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch
            # eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)
            Pia[p0:p1,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_p,C_q, in_ram=in_ram)

        if 'K' in calc:
            # eri3c_unzip_batch *= 2
            # eri3c_unzip_batch[pair_diag,:] *= 0.5
            # eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)

            Pij[p0:p1,:] = get_uvPCupCvq_to_Ppq_symmetry(eri3c_unzip_batch,C_p,C_p, in_ram=in_ram)
            Pab[p0:p1,:] = get_uvPCupCvq_to_Ppq_symmetry(eri3c_unzip_batch,C_q,C_q, in_ram=in_ram)
        
        log.info(f'batch {p0}-{p1} / {naux} done')

    if in_ram:
        tmp_einsum = einsum2dot 
        # upper_inv_eri2c = upper_inv_eri2c.get()
    else:
        tmp_einsum = contract

    if calc == 'J':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        return Tia

    if calc == 'K':
        Tij = (aux_coeff_lower_inv_eri2c.dot(upper_inv_eri2c)).dot(Pij)
        # Tab = upper_inv_eri2c.dot(Pab)
        return Tij, Pab

    if calc == 'JK':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        Tij = (aux_coeff_lower_inv_eri2c.dot(upper_inv_eri2c)).dot(Pij)
        # Tab = upper_inv_eri2c.dot(Pab)
        return Tia, Tij, Pab
     
def get_eri3c_bdiv(mol, auxmol, lower_inv_eri2c, 
                omega=None, alpha=None, beta=None,
                log=None, in_ram=True, single=True):

    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()
    # ao_idx = cp.asarray(int3c2e_opt.ao_idx)
    # print('ao_idx.shape', ao_idx.shape)

    nao, nao_orig = int3c2e_opt.coeff.shape
    print('nao, nao_orig', nao, nao_orig) # 220 208

    xp = np if in_ram else cp
    log.info(f'xp {xp}')
    int3c_dtype    = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    
    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff
    nauxao = aux_coeff.shape[0]
    # print('aux_coeff.shape', aux_coeff.shape)

    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.astype(cp_int3c_dtype, copy=False)
    aux_coeff_lower_inv_eri2c = cp.asfortranarray(aux_coeff_lower_inv_eri2c)
    # if in_ram:
    #     aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()

    # print('aux_coeff_lower_inv_eri2c.flag', aux_coeff_lower_inv_eri2c.flags)
    # have_holder = None
    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)

    naopair = ao_pair_mapping.shape[0]
    # eri3c = xp.empty((nauxao, naopair), dtype=int3c_dtype)
    eri3c = cp.empty((naopair, nauxao), dtype=int3c_dtype, order='F')

    byte_eri3c = naopair *nauxao * cp_int3c_dtype.itemsize

    log.info(f'eri3c shape {naopair, nauxao}, {byte_eri3c / (1024 ** 2):.0f} MB memory')
    log.info(f'compression rate: {naopair/(nao_orig*nao_orig):.4f}')
    p1 = 0
    for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=100):
        # print('eri3c_batch.shape', eri3c_batch.shape)
        cpu0 = log.init_timer()
        p0, p1 = p1, p1 + eri3c_batch.shape[1]

        eri3c_batch = cp.asarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
        eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)

        log.timer(' orbital_pair_cart2sph', *cpu0)

        cpu2 = log.init_timer()
        
        eri3c[:, p0:p1] = eri3c_batch

        log.info(f'batch {p0}-{p1} / {nauxao} done')
        log.timer(' eri3c_batch to holder', *cpu2)

    
    # eri3c = aux_coeff_lower_inv_eri2c.T.dot(eri3c)
    cpu3 = log.init_timer()
    eri3c = eri3c.dot(aux_coeff_lower_inv_eri2c)
    log.timer(' eri3c.dot(aux_coeff_lower_inv_eri2c)', *cpu3)

    if in_ram:
        cpu4 = log.init_timer()
        eri3c = eri3c.get()
        log.timer(' eri3c = eri3c.get()', *cpu4)
    return eri3c, int3c2e_opt

def get_int3c2e_opt(mol, auxmol, lower_inv_eri2c, C_p, C_q, 
           calc='JK', omega=None, alpha=None, beta=None,
           log=None, in_ram=True, single=True):

    mol.verbose -= 1
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()

    nao, nao_orig = int3c2e_opt.coeff.shape
    print('nao, nao_orig', nao, nao_orig) # 220 208

    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    
    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff

    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = cp.asarray(aux_coeff_lower_inv_eri2c, dtype=cp_int3c_dtype, order='F')
    if in_ram:
        aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
    return  int3c2e_opt, aux_coeff_lower_inv_eri2c

def gen_iajb_MVP_bdiv_backup(eri3c, int3c2e_opt, C_p, C_q, log): 
    '''
    (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
             = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    (ia|jb) in RKS

    V in shape (m, n_occ * n_vir)
    '''

    # def iajb_MVP(V):
    #     T_right_jb_V = einsum("Pjb,mjb->Pm", T_right, V)
    #     iajb_V = einsum("Pia,Pm->mia", T_left, T_right_jb_V)
    #     return iajb_V
    nao, nao_orig = int3c2e_opt.coeff.shape
    nauxao = eri3c.shape[1]

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    
    n_p = C_p.shape[1]

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=False)
    # print('ao_pair_mapping.shape', ao_pair_mapping.shape)

    rows, cols = divmod(ao_pair_mapping, nao_orig)

    def iajb_MVP(V, out):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V   (cupy.ndarray): Input tensor of shape (m, n_occ, n_vir).
            out (cupy.ndarray): output holder of shape (m, n_occ, n_vir).

        Returns:
            iajb_V (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''

        aux_batch_size = 10
        # print('iajb batchs', len(range(0, nauxao, aux_batch_size)))
        log.info(get_memory_info('  iajb_V before slicing aux')) 
        # Iterate over batchs of the auxao dimension
        for aux_start in range(0, nauxao, aux_batch_size):
            aux_end = min(aux_start + aux_batch_size, nauxao)
            tmp_aux_batch_size = aux_end - aux_start

            eri3c_batch = cp.asarray(eri3c[:, aux_start:aux_end], order='F')
            
            eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, tmp_aux_batch_size), order='F')

            eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
            eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch

            eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, tmp_aux_batch_size)

            tmp_Ppv = contract('uvP,up->Ppv', eri3c_unzip_batch, C_p)

            Tjb_batch = contract('Ppv,vq->Ppq', tmp_Ppv, C_q)
 
            Tjb_Vjb_batch = contract("Pjb,mjb->Pm", Tjb_batch, V)
            Tia_batch = Tjb_batch  
            
            out += 2*contract("Pia,Pm->mia", Tia_batch, Tjb_Vjb_batch)

            # Release intermediate variables and clean up memory, must!
            del eri3c_unzip_batch, Tjb_batch, Tia_batch, Tjb_Vjb_batch

            release_memory()

    log.info(get_memory_info('after generate iajb_MVP'))
    return iajb_MVP

def gen_iajb_MVP_bdiv_dev(int3c2e_opt, mol, auxmol, aux_coeff_lower_inv_eri2c, C_p, C_q, log,  single): 
    '''
    (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
             = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    (ia|jb) in RKS

    V in shape (m, n_occ * n_vir)
    '''

    # def iajb_MVP(V):
    #     T_right_jb_V = einsum("Pjb,mjb->Pm", T_right, V)
    #     iajb_V = einsum("Pia,Pm->mia", T_left, T_right_jb_V)
    #     return iajb_V
    nao, nao_orig = int3c2e_opt.coeff.shape

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    
    # siz_p = C_p.shape[1]
    # siz_q = C_q.shape[1]

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=False)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    def iajb_MVP(V, out):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V   (cupy.ndarray): Input tensor of shape (m, n_occ, n_vir).
            out (cupy.ndarray): output holder of shape (m, n_occ, n_vir).

        Returns:
            iajb_V (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = V.shape

        log.info(get_memory_info('  iajb_V before slicing aux')) 

        dms = contract('ui,mia->mua', C_p, V)
        dms = contract('va,mua->muv', C_q, dms)

        pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()
        dm_sparse  = dms[:, pair_rows, pair_cols]
        dm_sparse += dms[:, pair_cols, pair_rows]
        dm_sparse[:,pair_diag] *= 0.5

        aux_offset = 0

        nauxao = int3c2e_opt.aux_coeff.shape[0]
        T_right = cp.empty((n_state, nauxao))
        cpu0 = log.init_timer()

        p1 = 0
        for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=50):
            cpu1 = log.init_timer()
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            eri3c_batch = cp.asarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
            # aopair, aux_batch_size = eri3c_batch.shape

            p0, p1 = p1, p1 + eri3c_batch.shape[1]
            T_right[:,p0:p1] = dm_sparse.dot(eri3c_batch)   #  (nstate, npair)(npair|aux_batch_size)-> (nstate, aux_batch_size)
            # aux_offset += aux_batch_size
            # log.debug('eri3c_batch', *cpu1)

        log.timer('T_right', *cpu0)

        # debug = False
        # # debug = True
        # if debug:
        #     T_right_bug = T_right.copy()
        #     eri3c = next(int3c2e_opt.int3c2e_bdiv_generator(batch_size=50))
        #     eri3c = int3c2e_opt.orbital_pair_cart2sph(eri3c, inplace=True)

        #     aopair, nauxao = eri3c.shape
        #     eri3c_unzip = cp.zeros((nao_orig*nao_orig, nauxao), order='F')

        #     eri3c_unzip[ao_pair_mapping,   :] = eri3c
        #     eri3c_unzip[cols*nao_orig+rows,:] = eri3c
        #     eri3c_unzip = eri3c_unzip.reshape(nao_orig, nao_orig, nauxao)
        #     # T_right=cp.einsum('uvP, ui, va, mia->mP', eri3c_unzip, C_p, C_q, V)
        #     ref=cp.einsum('uvP, muv->mP', eri3c_unzip, dms.reshape(n_state, nao_orig, nao_orig)) 
            
        #     dm_sparse_new = cp.zeros((n_state, nao_orig*nao_orig))
        #     dms = dms.reshape(n_state, nao_orig*nao_orig)
        #     dm_sparse_new[:, ao_pair_mapping   ] = dms[:,ao_pair_mapping   ]
        #     dm_sparse_new[:, cols*nao_orig+rows] = dms[:,cols*nao_orig+rows]
        #     dm_sparse_new = dm_sparse_new.reshape(n_state, nao_orig, nao_orig)
            
        #     T_right=cp.einsum('uvP, muv->mP', eri3c_unzip, dm_sparse_new) 
        #     # T_direct_dm_sparse = dm_sparse.dot(eri3c)
        #     T_direct_dm_sparse = cp.einsum('mu, uP -> mP', dm_sparse, eri3c)

        #     print('check T_right', cp.linalg.norm(T_right-ref)) # 0.00 correct 
        #     print('check T_direct_dm_sparse', cp.linalg.norm(T_right-T_direct_dm_sparse)) # 0.18 wrong
        #     print('check T_right_bug', cp.linalg.norm(T_right-T_right_bug)) #
            
        T_right = contract('QP,mP->Qm', aux_coeff_lower_inv_eri2c.dot(aux_coeff_lower_inv_eri2c.T), T_right)  

        #(uv|m) (naopair,m)
        T_left = cp.zeros((len(ao_pair_mapping), n_state))
        p1 = 0
        for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=48):
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            p0, p1 = p1, p1 + eri3c_batch.shape[1]
            T_left += eri3c_batch.dot(T_right[p0:p1, :])
            release_memory()

        # rows, cols = divmod(ao_pair_mapping, nao_orig)

        for i in range(n_state):
            #(uv|m)
            J = cp.zeros((nao_orig,nao_orig))
            J[rows,cols] = J[cols,rows] = T_left[:,i]
            # J = J.reshape((nao_orig,nao_orig))
            # print('C_p.T.dot(J).dot(C_q).shape', (C_p.T.dot(J).dot(C_q)).shape)
            out[i,:,:] += 2* C_p.T.dot(J).dot(C_q)

        # for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=16):
        #     eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)

        #     aopair, aux_batch_size = eri3c_batch.shape
        #     # eri3c_batch = eri3c_batch.dot()
        #     eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, aux_batch_size), order='F')

        #     eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
        #     eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch

        #     eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)
            
        #     '''uvP -> Pia, AO->MO transform '''
        #     Pia_tmp = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_p,C_q, in_ram=False) 
        #     # print('Pia_tmp.shape', Pia_tmp.shape)
        #     out[:,:,:] += 2*contract('Pia,Pm->mia', Pia_tmp, T_right[aux_offset:aux_offset+aux_batch_size, :])           
        #     aux_offset += aux_batch_size
        #     release_memory()

    log.info(get_memory_info('after generate iajb_MVP'))
    return iajb_MVP

def gen_iajb_MVP_bdiv(int3c2e_opt, aux_coeff_lower_inv_eri2c, C_p, C_q, log,  single):  
    '''
    (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
             = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    (ia|jb) in RKS

    V in shape (m, n_occ * n_vir)
    '''

    # def iajb_MVP(V):
    #     T_right_jb_V = einsum("Pjb,mjb->Pm", T_right, V)
    #     iajb_V = einsum("Pia,Pm->mia", T_left, T_right_jb_V)
    #     return iajb_V
    nao, nao_orig = int3c2e_opt.coeff.shape

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=False)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    def iajb_MVP(V, out=None):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V   (cupy.ndarray): Input tensor of shape (m, n_occ, n_vir).
            out (cupy.ndarray): output holder of shape (m, n_occ, n_vir).

        Returns:
            iajb_V (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = V.shape
        if out is None: 
            out = cp.zeros_like(V)

        log.info(get_memory_info('  iajb_V before slicing aux')) 

        dms = contract('ui,mia->mua', C_p, V)
        dms = contract('va,mua->muv', C_q, dms)

        pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()
        dm_sparse  = dms[:, pair_rows, pair_cols]
        dm_sparse += dms[:, pair_cols, pair_rows]
        dm_sparse[:,pair_diag] *= 0.5

        aux_offset = 0

        nauxao = int3c2e_opt.aux_coeff.shape[0]
        T_right = cp.empty((n_state, nauxao))
        # cpu0 = log.init_timer()

        for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=50):
            cpu1 = log.init_timer()
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            eri3c_batch = cp.asarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
            aopair, aux_batch_size = eri3c_batch.shape
            T_right[:,aux_offset:aux_offset+aux_batch_size] = dm_sparse.dot(eri3c_batch)   #  (nstate, npair)(npair|aux_batch_size)-> (nstate, aux_batch_size)
            aux_offset += aux_batch_size
            # log.debug('eri3c_batch', *cpu1)

        # log.timer('T_right', *cpu0)
            
        T_right = contract('QP,mP->Qm', aux_coeff_lower_inv_eri2c.dot(aux_coeff_lower_inv_eri2c.T), T_right)  

        #(uv|m) (naopair,m)
        T_left = cp.zeros((len(ao_pair_mapping), n_state))
        p1 = 0
        for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=48):
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            p0, p1 = p1, p1 + eri3c_batch.shape[1]
            T_left += eri3c_batch.dot(T_right[p0:p1, :])
            release_memory()

        for i in range(n_state):
            #(uv|m)
            J = cp.zeros((nao_orig,nao_orig))
            J[rows,cols] = J[cols,rows] = T_left[:,i]
            out[i,:,:] += 2* (C_p.T.dot(J)).dot(C_q)

        return out
    log.info(get_memory_info('after generate iajb_MVP'))
    return iajb_MVP

def get_eri2c_inv_lower(auxmol, omega=0, alpha=None, beta=None, dtype=cp.float64):

    eri2c = auxmol.intor('int2c2e')

    if omega and omega != 0:

        with auxmol.with_range_coulomb(omega):
            eri2c_erf = auxmol.intor('int2c2e')

        eri2c = alpha * eri2c + beta * eri2c_erf

    eri2c = cp.asarray(eri2c)

    try:
        ''' we want lower_inv_eri2c = X
                X X.T = eri2c^-1
                (X X.T)^-1 = eri2c
                (X.T)^-1 X^-1 = eri2c = L L.T
                (X.T)^-1 = L
        need to solve  L_inv = L^-1
                X = L_inv.T
           
        '''
        L = cp.linalg.cholesky(eri2c)
        L_inv = cpx_linalg.solve_triangular(L, cp.eye(L.shape[0]), lower=True)
        lower_inv_eri2c = L_inv.T

    except cp.linalg.LinAlgError:
        ''' lower_inv_eri2c = eri2c ** -0.5
            LINEAR_EPSILON = 1e-8 to remove the linear dependency, sometimes the aux eri2c is not full rank.
        '''
        lower_inv_eri2c = math_helper.matrix_power(eri2c,-0.5,epsilon=LINEAR_EPSILON)

    lower_inv_eri2c = cp.asarray(lower_inv_eri2c, dtype=dtype, order='C')
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

def gen_iajb_MVP(T_ia):
    '''
    (ia|jb)V = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
             = Σ_P [ T_left_ia^P Σ_jb(T_right_jb^P V_jb^m) ]
    (ia|jb) in RKS

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
        # Get the shape of the tensors
        nauxao, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape
        # Initialize result tensor
        iajb_V = cp.zeros((n_state, n_occ, n_vir), dtype=V.dtype)

        # 1 denotes one auxao, we are slucing the auxao dimension.
        n_Tia_chunk = 1 * n_occ * n_vir
        n_TjbVjb_chunk = 1 * n_state
        n_iajb_V_chunk = n_state * n_occ * n_vir 

        estimated_chunk_size_bytes = (n_Tia_chunk + n_TjbVjb_chunk + n_iajb_V_chunk) * T_ia.itemsize 

        # Estimate the optimal chunk size based on available GPU memory
        aux_batch_size = int(get_avail_mem() * 0.8 // estimated_chunk_size_bytes)

        # Ensure the chunk size is at least 1 and doesn't exceed the total number of auxao
        aux_batch_size = max(1, min(nauxao, aux_batch_size))
        # print('iajb chunks', len(range(0, nauxao, aux_batch_size)))
        # print(get_memory_info('  iajb_V before slicing aux')) 
        # Iterate over chunks of the auxao dimension
        for aux_start in range(0, nauxao, aux_batch_size):
            aux_end = min(aux_start + aux_batch_size, nauxao)

            Tjb_chunk = cp.asarray(T_ia[aux_start:aux_end, :, :])   # Shape: (aux_range, n_occ * n_vir)
            Tjb_Vjb_chunk = contract("Pjb,mjb->Pm", Tjb_chunk, V)

            Tia_chunk = Tjb_chunk  # Shape: (aux_range, n_occ, n_vir)
            iajb_V += contract("Pia,Pm->mia", Tia_chunk, Tjb_Vjb_chunk)

            # Release intermediate variables and clean up memory, must!
            del Tjb_chunk, Tia_chunk, Tjb_Vjb_chunk
            release_memory()

        return iajb_V

    return iajb_MVP

def gen_ijab_MVP(T_ij, T_ab):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    # def ijab_MVP(V):
    #     T_ab_V = contract("Pab,mjb->Pamj", T_ab, V)
    #     ijab_V = contract("Pij,Pamj->mia", T_ij, T_ab_V)
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
        T_ij_gpu = cp.asarray(T_ij) # if T_ij was in RAM, upload to GPU on calling
        nauxao, n_occ, n_occ = T_ij.shape

        nauxao, n_vir, n_vir = T_ab.shape  # Dimensions of T_ab
        n_state, n_occ, n_vir = V.shape      # Dimensions of V

        # Initialize result tensor
        ijab_V = cp.empty((n_state, n_occ, n_vir), dtype=T_ab.dtype)

        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_mem()

        # 1 denotes one vir MO, we are slucing the n_vir dimension.
        n_T_ab_chunk = nauxao * 1 * n_vir
        n_T_ab_V_chunk = nauxao * 1 * n_state * n_occ 
        n_ijab_V_chunk = n_state * n_occ * 1

        bytes_per_vir = 2*( n_T_ab_chunk + n_T_ab_V_chunk + n_ijab_V_chunk) * T_ab.itemsize  
        # print('available_gpu_memory', available_gpu_memory)
        # print('bytes_per_vir', bytes_per_vir)
        vir_chunk_size = max(1, int(available_gpu_memory * 0.4 // bytes_per_vir)) 

        # print(get_memory_info('  ijab_V before slicing vir')) 
        # print('vir_chunk_size', vir_chunk_size)
            
        # print('chuncks', len(range(0, n_vir, vir_chunk_size)))

        # Iterate over chunks of the n_vir dimension
        # i = 0
        for vir_start in range(0, n_vir, vir_chunk_size):
            # print(' vir chunk', i,  available_gpu_memory)
            # i += 1

            vir_end = min(vir_start + vir_chunk_size, n_vir)
            # vir_range = vir_end - vir_start

            # Extract the corresponding chunk of T_ab
            T_ab_chunk = T_ab[:, vir_start:vir_end, :]  # Shape: (nauxao, vir_range, n_vir)

            # Compute T_ab_V for the current chunk
            T_ab_chunk_V = contract("Pab,mjb->Pamj", T_ab_chunk, V)

            # Compute ijab_V for the current chunk
            ijab_V[:, :, vir_start:vir_end] = contract("Pij,Pamj->mia", T_ij_gpu, T_ab_chunk_V)

            # Release intermediate variables and clean up memory, must!
            del T_ab_chunk, T_ab_chunk_V
            release_memory()

        del T_ij_gpu

        return ijab_V
        

    return ijab_MVP

def gen_ijab_MVP_out_non_symmetry(T_ij, T_ab, log):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    # def ijab_MVP(V):
    #     T_ab_V = contract("Pab,mjb->Pamj", T_ab, V)
    #     ijab_V = contract("Pij,Pamj->mia", T_ij, T_ab_V)
    #     return ijab_V

    def ijab_MVP(V, a_x, out):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
        by chunking along the n_vir dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        T_ij_gpu = cp.asarray(T_ij) # if T_ij was in RAM, upload to GPU on calling
        nauxao, n_occ, n_occ = T_ij.shape

        nauxao, n_vir, n_vir = T_ab.shape  # Dimensions of T_ab
        n_state, n_occ, n_vir = V.shape    # Dimensions of V

        # Initialize result tensor
        if out is None:
            ijab_V = cp.empty((n_state, n_occ, n_vir), dtype=T_ab.dtype)
        # else:
        #     out = out.reshape(n_state, n_occ, n_vir)
        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_mem()

        # 1 denotes one vir MO, we are slucing the n_vir dimension.
        n_T_ab_chunk = nauxao * 1 * n_vir
        n_T_ab_V_chunk = nauxao * 1 * n_state * n_occ 
        n_ijab_V_chunk = n_state * n_occ * 1

        bytes_per_vir = 2*( n_T_ab_chunk + n_T_ab_V_chunk + n_ijab_V_chunk) * T_ab.itemsize  
        # print('available_gpu_memory', available_gpu_memory)
        vir_chunk_size = max(1, int(available_gpu_memory * 0.4 // bytes_per_vir)) 

        # Iterate over chunks of the n_vir dimension
        # i = 0
        for vir_start in range(0, n_vir, vir_chunk_size):
            # print(' vir chunk', i,  available_gpu_memory)
            # i += 1

            vir_end = min(vir_start + vir_chunk_size, n_vir)
            # vir_range = vir_end - vir_start

            # Extract the corresponding chunk of T_ab
            T_ab_chunk = T_ab[:, vir_start:vir_end, :]  # Shape: (nauxao, vir_range, n_vir)

            # Compute T_ab_V for the current chunk
            T_ab_chunk_V = contract("Pab,mjb->Pamj", T_ab_chunk, V)

            # Compute ijab_V for the current chunk
            out[:, :, vir_start:vir_end] -= a_x*contract("Pij,Pamj->mia", T_ij_gpu, T_ab_chunk_V)

            # Release intermediate variables and clean up memory, must!
            del T_ab_chunk, T_ab_chunk_V
            release_memory()

        del T_ij_gpu

    log.info(get_memory_info('after generate ijab_MVP'))    
    return ijab_MVP

def unpack_symmetric(T_packed):
    '''
    Unpack a packed lower triangle to full symmetric matrix.
    
    Args:
        T_packed (cp.ndarray): Packed tensor of shape (nauxao, n_tri).
        n_dim (int): Dimension of the square matrix (n_occ or n_vir).
    
    Returns:
        cp.ndarray: Full symmetric matrix of shape (nauxao, n_dim, n_dim).
    '''

    pairs = T_packed.shape[1]
    n_dim = (-1 + cp.sqrt(1 + 8 * pairs))  / 2
    n_dim = int(n_dim)
    tril_i, tril_j = cp.tril_indices(n_dim)
    T_full = cp.zeros((T_packed.shape[0], n_dim, n_dim), dtype=T_packed.dtype)
    T_full[:, tril_i, tril_j] = T_packed
    T_full[:, tril_j, tril_i] = T_packed  # Fill upper triangle using symmetry (diagonal is duplicated but same)
    return T_full

def get_tri_indices(rows, cols):
    '''
    Vectorized computation of triangular indices for symmetric matrix (lower triangle, i >= j).
    
    Args:
        rows (np.ndarray): Array of row indices (shape can be broadcastable).
        cols (np.ndarray): Array of column indices (shape can be broadcastable).
        n_dim (int): Dimension of the square matrix (n_vir or n_occ).
    
    Returns:
        np.ndarray: Flat indices for the packed lower triangle.
    '''
    i = np.maximum(rows, cols)
    j = np.minimum(rows, cols)
    indices = i * (i + 1) // 2 + j
    return indices

def gen_ijab_MVP_out(T_ij, T_ab, log):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    # def ijab_MVP(V):
    #     T_ab_V = contract("Pab,mjb->Pamj", T_ab, V)
    #     ijab_V = contract("Pij,Pamj->mia", T_ij, T_ab_V)
    #     return ijab_V

    def ijab_MVP(V, a_x, out):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
        by chunking along the n_vir dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        T_ij_gpu = cp.asarray(T_ij) # if T_ij was in RAM, upload to GPU on calling
        nauxao = T_ij_gpu.shape[0]
        
        T_ij_gpu = unpack_symmetric(T_ij_gpu)
        # nauxao, ab_pairs = T_ab.shape  # Dimensions of T_ab

        n_state, n_occ, n_vir = V.shape    # Dimensions of V

        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_mem()

        # 1 denotes one vir MO, we are slucing the n_vir dimension.
        n_T_ab_chunk = nauxao * 1 * n_vir
        n_T_ab_V_chunk = nauxao * 1 * n_state * n_occ 
        n_ijab_V_chunk = n_state * n_occ * 1

        bytes_per_vir = 2*( n_T_ab_chunk + n_T_ab_V_chunk + n_ijab_V_chunk) * T_ab.itemsize  
        # print('available_gpu_memory', available_gpu_memory)
        vir_chunk_size = max(1, int(available_gpu_memory * 0.4 // bytes_per_vir)) 
        log.info(f'{int(n_vir/vir_chunk_size)} vir chunks')
        # Iterate over chunks of the n_vir dimension
        # i = 0
        for vir_start in range(0, n_vir, vir_chunk_size):
            # print(' vir chunk', i,  available_gpu_memory)
            # i += 1

            vir_end = min(vir_start + vir_chunk_size, n_vir)
            # vir_range = vir_end - vir_start

            # Compute triangular indices for this chunk: a in [vir_start, vir_end), b in [0, n_vir)
            a_vals = np.arange(vir_start, vir_end)[:, np.newaxis]  # (vir_range, 1)
            b_vals = np.arange(n_vir)[np.newaxis, :]  # (1, n_vir)
            indices = get_tri_indices(a_vals, b_vals)  # (vir_range, n_vir)

            # Extract the corresponding chunk of T_ab
            T_ab_chunk = T_ab[:, indices]  # Shape: (nauxao, vir_range, n_vir)
            # T_ab_chunk = unpack_symmetric(T_ab_chunk)
            # Compute T_ab_V for the current chunk
            T_ab_chunk_V = contract("Pab,mjb->Pamj", cp.asarray(T_ab_chunk), V)

            # Compute ijab_V for the current chunk
            out[:, :, vir_start:vir_end] -= a_x*contract("Pij,Pamj->mia", T_ij_gpu, T_ab_chunk_V)

            # Release intermediate variables and clean up memory, must!
            del T_ab_chunk, T_ab_chunk_V
            release_memory()

        del T_ij_gpu

    log.info(get_memory_info('after generate ijab_MVP'))    
    return ijab_MVP

def gen_ijab_MVP_eri3c(eri3c, int3c2e_opt, C_p, C_q, log, single):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    # def ijab_MVP(V):
    #     T_ab_V = contract("Pab,mjb->Pamj", T_ab, V)
    #     ijab_V = contract("Pij,Pamj->mia", T_ij, T_ab_V)
    #     return ijab_V
    nao, nao_orig = int3c2e_opt.coeff.shape

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=False)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    def ijab_MVP(V, a_x, out=None):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = V.shape
        if out is None: 
            out = cp.zeros_like(V)
        nauxao = int3c2e_opt.aux_coeff.shape[0]
        
        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_mem()

        batch_size = 16
        for p0 in range(0, nauxao, batch_size):
            p1 = min(p0+batch_size, nauxao)
            log.info(f'p0, p1 {p0,p1}')

            cpu = log.init_timer()
            eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, p1-p0), dtype=cp_int3c_dtype, order='F')
            log.timer(' eri3c_unzip_batch', *cpu)
            cpu = log.init_timer()

            eri3c_batch = cp.asarray(eri3c[:, p0:p1])
            log.timer(' cp.asarray(eri3c[:, p0:p1])', *cpu)
            cpu = log.init_timer()

            eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
            log.timer(' eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch', *cpu)
            cpu = log.init_timer()

            eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch
            log.timer(' eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch', *cpu)
            cpu = log.init_timer()
            eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, p1-p0)
            # print('eri3c_unzip_batch.shape C_q.shape', eri3c_unzip_batch.shape, C_q.shape)
            T_ab = contract('uvP,ua->avP', eri3c_unzip_batch, C_q)
            log.timer(' uvP,ua->avP', *cpu)
            cpu = log.init_timer()

            T_ab = contract('avP,vb->Pab', T_ab, C_q)
            log.timer(' avP,vb->Pab', *cpu)
            cpu = log.init_timer()

            # print('T_ab.shape V.shape', T_ab.shape, V.shape)
            T_ab_V = contract('Pab,mjb->mPja', T_ab, V)
            log.timer(' Pab,mjb->mPja', *cpu)
            cpu = log.init_timer()

            T_ij = contract('uvP,ui->ivP', eri3c_unzip_batch, C_p)
            log.timer(' uvP,ui->ivP', *cpu)
            cpu = log.init_timer()

            T_ij = contract('ivP,vj->Pij', T_ij, C_p)
            log.timer(' ivP,vj->Pij', *cpu)
            cpu = log.init_timer()          

            out += -a_x*contract('Pij,mPja->mia', T_ij, T_ab_V)
            log.timer(' Pij,mPja->mia', *cpu)
            cpu = log.init_timer()  
            release_memory()

        return out

    log.info(get_memory_info('after generate ijab_MVP'))    
    return ijab_MVP


def gen_ijab_MVP_eri3c_bad(eri3c, int3c2e_opt, C_p, C_q, log, single):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''

    # def ijab_MVP(V):
    #     T_ab_V = contract("Pab,mjb->Pamj", T_ab, V)
    #     ijab_V = contract("Pij,Pamj->mia", T_ij, T_ab_V)
    #     return ijab_V
    nao, nao_orig = int3c2e_opt.coeff.shape

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=False)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    def ijab_MVP(V, a_x, out=None):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
        using vectorized sparse AO pair indexing from orbital_pair_nonzero_indices,
        leveraging UV symmetry and GPU parallelism. Uses pair_rows == pair_cols for diagonal pairs.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = V.shape
        if out is None: 
            out = cp.zeros_like(V)
        nauxao = int3c2e_opt.aux_coeff.shape[0]
        nao_orig = int3c2e_opt.coeff.shape[1]  # Assuming coeff.shape = (naux, nao)
        
        # Get sparse pair indices
        pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()
        n_pairs = len(pair_rows)
        pair_rows = cp.asarray(pair_rows)  # (n_pairs,)
        pair_cols = cp.asarray(pair_cols)  # (n_pairs,)
        pair_diag = cp.asarray(pair_diag)  # Indices of diagonal pairs (u == v)
        
        # Debug: Check pair_diag
        log.info(f'n_pairs: {n_pairs}, nao_orig: {nao_orig}, n_vir: {n_vir}, n_occ: {n_occ}, n_diag: {len(pair_diag)}')
        if len(pair_diag) > 0:
            diag_rows = pair_rows[pair_diag].get()
            diag_cols = pair_cols[pair_diag].get()
            log.info(f'pair_diag sample: rows={diag_rows[:5]}, cols={diag_cols[:5]}')
            # Check which indices violate u == v
            violations = cp.where(pair_rows[pair_diag] != pair_cols[pair_diag])[0].get()
            if len(violations) > 0:
                log.info(f'Violations (u != v) at pair_diag indices: {violations[:5]}')
                log.info(f'Example violations: rows={diag_rows[violations[:5]]}, cols={diag_cols[violations[:5]]}')
        
        # Compute diagonal indices independently
        diag_mask = pair_rows == pair_cols  # Boolean mask for u == v
        n_diag = cp.sum(diag_mask)
        log.info(f'Computed n_diag (u == v): {n_diag}')
        
        batch_size = 16
        pair_batch_size = 10000  # Adjust based on GPU memory
        
        for p0 in range(0, nauxao, batch_size):
            p1 = min(p0 + batch_size, nauxao)
            log.info(f'p0, p1: {p0, p1}')

            # Sparse eri3c_batch: (n_pairs, batch_size)
            eri3c_batch = cp.asarray(eri3c[:, p0:p1], dtype=cp_int3c_dtype)
            log.info(f'eri3c_batch norm: {cp.linalg.norm(eri3c_batch)}')

            # Initialize temporary dense tensors
            T_avP = cp.zeros((n_vir, nao_orig, p1 - p0), dtype=cp_int3c_dtype)
            T_ivP = cp.zeros((n_occ, nao_orig, p1 - p0), dtype=cp_int3c_dtype)

            # Process pairs in blocks
            for pair_start in range(0, n_pairs, pair_batch_size):
                pair_end = min(pair_start + pair_batch_size, n_pairs)
                n_pairs_block = pair_end - pair_start
                rows_batch = pair_rows[pair_start:pair_end]  # (n_pairs_block,)
                cols_batch = pair_cols[pair_start:pair_end]  # (n_pairs_block,)
                diag_batch = diag_mask[pair_start:pair_end]  # (n_pairs_block,) boolean
                eri_batch = eri3c_batch[pair_start:pair_end, :]  # (n_pairs_block, batch_size)

                # Extract coefficients
                C_q_rows = C_q[rows_batch, :]  # (n_pairs_block, n_vir)
                C_q_cols = C_q[cols_batch, :]  # (n_pairs_block, n_vir)
                C_p_rows = C_p[rows_batch, :]  # (n_pairs_block, n_occ)
                C_p_cols = C_p[cols_batch, :]  # (n_pairs_block, n_occ)

                # Compute contributions for T_avP: T_avP[a, v, P] += C_q[u, a] * eri[u,v,P] (v = cols_batch[x])
                contrib_avP = cp.einsum('xa,xp->ap', C_q_rows, eri_batch)  # (n_vir, batch_size)
                for p in range(p1 - p0):
                    T_avP[:, cols_batch, p] += contrib_avP[:, p][:, cp.newaxis]
 
                # Symmetric contribution for non-diagonal pairs: T_avP[a, u, P] += C_q[v, a] * eri[v,u,P]
                non_diag = ~diag_batch
                if cp.any(non_diag):
                    contrib_avP_sym = cp.einsum('xa,xp->ap', C_q_cols[non_diag], eri_batch[non_diag])
                    for p in range(p1 - p0):
                        T_avP[:, rows_batch[non_diag], p] += contrib_avP_sym[:, p][:, cp.newaxis]

                # Compute contributions for T_ivP: T_ivP[i, v, P] += C_p[u, i] * eri[u,v,P]
                contrib_ivP = cp.einsum('xi,xp->ip', C_p_rows, eri_batch)
                for p in range(p1 - p0):
                    T_ivP[:, cols_batch, p] += contrib_ivP[:, p][:, cp.newaxis]

                # Symmetric contribution for non-diagonal pairs
                if cp.any(non_diag):
                    contrib_ivP_sym = cp.einsum('xi,xp->ip', C_p_cols[non_diag], eri_batch[non_diag])
                    for p in range(p1 - p0):
                        T_ivP[:, rows_batch[non_diag], p] += contrib_ivP_sym[:, p][:, cp.newaxis]

            # Debug: Check non-zero elements and norms
            log.info(f'T_avP non-zeros: {cp.count_nonzero(T_avP)}, T_ivP non-zeros: {cp.count_nonzero(T_ivP)}')
            log.info(f'T_avP norm: {cp.linalg.norm(T_avP)}, T_ivP norm: {cp.linalg.norm(T_ivP)}')

            # Contract to get T_ab (P,a,b) and T_ij (P,i,j)
            T_ab = contract('avP,vb->Pab', T_avP, C_q)
            T_ij = contract('ivP,vj->Pij', T_ivP, C_p)

            # Debug: Check shapes and norms
            log.info(f'T_ab shape: {T_ab.shape}, T_ij shape: {T_ij.shape}')
            log.info(f'T_ab norm: {cp.linalg.norm(T_ab)}, T_ij norm: {cp.linalg.norm(T_ij)}')

            # Rest unchanged
            T_ab_V = contract('Pab,mjb->mPja', T_ab, V)
            out += -a_x * contract('Pij,mPja->mia', T_ij, T_ab_V)

            # Release memory
            del T_avP, T_ivP, T_ab, T_ij, T_ab_V
            release_memory()

        # Debug: Check final output norm
        log.info(f'out norm: {cp.linalg.norm(out)}')
        return out
    log.info(get_memory_info('after generate ijab_MVP'))    
    return ijab_MVP

def gen_ibja_MVP(T_ia):
    '''
    the exchange (ib|ja) in B matrix
    (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
            = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]
    '''
    # def ibja_MVP(V):
    #     T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
    #     ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)
    #     return ibja_V

    def ibja_MVP(V):
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

        available_gpu_memory = get_avail_mem()

        n_T_ib_V_chunk = nauxao * n_occ * n_state * 1

        bytes_per_vir = 2 * n_T_ib_V_chunk * T_ia.itemsize 

        occ_chunk_size = max(1, int(available_gpu_memory * 0.8 // bytes_per_vir)) 

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
            del V_chunk, T_ia_chunk, T_ib_V_chunk
            release_memory()

        return ibja_V

    return ibja_MVP

def get_ab(td, mf, J_fit, K_fit, theta, mo_energy=None, mo_coeff=None, mo_occ=None, singlet=True):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454
    '''

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = cp.where(mo_occ==2)[0]
    viridx = cp.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = cp.hstack((orbo,orbv))

    e_ia = mo_energy[viridx] - mo_energy[occidx,None]
    a = cp.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = cp.zeros_like(a)
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    auxmol_J = get_auxmol(mol=mol, theta=theta, fitting_basis=J_fit)
    if K_fit == J_fit and (omega == 0 or omega is None):
        auxmol_K = auxmol_J
    else:
        auxmol_K = get_auxmol(mol=mol, theta=theta, fitting_basis=K_fit)

    def get_erimo(auxmol_i):
        naux = auxmol_i.nao
        int3c = int3c2e.get_int3c2e(mol, auxmol_i)
        int2c2e = auxmol_i.intor('int2c2e')
        int3c = cp.asarray(int3c)
        int2c2e = cp.asarray(int2c2e)
        df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def get_erimo_omega(auxmol_i, omega):
        naux = auxmol_i.nao
        int3c = int3c2e.get_int3c2e(mol, auxmol_i, omega=omega)
        with auxmol_i.with_range_coulomb(omega):
            int2c2e = auxmol_i.intor('int2c2e')
        int3c = cp.asarray(int3c)
        int2c2e = cp.asarray(int2c2e)
        df_coef = cp.linalg.solve(int2c2e, int3c.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        eri = contract('ijP,Pkl->ijkl', int3c, df_coef)
        eri_mo = contract('pjkl,pi->ijkl', eri, orbo.conj())
        eri_mo = contract('ipkl,pj->ijkl', eri_mo, mo)
        eri_mo = contract('ijpl,pk->ijkl', eri_mo, mo.conj())
        eri_mo = contract('ijkp,pl->ijkl', eri_mo, mo)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        return eri_mo
    def add_hf_(a, b, hyb=1):
        eri_mo_J = get_erimo(auxmol_J)
        eri_mo_K = get_erimo(auxmol_K)
        if singlet:
            a += cp.einsum('iabj->iajb', eri_mo_J[:nocc,nocc:,nocc:,:nocc]) * 2
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b += cp.einsum('iajb->iajb', eri_mo_J[:nocc,nocc:,:nocc,nocc:]) * 2
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb
        else:
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * hyb
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(td, 'with_solvent', None):
        raise NotImplementedError("PCM TDDFT RIS is not supported")

    if isinstance(mf, scf.hf.KohnShamDFT):
        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            eri_mo_K = get_erimo_omega(auxmol_K, omega)
            k_fac = alpha - hyb
            a -= cp.einsum('ijba->iajb', eri_mo_K[:nocc,:nocc,nocc:,nocc:]) * k_fac
            b -= cp.einsum('jaib->iajb', eri_mo_K[:nocc,nocc:,:nocc,nocc:]) * k_fac

        if mf.do_nlc():
            raise NotImplementedError('vv10 nlc not implemented in get_ab(). '
                                      'However the nlc contribution is small in TDDFT, '
                                      'so feel free to take the risk and comment out this line.')
    else:
        add_hf_(a, b)

    return a.get(), b.get()

def rescale_spin_free_amplitudes(xy, state_id):
    '''
    Rescales spin-free excitation amplitudes in TDDFT-ris to the normalization
    convention used in standard RKS-TDDFT.

    The original RKS-TDDFT formulation uses excitation amplitudes corresponding to
    the spin-up components only. The TDDFT-RIS implementation employs spin-free
    amplitudes that are not equivalent to the spin-up components and are
    normalized to 1.
    '''
    x, y = xy
    x = x[state_id] * .5**.5
    if y is not None: # TDDFT
        y = y[state_id] * .5**.5
    else: # TDA
        y = cp.zeros_like(x)
    return x, y

def as_scanner(td):
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)
    name = td.__class__.__name__ + TD_Scanner.__name_mixin__
    return lib.set_class(TD_Scanner(td), (TD_Scanner, td.__class__), name)


class TD_Scanner(lib.SinglePointScanner):
    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        assert self.device == 'gpu'
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        
        self.reset(mol)
        mf_scanner = self._scf
        mf_e = mf_scanner(mol)
        self.n_occ = None
        self.n_vir = None
        self.rest_occ = None
        self.rest_vir = None
        self.C_occ_notrunc = None
        self.C_vir_notrunc = None
        self.C_occ_Ktrunc = None
        self.C_vir_Ktrunc = None
        self.delta_hdiag = None
        self.hdiag = None
        self.eri_tag = None
        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None
        self.RKS = True
        self.UKS = False
        self.mo_coeff = cp.asarray(self._scf.mo_coeff, dtype=self.dtype)
        self.build()
        self.kernel()
        return mf_e + self.energies/HARTREE2EV


class RisBase(lib.StreamObject):
    def __init__(self, mf,  
                theta: float = 0.2, J_fit: str = 'sp', K_fit: str = 's', 
                Ktrunc: float = 40.0, a_x: float = None, omega: float = None, 
                alpha: float = None, beta: float = None, conv_tol: float = 1e-3, 
                nstates: int = 5, max_iter: int = 25, spectra: bool = False, 
                out_name: str = '', print_threshold: float = 0.05, gram_schmidt: bool = False, 
                single: bool = True, group_size: int = 256, group_size_aux: int = 256, 
                tensor_in_ram: bool = False, krylov_in_ram: bool = False, verbose=None):
        """
        Args:
            mf (object): Mean field object, typically obtained from a ground - state calculation.
            theta (float, optional): Global scaling factor for the fitting basis exponent. 
                                The relationship is defined as `alpha = theta/R_A^2`, where `alpha` is the Gaussian exponent 
                                and `R_A` is tabulated semi-empirical radii for element A. Defaults to 0.2.
            J_fit (str, optional): Fitting basis for the J matrix (`iajb` integrals). 
                                   's' means only one s orbital per atom, 'sp' means adding one extra p orbital per atom. 
                                   Defaults to 'sp', becasue more accurate than s.
            K_fit (str, optional): Fitting basis for the K matrix (`ijab` and `ibja` integrals). 
                                  's' means only one s orbital per atom, 
                                  'sp' means adding one extra p orbital per atom. 
                                   Defaults to 's', becasue 'sp' has no accuracy improvement.
            Ktrunc (float, optional): Truncation threshold for the K matrix. Orbitals are discarded if:
                                    - Occupied orbitals with energies < e_LUMO - Ktrunc
                                    - Virtual orbitals with energies > e_HOMO + Ktrunc. Defaults to 40.0.
            a_x (float, optional): Hartree-Fock component. By default, it will be assigned according 
                                    to the `mf.xc` attribute. 
                                    Will override the default value if provided.
            omega (float, optional): Range-separated hybrid functional parameter. By default, it will be 
                                    assigned according to the `mf.xc` attribute. 
                                    Will override the default value if provided.
            alpha (float, optional): Range-separated hybrid functional parameter. By default, it will be 
                                    assigned according to the `mf.xc` attribute.
                                    Will override the default value if provided.
            beta (float, optional): Range-separated hybrid functional parameter. By default, it will be 
                                    assigned according to the `mf.xc` attribute.
            conv_tol (float, optional): Convergence tolerance for the Davidson iteration. Defaults to 1e-3.
            nstates (int, optional): Number of excited states to be calculated. Defaults to 5.
            max_iter (int, optional): Maximum number of iterations for the Davidson iteration. Defaults to 25.
            spectra (bool, optional): Whether to calculate and dump the excitation spectra in G16 & Multiwfn style. 
                                     Defaults to False.
            out_name (str, optional): Output file name for the excitation spectra. Defaults to ''.
            print_threshold (float, optional): Threshold for printing the transition coefficients. Defaults to 0.05.
            gram_schmidt (bool, optional): Whether to calculate the ground state. Defaults to False.
            single (bool, optional): Whether to use single precision. Defaults to True.
            group_size (int, optional): Group size for the integral calculation. Defaults to 256.
            group_size_aux (int, optional): Group size for the auxiliary integral calculation. Defaults to 256.
            tensor_in_ram (bool, optional): Whether to store Tpq tensors in RAM. Defaults to False.
            krylov_in_ram (bool, optional): Whether to store Krylov vectors in RAM. Defaults to False.
            verbose (optional): Verbosity level of the logger. If None, it will use the verbosity of `mf`.
        """
        self.single = single

        if single:
            self.dtype = cp.dtype(cp.float32)
        else:
            self.dtype = cp.dtype(cp.float64)

        self._scf = mf
        self.chkfile = mf.chkfile
        self.singlet = True # TODO: add R-T excitation.
        self.exclude_nlc = False # TODO: exclude nlc functional 
        self.xy = None

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
        self.mo_coeff = cp.asarray(mf.mo_coeff, dtype=self.dtype)
        self.spectra = spectra
        self.out_name = out_name
        self.print_threshold = print_threshold
        self.gram_schmidt = gram_schmidt
        self.group_size = group_size
        self.group_size_aux = group_size_aux

        self.verbose = verbose if verbose else mf.verbose

        self.device = mf.device
        self.converged = None
        
        self._tensor_in_ram = tensor_in_ram
        self._krylov_in_ram = krylov_in_ram

        logger.TIMER_LEVEL = 4
        self.log = logger.new_logger(self)
        self.log.info(f'group_size {group_size}, group_size_aux {group_size_aux}')
    
        ''' following attributes will be initialized in self.build() '''
        self.n_occ = None
        self.n_vir = None
        self.rest_occ = None
        self.rest_vir = None

        self.C_occ_notrunc = None
        self.C_vir_notrunc = None
        self.C_occ_Ktrunc = None
        self.C_vir_Ktrunc = None

        self.delta_hdiag = None
        self.hdiag = None
        self.eri_tag = None

        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None

        self.RKS = True
        self.UKS = False

    def transition_dipole(self):
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

    def transition_magnetic_dipole(self):
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

    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.energies/HARTREE2EV

    def get_ab(self, mf=None):
        if mf is None:
            mf = self._scf
        J_fit = self.J_fit
        K_fit = self.K_fit
        theta = self.theta
        return get_ab(self, mf, J_fit, K_fit, theta, singlet=True)
    
    def build(self):
        log = self.log
        log.info(f'nstates: {self.nstates}')
        log.info(f'N atoms:{self._scf.mol.natm}')
        log.info(f'conv_tol: {self.conv_tol}')
        log.info(f'max_iter: {self.max_iter}')
        log.info(f'Ktrunc: {self.Ktrunc}')
        log.info(f'calculate and print UV-vis spectra info: {self.spectra}')
        if self.spectra:
            log.info(f'spectra files will be written and their name start with: {self.out_name}')
        log.info(f'store Tia Tij Tab in RAM: {self._tensor_in_ram}')

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

            omega, alpha_libxc, hyb_libxc = self._scf._numint.rsh_and_hybrid_coeff(self._scf.xc, spin=self._scf.mol.spin)
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
        log.info(f'gram_schmidt: {self.gram_schmidt}')
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
        log.info(f'cartesian or spherical electron integral: {self.eri_tag}')

        if self.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            n_occ = int(sum(self._scf.mo_occ>0))
            n_vir = int(sum(self._scf.mo_occ==0))
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = cp.asfortranarray(self.mo_coeff[:,:n_occ])
            self.C_vir_notrunc = cp.asfortranarray(self.mo_coeff[:,n_occ:])
            mo_energy = self._scf.mo_energy
            log.info(f'mo_energy.shape: {mo_energy.shape}')
            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)
            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
            delta_hdiag = cp.repeat(vir_ene, n_occ, axis=0) - cp.repeat(occ_ene, n_vir, axis=1)
            if self.single:
                delta_hdiag = cp.asarray(delta_hdiag, dtype=cp.float32)

            self.delta_hdiag = delta_hdiag
            self.hdiag = cp.asarray(delta_hdiag.reshape(-1))

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

            self.C_occ_Ktrunc = cp.asfortranarray(self.mo_coeff[:,n_occ-rest_occ:n_occ])
            self.C_vir_Ktrunc = cp.asfortranarray(self.mo_coeff[:,n_occ:n_occ+rest_vir])

            self.rest_occ = rest_occ
            self.rest_vir = rest_vir

        elif self.mo_coeff.ndim == 3:
            raise NotImplementedError('Does not support UKS method yet')
            ''' TODO UKS method '''
            self.RKS = False
            self.UKS = True
            self.n_occ_a = sum(self._scf.mo_occ[0]>0)
            self.n_vir_a = sum(self._scf.mo_occ[0]==0)
            self.n_occ_b = sum(self._scf.mo_occ[1]>0)
            self.n_vir_b = sum(self._scf.mo_occ[1]==0)
            log.info('n_occ for alpha spin = {self.n_occ_a}')
            log.info('n_vir for alpha spin = {self.n_vir_a}')
            log.info('n_occ for beta spin = {self.n_occ_b}')
            log.info('n_vir for beta spin = {self.n_vir_b}')

        auxmol_J = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.J_fit)
        log.info(f'n_bf in auxmol_J = {auxmol_J.nao_nr()}')
        self.auxmol_J = auxmol_J

        if self.a_x != 0:
            if self.K_fit == self.J_fit and (self.omega == 0 or self.omega is None):
                log.info('J and K use same aux basis, and they share same set of Tensors')
                auxmol_K = auxmol_J
                self._JK_share_aux = True

            else:
                log.info('either (1) J and K use different aux basis, or (2) RSH omega != 0')
                auxmol_K = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.K_fit) 
                self._JK_share_aux = False

            log.info(f'n_bf in auxmol_K = {auxmol_K.nao_nr()}')
            self.auxmol_K = auxmol_K

        log.info(f'self.dtype.itemsize,{self.dtype.itemsize}')
        byte_T_ia_J = self.auxmol_J.nao_nr() * self.n_occ * self.n_vir * self.dtype.itemsize
        log.info(f'T_ia_J will take {byte_T_ia_J / (1024 ** 2):.0f} MB memory') 

        self.lower_inv_eri2c_J = get_eri2c_inv_lower(self.auxmol_J, omega=0)

        if self.a_x != 0:
            
            byte_T_ij_K = auxmol_K.nao_nr() * self.rest_occ **2 * self.dtype.itemsize
            byte_T_ab_K = auxmol_K.nao_nr() * self.rest_vir **2 * self.dtype.itemsize
            log.info(f'T_ij_K will take {byte_T_ij_K / (1024 ** 2):.0f} MB memory')
            log.info(f'T_ab_K will take {byte_T_ab_K / (1024 ** 2):.0f} MB memory')

            byte_T_ia_K = auxmol_K.nao_nr() * self.rest_occ * self.rest_vir * self.dtype.itemsize
            log.info(f'(if full TDDFT) T_ia_K will take {byte_T_ia_K / (1024 ** 2):.0f} MB memory')

            if self._JK_share_aux:
                self.lower_inv_eri2c_K = self.lower_inv_eri2c_J
            else:
                self.lower_inv_eri2c_K = get_eri2c_inv_lower(auxmol_K, omega=self.omega, alpha=self.alpha, beta=self.beta)
             
        self.log = log

    def get_T_J(self):
        log = self.log
        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer()

        T_ia_J = get_Tpq(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J, 
                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, calc="J", omega=0, 
                        group_size = self.group_size, group_size_aux =self.group_size_aux,
                        in_ram=self._tensor_in_ram, single=self.single, log=log)

        log.timer('build T_ia_J', *cpu0)
        log.info(get_memory_info('after T_ia_J'))
        return T_ia_J
    
    def get_2T_K(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()

        T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K, 
                                C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, calc='K', 
                                omega=self.omega, alpha=self.alpha,beta=self.beta,
                                in_ram=self._tensor_in_ram, single=self.single,log=log)

        log.timer('T_ij_K T_ab_K', *cpu1)
        log.info(get_memory_info('after T_ij_K T_ab_K'))
        return T_ij_K, T_ab_K
    
    def get_3T_K(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()
        T_ia_K, T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K, 
                                C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, calc='JK', 
                                omega=self.omega, alpha=self.alpha,beta=self.beta,
                                group_size = self.group_size, group_size_aux =self.group_size_aux,
                                in_ram=self._tensor_in_ram, single=self.single,log=log)

        log.timer('T_ia_K T_ij_K T_ab_K', *cpu1)
        log.info(get_memory_info('after T_ia_K T_ij_K T_ab_K'))
        return T_ia_K, T_ij_K, T_ab_K

    def get_eri3c(self):
        log = self.log
        log.info('==================== eri3c ====================')
        cpu0 = log.init_timer()

        eri3c, int3c2e_opt = get_eri3c_bdiv(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K, 
                                 omega=0,  in_ram=self._tensor_in_ram, single=self.single, log=log)

        log.timer('build eri3c', *cpu0)
        log.info(get_memory_info('after eri3c'))
        return eri3c, int3c2e_opt

    def Gradients(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        return self.Gradients()

    def NAC(self):
        raise NotImplementedError

    def nac_method(self):
        return self.NAC()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    as_scanner = as_scanner

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

        
        int3c2e_opt_J, aux_coeff_lower_inv_eri2c = get_int3c2e_opt(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J, 
                            C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, calc="J", omega=0, 
                            in_ram=self._tensor_in_ram, single=self.single, log=log)
        log.info(get_memory_info('after int3c2e_opt_J'))

        eri3c_K, int3c2e_opt_K = self.get_eri3c()
        # T_ij_K, T_ab_K = self.get_2T_K()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        # iajb_MVP = gen_iajb_MVP(T_ia=T_ia_J)
        iajb_MVP = gen_iajb_MVP_bdiv(int3c2e_opt=int3c2e_opt_J, aux_coeff_lower_inv_eri2c=aux_coeff_lower_inv_eri2c, 
                                    C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)

        # ijab_MVP = gen_ijab_MVP_out(T_ij=T_ij_K, T_ab=T_ab_K, log=log)
        ijab_MVP = gen_ijab_MVP_eri3c(eri3c=eri3c_K, int3c2e_opt=int3c2e_opt_K, 
                        C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, log=log, single=self.single)


        def RKS_TDA_hybrid_MVP(X, out=None):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V) - a_x*ijab_MVP(V)
                for RSH, a_x = 1

                
                With MO truncation, most the occ and vir orbitals (transition pair) are neglected in the exchange part

                As shown below, * denotes the included transition pair
                         -------------------
                       /                  /
       original X =   /                  /  nstates
                     -------------------
                    |******************|
             n_occ  |******************|  
                    |******************|
                    |******************|
                    |------------------|
                            n_vir
        becomes:
                         -------------------
                       /                  /
                X' =  /                  /  nstates
                     -------------------
                    |                  |
     n_occ-rest_occ |                  |  
                    |-----|------------|
                    |*****|            |  
         rest_occ   |*****|            |
                    |-----|------------|
                  rest_vir

                (If no MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            
            # if out is None:
            #     cpu0 = log.init_timer()
            #     AX = hdiag_MVP(X) 
            #     AX += 2 * iajb_MVP(X, out=None) 
            #     log.timer('--iajb_MVP', *cpu0)

            #     cpu1 = log.init_timer()
            #     exchange = self.a_x * ijab_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            #     log.timer('--ijab_MVP', *cpu1)

            #     AX[:,self.n_occ-self.rest_occ:,:self.rest_vir] -= exchange
            #     AX = AX.reshape(nstates, self.n_occ*self.n_vir)
            #     return AX
            # if out is None:
            #     cpu0 = log.init_timer()
            #     out = hdiag_MVP(X) 
            #     iajb_MVP(X, out=out) 
            #     log.timer('--iajb_MVP', *cpu0)

            #     cpu1 = log.init_timer()
            #     ijab_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir],a_x=a_x out=out)
            #     log.timer('--ijab_MVP', *cpu1)

            #     return out

            # else:
            cpu0 = log.init_timer()
            if out is None:
                out_holder_provided = False
                out = hdiag_MVP(X)
            else:
                out_holder_provided = True
                out = out.reshape(nstates, self.n_occ, self.n_vir)
                out[:,:,:] += hdiag_MVP(X)

            iajb_MVP(X, out=out)
            log.timer('--iajb_MVP', *cpu0)

            cpu1 = log.init_timer()
            ijab_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir], a_x=self.a_x, out=out[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            log.timer('--ijab_MVP', *cpu1)

            if not out_holder_provided:
                out = out.reshape(nstates, self.n_occ*self.n_vir)
                return out

        return RKS_TDA_hybrid_MVP, self.hdiag
            
    
    ''' ===========  RKS pure =========== '''
    def get_RKS_TDA_pure_MVP(self):
        '''hybrid RKS TDA'''
        log = self.log  

        T_ia_J = self.get_T_J()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)
        iajb_MVP = gen_iajb_MVP(T_ia=T_ia_J)
        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            AX = hdiag_MVP(X) 
            cpu0 = log.init_timer()
            AX += 2 * iajb_MVP(X) 
            log.timer('--iajb_MVP', *cpu0)
            AX = AX.reshape(nstates, self.n_occ*self.n_vir)
            return AX

        return RKS_TDA_pure_MVP, self.hdiag
       
    #  TODO: UKS case

    def gen_vind(self):
        if self.RKS:
            self.build()
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_RKS_TDA_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDA_MVP, hdiag

    def kernel(self):

        '''for TDA, pure and hybrid share the same form of
                     AX = Xw
            always use the Davidson solver
            Unlike pure TDDFT, pure TDA is not using MZ=Zw^2 form
        '''
        log = self.log

        TDA_MVP, hdiag = self.gen_vind()
        converged, energies, X = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag, n_states=self.nstates, problem_type='eigenvalue',
                                              conv_tol=self.conv_tol, max_iter=self.max_iter, gram_schmidt=self.gram_schmidt,
                                              single=self.single, in_ram=self._krylov_in_ram, verbose=log)

        self.converged = converged
        log.debug(f'check orthonormality of X: {cp.linalg.norm(cp.dot(X, X.T) - cp.eye(X.shape[0])):.2e}')

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies,
                                                 X=X/(2**0.5), Y=None, P=self.transition_dipole(), mdpol=self.transition_magnetic_dipole(),
                                                 name=self.out_name+'_TDA_ris', RKS=self.RKS, spectra=self.spectra,
                                                 print_threshold = self.print_threshold, n_occ=self.n_occ, n_vir=self.n_vir, verbose=self.verbose)
        
        energies = energies*HARTREE2EV
        log.info(f'energies: {energies}')
        log.info(f'oscillator strength: {oscillator_strength}')
        log.info(CITATION_INFO)

        self.energies = energies
        self.xy = (X, None)
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        return energies, X, oscillator_strength, rotatory_strength

    def Gradients(self):
        if getattr(self._scf, 'with_df', None) is not None:
            from gpu4pyscf.df.grad import tdrks_ris
            return tdrks_ris.Gradients(self)
        else:
            from gpu4pyscf.grad import tdrks_ris
            return tdrks_ris.Gradients(self)

    def NAC(self):
        if getattr(self._scf, 'with_df', None) is not None:
            from gpu4pyscf.df.nac.tdrks_ris import NAC
            return NAC(self)
        else:
            from gpu4pyscf.nac.tdrks_ris import NAC
            return NAC(self)

    
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

        log.info(get_memory_info('before T_ia_J'))

        T_ia_J = self.get_T_J()

        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        iajb_MVP = gen_iajb_MVP(T_ia=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K, T_ab=T_ab_K)
        ibja_MVP = gen_ibja_MVP(T_ia=T_ia_K)

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
            n_occ, rest_occ = self.n_occ, self.rest_occ
            n_vir, rest_vir= self.n_vir, self.rest_vir

            X = X.reshape(nstates, n_occ, n_vir)
            Y = Y.reshape(nstates, n_occ, n_vir)

            XpY = X + Y
            XmY = X - Y
            ApB_XpY = hdiag_MVP(XpY)

            ApB_XpY += 4*iajb_MVP(XpY)

            ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir]) 

            ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])

            AmB_XmY = hdiag_MVP(XmY) 
            AmB_XmY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir]) 

            AmB_XmY[:,n_occ-rest_occ:,:rest_vir] += self.a_x*ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])

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
        return RKS_TDDFT_hybrid_MVP, self.hdiag

    ''' ===========  RKS pure =========== '''
    def gen_RKS_TDDFT_pure_MVP(self):
        log = self.log   
        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer()

        log.info(get_memory_info('before T_ia_J'))

        T_ia_J = self.get_T_J()
        
        log.timer('T_ia_J', *cpu0)

        hdiag_sqrt_MVP = gen_hdiag_MVP(hdiag=self.hdiag**0.5, n_occ=self.n_occ, n_vir=self.n_vir)
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)
        iajb_MVP = gen_iajb_MVP(T_ia=T_ia_J)
        hdiag_sq = self.hdiag**2
        def RKS_TDDFT_pure_MVP(Z):
            '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                    MZ = Z w^2
                M = (A-B)^1/2 (A+B) (A-B)^1/2
                X+Y = (A-B)^1/2 Z

                (A+B)(V) = hdiag_MVP(V) + 4*iajb_MVP(V)
                (A-B)^1/2(V) = hdiag_sqrt_MVP(V)
            '''
            nstates = Z.shape[0]
            Z = Z.reshape(nstates, self.n_occ, self.n_vir)
            AmB_sqrt_V = hdiag_sqrt_MVP(Z)
            ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V) + 4*iajb_MVP(AmB_sqrt_V)
            MZ = hdiag_sqrt_MVP(ApB_AmB_sqrt_V)
            MZ = MZ.reshape(nstates, self.n_occ*self.n_vir)
            return MZ

        return RKS_TDDFT_pure_MVP, hdiag_sq

    def gen_vind(self):
        if self.RKS:
            self.build()
            if self.a_x != 0:
                TDDFT_MVP, hdiag = self.gen_RKS_TDDFT_hybrid_MVP()

            elif self.a_x == 0:
                TDDFT_MVP, hdiag = self.gen_RKS_TDDFT_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDDFT_MVP, hdiag
    
    #  TODO: UKS 
    def kernel(self):
        self.build()
        log = self.log
        TDDFT_MVP, hdiag = self.gen_vind()
        if self.a_x != 0:
            '''hybrid TDDFT'''
            converged, energies, X, Y = _krylov_tools.ABBA_krylov_solver(matrix_vector_product=TDDFT_MVP, hdiag=hdiag,
                                                    n_states=self.nstates, conv_tol=self.conv_tol,
                                                    max_iter=self.max_iter, gram_schmidt=self.gram_schmidt,
                                                    single=self.single, verbose=self.verbose)
            self.converged = converged
            if not all(self.converged):
                log.info('TD-SCF states %s not converged.',
                            [i for i, x in enumerate(self.converged) if not x])
        elif self.a_x == 0:
            '''pure TDDFT'''
            hdiag_sq = hdiag
            converged, energies_sq, Z = _krylov_tools.krylov_solver(matrix_vector_product=TDDFT_MVP, hdiag=hdiag_sq,
                                            n_states=self.nstates, conv_tol=self.conv_tol, max_iter=self.max_iter,
                                            gram_schmidt=self.gram_schmidt, single=self.single, verbose=self.verbose)
            self.converged = converged
            if not all(self.converged):
                log.info('TD-SCF states %s not converged.',
                            [i for i, x in enumerate(self.converged) if not x])

            energies = energies_sq**0.5
            Z = (energies**0.5).reshape(-1,1) * Z

            X, Y = math_helper.XmY_2_XY(Z=Z, AmB_sq=hdiag_sq, omega=energies)

        log.debug(f'check normality of X^TX - Y^YY - I = {cp.linalg.norm( (cp.dot(X, X.T) - cp.dot(Y, Y.T)) - cp.eye(self.nstates) ):.2e}')


        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X/(2**0.5), Y=Y/(2**0.5),
                                                    P=self.transition_dipole(), mdpol=self.transition_magnetic_dipole(), name=self.out_name+'_TDDFT_ris',
                                                    spectra=self.spectra, RKS=self.RKS, print_threshold = self.print_threshold,
                                                    n_occ=self.n_occ, n_vir=self.n_vir, verbose=self.verbose)
        energies = energies*HARTREE2EV
        log.info(f'energies: {energies}')
        log.info(f'oscillator strength: {oscillator_strength}')
        log.info(CITATION_INFO)
        self.energies = energies
        self.xy = X, Y
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        return energies, X, Y, oscillator_strength, rotatory_strength

    Gradients = TDA.Gradients
    NAC = TDA.NAC


class StaticPolarizability(RisBase):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, **kwargs)
        log = self.log
        log.warn("Static Polarizability-ris is still in the experimental stage, and its APIs are subject to change in future releases.")
        log.info('Static Polarizability-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_ApB_hybrid_MVP(self):
        ''' RKS hybrid '''
        log = self.log

        T_ia_J = self.get_T_J()
        
        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        iajb_MVP = gen_iajb_MVP(T_ia=T_ia_J)
        ijab_MVP = gen_ijab_MVP(T_ij=T_ij_K, T_ab=T_ab_K)
        ibja_MVP = gen_ibja_MVP(T_ia=T_ia_K)

        def RKS_ApB_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                (A+B)X = hdiag_MVP(X) + 4*iajb_MVP(X) - a_x*[ijab_MVP(X) + ibja_MVP(X)]
                for RSH, a_x = 1
  
                if not MO truncation, then n_occ-rest_occ=0 and rest_vir=n_vir
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            cpu0 = log.init_timer()
            ApBX = hdiag_MVP(X) 
            ApBX += 4 * iajb_MVP(X) 
            log.timer('--iajb_MVP', *cpu0)

            cpu1 = log.init_timer()
            exchange =  ijab_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            exchange += ibja_MVP(X[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            log.timer('--ijab_MVP & ibja_MVP', *cpu1)

            ApBX[:,self.n_occ-self.rest_occ:,:self.rest_vir] -= self.a_x * exchange
            ApBX = ApBX.reshape(nstates, self.n_occ*self.n_vir)

            return ApBX

        return RKS_ApB_hybrid_MVP, self.hdiag

    def gen_vind(self):
        self.build()
        if self.RKS:
            if self.a_x != 0:
                TDA_MVP, hdiag = self.get_ApB_hybrid_MVP()

            elif self.a_x == 0:
                TDA_MVP, hdiag = self.get_ApB_pure_MVP()
        else:
            raise NotImplementedError('Does not support UKS method yet')
        return TDA_MVP, hdiag


    def kernel(self):
        '''for static polarizability, the problem is to solve
            (A+B)(X+Y) = -(P+Q)
            Q=P
        '''
        
        log = self.log

        TDA_MVP, hdiag = self.gen_vind()
        transition_dipole = self.transition_dipole()

        _, solver = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag, problem_type='linear',
                                        rhs=-transition_dipole, conv_tol=self.conv_tol, max_iter=self.max_iter, 
                                        gram_schmidt=self.gram_schmidt, single=self.single, verbose=log)
        X = solver.run()
        # actually X here means X+Y
        alpha = cp.dot(X, transition_dipole.T)*4

        self.xy = X
        self.alpha = alpha

        log.info(CITATION_INFO)
        return X

