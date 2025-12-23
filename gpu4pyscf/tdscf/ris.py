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

import pyscf, gpu4pyscf
import numpy as np
import cupy as cp
import gc, sys
import cupyx.scipy.linalg as cpx_linalg

from concurrent.futures import ThreadPoolExecutor
from pyscf import gto, lib
from gpu4pyscf import scf
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib.cupy_helper import asarray as cuasarray

from gpu4pyscf.tdscf import parameter, math_helper, spectralib, _krylov_tools
from gpu4pyscf.tdscf.math_helper import gpu_mem_info, release_memory, get_avail_gpumem
from pyscf.data.nist import HARTREE2EV
from gpu4pyscf.lib import logger
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df import int3c2e_bdiv

gpu4pyscf.lib.logger.TIMER_LEVEL = 5
# gpu4pyscf.lib.logger.WARN = 6
# pyscf.lib.logger.WARN=6


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


def get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis, excludeHs=False):
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

        if excludeHs:
            if atom == 'H':
                continue
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

def get_auxmol(mol, theta=0.2, fitting_basis='s', excludeHs=False):
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
    auxmol.verbose=0
    auxmol_basis_keys = mol._basis.keys()
    auxmol.basis = get_minimal_auxbasis(auxmol_basis_keys, theta, fitting_basis,excludeHs=excludeHs)
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

def get_uvPCupCvq_to_Ppq1(eri3c: cp.ndarray, C_p: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
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


def get_uvPCupCvq_to_Ppq(eri3c: cp.ndarray, C_pT: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''    
    eri3c : (uv|P) , P = nauxao
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    nao, nao, nauxao = eri3c.shape
    size_p,nao = C_pT.shape
    nao, size_q = C_q.shape

    # eri3c = eri3c.reshape(nao,  nao * nauxao)  # (u, vP)
    tmp = C_pT.dot(eri3c)  #(p,u) (u, vP) -> p,vP
    # tmp = tmp.reshape(size_p, nao, nauxao)
    # tmp = contract('uvP,up->Ppv', eri3c, C_p)
    Ppq_gpu = contract('pvP,vq->Ppq', tmp, C_q)

    del tmp

    if in_ram:
        Ppq_cpu = Ppq_gpu.get()
        del Ppq_gpu
        return Ppq_cpu
        
    else:
        return Ppq_gpu
    

   
def get_uvCupCvq_to_Ppq(eri3c: cp.ndarray, C_pT: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''    
    eri3c : (uv|1) , 
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    nao, nao = eri3c.shape
    size_p,nao = C_pT.shape
    nao, size_q = C_q.shape


    tmp = C_pT.dot(eri3c)  #(p,u) (u, v) -> p,v

    Ppq_gpu = tmp.dot(C_q)

    del tmp

    if in_ram:
        Ppq_cpu = Ppq_gpu.get()
        del Ppq_gpu
        return Ppq_cpu
        
    else:
        return Ppq_gpu



def get_uvPCupCvq_to_Ppq_symmetry(eri3c: cp.ndarray, C_p: cp.ndarray, C_q: cp.ndarray, in_ram: bool = False):
    '''    
    eri3c : (uv|P) , P = nauxao
    C_p and C_q:  C[:, :n_occ] or C[:, n_occ:], can be both

    Ppq = einsum("uvP,up,vq->Ppq", eri3c, Cp, C_q)
    '''
    tmp = contract('uvP,up->Ppv', eri3c, C_p)
    Ppq = contract('Ppv,vq->Ppq', tmp, C_q)
    del tmp
    release_memory()

    P, p, q = Ppq.shape
    tril_indices = cp.tril_indices(p)

    Ppq_lower = Ppq[:, tril_indices[0], tril_indices[1]]

    Ppq_flat = Ppq_lower.reshape(P, -1)

    if in_ram:
        Ppq_flat = Ppq_flat.get()
    return Ppq_flat

def einsum2dot(_, a, b):
    P, Q = a.shape
    Q, p, q = b.shape
    b_2d = b.reshape(Q, p*q)

    if P == Q:
        np.dot(a, b_2d, out=b_2d) 
        return b  
    else:
        out_2d = np.dot(a, b_2d)
        return out_2d.reshape(P, p, q)

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
    naopair = len(ao_pair_mapping)
    rows, cols = divmod(ao_pair_mapping, nao_orig)
    log.info(f' number of AO pairs {len(ao_pair_mapping)}')
    log.info(f' type(rows) {type(rows)}')

    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff
    naux = aux_coeff.shape[0]

    if omega and omega != 0:
        log.info(f'omega {omega}')
        # with mol.with_range_coulomb(omega):
        mol_omega = mol.copy()
        auxmol_omega = auxmol.copy()
        mol_omega.omega = omega

        int3c2e_opt_omega = int3c2e_bdiv.Int3c2eOpt(mol_omega, auxmol_omega).build()
        ao_pair_mapping_omega = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)
        # rows_omega, cols_omega = divmod(ao_pair_mapping, nao_orig)
        # log.info(f'number of AO pairs {len(ao_pair_mapping_omega)}')
        check = abs(ao_pair_mapping_omega - ao_pair_mapping).max()
        # log.info(f'abs(ao_pair_mapping_omega - ao_pair_mapping).max() {check}')
        assert check < 1e-10

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])

    siz_p = C_p.shape[1]
    siz_q = C_q.shape[1]

    C_pT = C_p.T
    C_qT = C_q.T

    xp = np if in_ram else cp
    log.info(f'xp {xp}')

    P_dtype = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    log.info(f'cp_int3c_dtype: {cp_int3c_dtype}')
    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.astype(cp_int3c_dtype, copy=False)
    upper_inv_eri2c = aux_coeff_lower_inv_eri2c.T

    if 'K' in calc:
        eri2c_inv = aux_coeff_lower_inv_eri2c.dot(upper_inv_eri2c)
        if in_ram:
            eri2c_inv = eri2c_inv.get()

    if in_ram:
        # aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
        upper_inv_eri2c = upper_inv_eri2c.get()

    if 'J' in calc:
        Pia = xp.empty((naux, siz_p, siz_q), dtype=P_dtype)

    if 'K' in calc:
        '''only store lower triangle of Tij and Tab'''
        # n_tri_p = (siz_p * (siz_p + 1)) // 2
        # n_tri_q = (siz_q * (siz_q + 1)) // 2
        # Pij = xp.empty((naux, n_tri_p), dtype=P_dtype)  
        # Pab = xp.empty((naux, n_tri_q), dtype=P_dtype)  
        Pij = xp.empty((naux, siz_p, siz_p), dtype=P_dtype)  
        Pab = xp.empty((naux, siz_q, siz_q), dtype=P_dtype)  

    byte_eri3c = nao_orig * nao_orig * cp_int3c_dtype.itemsize
    # pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()

    available_gpu_memory = get_avail_gpumem()
    n_eri3c_per_aux = naopair * 3 
    n_eri3c_unzip_per_aux = nao_orig * nao_orig * 1
    n_Ppq_per_aux = siz_p * nao_orig  + siz_p * siz_q * 1


    bytes_per_aux = ( n_eri3c_per_aux + n_eri3c_unzip_per_aux + n_Ppq_per_aux) * cp_int3c_dtype.itemsize  
    batch_size = min(naux, max(16, int(available_gpu_memory * 0.5 // bytes_per_aux)) )

    log.info(f'eri3c per aux dimension will take {byte_eri3c / 1e6:.0f} MB memory')
    log.info(f'batch_size for int3c2e_bdiv_generator (in aux dimension): {batch_size}')
    log.info(f'eri3c per aux batch will take {byte_eri3c * batch_size / 1e6:.0f} MB memory')
    log.info(gpu_mem_info('before int3c2e_bdiv_generator'))

    gen = int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size)

    if omega and omega != 0:
        gen_omega = int3c2e_opt_omega.int3c2e_bdiv_generator(batch_size=batch_size)
    p1 = 0
    while True:
        try:
            eri3c_batch_tmp = next(gen)
            eri3c_batch_tmp = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch_tmp, inplace=True)

            if omega and omega != 0:
                eri3c_batch_omega_tmp = next(gen_omega)
                eri3c_batch_omega_tmp = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch_omega_tmp, inplace=True)
                
                # eri3c_batch_tmp = alpha * eri3c_batch_tmp + beta * eri3c_batch_omega_tmp
                eri3c_batch_tmp *= alpha
                eri3c_batch_omega_tmp *= beta
                eri3c_batch_tmp += eri3c_batch_omega_tmp
                del eri3c_batch_omega_tmp
            
            eri3c_batch = cuasarray(eri3c_batch_tmp, dtype=cp_int3c_dtype, order='F')
            del eri3c_batch_tmp
            aopair, aux_batch_size = eri3c_batch.shape
            p0, p1 = p1, p1 + eri3c_batch.shape[1]
            release_memory()


            if aux_batch_size > 16:
                eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, aux_batch_size), dtype=cp_int3c_dtype, order='F')
                eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
                eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch
                
                eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, aux_batch_size)

                DEBUG = False 
                if DEBUG:
                    from pyscf.df import incore
                    ref = incore.aux_e2(mol, auxmol)

                    aux_coeff = cuasarray(int3c2e_opt.aux_coeff)
                    out = contract('uvP,PQ->uvQ', eri3c_unzip_batch, aux_coeff)
                    out = int3c2e_opt.unsort_orbitals(out, axis=(0,1))
                    log.warn(f'-------------eri3c DEBUG: out vs .incore.aux_e2(mol, auxmol) {abs(out.get()-ref).max()}')
                    assert abs(out.get()-ref).max() < 1e-10
                    

                '''Puv -> Ppq, AO->MO transform '''
                if 'J' in calc:
                    Pia[p0:p1,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_pT,C_q, in_ram=in_ram)

                if 'K' in calc:
                    Pij[p0:p1,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_pT,C_p, in_ram=in_ram)
                    Pab[p0:p1,:,:] = get_uvPCupCvq_to_Ppq(eri3c_unzip_batch,C_qT,C_q, in_ram=in_ram)

            else:
                ''' iterate over each aux function in the batch to save memory '''
                for aux_idx in range(aux_batch_size):
                    eri3c_aux_idx = eri3c_batch[:, aux_idx]  # (naopair, )
                    eri3c_unzip_idx = cp.zeros((nao_orig*nao_orig,), dtype=cp_int3c_dtype, order='F')
                    eri3c_unzip_idx[ao_pair_mapping] = eri3c_aux_idx
                    eri3c_unzip_idx[cols*nao_orig+rows] = eri3c_aux_idx
                
                    eri3c_unzip_idx = eri3c_unzip_idx.reshape(nao_orig, nao_orig)

                    '''Puv -> Ppq, AO->MO transform '''
                    if 'J' in calc:
                        Pia[p0+aux_idx,:,:] = get_uvCupCvq_to_Ppq(eri3c_unzip_idx,C_pT,C_q, in_ram=in_ram)

                    if 'K' in calc:
                        Pij[p0+aux_idx,:,:] = get_uvCupCvq_to_Ppq(eri3c_unzip_idx,C_pT,C_p, in_ram=in_ram)
                        Pab[p0+aux_idx,:,:] = get_uvCupCvq_to_Ppq(eri3c_unzip_idx,C_qT,C_q, in_ram=in_ram)

            last_reported = 0
            progress = int(100.0 * p1 / naux)

            if progress % 20 == 0 and progress != last_reported:
                log.last_reported = progress
                log.info(f'get_Tpq batch {p1} / {naux} done ({progress} percent)')
                
        except StopIteration:
            log.info(f' get_Tpq {calc} all batches processed')
            log.info(gpu_mem_info('after generate Ppq'))
            break

    if in_ram:
        tmp_einsum = einsum2dot 
    else:
        tmp_einsum = contract

    if calc == 'J':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        return Tia

    if calc == 'K':
        # Tij = tmp_einsum('PQ,Qij->Pij', upper_inv_eri2c, Pij)
        # Tab = tmp_einsum('PQ,Qab->Pab', upper_inv_eri2c, Pab)
        Tij = tmp_einsum('PQ,Qij->Pij', eri2c_inv, Pij)
        Tab = Pab
        return Tij, Tab

    if calc == 'JK':
        Tia = tmp_einsum('PQ,Qia->Pia', upper_inv_eri2c, Pia)
        # Tij = tmp_einsum('PQ,Qij->Pij', upper_inv_eri2c, Pij)
        # Tab = tmp_einsum('PQ,Qab->Pab', upper_inv_eri2c, Pab)
        Tij = tmp_einsum('PQ,Qij->Pij', eri2c_inv, Pij)
        Tab = Pab
        return Tia, Tij, Tab
     
def get_eri3c_bdiv(mol, auxmol, lower_inv_eri2c, 
                omega=None, alpha=None, beta=None,
                log=None, in_ram=True, single=True):
    ''' calculate lower part of uvP'''
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()

    nao, nao_orig = int3c2e_opt.coeff.shape
    log.info(f'nao, nao_orig {nao, nao_orig}') 

    xp = np if in_ram else cp
    log.info(f'xp {xp}')
    int3c_dtype    = xp.dtype(xp.float32 if single else xp.float64)
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    
    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff
    nauxao = aux_coeff.shape[0]

    if omega and omega != 0:
        log.info(f'omega {omega}')
        # with mol.with_range_coulomb(omega):
        mol_omega = mol.copy()
        auxmol_omega = auxmol.copy()
        mol_omega.omega = omega

        int3c2e_opt_omega = int3c2e_bdiv.Int3c2eOpt(mol_omega, auxmol_omega).build()

    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = cuasarray(aux_coeff_lower_inv_eri2c, dtype=cp_int3c_dtype, order='F')

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=mol.cart)

    naopair = len(ao_pair_mapping)

    eri3c_mem = naopair * nauxao * cp_int3c_dtype.itemsize  / 1e9
    log.info(f'eri3c shape {naopair, nauxao}, {eri3c_mem:.0f} GB memory')


    available_gpu_memory = get_avail_gpumem()
    bytes_per_aux = naopair *8 +  naopair * cp_int3c_dtype.itemsize 
    if omega and omega != 0:  
        bytes_per_aux *= 2
    batch_size = min(nauxao, max(1, int(available_gpu_memory * 0.2 // bytes_per_aux)) )
    log.info(f'int3c2e_bdiv_generator batch_size/nauxao: {batch_size} / {nauxao}')
    log.info(f'compression rate: {naopair/(nao_orig*nao_orig):.4f}')

    eri3c = cp.empty((naopair, nauxao), dtype=int3c_dtype, order='F')
    log.info(' generate eri3c in GPU, download to CPU only at final step')
    
    gen = int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size)

    if omega and omega != 0:
        gen_omega = int3c2e_opt_omega.int3c2e_bdiv_generator(batch_size=batch_size)

    p1 = 0
    while True:
        try:
            eri3c_batch = next(gen)
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)

            if omega and omega != 0:
                eri3c_batch_omega = next(gen_omega)
                eri3c_batch_omega = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch_omega, inplace=True)
                eri3c_batch = alpha * eri3c_batch + beta * eri3c_batch_omega
            
            eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
            
            p0, p1 = p1, p1 + eri3c_batch.shape[1]
            
            cpu2 = log.init_timer()
            
            eri3c[:, p0:p1] = eri3c_batch
            log.info(f' generate eri3c batch {p0}-{p1} / {nauxao} done')
            log.timer(' eri3c_batch to holder', *cpu2)
        except StopIteration:
            log.info('All batches processed')
            break

    cpu3 = log.init_timer()
    eri3c = eri3c.dot(aux_coeff_lower_inv_eri2c)
    log.timer(' eri3c.dot(aux_coeff_lower_inv_eri2c)', *cpu3)
    if in_ram:
        eri3c = eri3c.get()

    # if eri3c_mem >= gpu_memG_threshold:
    #     log.info(f' generate eri3c in GPU and download to CPU during all steps')
    #     eri3c = xp.empty((naopair, nauxao), dtype=int3c_dtype, order='F')

    #     p1 = 0
    #     for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size):

    #         p0, p1 = p1, p1 + eri3c_batch.shape[1]
    #         eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
    #         eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            
    #         cpu2 = log.init_timer()
            
    #         if in_ram:
    #             cpu = log.init_timer()
    #             # tmp = eri3c_batch.get()
    #             # log.timer(' tmp = eri3c_batch.get()', *cpu)
    #             # cpu = log.init_timer()
    #             eri3c[:, p0:p1] = eri3c_batch.get()
    #             log.timer(' eri3c[:, p0:p1] = tmp', *cpu)

    #         else:
    #             eri3c[:, p0:p1] = eri3c_batch
    #         log.info(f' generate eri3c batch {p0}-{p1} / {nauxao} done')
    #         log.timer(' eri3c_batch to holder', *cpu2)

    #     cpu3 = log.init_timer()
    #     if in_ram:
    #         aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
    #     eri3c = eri3c.dot(aux_coeff_lower_inv_eri2c)
    #     log.timer(' eri3c.dot(aux_coeff_lower_inv_eri2c)', *cpu3)
    
    return eri3c, int3c2e_opt

def get_int3c2e_eri2c(mol, auxmol, lower_inv_eri2c,
           log=None, in_ram=True, single=True):

    mol.verbose -= 1
    int3c2e_opt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol).build()

    nao, nao_orig = int3c2e_opt.coeff.shape
    log.info(f'nao, nao_orig {nao, nao_orig}') 

    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)
    
    aux_coeff = cp.array(int3c2e_opt.aux_coeff) # aux_coeff is contraction coeff of GTO, not the MO coeff

    aux_coeff_lower_inv_eri2c = aux_coeff.dot(lower_inv_eri2c)
    aux_coeff_lower_inv_eri2c = cuasarray(aux_coeff_lower_inv_eri2c, dtype=cp_int3c_dtype, order='F')
    if in_ram:
        aux_coeff_lower_inv_eri2c = aux_coeff_lower_inv_eri2c.get()
    return  int3c2e_opt, aux_coeff_lower_inv_eri2c


def get_eri2c_inv_lower(auxmol, omega=0, alpha=None, beta=None, dtype=cp.float64):

    eri2c = auxmol.intor('int2c2e')

    if omega and omega != 0:

        with auxmol.with_range_coulomb(omega):
            eri2c_erf = auxmol.intor('int2c2e')

        eri2c = alpha * eri2c + beta * eri2c_erf

    eri2c = cuasarray(eri2c)

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

    lower_inv_eri2c = cuasarray(lower_inv_eri2c, dtype=dtype, order='C')
    return lower_inv_eri2c

def get_inter_contract_C(int_tensor, C_occ, C_vir):

    P = get_PuvCupCvq_to_Ppq(int_tensor, C_occ, C_vir)

    ''' 3 for xyz three directions.
        reshape is helpful when calculating oscillator strength and polarizability.
    '''
    P = P.reshape(3,-1)
    return P


def gen_hdiag_MVP(hdiag, n_occ, n_vir):
    def hdiag_MVP(V):
        m = V.shape[0]
        V = V.reshape(m, n_occ*n_vir)
        # Use a local variable to avoid modifying closure variable
        if isinstance(V, np.ndarray):  # Check if V is on CPU
            hdiag_local = hdiag.get()  # Transfer from GPU to CPU if needed
        else:
            hdiag_local = hdiag  # Already on the correct device

        hdiag_v = hdiag_local[None,:] * V
        hdiag_v = cuasarray(hdiag_v).reshape(m, n_occ, n_vir)
        return hdiag_v

    return hdiag_MVP

def gen_K_diag(eri3c, int3c2e_opt, C_p, C_q, log, single):
    # nao, nao_orig = int3c2e_opt.coeff.shape

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=int3c2e_opt.mol.cart)
    naopair = ao_pair_mapping.shape[0]
    # compress_ratio = naopair/(nao_orig**2)
    # rows, cols = divmod(ao_pair_mapping, nao_orig)

    n_occ = C_p.shape[1]
    n_vir = C_q.shape[1]

    nauxao = int3c2e_opt.aux_coeff.shape[0]
    log.info(f' gen_K_diag cp_int3c_dtype {cp_int3c_dtype}')
    T_ii = cp.empty((nauxao, n_occ),dtype=cp_int3c_dtype)
    T_aa = cp.empty((nauxao, n_vir),dtype=cp_int3c_dtype)

    '''T_aa = cp.einsum('Puv,ua,va->Pa', eri3c, C_vir, C_vir)'''

    available_gpu_memory = get_avail_gpumem()

    pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()

    dtype_size = eri3c.itemsize  

    n_eri3c_batch = 1* naopair # per auxao

    bytes_per_eri3c = n_eri3c_batch * dtype_size

    aux_batch_size = min(nauxao, max(1, int(available_gpu_memory * 0.4 // bytes_per_eri3c)) )
    log.info(f'gen_K_diag aux_batch_size {aux_batch_size} / {nauxao}')

    n_C_tmp_batch = 4 * naopair * 1 # per mo
    bytes_per_mo = n_C_tmp_batch * dtype_size  

    mo_batch_size = min(max(n_occ, n_vir), max(1, int(available_gpu_memory * 0.4 // bytes_per_mo)))
    log.info(f'mo_batch_size={mo_batch_size}, n_occ={n_occ}, n_vir={n_vir}')

    cpu00 = log.init_timer()

    for auxp0 in range(0, nauxao, aux_batch_size):
        auxp1 = min(auxp0+aux_batch_size, nauxao)
        log.info(f' gen_K_diag auxao slicing: {auxp1} / {nauxao}')
        eri3c_batch = cuasarray(eri3c[:, auxp0:auxp1])   #(naopair, aux_batch_size)

        cpu0 = log.init_timer()
        for mop0 in range(0, n_vir, mo_batch_size):
            mop1 = min(mop0+mo_batch_size, n_vir)    
            log.info(f'        n_vir slicing: {mop1}/{n_vir}')
            cpu = log.init_timer()

            C_tmp_batch = C_q[:, mop0:mop1]
            dm_sparse = 2*C_tmp_batch[pair_rows, :] * C_tmp_batch[pair_cols, :]  # (naopair, mo_batch_size)

            dm_sparse[pair_diag, :] *= 0.5 

            log.timer(' dm_sparse', *cpu)

            T_aa[auxp0:auxp1, mop0:mop1] = contract('ka,kP->Pa', dm_sparse, eri3c_batch)
            cpu = log.init_timer()

        log.timer(' T_aa', *cpu0)

        cpu0 = log.init_timer()
        for mop0 in range(0, n_occ, mo_batch_size):
            mop1 = min(mop0+mo_batch_size, n_occ)    
            log.info(f'        n_occ slicing: {mop1}/{n_occ}')
            cpu = log.init_timer()

            C_tmp_batch = C_p[:, mop0:mop1]
            dm_sparse = 2*C_tmp_batch[pair_rows, :] * C_tmp_batch[pair_cols, :]  # (naopair, mo_batch_size)

            dm_sparse[pair_diag,:] *= 0.5 

            log.timer(' dm_sparse', *cpu)

            T_ii[auxp0:auxp1, mop0:mop1] = contract('ka,kP->Pa', dm_sparse, eri3c_batch)
            cpu = log.init_timer()

        log.timer(' T_ii', *cpu0)    

    K_diag = contract('Pi,Pa->ia', T_ii, T_aa)   
    log.timer(' build K_diag', *cpu00)
    return K_diag


def gen_iajb_MVP_bdiv(int3c2e_opt, aux_coeff_lower_inv_eri2c, C_p, C_q,  single, krylov_in_ram=False, log=None):  
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

    C_pT = C_p.T
    C_qT = C_q.T

    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=False)
    naopair = len(ao_pair_mapping)
    rows, cols = divmod(ao_pair_mapping, nao_orig)
    QP_= aux_coeff_lower_inv_eri2c.dot(aux_coeff_lower_inv_eri2c.T)
    QP = cuasarray(QP_, dtype=cp_int3c_dtype)
    del QP_
    release_memory()
    pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()
    nauxao = int3c2e_opt.aux_coeff.shape[0]
    log.info(gpu_mem_info('before def iajb_MVP'))

    def iajb_MVP(X, factor=2, out=None):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P X_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            X   (cupy.ndarray): Input tensor of shape (m, n_occ, n_vir).
            out (cupy.ndarray): output holder of shape (m, n_occ, n_vir).
            results are accumulated in out if provided.

        Returns:
            iajb_X (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = X.shape
        if out is None: 
            out = cp.zeros_like(X)

        log.info(gpu_mem_info('  iajb_X before build dm_sparse')) 

        # dms = contract('ui,mia->mua', C_p, X)
        # dms = contract('va,mua->muv', C_q, dms)

        # pair_rows, pair_cols, pair_diag = int3c2e_opt.orbital_pair_nonzero_indices()
        # dm_sparse  = dms[:, pair_rows, pair_cols]
        # dm_sparse += dms[:, pair_cols, pair_rows]
        # dm_sparse[:,pair_diag] *= 0.5

        cpu0 = log.init_timer()
        ''' build dm_sparse '''
        dm_sparse = cp.empty((n_state, len(pair_rows)),dtype=cp_int3c_dtype)
        log.info( f'     dm_sparse {dm_sparse.nbytes/1e9:.2f} GB')

        X_buffer    = cp.empty((n_occ, n_vir), dtype=cp_int3c_dtype)
        temp_buffer = cp.empty((C_p.shape[0], X.shape[2]), dtype=cp_int3c_dtype)
        dms_buffer  = cp.empty((C_p.shape[0], C_qT.shape[1]), dtype=cp_int3c_dtype) # uv
            
        for i in range(n_state):
            X_buffer[:,:] = cuasarray(X[i,:,:])
            cp.dot(C_p, X_buffer, out=temp_buffer)
            cp.dot(temp_buffer, C_qT, out=dms_buffer)
            dm_sparse[i,:]  = dms_buffer[pair_rows, pair_cols]
            dm_sparse[i,:] += dms_buffer[pair_cols, pair_rows]
            release_memory()

        # if krylov_in_ram:
        #     del X
        #     release_memory()
        #     log.info(gpu_mem_info('  iajb_X after del X'))

        del X_buffer, temp_buffer, dms_buffer
        release_memory()
        cp.cuda.Stream.null.synchronize()
        log.info(gpu_mem_info('  iajb_X after del buffers'))

        dm_sparse[:,pair_diag] *= 0.5
        log.timer(' dm_sparse', *cpu0)

        aux_offset = 0

        ''' (z|Q)X_mz mQ '''
        T_right = cp.empty((n_state, nauxao),dtype=cp_int3c_dtype)
        log.info( f'     T_right {T_right.nbytes/1e6:.2f} MB')

        # cpu0 = log.init_timer()

        available_gpu_memory = get_avail_gpumem()
        
        bytes_per_aux = ( naopair*3 + n_state) * cp_int3c_dtype.itemsize  
        batch_size = min(nauxao, max(16, int(available_gpu_memory * 0.7 // bytes_per_aux)) )
        log.info(f'   iajb_MVP: int3c2e_bdiv_generator batch_size first pass: {batch_size}')

        for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size):
            # cpu1 = log.init_timer()
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='F')
            # del eri3c_batch_
            release_memory()

            aopair, aux_batch_size = eri3c_batch.shape
            tmp = dm_sparse.dot(eri3c_batch)   
            T_right[:, aux_offset:aux_offset + aux_batch_size] = tmp

            del eri3c_batch, tmp
            release_memory()
            aux_offset += aux_batch_size
            # log.timer('eri3c_batch', *cpu1)

        del dm_sparse
        release_memory()
        cp.cuda.Stream.null.synchronize()
        # log.timer('T_right', *cpu0)
            
        T_right = cp.dot(T_right, QP, out=T_right) #mP,PQ->mQ   PQ symmetry
        T_right = cuasarray(T_right.T, order='F') #Pm
        #(z|P) @ (Pm),slice over P
        T_left = cp.zeros((len(ao_pair_mapping), n_state),dtype=cp_int3c_dtype)
        log.info( f'     T_left {T_left.nbytes/1e9:.2f} GB')
        

        # (z|P)mP -> zm  i.e.(uv|m)
        available_gpu_memory = get_avail_gpumem()
        bytes_per_aux = ( naopair*3  + n_state) * cp_int3c_dtype.itemsize  
        batch_size = min(nauxao, max(16, int(available_gpu_memory * 0.7 // bytes_per_aux)) )
        log.info(f'   iajb_MVP: int3c2e_bdiv_generator batch_size second pass: {batch_size}')
        p1 = 0
        for eri3c_batch in int3c2e_opt.int3c2e_bdiv_generator(batch_size=batch_size):
            eri3c_batch = int3c2e_opt.orbital_pair_cart2sph(eri3c_batch, inplace=True)
            eri3c_batch = cuasarray(eri3c_batch, dtype=cp_int3c_dtype, order='C')
            # del eri3c_batch_
            release_memory()

            p0, p1 = p1, p1 + eri3c_batch.shape[1]
            T_left = contract('zP,Pm->zm',eri3c_batch, T_right[p0:p1,:], alpha=1, beta=1, out=T_left)  #(uv|P) @ (Pm) -> (uv|m)
            del eri3c_batch
            release_memory()
  
        del T_right
        release_memory()

        # Cui Cva (uv|m) -> mia
        J_buffer = cp.empty((nao_orig, nao_orig), dtype=cp_int3c_dtype)
        temp_buffer = cp.empty((C_pT.shape[0], J_buffer.shape[1]), dtype=cp_int3c_dtype)
        # out_slice = cp.empty((C_pT.shape[0], C_q.shape[1]), dtype=cp_int3c_dtype)

        for i in range(n_state):
            #(uv|m)
            J_buffer.fill(0)                                    
            J_buffer[rows, cols] = T_left[:, i]
            J_buffer[cols, rows] = T_left[:, i]
            temp_buffer = cp.dot(C_pT, J_buffer, out=temp_buffer) # iu,uv->iv
            contract('iu,ua->ia',temp_buffer, C_q, alpha=factor, beta=1, out=out[i, :, :])
            # cp.dot(temp_buffer, C_q, out=out_slice)
            # out_slice *= factor
            # out[i, :, :] += out_slice
        del T_left, temp_buffer
        release_memory()
        log.info(gpu_mem_info('  iajb_MVP done')) 
        return out
        
    log.info(gpu_mem_info('after generate iajb_MVP'))
    return iajb_MVP

def gen_iajb_MVP_Tpq(T_ia, log=None):
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

    def iajb_MVP(V, factor=2, out=None):
        '''
        Optimized calculation of (ia|jb) = Σ_Pjb (T_left_ia^P T_right_jb^P V_jb^m)
        by chunking along the auxao dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (m, n_occ * n_vir).
            results are accumulated in out if provided.

        Returns:
            iajb_V (cupy.ndarray): Result tensor of shape (m, n_occ, n_vir).
        '''
        # Get the shape of the tensors
        nauxao, n_occ, n_vir = T_ia.shape
        n_state, n_occ, n_vir = V.shape
        # Initialize result tensor
        if out is None:
            out = cp.zeros_like(V)

        # 1 denotes one auxao, we are slucing the auxao dimension.
        n_Tia_chunk = 1 * n_occ * n_vir
        n_TjbVjb_chunk = 1 * n_state
        n_iajb_V_chunk = n_state * n_occ * n_vir 

        estimated_chunk_size_bytes = (n_Tia_chunk + n_TjbVjb_chunk + n_iajb_V_chunk) * T_ia.itemsize 

        # Estimate the optimal chunk size based on available GPU memory
        aux_batch_size = int(get_avail_gpumem() * 0.8 // estimated_chunk_size_bytes)

        # Ensure the chunk size is at least 1 and doesn't exceed the total number of auxao
        aux_batch_size = max(1, min(nauxao, aux_batch_size))

        # Iterate over chunks of the auxao dimension
        for aux_start in range(0, nauxao, aux_batch_size):
            aux_end = min(aux_start + aux_batch_size, nauxao)
            T_ia_slice = T_ia[aux_start:aux_end, :, :]
            Tjb_chunk = cuasarray(T_ia_slice)   
            del T_ia_slice

            Tjb_Vjb_chunk = contract("Pjb,mjb->Pm", Tjb_chunk, V)

            Tia_chunk = Tjb_chunk  # Shape: (aux_range, n_occ, n_vir)
            out = contract("Pia,Pm->mia", Tia_chunk, Tjb_Vjb_chunk, factor, 1, out=out)

            # Release intermediate variables and clean up memory, must!
            del Tjb_chunk, Tia_chunk, Tjb_Vjb_chunk
            release_memory()

        return out

    return iajb_MVP


def gen_ijab_MVP_Tpq(T_ij, T_ab, log=None):
    '''
    (ij|ab)V = Σ_Pjb (T_ij^P T_ab^P V_jb^m)
             = Σ_P [T_ij^P Σ_jb(T_ab^P V_jb^m)]
    V in shape (m, n_occ * n_vir)
    '''
    # @profile
    def ijab_MVP(X, a_x, out=None):
        '''
        Optimized calculation of (ij|ab) = Σ_Pjb (T_ij^P T_ab^P X_jb^m)
        by chunking along the P (nauxao) dimension for both T_ij and T_ab,
        uploading chunks to GPU to reduce memory usage.

        Parameters:
            X (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            a_x (float): Scaling factor.
            out (cupy.ndarray, optional): Output tensor of shape (n_state, n_occ, n_vir).

        Returns:
            ijab_X (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        nauxao, n_occ, n_occ = T_ij.shape
        nauxao, n_vir, n_vir = T_ab.shape  # Dimensions of T_ab
        n_state, n_occ, n_vir = X.shape    # Dimensions of X

        # Initialize result tensor
        if out is None:
            out = cp.zeros_like(X)

        # Get free memory and dynamically calculate chunk size
        available_gpu_memory = get_avail_gpumem()

        log.info(gpu_mem_info('          ijab_MVP start'))

        # Memory estimation for one P index
        n_T_ab_chunk = 1 * n_vir * n_vir  # T_ab chunk: (1, n_vir, n_vir)
        n_T_ij_chunk = 1 * n_occ * n_occ  # T_ij chunk: (1, n_occ, n_occ)
        n_T_ab_chunk_X = 1 * n_state * n_occ * n_vir  # T_ab_X chunk: (1, n_state, n_occ)
        n_ijab_chunk_X = n_state * n_occ * n_vir  # Full output size (accumulated)

        bytes_per_P = max(n_T_ab_chunk + n_T_ab_chunk_X,  n_T_ij_chunk + n_T_ab_chunk_X ) * T_ab.itemsize
        # log.info(f'bytes_per_P {bytes_per_P}')
        P_chunk_size = min(nauxao, max(1, int((available_gpu_memory * 0.8 - n_ijab_chunk_X * T_ab.itemsize) // bytes_per_P)))
        log.info(f'    ijab with Tij Tab, P_chunk_size = {P_chunk_size}')
        # Iterate over chunks of the P (nauxao) dimension
        for P_start in range(0, nauxao, P_chunk_size):
            P_end = min(P_start + P_chunk_size, nauxao)

            # log.info(gpu_mem_info(f'  ijab {P_start,P_end}'))
            # Extract and upload the corresponding chunks of T_ab and T_ij to GPU
            T_ab_slice = T_ab[P_start:P_end, :, :]
            T_ab_chunk = cuasarray(T_ab_slice)  # Shape: (P_chunk_size, n_vir, n_vir)
            del T_ab_slice
            gc.collect()
            # Compute T_ab_X for the current chunk
            T_ab_chunk_X = contract("Pab,mjb->Pamj", T_ab_chunk, X)
            del T_ab_chunk
            release_memory()

            T_ij_slice = T_ij[P_start:P_end, :, :]
            T_ij_chunk = cuasarray(T_ij_slice)  # Shape: (P_chunk_size, n_occ, n_occ)
            del T_ij_slice
            gc.collect()

            # Compute ijab_X for the current chunk and accumulate
            ijab_chunk_X =  contract("Pij,Pamj->mia", T_ij_chunk, T_ab_chunk_X)
            del T_ij_chunk, T_ab_chunk_X
            release_memory()
            ijab_chunk_X *= a_x
            out -= ijab_chunk_X
            del ijab_chunk_X
            release_memory()
            # Release intermediate variables and clean up memory
  
        log.info(gpu_mem_info('          ijab_MVP done'))
        return out

    return ijab_MVP

def gen_ijab_MVP_eri3c(eri3c, int3c2e_opt, C_p, C_q, single, log=None):
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

    size_p = C_p.shape[1]
    size_q = C_q.shape[1]

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=int3c2e_opt.mol.cart)
    naopair = len(ao_pair_mapping)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    C_full = cp.empty((nao_orig, size_p+size_q), dtype=C_p.dtype)
    C_full[:,:size_p] = C_p
    C_full[:,size_p:] = C_q
    # C_full = cp.hstack((C_p, C_q))
    log.info(f'C_full.flags {C_full.flags}')
    log.info(f'C_full.dtype {C_full.dtype}')
    log.info(f'C_full.shape {C_full.shape}')


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
        
        available_gpu_memory = get_avail_gpumem()

        mem_per_aux = (naopair + nao_orig*nao_orig + nao_orig*(size_q + size_p))*cp_int3c_dtype.itemsize
        batch_size = min(nauxao, max(1, int(available_gpu_memory * 0.8 // mem_per_aux)))
        log.info(f'     ijab batch_size for aux dimension {batch_size}')
        log.info(f'     ijab eri3c_unzip_batch {batch_size*nao_orig*nao_orig*cp_int3c_dtype.itemsize/1e9:.1f} GB')

        # batch_size = 24
        log.info(f'     TFLOPs Pvu,ur->Pvr: {int(2*batch_size*nao_orig*nao_orig*(n_vir+n_occ)*10**-12)}')
        # log.info(f'     TFLOPs uvP,ui->ivP: {int(2*nao_orig*n_occ*nao_orig*batch_size*10**-12)}')
        # logger.TIMER_LEVEL = 5
        for p0 in range(0, nauxao, batch_size):
            p1 = min(p0+batch_size, nauxao)
            # log.info(f' ijab slice over nauxao p0, p1 {p0,p1}/ {nauxao}')

            cpu0 = log.init_timer()

            cpu = log.init_timer()
            eri3c_batch = cuasarray(eri3c[:, p0:p1])
            eri3c_batch = cp.transpose(eri3c_batch, axes=(1, 0))
            log.timer(' cuasarray(eri3c[:, p0:p1])', *cpu)

            cpu = log.init_timer()
            eri3c_unzip_batch = cp.zeros((p1-p0, nao_orig*nao_orig), dtype=cp_int3c_dtype)
            log.timer(' eri3c_unzip_batch', *cpu)

            cpu = log.init_timer()
            eri3c_unzip_batch[:, ao_pair_mapping] = eri3c_batch
            eri3c_unzip_batch[:, cols*nao_orig+rows] = eri3c_batch
            # log.timer(' eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch', *cpu)  # 0.01s
            
            eri3c_unzip_batch = eri3c_unzip_batch.reshape(p1-p0, nao_orig, nao_orig)
            # log.info(f' eri3c_unzip_batch.shape {eri3c_unzip_batch.shape}')

            cpu = log.init_timer()
            T_rv = eri3c_unzip_batch.dot(C_full)  # Pva
            # cp.cuda.Stream.null.synchronize()
            log.timer(' Pvu,ur->Pvr', *cpu)
            cpu = log.init_timer()

            T_iv = T_rv[:,:,:n_occ]
            T_av = T_rv[:,:,n_occ:]

            del eri3c_unzip_batch
            release_memory()

            T_ab = contract('Pva,vb->Pab', T_av, C_q)
            log.timer(' Pav,vb->Pab', *cpu)
            cpu = log.init_timer()

            del T_av
            release_memory()

            T_ij = contract('Pvi,vj->Pij', T_iv, C_p)
            log.timer(' ivP,vj->Pij', *cpu)
            cpu = log.init_timer()  

            del T_iv
            release_memory()

            T_ab_V = contract('Pab,mjb->mPja', T_ab, V)
            del T_ab
            release_memory()
            log.timer(' Pab,mjb->mPja', *cpu)
            cpu = log.init_timer()

            out = contract('Pij,mPja->mia', T_ij, T_ab_V, -a_x, 1, out=out)
            del T_ij, T_ab_V
            release_memory()

            log.timer(' Pij,mPja->mia', *cpu)

            log.timer(f' ijab slice over nauxao {p1}/{nauxao}', *cpu0)

        return out

    log.info(gpu_mem_info('after generate ijab_MVP'))    
    return ijab_MVP



def gen_ijab_MVP_eri3c1(eri3c, int3c2e_opt, C_p, C_q, single, log=None):
    nao, nao_orig = int3c2e_opt.coeff.shape

    size_p = C_p.shape[1]
    size_q = C_q.shape[1]

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=int3c2e_opt.mol.cart)
    naopair = len(ao_pair_mapping)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    C_full = cp.empty((nao_orig, size_p+size_q), dtype=C_p.dtype)
    C_full[:,:size_p] = C_p
    C_full[:,size_p:] = C_q
    # C_full = cp.hstack((C_p, C_q))
    log.info(f'C_full.flags {C_full.flags}')
    log.info(f'C_full.dtype {C_full.dtype}')
    log.info(f'C_full.shape {C_full.shape}')

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
            out = cp.zeros_like(V)  # Assume out is on GPU 0 initially; will be shared via lock
        nauxao = int3c2e_opt.aux_coeff.shape[0]
        

        # Lock for protecting out writes
        # out_lock = threading.Lock()
        
        # Function for single GPU worker
        def gpu_worker(device_id, p_ranges):
            # nonlocal out
            with cp.cuda.Device(device_id):
                # Replicate data to this GPU
                cpu00 = log.init_timer()
                C_full_gpu = cuasarray(C_full)  # Transfer to this GPU
                C_p_gpu = cuasarray(C_p)
                C_q_gpu = cuasarray(C_q)
                V_gpu = cuasarray(V)  # Transfer V to this GPU
                out_gpu = cp.zeros_like(V_gpu)  # Local out on this GPU, later add to shared
                ao_pair_mapping_gpu = cuasarray(ao_pair_mapping)
                cols_gpu = cuasarray(cols)
                rows_gpu = cuasarray(rows)

                start_p, end_p = p_ranges[0]
                log.timer(f' ijab slice on GPU {device_id} initiate {start_p}-{end_p}', *cpu00)

                available_gpu_memory = get_avail_gpumem()  
                
                log.info(f'     device {device_id} available_gpu_memory {available_gpu_memory} GB')

                mem_per_aux = (naopair + nao_orig*nao_orig + nao_orig*(size_q + size_p)) * cp_int3c_dtype.itemsize
                batch_size = min(end_p - start_p, max(1, int(available_gpu_memory * 0.8 // mem_per_aux)))  # Adjust per slice if needed
                batch_size = (batch_size // 8) * 8
                # batch_size = 40
                log.info(f' GPU {device_id} batch_size {batch_size}, mem_per_aux {available_gpu_memory, mem_per_aux}')
                for p0 in range(start_p, end_p, batch_size):
                    p1 = min(p0 + batch_size, end_p)
                    # log.info(f'     device {device_id} p0:p1 {p0,p1}')
                    
                    # cpu0 = log.init_timer()
                    
                    # Transfer eri slice to this GPU
                    cpu = log.init_timer()
                    eri3c_batch = cuasarray(eri3c[:, p0:p1])  # Assume eri3c is CPU numpy
                    eri3c_batch = cp.transpose(eri3c_batch, axes=(1, 0))
                    log.timer(f' GPU {device_id} cuasarray(eri3c[:, p0:p1])', *cpu)
                    
                    # cpu = log.init_timer()
                    eri3c_unzip_batch = cp.zeros((p1 - p0, nao_orig * nao_orig), dtype=cp_int3c_dtype)
                    # log.timer(f' GPU {device_id} eri3c_unzip_batch', *cpu)

                    eri3c_unzip_batch[:, ao_pair_mapping_gpu] = eri3c_batch
                    eri3c_unzip_batch[:, cols_gpu * nao_orig + rows_gpu] = eri3c_batch
                    eri3c_unzip_batch = eri3c_unzip_batch.reshape(p1 - p0, nao_orig, nao_orig)
                    
                    # cpu = log.init_timer()
                    T_rv = eri3c_unzip_batch.dot(C_full_gpu)  # Pva
                    # log.timer(f' GPU {device_id} Pvu,ur->Pvr', *cpu)

                    T_iv = T_rv[:, :, :n_occ]
                    T_av = T_rv[:, :, n_occ:]
                    
                    # del eri3c_unzip_batch
                    # release_memory()
                    
                    # cpu = log.init_timer()
                    T_ab = contract('Pva,vb->Pab', T_av, C_q_gpu)
                    # del T_av
                    # release_memory()
                    
                
                    T_ij = contract('Pvi,vj->Pij', T_iv, C_p_gpu)
                    # del T_iv
                    # release_memory()
                    # log.timer(f' GPU {device_id} Pav,vb->Pab & Pvi,vj->Pij', *cpu)
                    
                    T_ab_V = contract('Pab,mjb->mPja', T_ab, V_gpu)
                    # del T_ab
                    # release_memory()
                    
                    out_gpu = contract('Pij,mPja->mia', T_ij, T_ab_V, a_x, 1 , out=out_gpu)
                    
                    # del T_ij, T_ab_V
                    # release_memory()
                    
                    # log.timer(f' ijab slice on GPU {device_id} over nauxao {p1}/{end_p}', *cpu0)
                log.timer(f' ijab slice on GPU {device_id} total {start_p}-{end_p}', *cpu00)
            
                return out_gpu

        cut = int(nauxao * 0.25)
        cut = (cut // 8) * 8
        ranges_gpu0 = [(0, cut)]
        ranges_gpu1 = [(cut, 2*cut)]
        ranges_gpu2 = [(2*cut, 3*cut)]
        ranges_gpu3 = [(3*cut, nauxao)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(gpu_worker, 0, ranges_gpu0),
                executor.submit(gpu_worker, 1, ranges_gpu1),
                executor.submit(gpu_worker, 2, ranges_gpu2),
                executor.submit(gpu_worker, 3, ranges_gpu3)
            ]
            # 收集返回的 out_gpu
            out_gpu_results = [f.result() for f in futures]

        with cp.cuda.Device(0):
            for out_gpu in out_gpu_results:
                out -= cp.array(out_gpu)  # Accumulate to GPU 0
        
        return out

    log.info(gpu_mem_info('after generate ijab_MVP'))    
    return ijab_MVP

def gen_ibja_MVP_Tpq(T_ia, log=None):
    '''
    the exchange (ib|ja) in B matrix
    (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
            = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]
    '''
    # def ibja_MVP(V):
    #     T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
    #     ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)
    #     return ibja_V

    def ibja_MVP(V, a_x, out=None):
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

        available_gpu_memory = get_avail_gpumem()



        bytes_per_aux = (n_occ * n_vir * 1 + n_state * n_occ * n_vir ) * T_ia.itemsize 

        batch_size = max(1, int(available_gpu_memory * 0.8 // bytes_per_aux)) 

        # Initialize result tensor
        # ibja_V = cp.empty((n_state, n_occ, n_vir), dtype=T_ia.dtype)
        if out is None:
            out = cp.zeros_like(V)
        # Iterate over chunks of the n_occ dimension
        for p0 in range(0, nauxao, batch_size):
            p1 = min(p0+batch_size, nauxao)

            # Extract the corresponding chunk of T_ia
            T_ib_chunk = T_ia[p0:p1, :, :]  # Shape: (batch_size, n_occ, n_vir)
            T_jb_chunk = T_ib_chunk

            T_ib_V_chunk = contract("Pib,mjb->mPij", T_ib_chunk, V)

            out = contract("Pja,mPij->mia", T_jb_chunk, T_ib_V_chunk, -a_x, 1, out=out)

            release_memory()

        return out

    return ibja_MVP


def gen_ibja_MVP_eri3c(eri3c, int3c2e_opt, C_p, C_q,  single, log=None):
    '''
    the exchange (ib|ja) in B matrix
    (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
            = Σ_P [T_ja^P Σ_jb(T_ib^P V_jb^m)]
    '''
    # def ibja_MVP(V):
    #     T_ib_V = einsum("Pib,mjb->Pimj", T_ia, V)
    #     ibja_V = einsum("Pja,Pimj->mia", T_ia, T_ib_V)
    #     return ibja_V

    nao, nao_orig = int3c2e_opt.coeff.shape

    size_p = C_p.shape[1]
    size_q = C_q.shape[1]

    C_p = int3c2e_opt.sort_orbitals(C_p, axis=[0])
    C_q = int3c2e_opt.sort_orbitals(C_q, axis=[0])
    cp_int3c_dtype = cp.dtype(cp.float32 if single else cp.float64)

    ao_pair_mapping = int3c2e_opt.create_ao_pair_mapping(cart=int3c2e_opt.mol.cart)
    naopair = len(ao_pair_mapping)
    rows, cols = divmod(ao_pair_mapping, nao_orig)

    def ibja_MVP(V, a_x, out=None):
        '''
        Optimized calculation of (ib|ja) = Σ_Pjb (T_ib^P T_ja^P V_jb^m)
        by chunking along the n_occ dimension to reduce memory usage.

        Parameters:
            V (cupy.ndarray): Input tensor of shape (n_state, n_occ, n_vir).
            occ_chunk_size (int): Chunk size for splitting the n_occ dimension.
            results are accumulated in out if provided.

        Returns:
            ibja_V (cupy.ndarray): Result tensor of shape (n_state, n_occ, n_vir).
        '''
        n_state, n_occ, n_vir = V.shape
        if out is None: 
            out = cp.zeros_like(V)
        nauxao = int3c2e_opt.aux_coeff.shape[0]
        
        available_gpu_memory = get_avail_gpumem()

        mem_per_aux = (naopair + nao_orig*nao_orig + nao_orig*size_q + nao_orig*size_p )*cp_int3c_dtype.itemsize
        batch_size = min(nauxao, max(1, int(available_gpu_memory * 0.8 // mem_per_aux)))
        log.info(f'     ibja batch_size for aux dimension {batch_size}')

        # batch_size = 24
        log.info(f'     TFLOPs uvP,ua->avP: {int(2*nao_orig*n_vir*nao_orig*batch_size*10**-12)}')
        log.info(f'     TFLOPs uvP,ui->ivP: {int(2*nao_orig*n_occ*nao_orig*batch_size*10**-12)}')

        for p0 in range(0, nauxao, batch_size):
            p1 = min(p0+batch_size, nauxao)
            # log.info(f' ibja slice over nauxao p0, p1 {p0,p1}/ {nauxao}')

            cpu0 = log.init_timer()

            cpu = log.init_timer()
            eri3c_batch = cuasarray(eri3c[:, p0:p1])
            # log.timer(' cuasarray(eri3c[:, p0:p1])', *cpu)

            cpu = log.init_timer()
            eri3c_unzip_batch = cp.zeros((nao_orig*nao_orig, p1-p0), dtype=cp_int3c_dtype, order='F')
            # log.timer(' eri3c_unzip_batch', *cpu)

            eri3c_unzip_batch[ao_pair_mapping,   :] = eri3c_batch
            eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch
            # log.timer(' eri3c_unzip_batch[cols*nao_orig+rows,:] = eri3c_batch', *cpu)
            cpu = log.init_timer()
            eri3c_unzip_batch = eri3c_unzip_batch.reshape(nao_orig, nao_orig, p1-p0)

            T_jv = contract('uvP,uj->Pjv', eri3c_unzip_batch, C_p)
            T_ja = contract('Pjv,va->Pja', T_jv, C_q)
            del T_jv
            release_memory()
            T_ib = T_ja
            T_ib_V = contract('Pib,mjb->mPij', T_ib, V)

            log.timer(' Pab,mjb->mPja', *cpu)
            cpu = log.init_timer()

            out = contract('Pja,mPij->mia', T_ja, T_ib_V, -a_x, 1, out=out)
            del T_ib, T_ja, T_ib_V
            release_memory()

            log.timer(' Pij,mPja->mia', *cpu)

            log.timer(f' ijab slice over nauxao {p1}/{nauxao}', *cpu0)

        return out

    log.info(gpu_mem_info('after generate ibja'))    
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

    mo_energy = cuasarray(mo_energy)
    mo_coeff = cuasarray(mo_coeff)
    mo_occ = cuasarray(mo_occ)
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
        int3c = cuasarray(int3c)
        int2c2e = cuasarray(int2c2e)
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
        int3c = cuasarray(int3c)
        int2c2e = cuasarray(int2c2e)
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
        # self.eri_tag = None
        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None
        self.RKS = True
        self.UKS = False
        self.mo_coeff = cuasarray(self._scf.mo_coeff, dtype=self.dtype)
        self.build()
        self.kernel()
        return mf_e + self.energies/HARTREE2EV

def get_nto(self,state_id):
    '''only for TDA'''
    orbo = self.C_occ_notrunc
    orbv = self.C_vir_notrunc
    nocc = self.n_occ
    nvir = self.n_vir

    cis_t1 = self.xy[0][state_id-1]

    # TDDFT (X,Y) has X^2-Y^2=1.
    # Renormalizing X (X^2=1) to map it to CIS coefficients
    cis_t1 *= 1. / cp.linalg.norm(cis_t1)

    cis_t1 = cis_t1.reshape(nocc, nvir)

    nto_o, w, nto_vT = cp.linalg.svd(cis_t1)
    nto_v = nto_vT.T
    weights = w**2
    print('weights',weights.shape)

    idx = cp.argmax(abs(nto_o), axis=0)
    nto_o[:,nto_o[idx,cp.arange(nocc)].real<0] *= -1
    idx = cp.argmax(abs(nto_v), axis=0)
    nto_v[:,nto_v[idx,cp.arange(nvir)].real<0] *= -1

    occupied_nto = cp.dot(orbo, nto_o)
    virtual_nto = cp.dot(orbv, nto_v)
    return weights, occupied_nto, virtual_nto


class RisBase(lib.StreamObject):
    def __init__(self, mf,  
                theta: float = 0.2, J_fit: str = 'sp', K_fit: str = 's', excludeHs=False,
                Ktrunc: float = 40.0, full_K_diag: bool = False, a_x: float = None, omega: float = None, 
                alpha: float = None, beta: float = None, conv_tol: float = 1e-3, 
                nstates: int = 5, max_iter: int = 25, spectra: bool = False, 
                out_name: str = '', print_threshold: float = 0.05, gram_schmidt: bool = True, 
                single: bool = True, store_Tpq_J: bool = True, store_Tpq_K: bool = False, tensor_in_ram: bool = False, krylov_in_ram: bool = False, 
                verbose=None, citation=True):
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
        # self.chkfile = mf.chkfile
        self.singlet = True # TODO: add R-T excitation.
        self.exclude_nlc = False # TODO: exclude nlc functional 
        self.xy = None

        self.theta = theta
        self.J_fit = J_fit
        self.K_fit = K_fit

        self.Ktrunc = Ktrunc
        self._excludeHs = excludeHs
        self._full_K_diag = full_K_diag
        self.a_x = a_x
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.conv_tol = conv_tol
        self.nstates = nstates
        self.max_iter = max_iter
        self.mol = mf.mol
        # self.mo_coeff = cuasarray(mf.mo_coeff, dtype=self.dtype)
        self.spectra = spectra
        self.out_name = out_name
        self.print_threshold = print_threshold
        self.gram_schmidt = gram_schmidt

        self.verbose = verbose if verbose else mf.verbose

        self.device = mf.device
        self.converged = None
        self._store_Tpq_J = store_Tpq_J
        self._store_Tpq_K = store_Tpq_K

        self._tensor_in_ram = tensor_in_ram
        self._krylov_in_ram = krylov_in_ram

        gpu4pyscf.lib.logger.WARN = 6
        pyscf.lib.logger.WARN=6

        self.log = gpu4pyscf.lib.logger.new_logger(verbose=self.verbose)
    
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
        if self.mol.cart:
            self.eri_tag = '_cart'
        else:
            self.eri_tag = '_sph'

        self.auxmol_J = None
        self.auxmol_K = None
        self.lower_inv_eri2c_J = None
        self.lower_inv_eri2c_K = None

        self.RKS = True
        self.UKS = False
        self._citation = citation

    def transition_dipole(self):
        '''
        transition dipole u
        '''
        int_r = self.mol.intor_symmetric('int1e_r' + self.eri_tag)
        int_r = cuasarray(int_r, dtype=self.dtype)
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
        int_rxp = cuasarray(int_rxp, dtype=self.dtype)

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
        # log.warn("TDA&TDDFT-ris is still in the experimental stage, and its APIs are subject to change in future releases.")

        log.info(f'nstates: {self.nstates}')
        log.info(f'N atoms:{self.mol.natm}')
        log.info(f'conv_tol: {self.conv_tol}')
        log.info(f'max_iter: {self.max_iter}')
        log.info(f'Ktrunc: {self.Ktrunc}')
        log.info(f'calculate and print UV-vis spectra info: {self.spectra}')
        log.info(gpu_mem_info('  after init of RisBase'))

        if self.spectra:
            log.info(f'spectra files will be written and their name start with: {self.out_name}')

        if self._store_Tpq_J:
            log.info(f'will calc Tia_J. In CPU RAM? {self._tensor_in_ram}')
        else:
            log.info('will calc J on-the-fly')

        if self._store_Tpq_K:
            log.info(f'will calc Tia_J (if full TDDFT) Tij_K Tab_K. In CPU RAM? {self._tensor_in_ram}')
        else:
            log.info('will store K eri3c ')

        if self.a_x or self.omega or self.alpha or self.beta:
            ''' user wants to define some XC parameters '''
            if self.a_x:
                if self.a_x == 0:
                    log.info('use pure XC functional, a_x = 0')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info(f'use hybrid XC functional, a_x = {self.a_x}')
                    if self.single:
                        self.a_x = cp.float32(self.a_x)
                elif self.a_x == 1:
                    log.info('use HF, a_x = 1')
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
            log.info(f'auto detect functional: {self._scf.xc}')

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
                    log.info('use pure XC functional, a_x = 0')
                elif self.a_x > 0 and self.a_x < 1:
                    log.info(f'use hybrid XC functional, a_x = {self.a_x}')
                elif self.a_x == 1:
                    log.info('use HF, a_x = 1')
                else:
                    log.info('a_x > 1, weird')

        log.info(f'omega: {self.omega}')
        log.info(f'alpha: {self.alpha}')
        log.info(f'beta: {self.beta}')
        log.info(f'a_x: {self.a_x}')
        log.info(f'gram_schmidt: {self.gram_schmidt}')
        log.info(f'single: {self.single}')

        if self.J_fit == self.K_fit:
            log.info(f'use same J and K fitting basis: {self.J_fit}')
        else:
            log.info(f'use different J and K fitting basis: J with {self.J_fit} and K with {self.K_fit}')


        log.info(f'cartesian or spherical electron integral: {self.eri_tag}')

        log.info(gpu_mem_info('  before process mo_coeff'))

        if self._scf.mo_coeff.ndim == 2:
            self.RKS = True
            self.UKS = False
            n_occ = int(sum(self._scf.mo_occ>0))
            n_vir = int(sum(self._scf.mo_occ==0))
            self.n_occ = n_occ
            self.n_vir = n_vir

            self.C_occ_notrunc = cuasarray(self._scf.mo_coeff[:,:n_occ], dtype=self.dtype, order='F')
            self.C_vir_notrunc = cuasarray(self._scf.mo_coeff[:,n_occ:], dtype=self.dtype, order='F')
            mo_energy = self._scf.mo_energy
            log.info(f'mo_energy.shape: {mo_energy.shape}')
            
            occ_ene = mo_energy[:n_occ].reshape(n_occ,1)
            vir_ene = mo_energy[n_occ:].reshape(1,n_vir)

            delta_hdiag = cp.repeat(vir_ene, n_occ, axis=0) - cp.repeat(occ_ene, n_vir, axis=1)
            if self.single:
                delta_hdiag = cuasarray(delta_hdiag, dtype=cp.float32)

            self.delta_hdiag = delta_hdiag
            self.hdiag = cuasarray(delta_hdiag.reshape(-1))

            log.info(f'n_occ = {n_occ}, E_HOMO ={occ_ene[-1,0]}')
            log.info(f'n_vir = {n_vir}, E_LOMO ={vir_ene[0,0]}')
            log.info(f'H-L gap = {(vir_ene[0,0] - occ_ene[-1,0])*HARTREE2EV:.2f} eV')     

            if self.Ktrunc > 0:
                log.info(f' MO truncation in K with threshold {self.Ktrunc} eV above HOMO and below LUMO')

                trunc_tol_au = self.Ktrunc/HARTREE2EV

                homo_vir_delta_ene = delta_hdiag[-1,:]
                occ_lumo_delta_ene = delta_hdiag[:,0]

                rest_occ = cp.sum(occ_lumo_delta_ene <= trunc_tol_au)
                rest_vir = cp.sum(homo_vir_delta_ene <= trunc_tol_au)

                # rest_occ = (rest_occ//2)*2
                # rest_vir = (rest_vir//2)*2
                assert rest_occ > 0 
                assert rest_vir > 0 


            elif self.Ktrunc == 0:
                log.info('no MO truncation in K')
                rest_occ = n_occ
                rest_vir = n_vir


            log.info(f'rest_occ = {rest_occ}')
            log.info(f'rest_vir = {rest_vir}')

            self.C_occ_Ktrunc = cuasarray(self._scf.mo_coeff[:,int(n_occ-rest_occ):int(n_occ)], dtype=self.dtype, order='F')
            self.C_vir_Ktrunc = cuasarray(self._scf.mo_coeff[:,int(n_occ):int(n_occ+rest_vir)], dtype=self.dtype, order='F')

            self.rest_occ = rest_occ
            self.rest_vir = rest_vir

        elif self._scf.mo_coeff.ndim == 3:
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
        self.lower_inv_eri2c_J = get_eri2c_inv_lower(self.auxmol_J, omega=0)
        byte_T_ia_J = self.auxmol_J.nao_nr() * self.n_occ * self.n_vir * self.dtype.itemsize
        log.info(f'FYI, storing T_ia_J will take {byte_T_ia_J / 1e6:.0f} MB memory')  


        if self.a_x != 0:

            auxmol_K = get_auxmol(mol=self.mol, theta=self.theta, fitting_basis=self.K_fit, excludeHs=self._excludeHs) 

            log.info(f'n_bf in auxmol_K = {auxmol_K.nao_nr()}')
            self.auxmol_K = auxmol_K

            self.lower_inv_eri2c_K = get_eri2c_inv_lower(auxmol_K, omega=self.omega, alpha=self.alpha, beta=self.beta)

            byte_T_ij_K = auxmol_K.nao_nr() * self.rest_occ **2 * self.dtype.itemsize
            byte_T_ab_K = auxmol_K.nao_nr() * self.rest_vir **2 * self.dtype.itemsize
            log.info(f'T_ij_K will take {byte_T_ij_K / 1e6:.0f} MB memory')
            log.info(f'T_ab_K will take {byte_T_ab_K / 1e6:.0f} MB memory')

            byte_T_ia_K = auxmol_K.nao_nr() * self.rest_occ * self.rest_vir * self.dtype.itemsize
            log.info(f'(if full TDDFT) T_ia_K will take {byte_T_ia_K / 1e6:.0f} MB memory')

        log.info(gpu_mem_info('  built ris obj'))
        self.log = log
        

    def get_T_J(self):
        log = self.log
        log.info('==================== RIJ ====================')
        cpu0 = log.init_timer() 

        T_ia_J = get_Tpq(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J, 
                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, calc="J", omega=0, 
                        in_ram=self._tensor_in_ram, single=self.single, log=log)

        log.timer('build T_ia_J', *cpu0)
        log.info(gpu_mem_info('after T_ia_J'))
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
        log.info(gpu_mem_info('after T_ij_K T_ab_K'))
        return T_ij_K, T_ab_K
    
    def get_3T_K(self):
        log = self.log
        log.info('==================== RIK ====================')
        cpu1 = log.init_timer()
        T_ia_K, T_ij_K, T_ab_K = get_Tpq(mol=self.mol, auxmol=self.auxmol_K, lower_inv_eri2c=self.lower_inv_eri2c_K, 
                                C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, calc='JK', 
                                omega=self.omega, alpha=self.alpha,beta=self.beta,
                                in_ram=self._tensor_in_ram, single=self.single,log=log)

        log.timer('T_ia_K T_ij_K T_ab_K', *cpu1)
        log.info(gpu_mem_info('after T_ia_K T_ij_K T_ab_K'))
        return T_ia_K, T_ij_K, T_ab_K

    def get_eri3c_K(self):
        log = self.log
        log.info('==================== eri3c K ====================')
        log.info('   eri3c = uvP.dot(Lower)')
        cpu0 = log.init_timer()
        if not hasattr(self, "eri3c_K"):
            self.eri3c_K, self.int3c2e_opt_K = get_eri3c_bdiv(mol=self.mol, auxmol=self.auxmol_K, 
                                            lower_inv_eri2c=self.lower_inv_eri2c_K, 
                                            omega=self.omega, alpha=self.alpha,beta=self.beta, 
                                            in_ram=self._tensor_in_ram, single=self.single, log=log)

        log.timer(' build eri3c K', *cpu0)
        log.info(gpu_mem_info('after eri3c'))
        return self.eri3c_K, self.int3c2e_opt_K

    def get_int3c2e_J(self):
        log = self.log
        log.info('==================== int3c2e for iajb ====================')
        cpu0 = log.init_timer()

        int3c2e_opt_J, aux_coeff_lower_inv_eri2c_J = get_int3c2e_eri2c(mol=self.mol, auxmol=self.auxmol_J, lower_inv_eri2c=self.lower_inv_eri2c_J, 
                                                                        in_ram=self._tensor_in_ram, single=self.single, log=log)
        log.timer('build int3c2e for iajb', *cpu0)
        log.info(gpu_mem_info('after build int3c2e for iajb'))
        return int3c2e_opt_J, aux_coeff_lower_inv_eri2c_J

    def get_hKdiag(self):
        log = self.log
        log.info('==================== build full K diag ====================')
        cpu0 = log.init_timer()
        if not hasattr(self, "K_diag"):
            self.get_eri3c_K()
            K_diag = gen_K_diag(eri3c=self.eri3c_K, int3c2e_opt=self.int3c2e_opt_K, C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)
            self.K_diag = K_diag

        DEBUG = False
        if DEBUG:
            T_ij_K, T_ab_K = self.get_2T_K()

            T_ij_K_diag = cp.diagonal(T_ij_K, axis1=1, axis2=2)  # shape: (P, i)
            T_ab_K_diag = cp.diagonal(T_ab_K, axis1=1, axis2=2)  # shape: (P, a)
            true_K_diag = contract('Pi,Pa->ia', T_ij_K_diag, T_ab_K_diag)
            rest_occ, rest_vir = true_K_diag.shape
            log.info(f'rest_occ, rest_vir = {rest_occ, rest_vir}')
            diff = true_K_diag - K_diag[self.n_occ - self.rest_occ:, :self.rest_vir]
            log.info(f'true_K_diag - K_diag norm {cp.linalg.norm(diff)}')
            assert cp.linalg.norm(diff) < 1e-3

        K_diag[self.n_occ - self.rest_occ:, :self.rest_vir] = 0
        self.hKdiag = self.hdiag - self.a_x * K_diag.reshape(-1)
        log.timer('build full K diag', *cpu0)
        log.info(gpu_mem_info('after build build full K diag'))
        return self.hKdiag

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
        log.info('TDA-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_RKS_TDA_hybrid_MVP(self):
        ''' TDA RKS hybrid '''
        log = self.log

        if self._store_Tpq_J:
            T_ia_J = self.get_T_J()
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)
        else:
            int3c2e_opt_J, aux_coeff_lower_inv_eri2c_J = self.get_int3c2e_J()
            iajb_MVP = gen_iajb_MVP_bdiv(int3c2e_opt=int3c2e_opt_J, aux_coeff_lower_inv_eri2c=aux_coeff_lower_inv_eri2c_J, 
                                        krylov_in_ram=self._krylov_in_ram,
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)

        if self._store_Tpq_K:
            T_ij_K, T_ab_K = self.get_2T_K()
            ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K, log=log)
        else:
            eri3c_K, int3c2e_opt_K = self.get_eri3c_K()
            ijab_MVP = gen_ijab_MVP_eri3c(eri3c=eri3c_K, int3c2e_opt=int3c2e_opt_K, 
                                        C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, log=log, single=self.single)

        if self._full_K_diag:
            _hdiag = self.get_hKdiag()
        else:
            _hdiag = self.hdiag

        hdiag_MVP = gen_hdiag_MVP(hdiag=_hdiag , n_occ=self.n_occ, n_vir=self.n_vir)

        def RKS_TDA_hybrid_MVP(X):
            ''' hybrid or range-sparated hybrid, a_x > 0
                return AX
                AX = hdiag_MVP(X) + 2*iajb_MVP(X) - a_x*ijab_MVP(X)
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
            
            cpu0 = log.init_timer()
            log.info(gpu_mem_info('       TDA MVP before hdiag_MVP')) 
            out = hdiag_MVP(X)

            log.info(gpu_mem_info('       TDA MVP before ijab'))   

            cpu1 = log.init_timer()
            X_trunc = cuasarray(X[:,int(self.n_occ-self.rest_occ):,:int(self.rest_vir)], dtype=self.dtype)
            ijab_MVP(X_trunc, a_x=self.a_x, out=out[:,self.n_occ-self.rest_occ:,:self.rest_vir])
            del X_trunc
            
            log.timer('--ijab_MVP', *cpu1)


            iajb_MVP(X, out=out) 

            gc.collect()
            cp.cuda.Stream.null.synchronize()
            release_memory()
            
            log.timer('--iajb_MVP', *cpu0)
            log.info(gpu_mem_info('       TDA MVP after iajb'))  

            out = out.reshape(nstates, self.n_occ*self.n_vir)
            return out

        return RKS_TDA_hybrid_MVP, self.hdiag
            
    
    ''' ===========  RKS pure =========== '''
    def get_RKS_TDA_pure_MVP(self):
        '''hybrid RKS TDA'''
        log = self.log  
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        if self._store_Tpq_J:
            T_ia_J = self.get_T_J()
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)
        else:
            int3c2e_opt_J, aux_coeff_lower_inv_eri2c_J = self.get_int3c2e_J()
            iajb_MVP = gen_iajb_MVP_bdiv(int3c2e_opt=int3c2e_opt_J, aux_coeff_lower_inv_eri2c=aux_coeff_lower_inv_eri2c_J, 
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)

        def RKS_TDA_pure_MVP(X):
            ''' pure functional, a_x = 0
                return AX
                AV = hdiag_MVP(V) + 2*iajb_MVP(V)
            '''
            nstates = X.shape[0]
            X = X.reshape(nstates, self.n_occ, self.n_vir)
            out = hdiag_MVP(X) 
            cpu0 = log.init_timer()
            # AX += 2 * iajb_MVP(X) 
            iajb_MVP(X, out=out) 
            log.timer('--iajb_MVP', *cpu0)
            out = out.reshape(nstates, self.n_occ*self.n_vir)
            return out

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
        if self._krylov_in_ram or self._tensor_in_ram:
            if hasattr(self, '_scf'):
                del self._scf
            if hasattr(self, 'mo_coeff'):
                del self._scf.mo_coeff
            gc.collect()
            release_memory()
        converged, energies, X = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag, n_states=self.nstates, problem_type='eigenvalue',
                                              conv_tol=self.conv_tol, max_iter=self.max_iter, gram_schmidt=self.gram_schmidt,
                                              single=self.single, in_ram=self._krylov_in_ram, verbose=log)

        self.converged = converged
        log.debug(f'check orthonormality of X: {cp.linalg.norm(cp.dot(X, X.T) - cp.eye(X.shape[0])):.2e}')

        cpu0 = log.init_timer()
        P = self.transition_dipole()
        log.timer('transition_dipole', *cpu0)
        cpu0 = log.init_timer()
        mdpol = self.transition_magnetic_dipole()
        log.timer('transition_magnetic_dipole', *cpu0)

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X, Y=None, 
                                                 P=P, mdpol=mdpol,
                                                 name=self.out_name+'_TDA_ris' if self.out_name else 'TDA_ris', 
                                                 RKS=self.RKS, spectra=self.spectra,
                                                 print_threshold = self.print_threshold, n_occ=self.n_occ, n_vir=self.n_vir, verbose=self.verbose)
        
        energies = energies*HARTREE2EV
        if self._citation:
            log.info(CITATION_INFO)

        self.energies = energies
        self.xy = (X, None)
        self.oscillator_strength = oscillator_strength
        self.rotatory_strength = rotatory_strength

        weights, occupied_nto, virtual_nto = get_nto(self, state_id=1)
        
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
        log.info('TDDFT-ris is initialized')

    ''' ===========  RKS hybrid =========== '''
    def gen_RKS_TDDFT_hybrid_MVP(self):
        '''hybrid RKS TDDFT'''
        log = self.log   

        log.info(gpu_mem_info('before T_ia_J'))

        if self._full_K_diag:
            _hdiag = self.get_hKdiag()
        else:
            _hdiag = self.hdiag
        hdiag_MVP = gen_hdiag_MVP(hdiag=_hdiag, n_occ=self.n_occ, n_vir=self.n_vir)


        if self._store_Tpq_J:
            T_ia_J = self.get_T_J()
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J, log=log)
        else:
            int3c2e_opt_J, aux_coeff_lower_inv_eri2c_J = self.get_int3c2e_J()
            iajb_MVP = gen_iajb_MVP_bdiv(int3c2e_opt=int3c2e_opt_J, aux_coeff_lower_inv_eri2c=aux_coeff_lower_inv_eri2c_J, 
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)


        if self._store_Tpq_K:
            T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()
            ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K, log=log)
            ibja_MVP = gen_ibja_MVP_Tpq(T_ia=T_ia_K, log=log)

        else:
            eri3c_K, int3c2e_opt_K = self.get_eri3c_K()
            ijab_MVP = gen_ijab_MVP_eri3c(eri3c=eri3c_K, int3c2e_opt=int3c2e_opt_K, C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, log=log, single=self.single)
            ibja_MVP = gen_ibja_MVP_eri3c(eri3c=eri3c_K, int3c2e_opt=int3c2e_opt_K, C_p=self.C_occ_Ktrunc, C_q=self.C_vir_Ktrunc, log=log, single=self.single)

        a_x = self.a_x
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

            # ApB_XpY += 4*iajb_MVP(XpY)

            # ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir]) 

            # ApB_XpY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir])

            # AmB_XmY = hdiag_MVP(XmY) 
            # AmB_XmY[:,n_occ-rest_occ:,:rest_vir] -= self.a_x*ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir]) 

            # AmB_XmY[:,n_occ-rest_occ:,:rest_vir] += self.a_x*ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir])

            iajb_MVP(XpY, factor=4, out=ApB_XpY)

            ijab_MVP(XpY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=ApB_XpY[:,n_occ-rest_occ:,:rest_vir])

            ibja_MVP(XpY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=ApB_XpY[:,n_occ-rest_occ:,:rest_vir])

            AmB_XmY = hdiag_MVP(XmY) 

            ijab_MVP(XmY[:,n_occ-rest_occ:,:rest_vir], a_x=a_x, out=AmB_XmY[:,n_occ-rest_occ:,:rest_vir])
 
            ibja_MVP(XmY[:,n_occ-rest_occ:,:rest_vir], a_x=-a_x, out=AmB_XmY[:,n_occ-rest_occ:,:rest_vir])


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
        

        log.info(gpu_mem_info('before T_ia_J'))

        hdiag_sq = self.hdiag**2
        hdiag_sqrt_MVP = gen_hdiag_MVP(hdiag=self.hdiag**0.5, n_occ=self.n_occ, n_vir=self.n_vir)
        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        if self._store_Tpq_J:
            cpu0 = log.init_timer()
            T_ia_J = self.get_T_J() 
            log.timer('T_ia_J', *cpu0)
            iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J)
            
        else:
            int3c2e_opt_J, aux_coeff_lower_inv_eri2c_J = self.get_int3c2e_J()
            log.info(gpu_mem_info('after int3c2e_opt_J'))
            iajb_MVP = gen_iajb_MVP_bdiv(int3c2e_opt=int3c2e_opt_J, aux_coeff_lower_inv_eri2c=aux_coeff_lower_inv_eri2c_J, 
                                        C_p=self.C_occ_notrunc, C_q=self.C_vir_notrunc, log=log, single=self.single)
        
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
            # ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V) + 4*iajb_MVP(AmB_sqrt_V)
            ApB_AmB_sqrt_V = hdiag_MVP(AmB_sqrt_V)
            iajb_MVP(AmB_sqrt_V, factor=4, out=ApB_AmB_sqrt_V)

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

        cpu0 = log.init_timer()
        P = self.transition_dipole()
        log.timer('transition_dipole', *cpu0)
        cpu0 = log.init_timer()
        mdpol = self.transition_magnetic_dipole()
        log.timer('transition_magnetic_dipole', *cpu0)

        oscillator_strength, rotatory_strength = spectralib.get_spectra(energies=energies, X=X, Y=Y,
                                                    P=P, mdpol=mdpol, 
                                                    name=self.out_name+'_TDDFT_ris' if self.out_name else 'TDA_ris',
                                                    spectra=self.spectra, RKS=self.RKS, print_threshold = self.print_threshold,
                                                    n_occ=self.n_occ, n_vir=self.n_vir, verbose=self.verbose)
        energies = energies*HARTREE2EV
        if self._citation:
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
        log.info('Static Polarizability-ris initialized')

    ''' ===========  RKS hybrid =========== '''
    def get_ApB_hybrid_MVP(self):
        ''' RKS hybrid '''
        log = self.log

        T_ia_J = self.get_T_J()
        
        T_ia_K, T_ij_K, T_ab_K = self.get_3T_K()

        hdiag_MVP = gen_hdiag_MVP(hdiag=self.hdiag, n_occ=self.n_occ, n_vir=self.n_vir)

        iajb_MVP = gen_iajb_MVP_Tpq(T_ia=T_ia_J)
        ijab_MVP = gen_ijab_MVP_Tpq(T_ij=T_ij_K, T_ab=T_ab_K)
        ibja_MVP = gen_ibja_MVP_Tpq(T_ia=T_ia_K)

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

        _, XpY = _krylov_tools.krylov_solver(matrix_vector_product=TDA_MVP,hdiag=hdiag, problem_type='linear',
                                        rhs=-transition_dipole, conv_tol=self.conv_tol, max_iter=self.max_iter, 
                                        gram_schmidt=self.gram_schmidt, single=self.single, verbose=log)

        alpha = cp.dot(XpY, transition_dipole.T)*4

        self.xy = XpY
        self.alpha = alpha

        if self._citation:
            log.info(CITATION_INFO)
        return XpY

